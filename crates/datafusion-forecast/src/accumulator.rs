//! Row buffers per group and the batched forecast at evaluation.
//!
//! [`ForecastGroupsAccumulator`] keeps every group's rows and, on evaluation, forecasts
//! groups of equal context length together in batches. [`ForecastAccumulator`] is the
//! same thing for a single group.
//!
//! The intermediate state is one list per input: timestamps (Int64) and the row-major
//! targets, past and known covariates (Float32, NaN for missing).

use std::{collections::BTreeMap, mem::size_of, sync::Arc};

use datafusion::{
    arrow::{
        array::{Array, ArrayRef, AsArray, BooleanArray, Float32Array, Int64Array, ListArray},
        buffer::OffsetBuffer,
        datatypes::{DataType, Field, Float32Type, Int64Type},
    },
    common::{Result, ScalarValue, exec_err},
    logical_expr::{Accumulator, EmitTo, GroupsAccumulator},
};

use crate::{
    convert::{GroupForecast, build_output, read_struct, read_ts},
    forecaster::{ForecastRequest, Forecaster},
    time::{Prepared, prepare},
};

/// Widths of one row and the horizon, fixed per query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Layout {
    pub variates: usize,
    pub past: usize,
    pub known: usize,
    pub horizon: usize,
}

/// The rows of one group, in arrival order; values are row-major per [`Layout`].
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Series {
    pub ts: Vec<i64>,
    pub targets: Vec<f32>,
    pub past: Vec<f32>,
    pub known: Vec<f32>,
}

impl Series {
    fn size(&self) -> usize {
        self.ts.capacity() * size_of::<i64>()
            + (self.targets.capacity() + self.past.capacity() + self.known.capacity())
                * size_of::<f32>()
    }
}

/// Everything an accumulator needs besides its rows.
#[derive(Clone)]
pub(crate) struct GroupForecaster {
    pub forecaster: Arc<dyn Forecaster>,
    pub batch_size: usize,
    pub layout: Layout,
    pub ts_type: DataType,
    pub variates: Vec<String>,
}

impl std::fmt::Debug for GroupForecaster {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GroupForecaster")
            .field("batch_size", &self.batch_size)
            .field("layout", &self.layout)
            .field("ts_type", &self.ts_type)
            .field("variates", &self.variates)
            .finish_non_exhaustive()
    }
}

impl GroupForecaster {
    /// Append input rows (`ts, targets, horizon[, past[, known]]`) to their groups.
    fn update(
        &self,
        groups: &mut [Series],
        values: &[ArrayRef],
        group_indices: &[usize],
        filter: Option<&BooleanArray>,
    ) -> Result<()> {
        let l = &self.layout;
        let ts = read_ts(&values[0])?;
        let targets = read_struct(&values[1], l.variates)?;
        let past = values
            .get(3)
            .map_or(Ok(vec![]), |a| read_struct(a, l.past))?;
        let known = values
            .get(4)
            .map_or(Ok(vec![]), |a| read_struct(a, l.known))?;
        for (row, (&g, ts)) in group_indices.iter().zip(ts).enumerate() {
            let Some(ts) = ts else { continue };
            if filter.is_some_and(|f| !f.is_valid(row) || !f.value(row)) {
                continue;
            }
            let s = &mut groups[g];
            s.ts.push(ts);
            s.targets
                .extend_from_slice(&targets[row * l.variates..(row + 1) * l.variates]);
            s.past
                .extend_from_slice(&past[row * l.past..(row + 1) * l.past]);
            s.known
                .extend_from_slice(&known[row * l.known..(row + 1) * l.known]);
        }
        Ok(())
    }

    /// Append intermediate states to their groups.
    fn merge(
        &self,
        groups: &mut [Series],
        states: &[ArrayRef],
        group_indices: &[usize],
    ) -> Result<()> {
        let lists: Vec<&ListArray> = states.iter().map(|s| s.as_list::<i32>()).collect();
        for (row, &g) in group_indices.iter().enumerate() {
            if lists[0].is_null(row) {
                continue;
            }
            let s = &mut groups[g];
            s.ts.extend(lists[0].value(row).as_primitive::<Int64Type>().values());
            for (list, out) in lists[1..]
                .iter()
                .zip([&mut s.targets, &mut s.past, &mut s.known])
            {
                out.extend(list.value(row).as_primitive::<Float32Type>().values());
            }
        }
        Ok(())
    }

    fn state(&self, groups: &[Series]) -> Vec<ArrayRef> {
        let ts = Int64Array::from_iter_values(groups.iter().flat_map(|s| s.ts.iter().copied()));
        let mut out = vec![list_of(groups, |s| s.ts.len(), Arc::new(ts))];
        for values in [
            (|s| &s.targets) as fn(&Series) -> &Vec<f32>,
            |s| &s.past,
            |s| &s.known,
        ] {
            let flat = Float32Array::from_iter_values(
                groups.iter().flat_map(|s| values(s).iter().copied()),
            );
            out.push(list_of(groups, |s| values(s).len(), Arc::new(flat)));
        }
        out
    }

    /// Forecast `groups`; groups of equal context length share forecaster calls.
    fn evaluate(&self, groups: &[Series]) -> Result<ArrayRef> {
        let max_context = self.forecaster.max_context();
        let prepared: Vec<Option<Prepared>> = groups
            .iter()
            .map(|s| prepare(s, &self.layout, max_context))
            .collect::<Result<_>>()?;
        let mut by_len: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (i, p) in prepared.iter().enumerate() {
            if let Some(p) = p {
                by_len.entry(p.t).or_default().push(i);
            }
        }
        let mut forecasts: Vec<Option<GroupForecast>> = groups.iter().map(|_| None).collect();
        for members in by_len.values() {
            for chunk in members.chunks(self.batch_size.max(1)) {
                let batch: Vec<&Prepared> = chunk
                    .iter()
                    .map(|&i| prepared[i].as_ref().unwrap())
                    .collect();
                let quantiles = self.forecast_batch(&batch)?;
                let l = &self.layout;
                let per_series = l.variates * l.horizon * self.forecaster.quantile_levels().len();
                if quantiles.len() != chunk.len() * per_series {
                    return exec_err!(
                        "forecaster returned {} values, expected {} (batch {} x variates {} x \
                         horizon {} x quantiles {})",
                        quantiles.len(),
                        chunk.len() * per_series,
                        chunk.len(),
                        l.variates,
                        l.horizon,
                        self.forecaster.quantile_levels().len()
                    );
                }
                for ((&i, p), qs) in chunk.iter().zip(&batch).zip(quantiles.chunks(per_series)) {
                    forecasts[i] = Some(GroupForecast {
                        ts: p.future_ts.clone(),
                        quantiles: qs.to_vec(),
                    });
                }
            }
        }
        build_output(
            &self.ts_type,
            &self.variates,
            self.forecaster.quantile_levels().len(),
            self.forecaster.median_index(),
            forecasts,
        )
    }

    /// One forecaster call over groups of equal context length; `(B, V, H, Q)` quantiles.
    fn forecast_batch(&self, batch: &[&Prepared]) -> Result<Vec<f32>> {
        let l = &self.layout;
        let concat = |part: fn(&Prepared) -> &Vec<f32>| -> Vec<f32> {
            batch.iter().flat_map(|p| part(p).iter().copied()).collect()
        };
        let (context, past, known) = (
            concat(|p| &p.context),
            concat(|p| &p.past),
            concat(|p| &p.known),
        );
        self.forecaster.forecast(&ForecastRequest {
            batch: batch.len(),
            variates: l.variates,
            context_len: batch[0].t,
            horizon: l.horizon,
            context: &context,
            past: (l.past > 0).then_some((&past[..], l.past)),
            known: (l.known > 0).then_some((&known[..], l.known)),
        })
    }
}

/// A list per group over the concatenated `values` of every group.
fn list_of(groups: &[Series], len: impl Fn(&Series) -> usize, values: ArrayRef) -> ArrayRef {
    Arc::new(ListArray::new(
        Arc::new(Field::new_list_field(values.data_type().clone(), true)),
        OffsetBuffer::from_lengths(groups.iter().map(len)),
        values,
        None,
    ))
}

pub(crate) struct ForecastGroupsAccumulator {
    forecaster: GroupForecaster,
    groups: Vec<Series>,
}

impl ForecastGroupsAccumulator {
    pub fn new(forecaster: GroupForecaster) -> Self {
        Self {
            forecaster,
            groups: vec![],
        }
    }

    fn resize(&mut self, total_num_groups: usize) {
        if self.groups.len() < total_num_groups {
            self.groups.resize_with(total_num_groups, Series::default);
        }
    }
}

impl GroupsAccumulator for ForecastGroupsAccumulator {
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        self.resize(total_num_groups);
        self.forecaster
            .update(&mut self.groups, values, group_indices, opt_filter)
    }

    fn evaluate(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        let groups = emit_to.take_needed(&mut self.groups);
        self.forecaster.evaluate(&groups)
    }

    fn state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        let groups = emit_to.take_needed(&mut self.groups);
        Ok(self.forecaster.state(&groups))
    }

    fn merge_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        total_num_groups: usize,
    ) -> Result<()> {
        self.resize(total_num_groups);
        self.forecaster
            .merge(&mut self.groups, values, group_indices)
    }

    /// Every row becomes the state of its own group.
    fn convert_to_state(
        &self,
        values: &[ArrayRef],
        opt_filter: Option<&BooleanArray>,
    ) -> Result<Vec<ArrayRef>> {
        let rows = values[0].len();
        let mut groups = vec![Series::default(); rows];
        let indices: Vec<usize> = (0..rows).collect();
        self.forecaster
            .update(&mut groups, values, &indices, opt_filter)?;
        Ok(self.forecaster.state(&groups))
    }

    fn size(&self) -> usize {
        self.groups.capacity() * size_of::<Series>()
            + self.groups.iter().map(Series::size).sum::<usize>()
    }
}

/// A single group, for plans without a `GROUP BY`.
#[derive(Debug)]
pub(crate) struct ForecastAccumulator {
    forecaster: GroupForecaster,
    group: [Series; 1],
}

impl ForecastAccumulator {
    pub fn new(forecaster: GroupForecaster) -> Self {
        Self {
            forecaster,
            group: Default::default(),
        }
    }
}

impl Accumulator for ForecastAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let indices = vec![0; values[0].len()];
        self.forecaster
            .update(&mut self.group, values, &indices, None)
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        ScalarValue::try_from_array(&self.forecaster.evaluate(&self.group)?, 0)
    }

    fn size(&self) -> usize {
        size_of::<Self>() + self.group[0].size()
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        self.forecaster
            .state(&self.group)
            .iter()
            .map(|a| ScalarValue::try_from_array(a, 0))
            .collect()
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        let indices = vec![0; states[0].len()];
        self.forecaster.merge(&mut self.group, states, &indices)
    }
}
