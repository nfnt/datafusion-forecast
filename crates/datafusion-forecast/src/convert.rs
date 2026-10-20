//! Arrow arrays in and out of the forecaster's flat slices.

use std::sync::Arc;

use datafusion::{
    arrow::{
        array::{
            Array, ArrayRef, AsArray, Date32Array, FixedSizeListArray, Float32Array, Int64Array,
            ListArray, StructArray,
        },
        buffer::{NullBuffer, OffsetBuffer},
        datatypes::{
            DataType, Date32Type, Date64Type, Field, Fields, Float32Type, Int64Type, TimeUnit,
            TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType,
            TimestampSecondType,
        },
    },
    common::{Result, exec_err, internal_err},
};

/// Name of the timestamp field of every forecast step.
pub const TS_FIELD: &str = "ts";
pub const MEDIAN_FIELD: &str = "median";
pub const QUANTILES_FIELD: &str = "quantiles";

/// Timestamps as `i64` in their native unit; NULL stays `None`.
pub(crate) fn read_ts(array: &ArrayRef) -> Result<Vec<Option<i64>>> {
    Ok(match array.data_type() {
        DataType::Int64 => array.as_primitive::<Int64Type>().iter().collect(),
        DataType::Date32 => array
            .as_primitive::<Date32Type>()
            .iter()
            .map(|d| d.map(i64::from))
            .collect(),
        DataType::Date64 => array.as_primitive::<Date64Type>().iter().collect(),
        DataType::Timestamp(TimeUnit::Second, _) => {
            array.as_primitive::<TimestampSecondType>().iter().collect()
        }
        DataType::Timestamp(TimeUnit::Millisecond, _) => array
            .as_primitive::<TimestampMillisecondType>()
            .iter()
            .collect(),
        DataType::Timestamp(TimeUnit::Microsecond, _) => array
            .as_primitive::<TimestampMicrosecondType>()
            .iter()
            .collect(),
        DataType::Timestamp(TimeUnit::Nanosecond, _) => array
            .as_primitive::<TimestampNanosecondType>()
            .iter()
            .collect(),
        other => return internal_err!("unexpected forecast timestamp type {other}"),
    })
}

/// Rebuild timestamps of type `ts_type` from their `i64` values.
pub(crate) fn ts_array(ts_type: &DataType, values: Vec<i64>) -> Result<ArrayRef> {
    let ints = Int64Array::from(values);
    Ok(match ts_type {
        DataType::Int64 => Arc::new(ints),
        DataType::Date32 => {
            let days: Result<Vec<i32>> = ints
                .values()
                .iter()
                .map(|&d| {
                    i32::try_from(d).or_else(|_| exec_err!("forecast date {d} is out of range"))
                })
                .collect();
            Arc::new(Date32Array::from(days?))
        }
        DataType::Date64 => Arc::new(ints.reinterpret_cast::<Date64Type>()),
        DataType::Timestamp(unit, tz) => match unit {
            TimeUnit::Second => Arc::new(
                ints.reinterpret_cast::<TimestampSecondType>()
                    .with_timezone_opt(tz.clone()),
            ),
            TimeUnit::Millisecond => Arc::new(
                ints.reinterpret_cast::<TimestampMillisecondType>()
                    .with_timezone_opt(tz.clone()),
            ),
            TimeUnit::Microsecond => Arc::new(
                ints.reinterpret_cast::<TimestampMicrosecondType>()
                    .with_timezone_opt(tz.clone()),
            ),
            TimeUnit::Nanosecond => Arc::new(
                ints.reinterpret_cast::<TimestampNanosecondType>()
                    .with_timezone_opt(tz.clone()),
            ),
        },
        other => return internal_err!("unexpected forecast timestamp type {other}"),
    })
}

/// Row-major `(rows, width)` values of a struct of Float32 fields; NULL becomes NaN.
/// A NULL-typed column (an omitted argument) has width 0.
pub(crate) fn read_struct(array: &ArrayRef, width: usize) -> Result<Vec<f32>> {
    if width == 0 {
        return Ok(vec![]);
    }
    let Some(st) = array.as_struct_opt() else {
        return internal_err!("expected a struct, got {}", array.data_type());
    };
    let mut out = vec![f32::NAN; st.len() * width];
    for (ci, column) in st.columns().iter().enumerate() {
        let column = column.as_primitive::<Float32Type>();
        for (row, x) in column.iter().enumerate() {
            if st.is_valid(row)
                && let Some(x) = x
            {
                out[row * width + ci] = x;
            }
        }
    }
    Ok(out)
}

/// Fields of one forecast step: the timestamp, then `{median, quantiles}` per variate.
pub(crate) fn step_fields(ts_type: &DataType, variates: &[String], quantiles: usize) -> Fields {
    let per_variate = Fields::from(vec![
        Field::new(MEDIAN_FIELD, DataType::Float32, false),
        Field::new(
            QUANTILES_FIELD,
            DataType::FixedSizeList(
                Arc::new(Field::new_list_field(DataType::Float32, false)),
                quantiles as i32,
            ),
            false,
        ),
    ]);
    std::iter::once(Field::new(TS_FIELD, ts_type.clone(), false))
        .chain(
            variates
                .iter()
                .map(|name| Field::new(name, DataType::Struct(per_variate.clone()), false)),
        )
        .collect()
}

/// The aggregate's return type: a list of steps.
pub(crate) fn output_type(ts_type: &DataType, variates: &[String], quantiles: usize) -> DataType {
    DataType::List(Arc::new(Field::new_list_field(
        DataType::Struct(step_fields(ts_type, variates, quantiles)),
        false,
    )))
}

/// Forecast of one group: `horizon` timestamps and `(V, horizon, Q)` quantiles.
pub(crate) struct GroupForecast {
    pub ts: Vec<i64>,
    pub quantiles: Vec<f32>,
}

/// Build the output column; `None` groups are NULL.
pub(crate) fn build_output(
    ts_type: &DataType,
    variates: &[String],
    num_quantiles: usize,
    median_index: usize,
    groups: Vec<Option<GroupForecast>>,
) -> Result<ArrayRef> {
    let fields = step_fields(ts_type, variates, num_quantiles);
    let v = variates.len();
    let q = num_quantiles;
    let mut lengths = Vec::with_capacity(groups.len());
    let mut ts = Vec::new();
    let mut medians: Vec<Vec<f32>> = vec![Vec::new(); v];
    let mut quantiles: Vec<Vec<f32>> = vec![Vec::new(); v];
    for g in &groups {
        let Some(g) = g else {
            lengths.push(0);
            continue;
        };
        let h = g.ts.len();
        lengths.push(h);
        ts.extend_from_slice(&g.ts);
        for vi in 0..v {
            let block = &g.quantiles[vi * h * q..(vi + 1) * h * q];
            medians[vi].extend(block.iter().skip(median_index).step_by(q));
            quantiles[vi].extend_from_slice(block);
        }
    }

    let mut columns: Vec<ArrayRef> = vec![ts_array(ts_type, ts)?];
    for (vi, (median, qs)) in medians.into_iter().zip(quantiles).enumerate() {
        let DataType::Struct(per_variate) = fields[vi + 1].data_type() else {
            unreachable!("variate fields are structs");
        };
        let DataType::FixedSizeList(item, _) = per_variate[1].data_type() else {
            unreachable!("quantiles are a fixed-size list");
        };
        let qs = FixedSizeListArray::try_new(
            item.clone(),
            q as i32,
            Arc::new(Float32Array::from(qs)),
            None,
        )?;
        columns.push(Arc::new(StructArray::try_new(
            per_variate.clone(),
            vec![Arc::new(Float32Array::from(median)), Arc::new(qs)],
            None,
        )?));
    }
    let steps = StructArray::try_new(fields, columns, None)?;
    let nulls = NullBuffer::from(groups.iter().map(Option::is_some).collect::<Vec<bool>>());
    let DataType::List(item) = output_type(ts_type, variates, q) else {
        unreachable!("output is a list");
    };
    Ok(Arc::new(ListArray::try_new(
        item,
        OffsetBuffer::from_lengths(lengths),
        Arc::new(steps),
        Some(nulls).filter(|n| n.null_count() > 0),
    )?))
}
