//! Turn one group's rows into the regular time grid the forecaster expects.
//!
//! Rows are sorted by timestamp. The grid step is the smallest gap between consecutive
//! timestamps; every timestamp must lie on the grid, and grid slots without a row are
//! missing values. With known-future covariates the last `horizon` slots are the horizon.

use datafusion::common::{Result, exec_err};

use crate::accumulator::{Layout, Series};

/// One group laid out for a [`ForecastRequest`](crate::ForecastRequest).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Prepared {
    /// Context length `T` after trimming to the forecaster's maximum context.
    pub t: usize,
    /// `(V, T)` targets, NaN for missing.
    pub context: Vec<f32>,
    /// `(C_past, T)` past covariates.
    pub past: Vec<f32>,
    /// `(C_known, T + horizon)` known covariates.
    pub known: Vec<f32>,
    /// Timestamps of the `horizon` forecast steps.
    pub future_ts: Vec<i64>,
}

/// Lay out a group on its time grid, keeping at most `max_context` context steps.
///
/// Returns `None` for groups that cannot be forecast: fewer than two rows, or no observed
/// target value in the kept context.
pub(crate) fn prepare(
    series: &Series,
    layout: &Layout,
    max_context: usize,
) -> Result<Option<Prepared>> {
    let n = series.ts.len();
    if n < 2 {
        return Ok(None);
    }
    let (v, cp, ck, h) = (layout.variates, layout.past, layout.known, layout.horizon);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by_key(|&i| series.ts[i]);

    let ts = |k: usize| series.ts[order[k]] as i128;
    let mut step = i128::MAX;
    for k in 1..n {
        let gap = ts(k) - ts(k - 1);
        if gap == 0 {
            return exec_err!("duplicate timestamp {} in a forecast group", ts(k));
        }
        step = step.min(gap);
    }
    let first = ts(0);
    let slot_of = |k: usize| -> Result<usize> {
        let offset = ts(k) - first;
        if offset % step != 0 {
            return exec_err!(
                "irregular timestamps in a forecast group: {} is not on the grid {first} + k * {step}; \
                 for calendar months forecast on an integer month number",
                ts(k)
            );
        }
        usize::try_from(offset / step).map_err(|_| {
            datafusion::error::DataFusionError::Execution("time grid too large".into())
        })
    };
    let slots = slot_of(n - 1)? + 1;

    let (t_full, future_ts) = if ck > 0 {
        if slots <= h {
            return exec_err!(
                "with known covariates each forecast group needs its {h} future rows after the context, \
                 got {slots} time steps"
            );
        }
        let t_full = slots - h;
        let future = (t_full..slots)
            .map(|s| grid_ts(first + s as i128 * step))
            .collect::<Result<_>>()?;
        (t_full, future)
    } else {
        let last = first + (slots - 1) as i128 * step;
        (
            slots,
            (1..=h)
                .map(|k| grid_ts(last + k as i128 * step))
                .collect::<Result<_>>()?,
        )
    };

    let start = t_full.saturating_sub(max_context);
    let t = t_full - start;
    let mut context = vec![f32::NAN; v * t];
    let mut past = vec![f32::NAN; cp * t];
    let mut known = vec![f32::NAN; ck * (t + h)];
    for (k, &row) in order.iter().enumerate() {
        let slot = slot_of(k)?;
        let targets = &series.targets[row * v..(row + 1) * v];
        if slot >= t_full {
            if targets.iter().any(|x| !x.is_nan()) {
                return exec_err!(
                    "with known covariates the last {h} time steps of a forecast group are the \
                     horizon, and their targets must be NULL"
                );
            }
        } else if slot < start {
            continue;
        }
        let s = slot - start;
        if slot < t_full {
            for (vi, &x) in targets.iter().enumerate() {
                context[vi * t + s] = x;
            }
            for ci in 0..cp {
                past[ci * t + s] = series.past[row * cp + ci];
            }
        }
        for ci in 0..ck {
            known[ci * (t + h) + s] = series.known[row * ck + ci];
        }
    }
    if context.iter().all(|x| !x.is_finite()) {
        return Ok(None);
    }
    Ok(Some(Prepared {
        t,
        context,
        past,
        known,
        future_ts,
    }))
}

/// A grid timestamp as `i64`, or an error if it overflows.
fn grid_ts(ts: i128) -> Result<i64> {
    i64::try_from(ts).or_else(|_| exec_err!("forecast timestamp {ts} overflows a 64-bit integer"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(variates: usize, known: usize, horizon: usize) -> Layout {
        Layout {
            variates,
            past: 0,
            known,
            horizon,
        }
    }

    fn series(ts: &[i64], targets: &[f32], known: &[f32]) -> Series {
        Series {
            ts: ts.to_vec(),
            targets: targets.to_vec(),
            past: vec![],
            known: known.to_vec(),
        }
    }

    #[test]
    fn sorts_and_fills_gaps() {
        let s = series(&[30, 10, 0], &[3.0, 1.0, 0.0], &[]);
        let p = prepare(&s, &layout(1, 0, 2), 100).unwrap().unwrap();
        assert_eq!(p.t, 4);
        assert_eq!(p.context[..2], [0.0, 1.0]);
        assert!(p.context[2].is_nan());
        assert_eq!(p.context[3], 3.0);
        assert_eq!(p.future_ts, vec![40, 50]);
    }

    #[test]
    fn variate_major_context_and_trimming() {
        let s = series(&[0, 1, 2], &[0.0, 10.0, 1.0, 11.0, 2.0, 12.0], &[]);
        let p = prepare(&s, &layout(2, 0, 1), 2).unwrap().unwrap();
        assert_eq!(p.context, vec![1.0, 2.0, 11.0, 12.0]);
        assert_eq!(p.future_ts, vec![3]);
    }

    #[test]
    fn known_future_rows_are_the_horizon() {
        let s = series(
            &[0, 1, 2, 3],
            &[1.0, 2.0, f32::NAN, f32::NAN],
            &[5.0, 6.0, 7.0, 8.0],
        );
        let p = prepare(&s, &layout(1, 1, 2), 100).unwrap().unwrap();
        assert_eq!(p.t, 2);
        assert_eq!(p.context, vec![1.0, 2.0]);
        assert_eq!(p.known, vec![5.0, 6.0, 7.0, 8.0]);
        assert_eq!(p.future_ts, vec![2, 3]);

        let s = series(&[0, 1, 2], &[1.0, 2.0, 3.0], &[5.0, 6.0, 7.0]);
        assert!(prepare(&s, &layout(1, 1, 1), 100).is_err());
    }

    #[test]
    fn rejects_bad_timestamps_and_skips_degenerate_groups() {
        let l = layout(1, 0, 1);
        assert!(prepare(&series(&[0, 2, 3], &[1.0; 3], &[]), &l, 9).is_ok());
        assert!(prepare(&series(&[0, 2, 5], &[1.0; 3], &[]), &l, 9).is_err());
        assert!(prepare(&series(&[0, 1, 1], &[1.0; 3], &[]), &l, 9).is_err());
        assert_eq!(prepare(&series(&[0], &[1.0], &[]), &l, 9).unwrap(), None);
        let s = series(&[0, 1], &[f32::NAN, f32::NAN], &[]);
        assert_eq!(prepare(&s, &l, 9).unwrap(), None);
    }

    #[test]
    fn rejects_future_timestamps_that_overflow() {
        let s = series(&[i64::MAX - 20, i64::MAX - 10], &[1.0; 2], &[]);
        assert!(prepare(&s, &layout(1, 0, 1), 9).is_ok());
        assert!(prepare(&s, &layout(1, 0, 2), 9).is_err());
    }
}
