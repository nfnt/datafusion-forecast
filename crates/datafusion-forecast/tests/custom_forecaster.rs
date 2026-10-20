//! The `forecast` aggregate with a model other than Timestral: a naive forecaster that
//! repeats the last observed value of each variate.

use std::sync::{Arc, Mutex};

use datafusion::{
    arrow::{
        array::AsArray,
        datatypes::{Date32Type, Float32Type, Float64Type, Int64Type},
    },
    common::{Result, exec_err},
    prelude::SessionContext,
};
use datafusion_forecast::{ForecastRequest, Forecaster, register};

const LEVELS: [f64; 3] = [0.1, 0.5, 0.9];

/// Quantiles `last - 1, last, last + 1`; records the context lengths it was called with.
#[derive(Default)]
struct Naive {
    calls: Mutex<Vec<(usize, usize)>>,
}

impl Forecaster for Naive {
    fn quantile_levels(&self) -> &[f64] {
        &LEVELS
    }

    fn max_context(&self) -> usize {
        3
    }

    fn forecast(&self, r: &ForecastRequest) -> Result<Vec<f32>> {
        if r.past.is_some() || r.known.is_some() {
            return exec_err!("naive forecaster: covariates are not supported");
        }
        self.calls.lock().unwrap().push((r.batch, r.context_len));
        let mut out = Vec::with_capacity(r.batch * r.variates * r.horizon * LEVELS.len());
        for series in r.context.chunks(r.context_len) {
            let last = series.iter().rev().find(|x| x.is_finite()).copied();
            let last = last.unwrap_or(f32::NAN);
            for _ in 0..r.horizon {
                out.extend([last - 1.0, last, last + 1.0]);
            }
        }
        Ok(out)
    }
}

async fn context(naive: Arc<Naive>) -> SessionContext {
    let ctx = SessionContext::new();
    register(&ctx, naive);
    ctx.sql(
        "CREATE TABLE t (id INT, ts BIGINT, a DOUBLE, b INT) AS VALUES \
         (1, 0, 1.0, 10), (1, 10, 2.0, 20), (1, 20, 3.0, 30), (1, 40, 5.0, NULL), \
         (2, 3, 7.0, 70), (2, 4, NULL, 80)",
    )
    .await
    .unwrap();
    ctx
}

#[tokio::test]
async fn custom_forecaster_drives_the_aggregate() {
    let naive = Arc::new(Naive::default());
    let ctx = context(naive.clone()).await;
    let batches = ctx
        .sql(
            "SELECT id, forecast(ts, named_struct('a', a, 'b', b), 2) AS f \
             FROM t GROUP BY id ORDER BY id",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let lists = batches[0].column_by_name("f").unwrap().as_list::<i32>();
    let expect = [
        // id 1: grid step 10, last at 40; b's last observed value is 30
        (vec![50, 60], [5.0, 30.0]),
        // id 2: grid step 1, last at 4; a's last observed value is 7
        (vec![5, 6], [7.0, 80.0]),
    ];
    for (row, (ts, last)) in expect.iter().enumerate() {
        let steps = lists.value(row);
        let steps = steps.as_struct();
        assert_eq!(
            steps.column(0).as_primitive::<Int64Type>().values(),
            &ts[..]
        );
        for (vi, &x) in last.iter().enumerate() {
            let variate = steps.column(vi + 1).as_struct();
            let median = variate.column(0).as_primitive::<Float32Type>();
            assert_eq!(median.values(), &[x, x]);
            let quantiles = variate.column(1).as_fixed_size_list();
            assert_eq!(quantiles.value_length(), 3);
            assert_eq!(
                quantiles.values().as_primitive::<Float32Type>().values(),
                &[x - 1.0, x, x + 1.0, x - 1.0, x, x + 1.0]
            );
        }
    }
    // id 1 spans five grid slots, trimmed to max_context = 3; id 2 has two
    let mut calls = naive.calls.lock().unwrap().clone();
    calls.sort();
    assert_eq!(calls, vec![(1, 2), (1, 3)]);

    let levels = ctx
        .sql("SELECT forecast_quantile_levels() AS l")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let levels = levels[0].column(0).as_list::<i32>().value(0);
    assert_eq!(levels.as_primitive::<Float64Type>().values(), &LEVELS);
}

#[tokio::test]
async fn forecaster_errors_reach_the_query() {
    let ctx = context(Arc::new(Naive::default())).await;
    let err = ctx
        .sql("SELECT forecast(ts, named_struct('a', a), 2, named_struct('b', b)) AS f FROM t")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("covariates are not supported"), "{err}");
}

async fn query_error(ctx: &SessionContext, sql: &str) -> String {
    ctx.sql(sql)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap_err()
        .to_string()
}

#[tokio::test]
async fn horizon_above_max_horizon_is_an_error() {
    let ctx = context(Arc::new(Naive::default())).await;
    let err = query_error(&ctx, "SELECT forecast(ts, named_struct('a', a), 4) FROM t").await;
    assert!(err.contains("horizon must be between 1 and 3"), "{err}");
}

/// Returns one value too few.
struct Short;

impl Forecaster for Short {
    fn quantile_levels(&self) -> &[f64] {
        &LEVELS
    }

    fn max_context(&self) -> usize {
        8
    }

    fn forecast(&self, r: &ForecastRequest) -> Result<Vec<f32>> {
        Ok(vec![
            0.0;
            r.batch * r.variates * r.horizon * LEVELS.len() - 1
        ])
    }
}

#[tokio::test]
async fn wrong_forecast_length_is_an_error() {
    let ctx = SessionContext::new();
    register(&ctx, Arc::new(Short));
    let err = query_error(
        &ctx,
        "SELECT forecast(ts, named_struct('a', a), 2) \
         FROM (VALUES (0, 1.0), (1, 2.0)) AS t(ts, a)",
    )
    .await;
    assert!(
        err.contains("forecaster returned 5 values, expected 6"),
        "{err}"
    );
}

#[test]
fn default_median_index_is_the_level_closest_to_one_half() {
    assert_eq!(Naive::default().median_index(), 1);
}

/// The README's workarounds for calendar months and sparse groups.
#[tokio::test]
async fn month_index_and_null_padding() {
    let ctx = SessionContext::new();
    register(&ctx, Arc::new(Naive::default()));
    let months = ctx
        .sql(
            "SELECT make_date(CAST(f.ts / 12 AS INT), CAST(f.ts % 12 + 1 AS INT), 1) AS month \
             FROM (SELECT unnest(forecast(m, named_struct('v', v), 2)) AS f FROM ( \
               SELECT CAST(date_part('year', d) * 12 + date_part('month', d) - 1 AS BIGINT) AS m, v \
               FROM (VALUES (DATE '2026-01-31', 1.0), (DATE '2026-02-28', 2.0), \
                            (DATE '2026-03-31', 3.0)) AS t(d, v)))",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let months = months[0].column(0).as_primitive::<Date32Type>();
    // 2026-04-01 and 2026-05-01
    assert_eq!(months.values(), &[20544, 20574]);

    // Every other day observed: padding with NULL rows keeps the step at one day.
    let padded = ctx
        .sql(
            "SELECT f.ts FROM (SELECT unnest(forecast(g.d, named_struct('v', t.v), 2)) AS f FROM \
               (SELECT CAST(unnest(range(DATE '2026-01-01', DATE '2026-01-06', INTERVAL '1 day')) \
                  AS DATE) AS d) g \
             LEFT JOIN (VALUES (DATE '2026-01-01', 1.0), (DATE '2026-01-03', 2.0), \
                               (DATE '2026-01-05', 3.0)) AS t(d, v) ON g.d = t.d)",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    // 2026-01-06 and 2026-01-07
    assert_eq!(
        padded[0].column(0).as_primitive::<Date32Type>().values(),
        &[20459, 20460]
    );
}
