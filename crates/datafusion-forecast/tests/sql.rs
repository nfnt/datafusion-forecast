//! The `forecast` aggregate through SQL on a tiny random-weight Timestral model, checked
//! against direct `TimestralForecaster::forecast` calls.

#![cfg(feature = "timestral")]

mod common;

use std::sync::Arc;

use datafusion::{
    arrow::{
        array::{
            Array, ArrayRef, AsArray, Date32Array, Float64Array, Int64Array, RecordBatch,
            TimestampNanosecondArray,
        },
        datatypes::{
            DataType, Date32Type, Field, Float32Type, Int64Type, Schema, TimeUnit,
            TimestampNanosecondType,
        },
    },
    datasource::MemTable,
    logical_expr::AggregateUDF,
    prelude::{SessionConfig, SessionContext},
};
use datafusion_forecast::{
    ForecastUdaf, Timestral,
    candle_timestral::{ForecastInput, ForecastOptions, TimestralForecaster},
    register,
};

use common::{series, tiny_model};

const HOUR_NS: i64 = 3_600_000_000_000;

fn forecaster() -> Arc<Timestral> {
    Arc::new(TimestralForecaster::new(tiny_model().0).into())
}

fn context(partitions: usize) -> SessionContext {
    SessionContext::new_with_config(SessionConfig::new().with_target_partitions(partitions))
}

/// One forecast group read back from the output: step timestamps and `(V, H, Q)` quantiles.
#[derive(Debug, PartialEq)]
struct Output {
    ts: Vec<i64>,
    quantiles: Vec<f32>,
    medians: Vec<f32>,
}

fn ts_values(array: &ArrayRef) -> Vec<i64> {
    match array.data_type() {
        DataType::Date32 => array
            .as_primitive::<Date32Type>()
            .values()
            .iter()
            .map(|&d| d as i64)
            .collect(),
        DataType::Timestamp(TimeUnit::Nanosecond, _) => array
            .as_primitive::<TimestampNanosecondType>()
            .values()
            .to_vec(),
        _ => array.as_primitive::<Int64Type>().values().to_vec(),
    }
}

/// `forecast` results in column `f`, by row; NULL rows are `None`.
async fn run(ctx: &SessionContext, sql: &str) -> datafusion::error::Result<Vec<Option<Output>>> {
    let batches = ctx.sql(sql).await?.collect().await?;
    let mut out = vec![];
    for batch in batches {
        let lists = batch.column_by_name("f").unwrap().as_list::<i32>();
        for row in 0..lists.len() {
            if lists.is_null(row) {
                out.push(None);
                continue;
            }
            let steps = lists.value(row);
            let steps = steps.as_struct();
            let mut o = Output {
                ts: ts_values(steps.column(0)),
                quantiles: vec![],
                medians: vec![],
            };
            for variate in &steps.columns()[1..] {
                let variate = variate.as_struct();
                o.medians
                    .extend(variate.column(0).as_primitive::<Float32Type>().values());
                o.quantiles.extend(
                    variate
                        .column(1)
                        .as_fixed_size_list()
                        .values()
                        .as_primitive::<Float32Type>()
                        .values(),
                );
            }
            out.push(Some(o));
        }
    }
    Ok(out)
}

fn assert_close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= 1e-4 * (1.0 + y.abs()),
            "element {i}: {x} vs {y}"
        );
    }
}

fn register_table(ctx: &SessionContext, name: &str, batch: RecordBatch, partitions: usize) {
    let rows = batch.num_rows();
    let chunk = rows.div_ceil(partitions);
    let parts: Vec<Vec<RecordBatch>> = (0..partitions)
        .map(|p| {
            let (start, end) = (p * chunk, ((p + 1) * chunk).min(rows));
            (start..end)
                .step_by(7)
                .map(|s| batch.slice(s, 7.min(end - s)))
                .collect()
        })
        .collect();
    let table = MemTable::try_new(batch.schema(), parts).unwrap();
    ctx.register_table(name, Arc::new(table)).unwrap();
}

/// A long-format table `(id, ts, a, b, p, k)` of hourly series with the given lengths,
/// rows interleaved across ids. `future` trailing rows per id have NULL targets.
fn table(lengths: &[usize], future: usize) -> (RecordBatch, Vec<Vec<f32>>) {
    let mut cols: [Vec<Option<f64>>; 4] = Default::default();
    let (mut ids, mut ts) = (vec![], vec![]);
    let mut raw = vec![];
    for (id, &len) in lengths.iter().enumerate() {
        // a, b, p over len + future steps; k too
        let data = series(1, 4, len + future, id as u64 + 3);
        raw.push(data.clone());
        for t in 0..len + future {
            ids.push(id as i64);
            ts.push(1_700_000_000_000_000_000 + t as i64 * HOUR_NS);
            for (c, col) in cols.iter_mut().enumerate() {
                let x = data[c * (len + future) + t] as f64;
                col.push(if c < 2 && t >= len { None } else { Some(x) });
            }
        }
    }
    // interleave the groups, newest rows first, so rows arrive out of order
    let mut order: Vec<usize> = (0..ids.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse((ts[i], ids[i])));
    let pick_i = |v: &Vec<i64>| Int64Array::from(order.iter().map(|&i| v[i]).collect::<Vec<_>>());
    let pick_f =
        |v: &Vec<Option<f64>>| Float64Array::from(order.iter().map(|&i| v[i]).collect::<Vec<_>>());
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
            false,
        ),
        Field::new("a", DataType::Float64, true),
        Field::new("b", DataType::Float64, true),
        Field::new("p", DataType::Float64, true),
        Field::new("k", DataType::Float64, true),
    ]);
    let ts = TimestampNanosecondArray::from(order.iter().map(|&i| ts[i]).collect::<Vec<_>>())
        .with_timezone("UTC");
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(pick_i(&ids)),
            Arc::new(ts),
            Arc::new(pick_f(&cols[0])),
            Arc::new(pick_f(&cols[1])),
            Arc::new(pick_f(&cols[2])),
            Arc::new(pick_f(&cols[3])),
        ],
    )
    .unwrap();
    (batch, raw)
}

/// Rows `c` of a raw `(4, len)` series, first `len` steps; rounded through f64 like the table.
fn rows(raw: &[f32], channels: std::ops::Range<usize>, total: usize, len: usize) -> Vec<f32> {
    channels
        .flat_map(|c| raw[c * total..c * total + len].to_vec())
        .collect()
}

#[tokio::test]
async fn univariate_matches_direct_forecast() {
    let fc = forecaster();
    let ctx = context(1);
    register(&ctx, fc.clone());
    let lengths = [60, 45, 60];
    let (batch, raw) = table(&lengths, 0);
    register_table(&ctx, "t", batch, 1);
    let h = 40;
    let out = run(
        &ctx,
        &format!(
            "SELECT id, forecast(ts, named_struct('a', a), {h}) AS f FROM t GROUP BY id ORDER BY id"
        ),
    )
    .await
    .unwrap();
    let q = fc.forecaster().quantile_levels().len();
    for (id, (o, &len)) in out.iter().zip(&lengths).enumerate() {
        let o = o.as_ref().unwrap();
        let ctx_values = rows(&raw[id], 0..1, len, len);
        let direct = fc
            .forecaster()
            .forecast(
                &ForecastInput::univariate(&ctx_values),
                &ForecastOptions::new(h),
            )
            .unwrap();
        assert_close(&o.quantiles, &direct.quantiles);
        assert_close(&o.medians, &direct.median());
        assert_eq!(o.quantiles.len(), h * q);
        let last = 1_700_000_000_000_000_000 + (len as i64 - 1) * HOUR_NS;
        assert_eq!(
            o.ts,
            (1..=h as i64)
                .map(|k| last + k * HOUR_NS)
                .collect::<Vec<_>>()
        );
    }
}

#[tokio::test]
async fn multivariate_with_covariates_matches_direct_forecast() {
    let fc = forecaster();
    let ctx = context(1);
    register(&ctx, fc.clone());
    let (len, h) = (50, 12);
    let (batch, raw) = table(&[len, len], h);
    register_table(&ctx, "t", batch, 1);
    let out = run(
        &ctx,
        &format!(
            "SELECT id, forecast(ts, named_struct('a', a, 'b', b), {h}, named_struct('p', p), \
             named_struct('k', k)) AS f FROM t GROUP BY id ORDER BY id"
        ),
    )
    .await
    .unwrap();
    for (id, o) in out.iter().enumerate() {
        let total = len + h;
        let context = rows(&raw[id], 0..2, total, len);
        let past = rows(&raw[id], 2..3, total, len);
        let known = rows(&raw[id], 3..4, total, total);
        let input = ForecastInput::new(&context, 1, 2)
            .with_past_covariates(&past, 1)
            .with_known_covariates(&known, 1);
        let direct = fc
            .forecaster()
            .forecast(&input, &ForecastOptions::new(h))
            .unwrap();
        assert_close(&o.as_ref().unwrap().quantiles, &direct.quantiles);
    }

    let out = run(
        &ctx,
        &format!("SELECT forecast(ts, named_struct('a', a), {h}, NULL, named_struct('k', k)) AS f FROM t WHERE id = 1"),
    )
    .await
    .unwrap();
    let total = len + h;
    let context = rows(&raw[1], 0..1, total, len);
    let known = rows(&raw[1], 3..4, total, total);
    let input = ForecastInput::univariate(&context).with_known_covariates(&known, 1);
    let direct = fc
        .forecaster()
        .forecast(&input, &ForecastOptions::new(h))
        .unwrap();
    assert_close(&out[0].as_ref().unwrap().quantiles, &direct.quantiles);
}

#[tokio::test]
async fn gaps_are_missing_values() {
    let fc = forecaster();
    let ctx = context(1);
    register(&ctx, fc.clone());
    let len = 48;
    let (batch, raw) = table(&[len], 0);
    register_table(&ctx, "t", batch, 1);
    let start = 1_700_000_000_000_000_000i64;
    let dropped = [5i64, 6, 20];
    let filter = dropped
        .iter()
        .map(|t| format!("CAST(ts AS BIGINT) <> {}", start + t * HOUR_NS))
        .collect::<Vec<_>>()
        .join(" AND ");
    let sql = format!("SELECT forecast(ts, named_struct('a', a), 8) AS f FROM t WHERE {filter}");
    let out = run(&ctx, &sql).await.unwrap();
    let mut values = rows(&raw[0], 0..1, len, len);
    for &t in &dropped {
        values[t as usize] = f32::NAN;
    }
    let direct = fc
        .forecaster()
        .forecast(
            &ForecastInput::univariate(&values),
            &ForecastOptions::new(8),
        )
        .unwrap();
    assert_close(&out[0].as_ref().unwrap().quantiles, &direct.quantiles);
}

#[tokio::test]
async fn batching_and_partitioning_do_not_change_results() {
    let fc = forecaster();
    let lengths: Vec<usize> = (0..12).map(|i| [40, 64, 300][i % 3]).collect();
    let (batch, _) = table(&lengths, 0);
    let sql = "SELECT id, forecast(ts, named_struct('a', a, 'b', b), 20) AS f FROM t GROUP BY id ORDER BY id";

    let mut results = vec![];
    for (batch_size, partitions) in [(1, 1), (64, 1), (64, 4), (5, 3)] {
        let ctx = context(partitions);
        ctx.register_udaf(AggregateUDF::from(
            ForecastUdaf::new(fc.clone()).with_batch_size(batch_size),
        ));
        register_table(&ctx, "t", batch.clone(), partitions);
        results.push(run(&ctx, sql).await.unwrap());
    }
    for r in &results[1..] {
        assert_eq!(r.len(), lengths.len());
        for (a, b) in r.iter().zip(&results[0]) {
            let (a, b) = (a.as_ref().unwrap(), b.as_ref().unwrap());
            assert_eq!(a.ts, b.ts);
            assert_close(&a.quantiles, &b.quantiles);
        }
    }
}

#[tokio::test]
async fn date_and_integer_time() {
    let fc = forecaster();
    let ctx = context(1);
    register(&ctx, fc.clone());
    let n = 30;
    let values = series(1, 1, n, 9);
    let schema = Arc::new(Schema::new(vec![
        Field::new("d", DataType::Date32, false),
        Field::new("i", DataType::Int32, false),
        Field::new("v", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Date32Array::from(
                (0..n as i32).map(|k| 19_000 + 7 * k).collect::<Vec<_>>(),
            )),
            Arc::new(datafusion::arrow::array::Int32Array::from(
                (0..n as i32).map(|k| 3 * k).collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                values.iter().map(|&x| x as f64).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();
    ctx.register_batch("t", batch).unwrap();

    let weekly = run(
        &ctx,
        "SELECT forecast(d, named_struct('v', v), 3) AS f FROM t",
    )
    .await
    .unwrap();
    let last = 19_000 + 7 * (n as i64 - 1);
    assert_eq!(
        weekly[0].as_ref().unwrap().ts,
        vec![last + 7, last + 14, last + 21]
    );

    let ints = run(
        &ctx,
        "SELECT forecast(i, named_struct('v', v), 2) AS f FROM t",
    )
    .await
    .unwrap();
    let last = 3 * (n as i64 - 1);
    assert_eq!(ints[0].as_ref().unwrap().ts, vec![last + 3, last + 6]);
    assert_close(
        &ints[0].as_ref().unwrap().quantiles,
        &fc.forecaster()
            .forecast(
                &ForecastInput::univariate(&values),
                &ForecastOptions::new(2),
            )
            .unwrap()
            .quantiles,
    );
}

#[tokio::test]
async fn quantile_levels_and_null_groups() {
    let fc = forecaster();
    let ctx = context(1);
    register(&ctx, fc.clone());
    let batches = ctx
        .sql("SELECT forecast_quantile_levels() AS l")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let levels = batches[0].column(0).as_list::<i32>().value(0);
    assert_eq!(
        levels
            .as_primitive::<datafusion::arrow::datatypes::Float64Type>()
            .values()
            .to_vec(),
        fc.forecaster().quantile_levels()
    );

    ctx.sql("CREATE TABLE t (id INT, ts BIGINT, v DOUBLE) AS VALUES (1, 0, 1.0), (2, 0, NULL), (2, 1, NULL)")
        .await
        .unwrap();
    let out = run(
        &ctx,
        "SELECT id, forecast(ts, named_struct('v', v), 4) AS f FROM t GROUP BY id ORDER BY id",
    )
    .await
    .unwrap();
    assert_eq!(out, vec![None, None]);
}

#[tokio::test]
async fn invalid_queries_are_errors() {
    let ctx = context(1);
    register(&ctx, forecaster());
    ctx.sql(
        "CREATE TABLE t (id INT, ts BIGINT, v DOUBLE, h BIGINT) AS VALUES \
         (1, 0, 1.0, 2), (1, 2, 2.0, 2), (1, 5, 3.0, 2), (2, 0, 1.0, 2), (2, 0, 2.0, 2), \
         (3, 0, 1.0, 2), (3, 1, 2.0, 2), (3, 2, 3.0, 2)",
    )
    .await
    .unwrap();
    let err = |sql: &'static str| {
        let ctx = ctx.clone();
        async move { run(&ctx, sql).await.expect_err(sql).to_string() }
    };
    let e = err("SELECT forecast(ts, named_struct('v', v), 2) AS f FROM t WHERE id = 1").await;
    assert!(e.contains("irregular"), "{e}");
    let e = err("SELECT forecast(ts, named_struct('v', v), 2) AS f FROM t WHERE id = 2").await;
    assert!(e.contains("duplicate"), "{e}");
    let e = err("SELECT forecast(ts, named_struct('v', v), h) AS f FROM t").await;
    assert!(e.contains("literal"), "{e}");
    let e = err("SELECT forecast(ts, named_struct('v', v), 1, NULL, named_struct('k', v)) AS f FROM t WHERE id = 3").await;
    assert!(e.contains("must be NULL"), "{e}");
    let e = err("SELECT forecast(ts, v, 2) AS f FROM t").await;
    assert!(e.contains("struct"), "{e}");
    let e = err("SELECT forecast(ts, named_struct('ts', v), 2) AS f FROM t").await;
    assert!(e.contains("cannot be named"), "{e}");
}
