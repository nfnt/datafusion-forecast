//! Forecast two stores' hourly sales in SQL, with a promotion flag known in advance.
//!
//! ```sh
//! cargo run --release -p datafusion-forecast --example sql -- path/to/Timestral
//! cargo run --release -p datafusion-forecast --features hub --example sql -- mistralai/Timestral-5M
//! ```

use std::sync::Arc;

use candle_core::Device;
use datafusion::{
    arrow::{
        array::{Float64Array, Int64Array, RecordBatch, StringArray},
        datatypes::{DataType, Field, Schema},
    },
    error::{DataFusionError, Result},
    prelude::SessionContext,
};
use datafusion_forecast::{
    Timestral,
    candle_timestral::{TimestralForecaster, loading::resolve_checkpoint},
};

const CONTEXT: usize = 24 * 21;
const HORIZON: usize = 24;

#[tokio::main]
async fn main() -> Result<()> {
    let checkpoint = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "mistralai/Timestral-5M".to_string());
    let forecaster = resolve_checkpoint(&checkpoint)
        .and_then(|dir| TimestralForecaster::from_dir(dir, &Device::Cpu))
        .map_err(|e| DataFusionError::External(Box::new(e)))?;

    let ctx = SessionContext::new();
    datafusion_forecast::register(&ctx, Arc::new(Timestral::from(forecaster)));
    ctx.register_batch("sales", sales())?;

    // The last HORIZON rows of each store carry only the promo flag: they are the future.
    ctx.sql(&format!(
        "SELECT store, f.ts AS ts, round(f.sales.median, 1) AS median,
                round(f.sales.quantiles[3], 1) AS p10, round(f.sales.quantiles[19], 1) AS p90
         FROM (
             SELECT store, unnest(forecast(ts, named_struct('sales', sales), {HORIZON},
                                           NULL, named_struct('promo', promo))) AS f
             FROM sales GROUP BY store
         )
         ORDER BY ts, store
         LIMIT 12"
    ))
    .await?
    .show()
    .await?;

    ctx.sql("SELECT forecast_quantile_levels() AS levels")
        .await?
        .show()
        .await?;
    Ok(())
}

/// `(store, ts, sales, promo)`: hourly sales with a daily cycle that jump during
/// promotions, plus HORIZON future rows per store with NULL sales.
fn sales() -> RecordBatch {
    let (mut store, mut ts, mut sales, mut promo) = (vec![], vec![], vec![], vec![]);
    for (name, base) in [("berlin", 120.0), ("paris", 80.0)] {
        for t in 0..CONTEXT + HORIZON {
            let on = (t / 24) % 7 == 5;
            store.push(name);
            ts.push(t as i64);
            promo.push(on as i64 as f64);
            sales.push((t < CONTEXT).then(|| {
                let daily = (2.0 * std::f64::consts::PI * t as f64 / 24.0).sin();
                base + 30.0 * daily
                    + if on { 40.0 } else { 0.0 }
                    + 5.0 * (t as f64 * 12.9898).sin().fract()
            }));
        }
    }
    let schema = Schema::new(vec![
        Field::new("store", DataType::Utf8, false),
        Field::new("ts", DataType::Int64, false),
        Field::new("sales", DataType::Float64, true),
        Field::new("promo", DataType::Float64, false),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(store)),
            Arc::new(Int64Array::from(ts)),
            Arc::new(Float64Array::from(sales)),
            Arc::new(Float64Array::from(promo)),
        ],
    )
    .expect("columns match the schema")
}
