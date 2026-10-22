# datafusion-forecast

[![crates.io](https://img.shields.io/crates/v/datafusion-forecast.svg)](https://crates.io/crates/datafusion-forecast)
[![docs.rs](https://img.shields.io/docsrs/datafusion-forecast)](https://docs.rs/datafusion-forecast)
[![CI](https://github.com/nfnt/datafusion-forecast/actions/workflows/check.yml/badge.svg)](https://github.com/nfnt/datafusion-forecast/actions/workflows/check.yml)
[![license](https://img.shields.io/crates/l/datafusion-forecast.svg)](https://github.com/nfnt/datafusion-forecast/blob/main/LICENSE)
[![MSRV](https://img.shields.io/crates/msrv/datafusion-forecast.svg)](https://crates.io/crates/datafusion-forecast)

Forecasts with Mistral AI's [Timestral](https://github.com/mistralai/timestral) models as an
[Apache DataFusion](https://datafusion.apache.org) aggregate function. The model runs in Rust on top of the
[`candle-timestral`](https://github.com/nfnt/datafusion-forecast/tree/main/crates/candle-timestral) crate.

```rust
use std::sync::Arc;
use candle_core::Device;
use datafusion::prelude::SessionContext;
use datafusion_forecast::{candle_timestral::TimestralForecaster, Timestral};

let forecaster = TimestralForecaster::from_dir("path/to/Timestral", &Device::Cpu)?;
let ctx = SessionContext::new();
datafusion_forecast::register(&ctx, Arc::new(Timestral::from(forecaster)));
```

```sql
SELECT store, f.ts AS ts, f.sales.median, f.sales.quantiles[19] AS p90
FROM (
    SELECT store, unnest(forecast(ts, named_struct('sales', sales, 'returns', returns), 48)) AS f
    FROM sales GROUP BY store
);
```

`forecast(ts, targets, horizon [, past [, known]])` treats each group as a series and returns, for each of the
`horizon` steps, a median and quantiles for every field of `targets`. `past` and `known` are optional covariates
known over the context only, or over the context and the horizon. The arguments, the handling of calendar steps
and sparse groups, the options and the limitations are documented in the
[repository README](https://github.com/nfnt/datafusion-forecast#forecastts-targets-horizon--past--known).

The features `hub`, `metal`, `cuda`, `cudnn`, `accelerate` and `mkl` are forwarded to `candle-timestral`; see
[its README](https://github.com/nfnt/datafusion-forecast/tree/main/crates/candle-timestral) for what they do.

## Example and tests

```sh
cargo run --release -p datafusion-forecast --features hub --example sql -- mistralai/Timestral-5M
cargo test -p datafusion-forecast
```

The tests run SQL queries on a tiny random-weight Timestral model and compare the results with direct
`TimestralForecaster::forecast` calls.
