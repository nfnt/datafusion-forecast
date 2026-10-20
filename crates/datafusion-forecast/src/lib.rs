//! Time-series forecasts in [Apache DataFusion](https://datafusion.apache.org) SQL.
//!
//! [`register`] adds the `forecast` aggregate and `forecast_quantile_levels()`:
//!
//! ```sql
//! SELECT store, f.ts, f.sales.median, f.sales.quantiles[19] AS p90
//! FROM (
//!     SELECT store, unnest(forecast(ts, named_struct('sales', sales), 48)) AS f
//!     FROM sales GROUP BY store
//! );
//! ```
//!
//! Each group is one series; the fields of the `targets` struct are its variates and are
//! forecast jointly. See [`ForecastUdaf`] for the arguments, covariates and output.
//!
//! The model is any [`Forecaster`]. [Timestral](https://github.com/mistralai/timestral)
//! is built in as `Timestral` with the `timestral` feature (on by default).

mod accumulator;
mod convert;
mod forecaster;
mod time;
#[cfg(feature = "timestral")]
mod timestral;
mod udaf;

use std::sync::Arc;

use datafusion::{
    logical_expr::{AggregateUDF, ScalarUDF},
    prelude::SessionContext,
};

#[cfg(feature = "timestral")]
pub use candle_timestral;
pub use convert::{MEDIAN_FIELD, QUANTILES_FIELD, TS_FIELD};
pub use forecaster::{ForecastRequest, Forecaster};
#[cfg(feature = "timestral")]
pub use timestral::Timestral;
pub use udaf::{ForecastUdaf, QuantileLevelsUdf};

/// The `forecast` aggregate with default options.
pub fn forecast_udaf(forecaster: Arc<dyn Forecaster>) -> AggregateUDF {
    AggregateUDF::from(ForecastUdaf::new(forecaster))
}

/// The `forecast_quantile_levels()` scalar function.
pub fn quantile_levels_udf(forecaster: &dyn Forecaster) -> ScalarUDF {
    ScalarUDF::from(QuantileLevelsUdf::new(forecaster))
}

/// Register `forecast` and `forecast_quantile_levels` with default options.
pub fn register(ctx: &SessionContext, forecaster: Arc<dyn Forecaster>) {
    ctx.register_udf(quantile_levels_udf(forecaster.as_ref()));
    ctx.register_udaf(forecast_udaf(forecaster));
}
