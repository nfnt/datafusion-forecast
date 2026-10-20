//! [Timestral](https://github.com/mistralai/timestral) as a [`Forecaster`], through the
//! [`candle_timestral`] crate.

use candle_timestral::{ForecastInput, ForecastOptions, TimestralForecaster};
use datafusion::common::{DataFusionError, Result};

use crate::forecaster::{ForecastRequest, Forecaster};

/// A [`TimestralForecaster`] with its forecast options.
///
/// ```no_run
/// # async fn run() -> datafusion::error::Result<()> {
/// use std::sync::Arc;
/// use candle_core::Device;
/// use datafusion::prelude::SessionContext;
/// use datafusion_forecast::{candle_timestral::TimestralForecaster, Timestral};
///
/// let forecaster = TimestralForecaster::from_dir("models/Timestral-5M", &Device::Cpu)
///     .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
/// let ctx = SessionContext::new();
/// datafusion_forecast::register(&ctx, Arc::new(Timestral::from(forecaster)));
/// # Ok(()) }
/// ```
pub struct Timestral {
    forecaster: TimestralForecaster,
    symmetric_averaging: bool,
    clip_nonnegative: bool,
}

impl Timestral {
    pub fn new(forecaster: TimestralForecaster) -> Self {
        Self {
            forecaster,
            symmetric_averaging: true,
            clip_nonnegative: true,
        }
    }

    /// See [`ForecastOptions::symmetric_averaging`]; on by default.
    pub fn with_symmetric_averaging(mut self, on: bool) -> Self {
        self.symmetric_averaging = on;
        self
    }

    /// See [`ForecastOptions::clip_nonnegative`]; on by default.
    pub fn with_clip_nonnegative(mut self, on: bool) -> Self {
        self.clip_nonnegative = on;
        self
    }

    pub fn forecaster(&self) -> &TimestralForecaster {
        &self.forecaster
    }
}

impl From<TimestralForecaster> for Timestral {
    fn from(forecaster: TimestralForecaster) -> Self {
        Self::new(forecaster)
    }
}

impl std::fmt::Debug for Timestral {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Timestral")
            .field("max_context", &self.forecaster.max_context)
            .field("symmetric_averaging", &self.symmetric_averaging)
            .field("clip_nonnegative", &self.clip_nonnegative)
            .finish_non_exhaustive()
    }
}

impl Forecaster for Timestral {
    fn quantile_levels(&self) -> &[f64] {
        self.forecaster.quantile_levels()
    }

    fn median_index(&self) -> usize {
        self.forecaster.config().median_index()
    }

    fn max_context(&self) -> usize {
        self.forecaster.max_context
    }

    fn forecast(&self, request: &ForecastRequest) -> Result<Vec<f32>> {
        let mut input = ForecastInput::new(request.context, request.batch, request.variates);
        if let Some((values, channels)) = request.past {
            input = input.with_past_covariates(values, channels);
        }
        if let Some((values, channels)) = request.known {
            input = input.with_known_covariates(values, channels);
        }
        let options = ForecastOptions {
            prediction_length: request.horizon,
            symmetric_averaging: self.symmetric_averaging,
            clip_nonnegative: self.clip_nonnegative,
        };
        let forecast = self
            .forecaster
            .forecast(&input, &options)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        Ok(forecast.quantiles)
    }
}
