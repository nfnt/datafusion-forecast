//! Inference for [Timestral](https://github.com/mistralai/timestral) time-series forecasting
//! models with [Candle](https://github.com/huggingface/candle).
//!
//! The network runs in Candle; normalization, masks and the block-autoregressive decode
//! loop around it are plain Rust, in the float64 precision the reference implementation
//! uses for its statistics.
//!
//! ```no_run
//! use candle_core::Device;
//! use candle_timestral::{ForecastInput, ForecastOptions, TimestralForecaster};
//!
//! let forecaster = TimestralForecaster::from_dir("models/Timestral-5M", &Device::Cpu)?;
//! let context: Vec<f32> = (0..500).map(|t| (t as f32 / 24.0).sin()).collect();
//! let forecast = forecaster.forecast(&ForecastInput::univariate(&context), &ForecastOptions::new(96))?;
//! let median = forecast.median(); // (batch, variates, horizon) flattened
//! # Ok::<(), candle_timestral::Error>(())
//! ```

mod config;
mod forecaster;
mod layers;
pub mod loading;
mod masks;
mod model;
mod scaler;

pub use config::{AttentionGate, Config, LayerKind};
pub use forecaster::{Covariates, Forecast, ForecastInput, ForecastOptions, TimestralForecaster};
pub use loading::load_model;
pub use model::{Role, TimestralModel};

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error(transparent)]
    Candle(#[from] candle_core::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("invalid config: {0}")]
    Config(String),
    #[error("invalid input: {0}")]
    Input(String),
    #[error("{0}")]
    Checkpoint(String),
    #[cfg(feature = "hub")]
    #[error(transparent)]
    Hub(#[from] hf_hub::HFError),
}

pub type Result<T> = std::result::Result<T, Error>;
