//! Forecast a noisy daily-seasonal series with a Timestral checkpoint.
//!
//! ```sh
//! cargo run --release --features hub --example forecast -- mistralai/Timestral-5M
//! ```

use std::time::Instant;

use candle_core::Device;
use candle_timestral::{ForecastInput, ForecastOptions, TimestralForecaster};

fn main() -> candle_timestral::Result<()> {
    let repo_id = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "mistralai/Timestral-5M".to_string());
    let forecaster = TimestralForecaster::from_hub(&repo_id, &Device::Cpu)?;

    // Hourly series with a daily cycle, a trend and some deterministic noise.
    let context: Vec<f32> = (0..2000)
        .map(|t| {
            let t = t as f32;
            50.0 + 0.01 * t
                + 10.0 * (2.0 * std::f32::consts::PI * t / 24.0).sin()
                + 2.0 * (t * 12.9898).sin().fract()
        })
        .collect();

    let start = Instant::now();
    let forecast = forecaster.forecast(
        &ForecastInput::univariate(&context),
        &ForecastOptions::new(48),
    )?;
    println!(
        "forecast 48 steps from {} in {:.2?}",
        context.len(),
        start.elapsed()
    );

    let (p10, median, p90) = (
        forecast.quantile(0.1)?,
        forecast.median(),
        forecast.quantile(0.9)?,
    );
    println!("{:>5} {:>9} {:>9} {:>9}", "step", "p10", "median", "p90");
    for step in 0..forecast.horizon {
        println!(
            "{step:>5} {:>9.3} {:>9.3} {:>9.3}",
            p10[step], median[step], p90[step]
        );
    }
    Ok(())
}
