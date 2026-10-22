//! The interface between the SQL functions and a forecasting model.

use datafusion::common::Result;

/// One batch of series with equal context length `T`; every slice is NaN for missing.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct ForecastRequest<'a> {
    /// Number of series `B`.
    pub batch: usize,
    /// Targets per series `V`.
    pub variates: usize,
    /// Context length `T`.
    pub context_len: usize,
    /// Steps to forecast `H`.
    pub horizon: usize,
    /// `(B, V, T)` targets.
    pub context: &'a [f32],
    /// `(B, C, T)` covariates known over the context only, and their channel count `C`.
    pub past: Option<(&'a [f32], usize)>,
    /// `(B, C, T + H)` covariates known over the context and the horizon, and `C`.
    pub known: Option<(&'a [f32], usize)>,
}

impl<'a> ForecastRequest<'a> {
    /// `context` shaped `(batch, variates, T)`, forecasting `horizon` steps.
    pub fn new(context: &'a [f32], batch: usize, variates: usize, horizon: usize) -> Self {
        Self {
            batch,
            variates,
            context_len: context.len() / (batch * variates).max(1),
            horizon,
            context,
            past: None,
            known: None,
        }
    }

    /// `(batch, channels, T)` covariates known over the context only.
    pub fn with_past(mut self, values: &'a [f32], channels: usize) -> Self {
        self.past = Some((values, channels));
        self
    }

    /// `(batch, channels, T + horizon)` covariates known over the context and the horizon.
    pub fn with_known(mut self, values: &'a [f32], channels: usize) -> Self {
        self.known = Some((values, channels));
        self
    }
}

/// A model behind the `forecast` aggregate.
///
/// Groups are trimmed to [`max_context`](Self::max_context) steps and passed to
/// [`forecast`](Self::forecast) in batches of equal context length.
pub trait Forecaster: Send + Sync + 'static {
    /// Probability levels of the quantiles returned by [`forecast`](Self::forecast),
    /// ascending; fixed for the model.
    fn quantile_levels(&self) -> &[f64];

    /// Index of the median in [`quantile_levels`](Self::quantile_levels); by default the
    /// level closest to 0.5.
    fn median_index(&self) -> usize {
        let levels = self.quantile_levels();
        (0..levels.len())
            .min_by(|&a, &b| (levels[a] - 0.5).abs().total_cmp(&(levels[b] - 0.5).abs()))
            .unwrap_or(0)
    }

    /// Most context steps per series; older rows of a group are dropped.
    fn max_context(&self) -> usize;

    /// Largest horizon a query may ask for; by default [`max_context`](Self::max_context).
    fn max_horizon(&self) -> usize {
        self.max_context()
    }

    /// `(B, V, H, Q)` quantiles of `request`, `Q` the number of
    /// [`quantile_levels`](Self::quantile_levels). Inputs the model does not support, such
    /// as covariates, are errors.
    fn forecast(&self, request: &ForecastRequest) -> Result<Vec<f32>>;
}
