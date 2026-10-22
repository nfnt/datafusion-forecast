//! Probabilistic forecasting with Timestral, including the clampsym test-time augmentation.
//!
//! [`TimestralForecaster::forecast`] takes a context `(B, V, T)` (series, variates, time;
//! NaN marks a missing value) and returns quantile forecasts `(B, V, H, Q)`. Horizons
//! longer than one decode block are forecast block by block, feeding each block's median
//! back as context.
//!
//! Clampsym (both options on by default):
//!
//! - Symmetric averaging forecasts the context and its negation, reflects the second
//!   forecast back (`-Q(1 - p)`), and averages the two. Every covariate row is negated
//!   with the context.
//! - Non-negative clipping clamps the forecast at 0 for every series whose observed
//!   context is entirely non-negative.
//!
//! All arrays are flat, row-major slices.

use std::path::Path;

use candle_core::Device;

use crate::{
    Config, Error, Result,
    model::{Role, TimestralModel},
    scaler::{advancing_block_frames, causal_patch_loc_scale, clamp_nonfinite},
};

/// Covariate rows `(B, channels, length)`; NaN marks a missing value.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct Covariates<'a> {
    pub values: &'a [f32],
    pub channels: usize,
}

/// The context to forecast from, with optional covariates.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct ForecastInput<'a> {
    /// `(batch, variates, T)` values; NaN marks a missing value. The variates of a series
    /// are forecast jointly.
    pub context: &'a [f32],
    pub batch: usize,
    pub variates: usize,
    /// Optional mask shaped like `context`; `false` marks additional missing values.
    pub observation_mask: Option<&'a [bool]>,
    /// `(batch, C_past, T)` covariates known over the context only.
    pub past_covariates: Option<Covariates<'a>>,
    /// `(batch, C_known, T + prediction_length)` covariates known over the context and the
    /// horizon. Rows whose values are all 0 or 1 are treated as indicators.
    pub known_covariates: Option<Covariates<'a>>,
}

impl<'a> ForecastInput<'a> {
    /// `context` shaped `(batch, variates, T)`.
    pub fn new(context: &'a [f32], batch: usize, variates: usize) -> Self {
        Self {
            context,
            batch,
            variates,
            observation_mask: None,
            past_covariates: None,
            known_covariates: None,
        }
    }

    /// One series with one variate.
    pub fn univariate(context: &'a [f32]) -> Self {
        Self::new(context, 1, 1)
    }

    pub fn with_observation_mask(mut self, mask: &'a [bool]) -> Self {
        self.observation_mask = Some(mask);
        self
    }

    pub fn with_past_covariates(mut self, values: &'a [f32], channels: usize) -> Self {
        self.past_covariates = Some(Covariates { values, channels });
        self
    }

    pub fn with_known_covariates(mut self, values: &'a [f32], channels: usize) -> Self {
        self.known_covariates = Some(Covariates { values, channels });
        self
    }

    /// Context length `T`.
    pub fn len(&self) -> usize {
        self.context.len() / (self.batch * self.variates).max(1)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct ForecastOptions {
    /// Horizon in time steps.
    pub prediction_length: usize,
    /// Average with the reflected forecast of `-context`, with every covariate negated too.
    pub symmetric_averaging: bool,
    /// Clamp at 0 the series whose observed context is non-negative.
    pub clip_nonnegative: bool,
}

impl ForecastOptions {
    pub fn new(prediction_length: usize) -> Self {
        Self {
            prediction_length,
            symmetric_averaging: true,
            clip_nonnegative: true,
        }
    }

    pub fn with_symmetric_averaging(mut self, enabled: bool) -> Self {
        self.symmetric_averaging = enabled;
        self
    }

    pub fn with_clip_nonnegative(mut self, enabled: bool) -> Self {
        self.clip_nonnegative = enabled;
        self
    }
}

/// Quantile forecast `(batch, variates, horizon, Q)`, sorted along the quantile axis.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Forecast {
    pub quantiles: Vec<f32>,
    pub batch: usize,
    pub variates: usize,
    pub horizon: usize,
    /// The `Q` probability levels.
    pub quantile_levels: Vec<f64>,
}

impl Forecast {
    pub fn num_quantiles(&self) -> usize {
        self.quantile_levels.len()
    }

    pub fn get(&self, series: usize, variate: usize, step: usize, quantile: usize) -> f32 {
        let q = self.num_quantiles();
        self.quantiles[((series * self.variates + variate) * self.horizon + step) * q + quantile]
    }

    /// `(batch, variates, horizon)` forecast at one of the model's quantile levels.
    pub fn quantile(&self, level: f64) -> Result<Vec<f32>> {
        let index = self
            .quantile_levels
            .iter()
            .position(|&l| l == level)
            .ok_or_else(|| {
                Error::Input(format!(
                    "level {level} is not one of {:?}",
                    self.quantile_levels
                ))
            })?;
        Ok(self
            .quantiles
            .iter()
            .skip(index)
            .step_by(self.num_quantiles())
            .copied()
            .collect())
    }

    /// `(batch, variates, horizon)` median forecast.
    pub fn median(&self) -> Vec<f32> {
        self.quantile(0.5).expect("quantile levels contain 0.5")
    }
}

/// Packed rows `(R, len)`.
#[derive(Debug, Clone)]
struct Rows {
    values: Vec<f32>,
    observed: Vec<bool>,
    len: usize,
}

impl Rows {
    fn row(&self, r: usize) -> (&[f32], &[bool]) {
        let span = r * self.len..(r + 1) * self.len;
        (&self.values[span.clone()], &self.observed[span])
    }
}

/// Target and covariate rows of every series, ready for the decode loop.
struct Assembled {
    rows: Rows,
    /// Rows per series: targets, then past covariates, then known covariates.
    rows_per_series: usize,
    roles: Option<Vec<Role>>,
    /// `(R, horizon padded to whole patches)` known-future values and observation mask.
    known_future: Option<(Vec<f32>, Vec<bool>)>,
}

/// Block-autoregressive quantile forecaster around a [`TimestralModel`].
pub struct TimestralForecaster {
    model: TimestralModel,
    /// Context cap in time steps (most recent kept); defaults to the trained context length.
    max_context: usize,
    per_shot_horizon_patches: usize,
    /// Clamp forecasts to `[min - k * s, max + k * s]` of the observed context, `s` being
    /// the scale of the last context patch. 0 disables.
    quantile_real_cap_k: f32,
    /// Fed-back medians at or above this magnitude become 0.
    extreme_value_threshold: f32,
}

impl std::fmt::Debug for TimestralForecaster {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TimestralForecaster")
            .field("max_context", &self.max_context)
            .field("per_shot_horizon_patches", &self.per_shot_horizon_patches)
            .field("quantile_real_cap_k", &self.quantile_real_cap_k)
            .field("extreme_value_threshold", &self.extreme_value_threshold)
            .finish_non_exhaustive()
    }
}

impl TimestralForecaster {
    pub fn new(model: TimestralModel) -> Self {
        let c = model.config();
        Self {
            max_context: c.context_length,
            per_shot_horizon_patches: c.block_size.min(32),
            quantile_real_cap_k: 1e4,
            extreme_value_threshold: 1e30,
            model,
        }
    }

    /// Load a checkpoint directory holding `config.json` and `model.safetensors`.
    pub fn from_dir(dir: impl AsRef<Path>, device: &Device) -> Result<Self> {
        Ok(Self::new(crate::load_model(dir, device)?))
    }

    /// Load a local checkpoint directory or a Hugging Face Hub repo id such as
    /// `mistralai/Timestral-5M`; files are cached by `hf-hub`.
    #[cfg(feature = "hub")]
    pub fn from_pretrained(path_or_repo_id: &str, device: &Device) -> Result<Self> {
        Self::from_dir(crate::loading::resolve_checkpoint(path_or_repo_id)?, device)
    }

    pub fn with_max_context(mut self, max_context: usize) -> Self {
        self.max_context = max_context.max(1);
        self
    }

    /// Clamp forecasts to `[min - k * s, max + k * s]` of the observed context, `s` being
    /// the scale of the last context patch. 0 disables; defaults to 1e4.
    pub fn with_quantile_real_cap_k(mut self, k: f32) -> Self {
        self.quantile_real_cap_k = k;
        self
    }

    /// Fed-back medians at or above this magnitude become 0; defaults to 1e30.
    pub fn with_extreme_value_threshold(mut self, threshold: f32) -> Self {
        self.extreme_value_threshold = threshold;
        self
    }

    /// Patches decoded per block, capped at the trained block size.
    pub fn with_per_shot_horizon_patches(mut self, patches: usize) -> Self {
        self.per_shot_horizon_patches = patches.clamp(1, self.config().block_size);
        self
    }

    /// Context cap in time steps; see [`with_max_context`](Self::with_max_context).
    pub fn max_context(&self) -> usize {
        self.max_context
    }

    /// See [`with_quantile_real_cap_k`](Self::with_quantile_real_cap_k).
    pub fn quantile_real_cap_k(&self) -> f32 {
        self.quantile_real_cap_k
    }

    /// See [`with_extreme_value_threshold`](Self::with_extreme_value_threshold).
    pub fn extreme_value_threshold(&self) -> f32 {
        self.extreme_value_threshold
    }

    /// See [`with_per_shot_horizon_patches`](Self::with_per_shot_horizon_patches).
    pub fn per_shot_horizon_patches(&self) -> usize {
        self.per_shot_horizon_patches
    }

    pub fn model(&self) -> &TimestralModel {
        &self.model
    }

    pub fn config(&self) -> &Config {
        self.model.config()
    }

    pub fn quantile_levels(&self) -> &[f64] {
        &self.config().quantile_knots
    }

    /// Forecast `options.prediction_length` steps after the context.
    pub fn forecast(&self, input: &ForecastInput, options: &ForecastOptions) -> Result<Forecast> {
        let h = options.prediction_length;
        let (b, v) = (input.batch, input.variates);
        if h == 0 {
            return Err(Error::Input("prediction_length must be >= 1".into()));
        }
        if b == 0
            || v == 0
            || input.context.is_empty()
            || !input.context.len().is_multiple_of(b * v)
        {
            return Err(Error::Input(format!(
                "context of {} values is not (batch={b}, variates={v}, T>0)",
                input.context.len()
            )));
        }
        let t = input.len();
        if input
            .observation_mask
            .is_some_and(|m| m.len() != input.context.len())
        {
            return Err(Error::Input(
                "observation_mask must be shaped like context".into(),
            ));
        }
        let observed: Vec<bool> = match input.observation_mask {
            Some(mask) => input
                .context
                .iter()
                .zip(mask)
                .map(|(x, &m)| x.is_finite() && m)
                .collect(),
            None => input.context.iter().map(|x| x.is_finite()).collect(),
        };
        let values: Vec<f32> = input
            .context
            .iter()
            .zip(&observed)
            .map(|(&x, &o)| if o { x } else { 0.0 })
            .collect();
        let past = self.check_covariates(input.past_covariates, b, t, "past_covariates")?;
        let known = self.check_covariates(input.known_covariates, b, t + h, "known_covariates")?;
        if known.is_some() && !self.config().supports_known_covariates() {
            return Err(Error::Input(
                "this checkpoint cannot condition on known-future covariates".into(),
            ));
        }

        let mut quantiles = if options.symmetric_averaging {
            let levels = self.quantile_levels();
            if levels
                .iter()
                .zip(levels.iter().rev())
                .any(|(lo, hi)| (lo + hi - 1.0).abs() >= 1e-9)
            {
                return Err(Error::Config(format!(
                    "symmetric averaging needs quantile levels symmetric about 0.5, got {levels:?}"
                )));
            }
            let mut quantiles = self.forecast_raw(&values, &observed, b, t, h, past, known)?;
            // The mirror pass forecasts `-context` with every covariate negated. Batching it
            // with the first pass gives the same values but doubles the attention masks.
            let negate = |x: &[f32]| x.iter().map(|v| -v).collect::<Vec<f32>>();
            let (neg_past, neg_known) = (
                past.map(|c| negate(c.values)),
                known.map(|c| negate(c.values)),
            );
            let mirrored = self.forecast_raw(
                &negate(&values),
                &observed,
                b,
                t,
                h,
                past.zip(neg_past.as_deref())
                    .map(|(c, values)| Covariates { values, ..c }),
                known
                    .zip(neg_known.as_deref())
                    .map(|(c, values)| Covariates { values, ..c }),
            )?;
            let q = levels.len();
            for (out, mirror) in quantiles.chunks_exact_mut(q).zip(mirrored.chunks_exact(q)) {
                for (i, o) in out.iter_mut().enumerate() {
                    *o = 0.5 * (*o + -mirror[q - 1 - i]);
                }
            }
            quantiles
        } else {
            self.forecast_raw(&values, &observed, b, t, h, past, known)?
        };
        if options.clip_nonnegative {
            let per_row = h * self.quantile_levels().len();
            for (r, out) in quantiles.chunks_exact_mut(per_row).enumerate() {
                let (row, obs) = (&values[r * t..(r + 1) * t], &observed[r * t..(r + 1) * t]);
                let any = obs.iter().any(|&o| o);
                if any && row.iter().zip(obs).all(|(&x, &o)| !o || x >= 0.0) {
                    out.iter_mut().for_each(|x| *x = x.max(0.0));
                }
            }
        }
        Ok(Forecast {
            quantiles,
            batch: b,
            variates: v,
            horizon: h,
            quantile_levels: self.quantile_levels().to_vec(),
        })
    }

    fn check_covariates<'a>(
        &self,
        covariates: Option<Covariates<'a>>,
        batch: usize,
        length: usize,
        name: &str,
    ) -> Result<Option<Covariates<'a>>> {
        let Some(c) = covariates.filter(|c| c.channels > 0) else {
            return Ok(None);
        };
        if !self.config().covariate_enabled {
            return Err(Error::Input(format!(
                "this checkpoint was not trained with covariates; got {name}"
            )));
        }
        if c.values.len() != batch * c.channels * length {
            return Err(Error::Input(format!(
                "{name} must be (B={batch}, C={}, {length}), got {} values",
                c.channels,
                c.values.len()
            )));
        }
        Ok(Some(c))
    }

    /// One forecast pass: `(B, V, T)` inputs to `(B, V, H, Q)` raw-space quantiles.
    #[allow(clippy::too_many_arguments)]
    fn forecast_raw(
        &self,
        values: &[f32],
        observed: &[bool],
        b: usize,
        t: usize,
        h: usize,
        past: Option<Covariates>,
        known: Option<Covariates>,
    ) -> Result<Vec<f32>> {
        let v = values.len() / (b * t);
        let assembled = self.assemble_rows(values, observed, b, t, h, past, known);
        let decoded = self.block_ar_decode(&assembled, h)?;
        let q = self.quantile_levels().len();
        let per_row = h * q;
        let mut out = Vec::with_capacity(b * v * per_row);
        for bi in 0..b {
            for vi in 0..v {
                let r = bi * assembled.rows_per_series + vi;
                let start = out.len();
                out.extend_from_slice(&decoded[r * per_row..(r + 1) * per_row]);
                if self.quantile_real_cap_k > 0.0 {
                    let (row, obs) = assembled.rows.row(r);
                    self.cap_quantiles(&mut out[start..], row, obs);
                }
            }
        }
        let nans = out.iter().filter(|x| x.is_nan()).count();
        if nans > 0 {
            return Err(Error::Input(format!(
                "forecast produced {nans} NaN values; the input is degenerate"
            )));
        }
        Ok(out)
    }

    /// Clamp one row's quantiles to `[min - k * s, max + k * s]` of its observed context.
    fn cap_quantiles(&self, quantiles: &mut [f32], row: &[f32], observed: &[bool]) {
        let obs = || {
            row.iter()
                .zip(observed)
                .filter(|&(_, &o)| o)
                .map(|(&x, _)| x)
        };
        let ctx_max = obs().fold(f32::NEG_INFINITY, f32::max);
        let ctx_min = obs().fold(f32::INFINITY, f32::min);
        let (ctx_max, ctx_min) = if ctx_max.is_finite() {
            (ctx_max, ctx_min)
        } else {
            (0.0, 0.0)
        };
        let c = self.config();
        let (_, scale) = causal_patch_loc_scale(row, observed, c.patch_size, c.minimum_scale);
        let anchor = scale[scale.len() - 1] * self.quantile_real_cap_k;
        let (lo, hi) = (ctx_min - anchor, ctx_max + anchor);
        for x in quantiles.iter_mut() {
            // Comparisons keep NaN, like torch.clamp.
            if *x < lo {
                *x = lo;
            } else if *x > hi {
                *x = hi;
            }
        }
    }

    /// Stack target and covariate rows of every series, capped at `max_context` and
    /// left-padded to whole patches.
    #[allow(clippy::too_many_arguments)]
    fn assemble_rows(
        &self,
        values: &[f32],
        observed: &[bool],
        b: usize,
        t: usize,
        h: usize,
        past: Option<Covariates>,
        known: Option<Covariates>,
    ) -> Assembled {
        let ps = self.config().patch_size;
        let v = values.len() / (b * t);
        let n_past = past.map_or(0, |c| c.channels);
        let n_known = known.map_or(0, |c| c.channels);
        let rows_per_series = v + n_past + n_known;
        let start = t.saturating_sub(self.max_context);
        let len = t - start;
        let pad = (ps - len % ps) % ps;
        let padded = pad + len;

        let mut rows = Rows {
            values: Vec::with_capacity(b * rows_per_series * padded),
            observed: Vec::with_capacity(b * rows_per_series * padded),
            len: padded,
        };
        let mut push = |vals: &[f32], obs: &[bool]| {
            rows.values
                .extend(std::iter::repeat_n(0.0, pad).chain(vals[start..t].iter().copied()));
            rows.observed
                .extend(std::iter::repeat_n(false, pad).chain(obs[start..t].iter().copied()));
        };
        let cov_row = |vals: &[f32]| -> (Vec<f32>, Vec<bool>) {
            (
                vals.iter()
                    .map(|x| if x.is_nan() { 0.0 } else { *x })
                    .collect(),
                vals.iter().map(|x| !x.is_nan()).collect(),
            )
        };
        for bi in 0..b {
            for vi in 0..v {
                // The reference forward-fills targets here; nothing reads unobserved values,
                // so they stay 0.
                let span = (bi * v + vi) * t..(bi * v + vi + 1) * t;
                push(&values[span.clone()], &observed[span]);
            }
            for ci in 0..n_past {
                let c = past.unwrap();
                let (vals, obs) = cov_row(&c.values[(bi * c.channels + ci) * t..][..t]);
                push(&vals, &obs);
            }
            for ci in 0..n_known {
                let c = known.unwrap();
                let (vals, obs) = cov_row(&c.values[(bi * c.channels + ci) * (t + h)..][..t]);
                push(&vals, &obs);
            }
        }
        if n_past == 0 && n_known == 0 {
            return Assembled {
                rows,
                rows_per_series,
                roles: None,
                known_future: None,
            };
        }

        let series_roles = std::iter::repeat_n(Role::Target, v)
            .chain(std::iter::repeat_n(Role::PastCovariate, n_past))
            .chain(std::iter::repeat_n(Role::KnownCovariate, n_known));
        let roles: Vec<Role> = (0..b).flat_map(|_| series_roles.clone()).collect();
        let known_future = known.map(|c| {
            let horizon = h.div_ceil(ps) * ps;
            let mut kf = vec![0.0f32; b * rows_per_series * horizon];
            let mut kfm = vec![false; b * rows_per_series * horizon];
            for bi in 0..b {
                for ci in 0..c.channels {
                    let (vals, obs) =
                        cov_row(&c.values[(bi * c.channels + ci) * (t + h) + t..][..h]);
                    let r = bi * rows_per_series + v + n_past + ci;
                    kf[r * horizon..][..h].copy_from_slice(&vals);
                    kfm[r * horizon..][..h].copy_from_slice(&obs);
                }
            }
            (kf, kfm)
        });
        Assembled {
            rows,
            rows_per_series,
            roles: Some(roles),
            known_future,
        }
    }

    /// Decode block by block; returns `(R, H, Q)`.
    ///
    /// After each block the context grows by the block's feedback: targets carry their
    /// median, known-future covariates their real future, past-only covariates a blank.
    /// Past `max_context` the oldest steps are dropped.
    fn block_ar_decode(&self, assembled: &Assembled, h: usize) -> Result<Vec<f32>> {
        let c = self.config();
        let ps = c.patch_size;
        let q = c.num_quantiles();
        let m = c.median_index();
        let mut ctx = assembled.rows.clone();
        let r_count = ctx.values.len() / ctx.len;
        let group_ids: Vec<usize> = (0..r_count)
            .map(|r| r / assembled.rows_per_series)
            .collect();
        let roles = assembled.roles.as_deref();
        let target_rows: Option<Vec<bool>> =
            roles.map(|r| r.iter().map(|&r| r == Role::Target).collect());
        let horizon = h.div_ceil(ps) * ps;

        let mut out: Vec<Vec<f32>> = vec![Vec::with_capacity(horizon * q); r_count];
        let mut remaining = h.div_ceil(ps);
        let mut offset = 0;
        while remaining > 0 {
            let block = self.per_shot_horizon_patches.min(remaining);
            let blen = block * ps;
            let kf_block = assembled.known_future.as_ref().map(|(kf, kfm)| {
                (
                    columns(kf, horizon, offset, blen),
                    columns(kfm, horizon, offset, blen),
                )
            });
            let kf_ref = kf_block.as_ref().map(|(v, o)| (&v[..], &o[..]));
            let mut quantiles = self.model.predict_block(
                &ctx.values,
                &ctx.observed,
                &group_ids,
                block,
                roles,
                kf_ref,
            )?;
            self.refine_block(&mut quantiles, &ctx, target_rows.as_deref());
            for (r, chunk) in quantiles.chunks_exact(blen * q).enumerate() {
                out[r].extend_from_slice(chunk);
            }
            remaining -= block;
            offset += blen;
            if remaining == 0 {
                break;
            }

            let new_len = ctx.len + blen;
            let keep = if new_len > self.max_context {
                ps.max(self.max_context / ps * ps)
            } else {
                new_len
            };
            let mut next = Rows {
                values: Vec::with_capacity(r_count * keep),
                observed: Vec::with_capacity(r_count * keep),
                len: keep,
            };
            for r in 0..r_count {
                let mut feedback: Vec<f32> = quantiles[r * blen * q..(r + 1) * blen * q]
                    .iter()
                    .skip(m)
                    .step_by(q)
                    .map(|&x| {
                        if x.is_finite() && x.abs() < self.extreme_value_threshold {
                            x
                        } else {
                            0.0
                        }
                    })
                    .collect();
                let mut feedback_mask = vec![true; blen];
                match roles.map(|roles| roles[r]) {
                    Some(Role::KnownCovariate) => {
                        let (vals, obs) = kf_block
                            .as_ref()
                            .expect("known rows come with known_future");
                        feedback.copy_from_slice(&vals[r * blen..(r + 1) * blen]);
                        feedback_mask.copy_from_slice(&obs[r * blen..(r + 1) * blen]);
                    }
                    Some(Role::PastCovariate) => {
                        feedback.fill(0.0);
                        feedback_mask.fill(false);
                    }
                    _ => {}
                }
                let (vals, obs) = ctx.row(r);
                let skip = new_len - keep;
                next.values.extend(vals.iter().chain(&feedback).skip(skip));
                next.observed
                    .extend(obs.iter().chain(&feedback_mask).skip(skip));
            }
            ctx = next;
        }
        Ok(out
            .into_iter()
            .flat_map(|row| row.into_iter().take(h * q))
            .collect())
    }

    /// Move one decoded block `(R, H * patch_size, Q)` from the anchor frame to the
    /// advancing refine frame it was trained in.
    ///
    /// The model denormalizes the whole block with the frame of the last context patch.
    /// The first horizon patch keeps that frame; each later one is re-denormalized with
    /// [`advancing_block_frames`]. Covariate rows keep their quantiles.
    fn refine_block(&self, quantiles: &mut [f32], ctx: &Rows, target_rows: Option<&[bool]>) {
        let c = self.config();
        let (ps, q, m) = (c.patch_size, c.num_quantiles(), c.median_index());
        let r_count = ctx.values.len() / ctx.len;
        let row_len = quantiles.len() / r_count;
        let n_pred = row_len / q / ps;
        if n_pred <= 1 {
            return;
        }
        for (r, row_q) in quantiles.chunks_exact_mut(row_len).enumerate() {
            if target_rows.is_some_and(|t| !t[r]) {
                continue;
            }
            let (vals, obs) = ctx.row(r);
            let (loc, scale) = causal_patch_loc_scale(vals, obs, ps, c.minimum_scale);
            let (loc0, scale0) = (loc[loc.len() - 1], scale[scale.len() - 1]);
            let sinh_median: Vec<f64> = row_q
                .iter()
                .skip(m)
                .step_by(q)
                .map(|&x| (x as f64 - loc0 as f64) / scale0 as f64)
                .collect();
            let (loc_j, scale_j) =
                advancing_block_frames(vals, obs, &sinh_median, ps, c.minimum_scale);
            for (step, quantiles) in row_q.chunks_exact_mut(q).enumerate() {
                let p = step / ps;
                let (l, s) = if p == 0 {
                    (loc0, scale0)
                } else {
                    (loc_j[p] as f32, scale_j[p] as f32)
                };
                for x in quantiles.iter_mut() {
                    *x = (*x - loc0) / scale0 * s + l;
                }
                clamp_nonfinite(quantiles);
            }
        }
    }
}

/// Columns `offset..offset + width` of every row of a `(R, len)` array.
fn columns<T: Copy>(x: &[T], len: usize, offset: usize, width: usize) -> Vec<T> {
    x.chunks_exact(len)
        .flat_map(|row| row[offset..offset + width].iter().copied())
        .collect()
}
