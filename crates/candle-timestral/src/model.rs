//! The Timestral network and its single-block forecast primitive.
//!
//! The model normalizes each row causally per patch, embeds patches from `[values,
//! missing]` channels, runs a time/variate transformer over `[Enc ; Dec]` tokens (context
//! patches followed by learned forecast-query tokens), and reads quantiles for every
//! horizon time step from the decoder tokens.
//!
//! Inputs use the packed layout: `R` rows (one per variate of every series), each a
//! row-major slice of `T` time steps.

use candle_core::{Device, Module, Tensor};
use candle_nn::VarBuilder;

use crate::{
    Error, Result,
    config::Config,
    layers::{AttentionInputs, ResidualBlock, TimeBias, TimeMask, Transformer, VariateMask},
    masks::{self, Mask},
    scaler::{self, affine_asinh, causal_patch_loc_scale, clamp_nonfinite, invert_affine_asinh},
};

/// What a packed row is to the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// A series to forecast.
    Target,
    /// A covariate known over the context only.
    PastCovariate,
    /// A covariate known over the context and the horizon.
    KnownCovariate,
}

pub struct TimestralModel {
    config: Config,
    patch_proj: ResidualBlock,
    forecast_query: Tensor,
    transformer: Transformer,
    head: ResidualBlock,
    /// `width_factor * dp` per quantile level, `dp` being the probability gap to the next level inward.
    quantile_widths: Vec<f32>,
    device: Device,
}

impl TimestralModel {
    /// Build the model from a validated config and a `VarBuilder` over the released weights.
    pub fn new(config: Config, vb: VarBuilder) -> Result<Self> {
        config.validate()?;
        let c = &config;
        let repr = vb.pp("representation");
        let patch_proj = ResidualBlock::new(
            2 * c.patch_size,
            4 * c.d_model,
            c.d_model,
            true,
            repr.pp("patch_proj"),
        )?;
        let forecast_query = repr.get(c.d_model, "forecast_query")?;
        let transformer = Transformer::new(c, vb.pp("transformer"))?;
        let head = ResidualBlock::new(
            c.d_model,
            4 * c.d_model,
            c.patch_size * c.num_quantiles(),
            false,
            vb.pp("head").pp("projection"),
        )?;
        let quantile_widths = quantile_widths(&c.quantile_knots, c.quantile_width_factor);
        Ok(Self {
            device: vb.device().clone(),
            config,
            patch_proj,
            forecast_query,
            transformer,
            head,
            quantile_widths,
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Causal per-patch normalization of packed rows of length `len`.
    ///
    /// Returns `(scaled, loc, scale)`: `scaled` is `(R, len)`, `loc` and `scale` are per
    /// patch, `(R, len / patch_size)`.
    pub fn scale(
        &self,
        inputs: &[f32],
        observed: &[bool],
        len: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let c = &self.config;
        let ps = c.patch_size;
        let (mut scaled, mut locs, mut scales) =
            (Vec::with_capacity(inputs.len()), Vec::new(), Vec::new());
        for (row, obs) in inputs.chunks_exact(len).zip(observed.chunks_exact(len)) {
            let (loc, scale) = causal_patch_loc_scale(row, obs, ps, c.minimum_scale);
            for (t, (&x, &o)) in row.iter().zip(obs).enumerate() {
                scaled.push(affine_asinh(x, loc[t / ps], scale[t / ps], o, c.use_asinh));
            }
            locs.extend(loc);
            scales.extend(scale);
        }
        (scaled, locs, scales)
    }

    /// Forecast one block of `horizon_patches` patches after a packed context.
    ///
    /// - `inputs`, `observed`: `(R, T)` context values and observation mask, `T` a multiple
    ///   of the patch size.
    /// - `group_ids`: `(R,)` series id per row; rows of one series attend each other.
    /// - `roles`: `(R,)` row roles, or None when every row is a target.
    /// - `known_future`: `(R, horizon_patches * patch_size)` known-future covariate values
    ///   and observation mask over this block; zeros/false on other rows.
    ///
    /// Returns raw-space quantiles `(R, horizon_patches * patch_size, Q)`, non-decreasing
    /// along `Q`.
    pub fn predict_block(
        &self,
        inputs: &[f32],
        observed: &[bool],
        group_ids: &[usize],
        horizon_patches: usize,
        roles: Option<&[Role]>,
        known_future: Option<(&[f32], &[bool])>,
    ) -> Result<Vec<f32>> {
        let rows = group_ids.len();
        let ps = self.config.patch_size;
        if rows == 0 || !inputs.len().is_multiple_of(rows) || observed.len() != inputs.len() {
            return Err(Error::Input(format!(
                "inputs ({}) and observed ({}) must be (R={rows}, T)",
                inputs.len(),
                observed.len()
            )));
        }
        let t = inputs.len() / rows;
        if t == 0 || !t.is_multiple_of(ps) {
            return Err(Error::Input(format!(
                "context length {t} must be a positive multiple of patch_size {ps}"
            )));
        }
        if horizon_patches == 0 {
            return Err(Error::Input("horizon_patches must be >= 1".into()));
        }
        if roles.is_some_and(|r| r.len() != rows) {
            return Err(Error::Input(format!("roles must have {rows} entries")));
        }
        let all_targets = roles.is_none_or(|r| r.iter().all(|&r| r == Role::Target));
        if all_targets || !self.config.covariate_enabled {
            return self.predict_context_only(inputs, observed, t, group_ids, horizon_patches);
        }
        let horizon_len = horizon_patches * ps;
        let (zeros, blank);
        let (kf, kfm) = match known_future {
            Some(kf) => kf,
            None => {
                zeros = vec![0.0f32; rows * horizon_len];
                blank = vec![false; rows * horizon_len];
                (&zeros[..], &blank[..])
            }
        };
        if kf.len() != rows * horizon_len || kfm.len() != kf.len() {
            return Err(Error::Input(format!(
                "known_future must be ({rows}, {horizon_len})"
            )));
        }
        let w = t + horizon_len;
        let mut full_inputs = Vec::with_capacity(rows * w);
        let mut full_observed = Vec::with_capacity(rows * w);
        for r in 0..rows {
            full_inputs.extend_from_slice(&inputs[r * t..(r + 1) * t]);
            full_inputs.extend_from_slice(&kf[r * horizon_len..(r + 1) * horizon_len]);
            full_observed.extend_from_slice(&observed[r * t..(r + 1) * t]);
            full_observed.extend_from_slice(&kfm[r * horizon_len..(r + 1) * horizon_len]);
        }
        self.predict_window(
            &full_inputs,
            &full_observed,
            w,
            group_ids,
            horizon_patches,
            roles.unwrap(),
        )
    }

    /// No covariates: encoder over the context, `h` forecast-query tokens after it.
    fn predict_context_only(
        &self,
        inputs: &[f32],
        observed: &[bool],
        t: usize,
        group_ids: &[usize],
        h: usize,
    ) -> Result<Vec<f32>> {
        let c = &self.config;
        let rows = group_ids.len();
        let ll = t / c.patch_size;

        let (scaled, loc, scale) = self.scale(inputs, observed, t);
        let x_enc = self.embed(&scaled, observed, rows, ll)?;
        let x_dec = self.forecast_tokens(rows, h)?;

        let obs_per_patch = masks::observed_per_patch(observed, c.patch_size);
        let key_valid = masks::enc_key_validity(&obs_per_patch, ll, c.attend_missing_patches);
        let time_mask = self.time_mask(
            &masks::forecast_time_pattern(ll, h),
            None,
            masks::time_key_bias(&key_valid, ll, h),
            ll,
        )?;
        let variate_mask = self.variate_mask(group_ids, &key_valid, ll, h, None, None)?;
        let first_obs = masks::group_first_observed_patch(&obs_per_patch, ll, group_ids);
        let positions = masks::forecast_positions(ll, h, &first_obs);

        let mut preds = self.run_two_stream(x_enc, x_dec, time_mask, variate_mask, &positions)?;
        let anchors: Vec<(f32, f32)> = (0..rows)
            .map(|r| (loc[r * ll + ll - 1], scale[r * ll + ll - 1]))
            .collect();
        self.denormalize(&mut preds, &anchors);
        Ok(preds)
    }

    /// With covariates: encoder over context plus horizon, so known-future values are keys.
    ///
    /// Target and past-only rows keep a blank horizon. The time and variate masks keep
    /// target information out of covariate rows and keep any row's future out of the
    /// targets.
    fn predict_window(
        &self,
        full_inputs: &[f32],
        full_observed: &[bool],
        w: usize,
        group_ids: &[usize],
        h: usize,
        roles: &[Role],
    ) -> Result<Vec<f32>> {
        let c = &self.config;
        let ps = c.patch_size;
        let rows = group_ids.len();
        if h > c.block_size {
            return Err(Error::Input(format!(
                "decode block {h} must be in [1, {}] with covariates",
                c.block_size
            )));
        }
        let n = w / ps;
        let ll = n - h;
        let target_rows: Vec<bool> = roles.iter().map(|&r| r == Role::Target).collect();
        let known_rows: Vec<bool> = roles.iter().map(|&r| r == Role::KnownCovariate).collect();

        let (mut scaled, mut loc, mut scale) = self.scale(full_inputs, full_observed, w);
        // Known-future covariate rows use non-causal whole-row statistics; indicator rows skip the asinh squash.
        for r in (0..rows).filter(|&r| known_rows[r]) {
            let (values, obs) = (
                &full_inputs[r * w..(r + 1) * w],
                &full_observed[r * w..(r + 1) * w],
            );
            let (loc_c, scale_c, is_binary) =
                scaler::instance_norm_stats(values, obs, c.minimum_scale);
            for t in 0..w {
                let v = affine_asinh(values[t], loc_c, scale_c, obs[t], false);
                scaled[r * w + t] = if c.use_asinh && obs[t] && !is_binary {
                    v.asinh()
                } else {
                    v
                };
            }
            loc[r * n..(r + 1) * n].fill(loc_c);
            scale[r * n..(r + 1) * n].fill(scale_c);
        }
        let x_enc = self.embed(&scaled, full_observed, rows, n)?;
        let x_dec = self.forecast_tokens(rows, n)?;

        let obs_per_patch = masks::observed_per_patch(full_observed, ps);
        let key_valid = masks::enc_key_validity(&obs_per_patch, n, c.attend_missing_patches);
        let time_mask = self.time_mask(
            &masks::two_stream_time_patterns(ll, n, c.block_size),
            Some(known_rows.iter().map(|&k| u32::from(k)).collect()),
            masks::time_key_bias(&key_valid, n, n),
            n,
        )?;
        let variate_mask = self.variate_mask(
            group_ids,
            &key_valid,
            n,
            n,
            Some(&target_rows),
            Some(&known_rows),
        )?;
        let first_obs = masks::group_first_observed_patch(&obs_per_patch, n, group_ids);
        let positions = masks::two_stream_positions(n, &first_obs);

        let preds = self.run_two_stream(x_enc, x_dec, time_mask, variate_mask, &positions)?;
        let q = c.num_quantiles();
        let row_len = n * ps * q;
        let mut horizon_preds = Vec::with_capacity(rows * h * ps * q);
        for r in 0..rows {
            horizon_preds.extend_from_slice(&preds[r * row_len + ll * ps * q..(r + 1) * row_len]);
        }
        let anchors: Vec<(f32, f32)> = (0..rows)
            .map(|r| (loc[r * n + ll - 1], scale[r * n + ll - 1]))
            .collect();
        self.denormalize(&mut horizon_preds, &anchors);
        Ok(horizon_preds)
    }

    /// Time-attention bias over the `[Enc ; Dec]` sequence: `patterns` `(K, S, S)` per row
    /// kind, the kind of each row (None: all 0), and the `(R, S)` key bias. Split at
    /// `enc_len` when the config splits context and horizon attention.
    fn time_mask(
        &self,
        patterns: &Mask,
        kinds: Option<Vec<u32>>,
        keys: Vec<f32>,
        enc_len: usize,
    ) -> Result<TimeMask> {
        let dev = &self.device;
        let (k, s) = (patterns.batch, patterns.keys);
        let rows = keys.len() / s;
        let keys = Tensor::from_vec(keys, (rows, 1, 1, s), dev)?;
        let kinds = kinds
            .map(|kinds| Tensor::from_vec(kinds, rows, dev))
            .transpose()?;
        let bias = |q: std::ops::Range<usize>, k_len: usize| -> Result<TimeBias> {
            Ok(TimeBias {
                patterns: Tensor::from_vec(
                    patterns.bias(q.clone(), 0..k_len),
                    (k, 1, q.len(), k_len),
                    dev,
                )?,
                kinds: kinds.clone(),
                keys: keys.narrow(3, 0, k_len)?,
            })
        };
        Ok(if self.config.split_context_horizon_attn {
            TimeMask::Split {
                enc: bias(0..enc_len, enc_len)?,
                dec: bias(enc_len..s, s)?,
                enc_len,
            }
        } else {
            TimeMask::Full(bias(0..s, s)?)
        })
    }

    /// Variate-attention mask over `n_enc + n_dec` slots; see [`VariateMask`].
    fn variate_mask(
        &self,
        group_ids: &[usize],
        enc_key_valid: &[bool],
        n_enc: usize,
        n_dec: usize,
        target_rows: Option<&[bool]>,
        bidirectional_rows: Option<&[bool]>,
    ) -> Result<VariateMask> {
        let dev = &self.device;
        Ok(match masks::uniform_groups(group_ids) {
            Some(1) => VariateMask::SelfOnly,
            Some(size) => {
                let m = masks::grouped_variate_mask(
                    size,
                    enc_key_valid,
                    n_enc,
                    n_dec,
                    target_rows,
                    bidirectional_rows,
                );
                let bias =
                    Tensor::from_vec(m.bias(0..size, 0..size), (m.batch, 1, size, size), dev)?;
                VariateMask::Groups { size, bias }
            }
            None => {
                let rows = group_ids.len();
                let m = masks::two_stream_variate_mask(
                    group_ids,
                    enc_key_valid,
                    n_enc,
                    n_dec,
                    target_rows,
                    bidirectional_rows,
                )
                .expect("several rows when series are not uniform runs");
                VariateMask::Dense(Tensor::from_vec(
                    m.bias(0..rows, 0..rows),
                    (m.batch, 1, rows, rows),
                    dev,
                )?)
            }
        })
    }

    /// Patch embedding over per-step `[values, missing]` channels: `(R, n * ps)` to `(R, n, D)`.
    fn embed(&self, scaled: &[f32], observed: &[bool], rows: usize, n: usize) -> Result<Tensor> {
        let ps = self.config.patch_size;
        let mut x = Vec::with_capacity(rows * n * 2 * ps);
        for (values, obs) in scaled.chunks_exact(ps).zip(observed.chunks_exact(ps)) {
            x.extend_from_slice(values);
            x.extend(obs.iter().map(|&o| if o { 0.0f32 } else { 1.0 }));
        }
        let x = Tensor::from_vec(x, (rows, n, 2 * ps), &self.device)?;
        Ok(self.patch_proj.forward(&x)?)
    }

    fn forecast_tokens(&self, rows: usize, n: usize) -> Result<Tensor> {
        Ok(self
            .forecast_query
            .reshape((1, 1, self.config.d_model))?
            .repeat((rows, n, 1))?)
    }

    /// Transformer over `[Enc ; Dec]`, then the head on the decoder tokens.
    ///
    /// Returns normalized-space quantiles `(R, N_dec * patch_size, Q)`.
    fn run_two_stream(
        &self,
        x_enc: Tensor,
        x_dec: Tensor,
        time_mask: TimeMask,
        variate_mask: VariateMask,
        positions: &[i64],
    ) -> Result<Vec<f32>> {
        let (rows, enc_len, _) = x_enc.dims3()?;
        let x = Tensor::cat(&[x_enc, x_dec], 1)?;
        let s = x.dim(1)?;
        let dev = &self.device;
        let (cos, sin) = self.transformer.rope().tables(positions, rows, dev)?;
        let inputs = AttentionInputs {
            time_mask,
            variate_mask,
            cos,
            sin,
        };
        let h = self.transformer.forward(x, &inputs)?;
        let raw = self.head.forward(&h.narrow(1, enc_len, s - enc_len)?)?;
        // (R, N, ps * Q) has the same memory layout as (R, N * ps, Q).
        let mut raw: Vec<f32> = raw.flatten_all()?.to_vec1()?;
        let m = self.config.median_index();
        for quantiles in raw.chunks_exact_mut(self.config.num_quantiles()) {
            monotone_quantiles(quantiles, m, &self.quantile_widths);
        }
        Ok(raw)
    }

    /// Invert the normalization per row with its anchor `(loc, scale)` and clamp infinities.
    fn denormalize(&self, preds: &mut [f32], anchors: &[(f32, f32)]) {
        let q = self.config.num_quantiles();
        let row_len = preds.len() / anchors.len();
        for (row, &(loc, scale)) in preds.chunks_exact_mut(row_len).zip(anchors) {
            for quantiles in row.chunks_exact_mut(q) {
                for v in quantiles.iter_mut() {
                    *v = invert_affine_asinh(*v, loc, scale, self.config.use_asinh);
                }
                clamp_nonfinite(quantiles);
            }
        }
    }
}

fn quantile_widths(levels: &[f64], width_factor: f64) -> Vec<f32> {
    let m = levels
        .iter()
        .position(|&q| q == 0.5)
        .expect("validated: knots contain 0.5");
    (0..levels.len())
        .map(|i| {
            let dp = match i.cmp(&m) {
                std::cmp::Ordering::Greater => levels[i] - levels[i - 1],
                std::cmp::Ordering::Less => levels[i + 1] - levels[i],
                std::cmp::Ordering::Equal => 0.0,
            };
            width_factor as f32 * dp as f32
        })
        .collect()
}

/// `torch.nn.functional.softplus` with its default threshold of 20.
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

/// Non-crossing quantiles from raw head outputs, in place (`monotone_cumsum`).
///
/// The median channel `m` is free. The others fan out from it as
/// `median -/+ cumsum(width_factor * dp * softplus(raw))`.
pub fn monotone_quantiles(raw: &mut [f32], m: usize, widths: &[f32]) {
    let median = raw[m];
    let mut acc = 0.0f32;
    for i in m + 1..raw.len() {
        acc += widths[i] * softplus(raw[i]);
        raw[i] = median + acc;
    }
    acc = 0.0;
    for i in (0..m).rev() {
        acc += widths[i] * softplus(raw[i]);
        raw[i] = median - acc;
    }
    for v in raw.iter_mut() {
        if v.is_infinite() {
            *v = v.signum() * f32::MAX;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monotone_quantiles_are_sorted() {
        let levels = [0.1, 0.25, 0.5, 0.75, 0.9];
        let widths = quantile_widths(&levels, 2.0);
        assert_eq!(widths[2], 0.0);
        assert!((widths[0] - 0.3).abs() < 1e-6 && (widths[4] - 0.3).abs() < 1e-6);
        let mut raw = [3.0f32, -40.0, 0.7, 25.0, -1.0];
        monotone_quantiles(&mut raw, 2, &widths);
        assert_eq!(raw[2], 0.7);
        assert!(raw.windows(2).all(|w| w[0] <= w[1]), "{raw:?}");
    }
}
