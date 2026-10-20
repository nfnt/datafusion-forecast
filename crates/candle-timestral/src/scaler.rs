//! Normalization for Timestral: causal per-patch mean/std scaling with an asinh squash.
//!
//! Each patch is normalized with the mean and std of every observed value up to the end of
//! that patch, so the context never sees future statistics. Forecasts come back in the
//! normalized (asinh) space of the last context patch and are inverted with `sinh`.
//!
//! All functions work on one row (a single variate) at a time. Statistics are accumulated
//! in float64 as in the reference implementation; per-element transforms run in float32.

/// Causal running mean and std of one row (Welford form, in float64).
///
/// Position `t` uses the observed values at positions `0..=t` only. The variance uses
/// Bessel's correction and the std is clamped to `minimum_scale` after the sqrt.
pub fn causal_statistics(
    data: &[f32],
    observed: &[bool],
    minimum_scale: f64,
) -> (Vec<f32>, Vec<f32>) {
    let mut means = Vec::with_capacity(data.len());
    let mut scales = Vec::with_capacity(data.len());
    let (mut cum_w, mut cum_x, mut m2, mut prev_mean) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (&x, &obs) in data.iter().zip(observed) {
        let w = if obs { 1.0 } else { 0.0 };
        let x = if obs { x as f64 } else { 0.0 };
        cum_w += w;
        cum_x += w * x;
        let denominator = cum_w.max(1.0);
        let mean = cum_x / denominator;
        m2 += (x - prev_mean) * (x - mean) * w;
        let variance = m2 / (denominator - 1.0).max(1.0);
        means.push(mean as f32);
        scales.push(variance.sqrt().max(minimum_scale) as f32);
        prev_mean = mean;
    }
    (means, scales)
}

/// Per-patch `(loc, scale)` of one row: the causal statistics at each patch's last step.
pub fn causal_patch_loc_scale(
    data: &[f32],
    observed: &[bool],
    patch_size: usize,
    minimum_scale: f64,
) -> (Vec<f32>, Vec<f32>) {
    debug_assert_eq!(data.len() % patch_size, 0);
    let (means, scales) = causal_statistics(data, observed, minimum_scale);
    let last = |v: &[f32]| {
        v.chunks_exact(patch_size)
            .map(|c| c[patch_size - 1])
            .collect()
    };
    (last(&means), last(&scales))
}

/// `asinh((x - loc) / scale)` on an observed position, 0 on an unobserved one.
#[inline]
pub fn affine_asinh(x: f32, loc: f32, scale: f32, observed: bool, use_asinh: bool) -> f32 {
    if !observed {
        return 0.0;
    }
    let scaled = (x - loc) / scale.max(f32::EPSILON);
    if use_asinh { scaled.asinh() } else { scaled }
}

/// Inverse of the forward transform: `sinh(x) * scale + loc`.
#[inline]
pub fn invert_affine_asinh(x: f32, loc: f32, scale: f32, use_asinh: bool) -> f32 {
    let x = if use_asinh { x.sinh() } else { x };
    x * scale + loc
}

/// Map +/-inf (a sinh overflow) to the nearest finite value of the slice, in place.
///
/// A slice with no finite value maps infinities to 0. NaN is left untouched so that real
/// instability still surfaces downstream.
pub fn clamp_nonfinite(vals: &mut [f32]) {
    if vals.iter().all(|v| !v.is_infinite()) {
        return;
    }
    let finite = || vals.iter().copied().filter(|v| v.is_finite());
    let hi = finite().fold(f32::NEG_INFINITY, f32::max);
    let lo = finite().fold(f32::INFINITY, f32::min);
    for v in vals.iter_mut() {
        if *v == f32::INFINITY {
            *v = hi;
        } else if *v == f32::NEG_INFINITY {
            *v = lo;
        }
        if v.is_infinite() {
            *v = 0.0;
        }
    }
}

/// Non-causal mean/std of one row over its observed positions, for known-future covariates.
///
/// Rows whose observed values are all 0 or 1 (indicator covariates) keep `loc=0`,
/// `scale=1`. Returns `(loc, scale, is_binary)`.
pub fn instance_norm_stats(
    values: &[f32],
    observed: &[bool],
    minimum_scale: f64,
) -> (f32, f32, bool) {
    let n = observed.iter().filter(|&&o| o).count();
    let is_binary = n > 0
        && values
            .iter()
            .zip(observed)
            .all(|(&v, &o)| !o || v == 0.0 || v == 1.0);
    if is_binary {
        return (0.0, 1.0, true);
    }
    let n = (n as f32).max(1.0);
    let sum: f64 = values
        .iter()
        .zip(observed)
        .filter(|&(_, &o)| o)
        .map(|(&v, _)| v as f64)
        .sum();
    let loc = (sum as f32) / n;
    let ss: f64 = values
        .iter()
        .zip(observed)
        .map(|(&v, &o)| if o { ((v - loc) as f64).powi(2) } else { 0.0 })
        .sum();
    let scale = ((ss as f32) / n).sqrt().max(minimum_scale as f32);
    (loc, scale, false)
}

/// A Welford state `(weight, mean, m2)`; the default `(0, 0, 0)` is the identity.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Welford {
    pub weight: f64,
    pub mean: f64,
    pub m2: f64,
}

impl Welford {
    /// Moments of the observed values of one patch.
    pub fn of_patch(values: &[f32], observed: &[bool]) -> Self {
        let weight = observed.iter().filter(|&&o| o).count() as f64;
        if weight == 0.0 {
            return Self::default();
        }
        let obs = || {
            values
                .iter()
                .zip(observed)
                .filter(|&(_, &o)| o)
                .map(|(&v, _)| v as f64)
        };
        let mean = obs().sum::<f64>() / weight;
        let m2 = obs().map(|v| (v - mean).powi(2)).sum();
        Self { weight, mean, m2 }
    }

    /// Merge `other` into this state.
    pub fn combine(self, other: Self) -> Self {
        let weight = self.weight + other.weight;
        if weight == 0.0 {
            return Self::default();
        }
        let delta = other.mean - self.mean;
        Self {
            weight,
            mean: self.mean + delta * other.weight / weight,
            m2: self.m2 + other.m2 + delta * delta * self.weight * other.weight / weight,
        }
    }
}

/// Per-patch `(loc, scale)` of one decoded block in the advancing refine frame (float64).
///
/// Horizon patch `j` is normalized with the statistics of the observed context plus the
/// block's medians of patches `0..j`, each median denormalized in its own frame before it
/// is folded in. With perfect medians this is causal per-patch normalization of the true
/// series.
///
/// `sinh_median` is the block's median `(H, patch_size)` in the last context patch's frame
/// with the asinh undone, `(median - loc) / scale`.
pub fn advancing_block_frames(
    context: &[f32],
    observed: &[bool],
    sinh_median: &[f64],
    patch_size: usize,
    minimum_scale: f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut state = context
        .chunks_exact(patch_size)
        .zip(observed.chunks_exact(patch_size))
        .map(|(v, o)| Welford::of_patch(v, o))
        .fold(Welford::default(), Welford::combine);

    let offset = state.mean; // medians are clamped relative to the context mean, so a large level survives
    let clip = (f32::MAX as f64).sqrt() / 1e3;
    let n_pred = sinh_median.len() / patch_size;
    let (mut locs, mut scales) = (Vec::with_capacity(n_pred), Vec::with_capacity(n_pred));
    let mut folded = vec![0.0f64; patch_size];
    for median in sinh_median.chunks_exact(patch_size) {
        let var = (state.m2 / (state.weight - 1.0).max(1.0)).max(0.0);
        let scale = var.sqrt().max(minimum_scale);
        locs.push(state.mean);
        scales.push(scale);
        for (f, &m) in folded.iter_mut().zip(median) {
            *f = (m * scale + state.mean - offset).clamp(-clip, clip) + offset;
        }
        let b_mean = folded.iter().sum::<f64>() / patch_size as f64;
        let b_m2 = folded.iter().map(|f| (f - b_mean).powi(2)).sum();
        state = state.combine(Welford {
            weight: patch_size as f64,
            mean: b_mean,
            m2: b_m2,
        });
    }
    (locs, scales)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_statistics_match_direct_computation() {
        let data = [1.0f32, 5.0, 2.0, 100.0, 4.0, 7.0];
        let observed = [true, true, false, false, true, true];
        let (means, scales) = causal_statistics(&data, &observed, 1e-6);
        let obs: Vec<f64> = vec![1.0, 5.0, 4.0, 7.0];
        let expected_at = [1usize, 2, 2, 2, 3, 4];
        for (t, &k) in expected_at.iter().enumerate() {
            let xs = &obs[..k];
            let mean = xs.iter().sum::<f64>() / k as f64;
            let var =
                xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / ((k as f64 - 1.0).max(1.0));
            assert!((means[t] as f64 - mean).abs() < 1e-6, "mean at {t}");
            assert!(
                (scales[t] as f64 - var.sqrt().max(1e-6)).abs() < 1e-6,
                "scale at {t}"
            );
        }
    }

    #[test]
    fn welford_combine_matches_pooled() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [10.0f32, 20.0];
        let all = [1.0f32, 2.0, 3.0, 10.0, 20.0];
        let merged = Welford::of_patch(&a, &[true; 3]).combine(Welford::of_patch(&b, &[true; 2]));
        let direct = Welford::of_patch(&all, &[true; 5]);
        assert!((merged.mean - direct.mean).abs() < 1e-12);
        assert!((merged.m2 - direct.m2).abs() < 1e-9);
        assert_eq!(
            Welford::default().combine(Welford::default()),
            Welford::default()
        );
    }

    #[test]
    fn clamp_nonfinite_maps_infinities() {
        let mut v = [f32::NEG_INFINITY, -1.0, 2.0, f32::INFINITY];
        clamp_nonfinite(&mut v);
        assert_eq!(v, [-1.0, -1.0, 2.0, 2.0]);
        let mut v = [f32::INFINITY, f32::INFINITY];
        clamp_nonfinite(&mut v);
        assert_eq!(v, [0.0, 0.0]);
    }

    #[test]
    fn instance_norm_detects_indicators() {
        assert_eq!(
            instance_norm_stats(&[0.0, 1.0, 5.0], &[true, true, false], 1e-6),
            (0.0, 1.0, true)
        );
        let (loc, scale, bin) = instance_norm_stats(&[1.0, 3.0], &[true, true], 1e-6);
        assert_eq!((loc, scale, bin), (2.0, 1.0, false));
    }
}
