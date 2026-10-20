//! Inference configuration of a Timestral checkpoint (`config.json`).
//!
//! Only the fields that change the forward pass at inference are kept. One family of
//! models is supported: causal modeling trained with Decoupled Horizon Forecasting (DHF),
//! with monotone quantiles, decoded in the advancing refine frame.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// The attention sublayers a transformer layer runs before its shared MLP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerKind {
    Time,
    Variate,
    TimeVariate,
    VariateTime,
}

impl LayerKind {
    /// Ordered attention axes of the layer; `true` is the time axis.
    pub fn axes(self) -> &'static [bool] {
        match self {
            LayerKind::Time => &[true],
            LayerKind::Variate => &[false],
            LayerKind::TimeVariate => &[true, false],
            LayerKind::VariateTime => &[false, true],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionGate {
    None,
    /// The attention output is multiplied by `sigmoid(x @ W_g)` before the output projection.
    SdpaSigmoid,
}

/// Architecture and decode settings of a Timestral checkpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Time steps per patch.
    pub patch_size: usize,
    /// Model width.
    pub d_model: usize,
    /// Attention heads per attention sublayer.
    pub num_heads: usize,
    /// Hidden width of the SwiGLU MLP.
    pub mlp_hidden_dim: usize,
    /// One entry per layer.
    pub block_schedule: Vec<LayerKind>,
    /// Quantile levels of the head, symmetric about 0.5.
    pub quantile_knots: Vec<f64>,
    /// Trained DHF block size, the maximum patches decoded per block.
    pub block_size: usize,
    /// Trained context length, the default context cap.
    pub context_length: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_rope_partial_factor")]
    pub rope_partial_factor: f64,
    #[serde(default = "default_eps")]
    pub norm_eps: f64,
    #[serde(default)]
    pub attn_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    #[serde(default = "default_gate")]
    pub time_attn_gate: AttentionGate,
    #[serde(default = "default_gate")]
    pub variate_attn_gate: AttentionGate,
    #[serde(default = "default_eps")]
    pub minimum_scale: f64,
    #[serde(default = "default_true")]
    pub use_asinh: bool,
    #[serde(default)]
    pub stabilize_with_global: bool,
    #[serde(default = "default_true")]
    pub covariate_enabled: bool,
    #[serde(default = "default_true")]
    pub split_context_horizon_attn: bool,
    #[serde(default = "default_true")]
    pub attend_missing_patches: bool,
    /// Increment scale of the monotone quantile head.
    #[serde(default = "default_quantile_width_factor")]
    pub quantile_width_factor: f64,
}

fn default_rope_theta() -> f64 {
    10000.0
}
fn default_rope_partial_factor() -> f64 {
    0.5
}
fn default_eps() -> f64 {
    1e-6
}
fn default_gate() -> AttentionGate {
    AttentionGate::SdpaSigmoid
}
fn default_true() -> bool {
    true
}
fn default_quantile_width_factor() -> f64 {
    2.0
}

impl Config {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let config: Self = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        let fail = |msg: String| Err(Error::Config(msg));
        if self.num_heads == 0 || !self.d_model.is_multiple_of(self.num_heads) {
            return fail(format!(
                "d_model={} not divisible by num_heads={}",
                self.d_model, self.num_heads
            ));
        }
        if self.rope_dim() == 0 || !self.rope_dim().is_multiple_of(2) {
            return fail(format!(
                "rope_partial_factor={} gives an invalid rope_dim={}",
                self.rope_partial_factor,
                self.rope_dim()
            ));
        }
        if !self.quantile_knots.contains(&0.5) {
            return fail(format!(
                "quantile_knots must contain 0.5, got {:?}",
                self.quantile_knots
            ));
        }
        if self.quantile_knots.windows(2).any(|w| w[0] >= w[1]) {
            return fail(format!(
                "quantile_knots must be increasing, got {:?}",
                self.quantile_knots
            ));
        }
        if self.stabilize_with_global {
            return fail("the advancing refine frame needs stabilize_with_global=false".into());
        }
        if self.patch_size == 0 || self.block_size == 0 {
            return fail("patch_size and block_size must be positive".into());
        }
        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.d_model / self.num_heads
    }

    /// Head dimensions rotated by RoPE; the rest pass through unrotated.
    pub fn rope_dim(&self) -> usize {
        (self.head_dim() as f64 * self.rope_partial_factor) as usize
    }

    pub fn num_quantiles(&self) -> usize {
        self.quantile_knots.len()
    }

    pub fn median_index(&self) -> usize {
        self.quantile_knots
            .iter()
            .position(|&q| q == 0.5)
            .expect("validated: knots contain 0.5")
    }

    /// Whether the layer schedule can route a known covariate's future to the targets.
    ///
    /// Some time op must be followed by at least one more time op and at least one
    /// variate op; otherwise the future of a known covariate never reaches a target.
    pub fn supports_known_covariates(&self) -> bool {
        if !self.covariate_enabled {
            return false;
        }
        let ops: Vec<bool> = self
            .block_schedule
            .iter()
            .flat_map(|k| k.axes().iter().copied())
            .collect();
        ops.iter().enumerate().any(|(i, &is_time)| {
            let rest = &ops[i + 1..];
            is_time && rest.iter().any(|&t| t) && !rest.iter().all(|&t| t)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG_5M: &str = r#"{
      "patch_size": 32, "d_model": 192, "num_heads": 3, "mlp_hidden_dim": 512,
      "block_schedule": ["time", "time", "time", "time_variate", "time", "time", "time", "time_variate"],
      "quantile_knots": [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                         0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99],
      "block_size": 32, "context_length": 32768, "rope_theta": 10000.0, "rope_partial_factor": 0.5,
      "norm_eps": 1e-06, "attn_bias": false, "mlp_bias": false, "time_attn_gate": "sdpa_sigmoid",
      "variate_attn_gate": "sdpa_sigmoid", "minimum_scale": 1e-06, "use_asinh": true,
      "stabilize_with_global": false, "covariate_enabled": true, "split_context_horizon_attn": true,
      "attend_missing_patches": true, "quantile_width_factor": 3.0
    }"#;

    #[test]
    fn parses_released_config() {
        let c: Config = serde_json::from_str(CONFIG_5M).unwrap();
        c.validate().unwrap();
        assert_eq!(c.head_dim(), 64);
        assert_eq!(c.rope_dim(), 32);
        assert_eq!(c.num_quantiles(), 21);
        assert_eq!(c.median_index(), 10);
        assert_eq!(c.quantile_width_factor, 3.0);
        assert!(c.supports_known_covariates());
    }

    #[test]
    fn rejects_unknown_keys() {
        let json = CONFIG_5M.replacen("\"patch_size\"", "\"bogus\": 1, \"patch_size\"", 1);
        assert!(serde_json::from_str::<Config>(&json).is_err());
    }
}
