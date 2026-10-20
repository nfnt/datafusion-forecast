//! A tiny randomly initialized Timestral model, as in the `candle-timestral` crate's tests.

#![allow(dead_code)]

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use datafusion_forecast::candle_timestral::{Config, TimestralModel};

pub const KNOTS: [f64; 9] = [0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95];

/// Deterministic standard-normal samples (xorshift + Box-Muller).
pub struct Rng(pub u64);

impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    pub fn normal(&mut self) -> f32 {
        let (u, v) = (self.uniform(), self.uniform());
        ((-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()) as f32
    }
}

pub fn tiny_config() -> Config {
    serde_json::from_value(serde_json::json!({
        "patch_size": 8,
        "d_model": 32,
        "num_heads": 4,
        "mlp_hidden_dim": 64,
        "block_schedule": ["time", "time_variate", "time", "time_variate"],
        "quantile_knots": KNOTS,
        "block_size": 4,
        "context_length": 256,
    }))
    .unwrap()
}

/// A tiny model with every parameter drawn from N(0, 0.05²), and its weights.
pub fn tiny_model() -> (TimestralModel, VarMap) {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    TimestralModel::new(tiny_config(), vb).unwrap();
    {
        let vars = varmap.data().lock().unwrap();
        let mut names: Vec<&String> = vars.keys().collect();
        names.sort();
        let mut rng = Rng(42);
        for name in names {
            let var = &vars[name];
            let values: Vec<f32> = (0..var.elem_count()).map(|_| 0.05 * rng.normal()).collect();
            var.set(&Tensor::from_vec(values, var.shape(), &Device::Cpu).unwrap())
                .unwrap();
        }
    }
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    (TimestralModel::new(tiny_config(), vb).unwrap(), varmap)
}

pub fn series(batch: usize, variates: usize, len: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed * 7919 + 1);
    let mut out = Vec::with_capacity(batch * variates * len);
    for _ in 0..batch * variates {
        let mut walk = 0.0f32;
        for t in 0..len {
            walk += rng.normal();
            out.push(10.0 * (2.0 * std::f32::consts::PI * t as f32 / 24.0).sin() + walk);
        }
    }
    out
}
