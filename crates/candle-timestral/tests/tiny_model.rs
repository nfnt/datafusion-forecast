//! Forecaster behavior on a tiny randomly initialized model; needs no checkpoint.

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use candle_timestral::{
    Config, ForecastInput, ForecastOptions, LayerKind, TimestralForecaster, TimestralModel,
};

const KNOTS: [f64; 9] = [0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95];

/// Deterministic standard-normal samples (xorshift + Box-Muller).
struct Rng(u64);

impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f32 {
        let (u, v) = (self.uniform(), self.uniform());
        ((-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()) as f32
    }
}

fn tiny_config() -> Config {
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
fn tiny_model() -> (TimestralModel, VarMap) {
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

fn series(batch: usize, variates: usize, len: usize, seed: u64) -> Vec<f32> {
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

#[test]
fn forecast_shapes_finite_and_sorted() {
    let (model, _) = tiny_model();
    let fc = TimestralForecaster::new(model);
    // one block; context past the cap; three blocks after a short context
    for (variates, len, horizon) in [(1, 100, 12), (3, 300, 50), (2, 37, 70)] {
        let mut x = series(2, variates, len, 0);
        x[5..9].fill(f32::NAN);
        let out = fc
            .forecast(
                &ForecastInput::new(&x, 2, variates),
                &ForecastOptions::new(horizon),
            )
            .unwrap();
        assert_eq!(out.quantiles.len(), 2 * variates * horizon * KNOTS.len());
        assert!(out.quantiles.iter().all(|q| q.is_finite()));
        for qs in out.quantiles.as_chunks::<{ KNOTS.len() }>().0 {
            assert!(
                qs.windows(2).all(|w| w[0] <= w[1]),
                "quantiles must be sorted: {qs:?}"
            );
        }
        assert_eq!(out.median().len(), 2 * variates * horizon);
    }
}

#[test]
fn symmetric_averaging_is_odd_with_covariates() {
    // The negated pass negates the covariates too, so negating every input reflects the forecast exactly.
    let (model, _) = tiny_model();
    let fc = TimestralForecaster::new(model);
    let (x, past, known) = (
        series(2, 1, 120, 3),
        series(2, 2, 120, 1),
        series(2, 1, 144, 2),
    );
    let neg = |v: &[f32]| v.iter().map(|x| -x).collect::<Vec<f32>>();
    let opts = ForecastOptions {
        clip_nonnegative: false,
        ..ForecastOptions::new(24)
    };
    let input = ForecastInput::new(&x, 2, 1)
        .with_past_covariates(&past, 2)
        .with_known_covariates(&known, 1);
    let pos = fc.forecast(&input, &opts).unwrap();
    let (nx, np, nk) = (neg(&x), neg(&past), neg(&known));
    let input = ForecastInput::new(&nx, 2, 1)
        .with_past_covariates(&np, 2)
        .with_known_covariates(&nk, 1);
    let negated = fc.forecast(&input, &opts).unwrap();
    let (negated, pos) = (
        negated.quantiles.as_chunks::<{ KNOTS.len() }>().0,
        pos.quantiles.as_chunks::<{ KNOTS.len() }>().0,
    );
    for (n, p) in negated.iter().zip(pos) {
        let reflected: Vec<f32> = p.iter().rev().map(|v| -v).collect();
        assert_eq!(n, &reflected[..]);
    }
}

#[test]
fn nonnegative_series_are_clipped() {
    let (model, _) = tiny_model();
    let fc = TimestralForecaster::new(model);
    let x: Vec<f32> = series(1, 1, 64, 5).iter().map(|v| v.abs() * 0.01).collect();
    let out = fc
        .forecast(&ForecastInput::univariate(&x), &ForecastOptions::new(40))
        .unwrap();
    assert!(out.quantiles.iter().all(|&q| q >= 0.0));
}

#[test]
fn invalid_inputs_are_errors() {
    let (model, _) = tiny_model();
    let fc = TimestralForecaster::new(model);
    let x = series(1, 1, 50, 0);
    assert!(
        fc.forecast(&ForecastInput::univariate(&x), &ForecastOptions::new(0))
            .is_err()
    );
    assert!(
        fc.forecast(&ForecastInput::new(&x, 3, 1), &ForecastOptions::new(4))
            .is_err()
    );
    let past = vec![0.0; 49];
    let input = ForecastInput::univariate(&x).with_past_covariates(&past, 1);
    assert!(fc.forecast(&input, &ForecastOptions::new(4)).is_err());
}

#[test]
fn symmetric_averaging_needs_symmetric_levels() {
    let mut config = tiny_config();
    config.quantile_knots = vec![0.1, 0.5, 0.8];
    let vb = VarBuilder::from_varmap(&VarMap::new(), DType::F32, &Device::Cpu);
    let fc = TimestralForecaster::new(TimestralModel::new(config, vb).unwrap());
    let x = series(1, 1, 50, 0);
    let input = ForecastInput::univariate(&x);
    let err = fc.forecast(&input, &ForecastOptions::new(4)).unwrap_err();
    assert!(err.to_string().contains("symmetric about 0.5"), "{err}");
    let options = ForecastOptions {
        symmetric_averaging: false,
        ..ForecastOptions::new(4)
    };
    assert!(fc.forecast(&input, &options).is_ok());
}

#[test]
fn checkpoint_round_trip_is_strict() {
    let (model, varmap) = tiny_model();
    let dir = std::env::temp_dir().join(format!("timestral-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string(model.config()).unwrap(),
    )
    .unwrap();
    varmap.save(dir.join("model.safetensors")).unwrap();

    let x = series(2, 1, 100, 9);
    let opts = ForecastOptions::new(12);
    let expected = TimestralForecaster::new(model)
        .forecast(&ForecastInput::new(&x, 2, 1), &opts)
        .unwrap();
    let loaded = TimestralForecaster::from_dir(&dir, &Device::Cpu).unwrap();
    assert_eq!(
        loaded
            .forecast(&ForecastInput::new(&x, 2, 1), &opts)
            .unwrap(),
        expected
    );

    let mut tensors: std::collections::HashMap<String, Tensor> = varmap
        .data()
        .lock()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.as_tensor().clone()))
        .collect();
    tensors.insert(
        "transformer.layers.9.extra".into(),
        Tensor::zeros(3, DType::F32, &Device::Cpu).unwrap(),
    );
    candle_core::safetensors::save(&tensors, dir.join("model.safetensors")).unwrap();
    let err = TimestralForecaster::from_dir(&dir, &Device::Cpu)
        .err()
        .expect("unused tensor must fail");
    assert!(
        err.to_string().contains("transformer.layers.9.extra"),
        "{err}"
    );

    // Every released layer kind is accepted by the config.
    for kind in ["time", "variate", "time_variate", "variate_time"] {
        let _: LayerKind = serde_json::from_value(serde_json::json!(kind)).unwrap();
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
