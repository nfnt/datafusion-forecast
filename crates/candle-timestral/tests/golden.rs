//! Parity with the Python reference implementation on the Timestral checkpoints.
//!
//! The fixtures come from `scripts/make_golden.py`. The tests need the checkpoints, so they are
//! ignored by default; run them with `cargo test --test golden -- --ignored`, optionally filtered
//! to one checkpoint (e.g. `golden_5m::`). Each checkpoint's directory is read from
//! `TIMESTRAL_5M_DIR`, `TIMESTRAL_28M_DIR`, `TIMESTRAL_123M_DIR` or `TIMESTRAL_209M_DIR`, and a
//! test fails if its variable is unset.

use std::collections::HashMap;

use candle_core::{Device, Tensor};
use candle_timestral::{ForecastInput, ForecastOptions, Role, TimestralForecaster};

/// Max abs error relative to the largest expected magnitude.
const TOLERANCE: f32 = 1e-4;

struct Golden {
    name: &'static str,
    arrays: HashMap<String, Tensor>,
}

impl Golden {
    fn load(name: &'static str) -> Self {
        let path = format!(
            "{}/tests/fixtures/{name}.safetensors",
            env!("CARGO_MANIFEST_DIR")
        );
        Self {
            name,
            arrays: candle_core::safetensors::load(path, &Device::Cpu).expect("fixtures"),
        }
    }

    fn tensor(&self, name: &str) -> &Tensor {
        self.arrays
            .get(name)
            .unwrap_or_else(|| panic!("missing fixture {name}"))
    }

    fn f32(&self, name: &str) -> Vec<f32> {
        self.tensor(name).flatten_all().unwrap().to_vec1().unwrap()
    }

    fn bool(&self, name: &str) -> Vec<bool> {
        let v: Vec<u8> = self.tensor(name).flatten_all().unwrap().to_vec1().unwrap();
        v.into_iter().map(|x| x != 0).collect()
    }

    fn i64(&self, name: &str) -> Vec<i64> {
        self.tensor(name).to_vec1().unwrap()
    }

    fn dims(&self, name: &str) -> Vec<usize> {
        self.tensor(name).dims().to_vec()
    }
}

/// Run `case` on the checkpoint in the directory named by `var`, against the `fixtures`.
fn run(var: &str, fixtures: &'static str, case: fn(TimestralForecaster, &Golden)) {
    let dir = std::env::var(var)
        .unwrap_or_else(|_| panic!("{var} must point to the Timestral checkpoint directory"));
    let fc = TimestralForecaster::from_dir(&dir, &Device::Cpu).expect("load checkpoint");
    case(fc, &Golden::load(fixtures));
}

fn assert_close(g: &Golden, name: &str, actual: &[f32], expected: &[f32]) {
    let name = format!("{} {name}", g.name);
    assert_eq!(actual.len(), expected.len(), "{name}: length");
    let magnitude = expected.iter().fold(1.0f32, |m, x| m.max(x.abs()));
    let (worst, at) = actual
        .iter()
        .zip(expected)
        .enumerate()
        .map(|(i, (a, e))| ((a - e).abs(), i))
        .fold((0.0f32, 0), |acc, x| if x.0 > acc.0 { x } else { acc });
    let rel = worst / magnitude;
    eprintln!("{name}: max abs err {worst:.3e} (rel {rel:.3e}) at {at}");
    assert!(
        rel < TOLERANCE,
        "{name}: max abs err {worst} at {at}: {} vs {}",
        actual[at],
        expected[at]
    );
}

/// The test cases, run on each checkpoint by `checkpoint!`.
mod cases {
    use super::*;

    pub fn predict_block_context_only(fc: TimestralForecaster, g: &Golden) {
        let groups: Vec<usize> = g
            .i64("d1.group_ids")
            .into_iter()
            .map(|x| x as usize)
            .collect();
        let q = fc
            .model()
            .predict_block(
                &g.f32("d1.inputs"),
                &g.bool("d1.observed"),
                &groups,
                4,
                None,
                None,
            )
            .unwrap();
        assert_close(g, "d1", &q, &g.f32("d1.quantiles"));
    }

    pub fn predict_block_with_covariates(fc: TimestralForecaster, g: &Golden) {
        let groups: Vec<usize> = g
            .i64("d2.group_ids")
            .into_iter()
            .map(|x| x as usize)
            .collect();
        let roles: Vec<Role> = g
            .i64("d2.roles")
            .into_iter()
            .map(|r| [Role::Target, Role::PastCovariate, Role::KnownCovariate][r as usize])
            .collect();
        let (kf, kfm) = (g.f32("d2.known_future"), g.bool("d2.known_future_mask"));
        let q = fc
            .model()
            .predict_block(
                &g.f32("d2.inputs"),
                &g.bool("d2.observed"),
                &groups,
                4,
                Some(&roles),
                Some((&kf, &kfm)),
            )
            .unwrap();
        assert_close(g, "d2", &q, &g.f32("d2.quantiles"));
    }

    pub fn univariate_forecast(fc: TimestralForecaster, g: &Golden) {
        let ctx = g.f32("a.context");
        let out = fc
            .forecast(&ForecastInput::univariate(&ctx), &ForecastOptions::new(96))
            .unwrap();
        assert_close(g, "a", &out.quantiles, &g.f32("a.quantiles"));
        let raw_opts = ForecastOptions {
            symmetric_averaging: false,
            clip_nonnegative: false,
            ..ForecastOptions::new(96)
        };
        let raw = fc
            .forecast(&ForecastInput::univariate(&ctx), &raw_opts)
            .unwrap();
        assert_close(g, "a.raw", &raw.quantiles, &g.f32("a.raw"));
    }

    pub fn multivariate_two_blocks_with_missing_values(fc: TimestralForecaster, g: &Golden) {
        let dims = g.dims("b.context");
        let ctx = g.f32("b.context");
        let out = fc
            .forecast(
                &ForecastInput::new(&ctx, dims[0], dims[1]),
                &ForecastOptions::new(1100),
            )
            .unwrap();
        assert_close(g, "b", &out.quantiles, &g.f32("b.quantiles"));
    }

    pub fn covariates_two_blocks(fc: TimestralForecaster, g: &Golden) {
        let (ctx, past, known) = (g.f32("c.context"), g.f32("c.past"), g.f32("c.known"));
        let input = ForecastInput::new(&ctx, 2, 1)
            .with_past_covariates(&past, g.dims("c.past")[1])
            .with_known_covariates(&known, g.dims("c.known")[1]);
        let out = fc.forecast(&input, &ForecastOptions::new(1100)).unwrap();
        assert_close(g, "c", &out.quantiles, &g.f32("c.quantiles"));
    }

    pub fn capped_context_nonnegative(fc: TimestralForecaster, g: &Golden) {
        let fc = fc.with_max_context(1024);
        let ctx = g.f32("e.context");
        let out = fc
            .forecast(&ForecastInput::new(&ctx, 3, 1), &ForecastOptions::new(1100))
            .unwrap();
        assert_close(g, "e", &out.quantiles, &g.f32("e.quantiles"));
    }
}

/// One ignored test per case on the checkpoint whose directory is named by `$var`.
macro_rules! checkpoint {
    ($module:ident, $var:literal) => {
        mod $module {
            use super::*;

            #[test]
            #[ignore = "needs the Timestral checkpoint; run with --ignored"]
            fn predict_block_context_only() {
                run($var, stringify!($module), cases::predict_block_context_only);
            }

            #[test]
            #[ignore = "needs the Timestral checkpoint; run with --ignored"]
            fn predict_block_with_covariates() {
                run(
                    $var,
                    stringify!($module),
                    cases::predict_block_with_covariates,
                );
            }

            #[test]
            #[ignore = "needs the Timestral checkpoint; run with --ignored"]
            fn univariate_forecast() {
                run($var, stringify!($module), cases::univariate_forecast);
            }

            #[test]
            #[ignore = "needs the Timestral checkpoint; run with --ignored"]
            fn multivariate_two_blocks_with_missing_values() {
                run(
                    $var,
                    stringify!($module),
                    cases::multivariate_two_blocks_with_missing_values,
                );
            }

            #[test]
            #[ignore = "needs the Timestral checkpoint; run with --ignored"]
            fn covariates_two_blocks() {
                run($var, stringify!($module), cases::covariates_two_blocks);
            }

            #[test]
            #[ignore = "needs the Timestral checkpoint; run with --ignored"]
            fn capped_context_nonnegative() {
                run($var, stringify!($module), cases::capped_context_nonnegative);
            }
        }
    };
}

checkpoint!(golden_5m, "TIMESTRAL_5M_DIR");
checkpoint!(golden_28m, "TIMESTRAL_28M_DIR");
checkpoint!(golden_123m, "TIMESTRAL_123M_DIR");
checkpoint!(golden_209m, "TIMESTRAL_209M_DIR");
