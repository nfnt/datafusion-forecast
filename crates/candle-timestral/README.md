# candle-timestral

Inference for Timestral checkpoints (e.g. [`mistralai/Timestral-5M`](https://huggingface.co/mistralai/Timestral-5M))
in Rust with [Candle](https://github.com/huggingface/candle), ported from the reference Python package. It loads the
released `config.json` + `model.safetensors` and reproduces `TimestralForecaster.forecast`: multivariate context,
past-only and known-future covariates, 21 quantiles per step, block-by-block decoding for long horizons, symmetric
averaging and non-negative clipping. Categorical target encoding is not ported; pass encoded values as numeric
covariates.

```rust
use candle_core::Device;
use candle_timestral::{ForecastInput, ForecastOptions, TimestralForecaster};

let forecaster = TimestralForecaster::from_dir("path/to/Timestral-5M", &Device::Cpu)?;
// or, with the `hub` feature: TimestralForecaster::from_pretrained("mistralai/Timestral-5M", &Device::Cpu)?

// context: (batch, variates, T) flattened, NaN = missing
let input = ForecastInput::new(&context, batch, variates)
    .with_past_covariates(&past, n_past)      // (batch, n_past, T)
    .with_known_covariates(&known, n_known);  // (batch, n_known, T + horizon)
let forecast = forecaster.forecast(&input, &ForecastOptions::new(96))?;
forecast.quantiles;       // (batch, variates, 96, 21) flattened
forecast.median();        // (batch, variates, 96)
forecast.quantile(0.9)?;  // (batch, variates, 96)
```

`TimestralModel::predict_block` exposes the raw single-block network as well.

Features:

| feature      |                                                                              |
| ------------ | ---------------------------------------------------------------------------- |
| `hub`        | `from_pretrained` and Hub ids in `loading::resolve_checkpoint`, via `hf-hub` |
| `metal`      | Apple GPUs (`Device::new_metal(0)?`)                                         |
| `cuda`       | NVIDIA GPUs (`Device::new_cuda(0)?`)                                         |
| `cudnn`      | `cuda` plus cuDNN kernels                                                    |
| `accelerate` | Apple's Accelerate framework for CPU matrix multiplication                   |
| `mkl`        | Intel MKL for CPU matrix multiplication                                      |

```sh
cargo run --release -p candle-timestral --features hub --example forecast -- mistralai/Timestral-5M
```

## Layout

| file                            | contents                                                                                                        |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `src/layers.rs`, `src/model.rs` | the network in Candle (RMSNorm, gated time/variate attention with partial RoPE, SwiGLU, monotone quantile head) |
| `src/masks.rs`, `src/scaler.rs` | DHF attention masks and positions; causal per-patch scaling, refine frames (plain Rust, float64 statistics)     |
| `src/forecaster.rs`             | row assembly, block-autoregressive decoding, clampsym                                                           |
| `src/loading.rs`                | strict safetensors loading (missing or unused tensors are errors), optional Hub download                        |

## Tests

```sh
cargo test -p candle-timestral   # unit tests + tiny random-weight model
# + parity with the Python reference on all checkpoints (absolute paths)
TIMESTRAL_5M_DIR=/abs/path/to/Timestral-5M TIMESTRAL_28M_DIR=/abs/path/to/Timestral-28M \
  TIMESTRAL_123M_DIR=/abs/path/to/Timestral-123M TIMESTRAL_209M_DIR=/abs/path/to/Timestral-209M \
  cargo test --release -p candle-timestral --test golden -- --ignored
# or on one of them, e.g. Timestral-5M at the revision CI uses
scripts/fetch_checkpoint.sh Timestral-5M bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f /tmp/Timestral-5M
TIMESTRAL_5M_DIR=/tmp/Timestral-5M cargo test -p candle-timestral --test golden -- --ignored golden_5m::
```

The parity tests are ignored by default and fail when run without their checkpoint directory. CI runs them on
Timestral-5M on every push; the manual `Golden` workflow runs all four. Both download the checkpoints at the Hub
revisions pinned in the workflows.

`tests/fixtures/golden_{5m,28m,123m,209m}.safetensors` hold reference outputs of the Python package on Timestral-5M,
-28M, -123M and -209M, regenerated with `scripts/make_golden.py` (see its docstring). Rust matches them to a relative
error below 1e-4.

## Differences from the reference

The outputs match the Python package, but attention is computed with less memory:

- Time attention masks are not materialized per row. Each is a pattern shared by all rows of one kind (causal or
  known-future) plus a per-row key bias, expanded for one block of rows at a time; attention runs over blocks of rows
  or heads so that at most 16 MiB of scores exist at once. At the trained context of 32768 steps this uses about half
  the memory of the reference.
- Variate attention runs within each series instead of over all rows of the batch with a mask, and is skipped down to
  the value projection for series of a single row, whose attention weight is exactly 1.

## License

The code is a port of [mistralai/timestral](https://github.com/mistralai/timestral) (Apache 2.0). The Timestral weights
are distributed separately on the Hugging Face Hub under a Modified MIT license.
