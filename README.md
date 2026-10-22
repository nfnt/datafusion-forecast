# datafusion-forecast

Forecasts with Mistral AI's [Timestral](https://github.com/mistralai/timestral) time-series models inside
[Apache DataFusion](https://datafusion.apache.org) SQL. The `forecast` aggregate function treats each group as a series
and returns a median and 21 quantiles for each step ahead. The model runs natively in Rust on
[Candle](https://github.com/huggingface/candle), loads the released checkpoints, and matches the reference Python
package; no Python is needed at inference time.

```sql
SELECT store, f.ts AS ts, f.sales.median, f.sales.quantiles[19] AS p90
FROM (
    SELECT store, unnest(forecast(ts, named_struct('sales', sales), 48)) AS f
    FROM sales GROUP BY store
);
```

## Crates

| crate                                               |                                                                                                                        |
| --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| [`datafusion-forecast`](crates/datafusion-forecast) | the `forecast` aggregate function for DataFusion SQL, with multivariate targets, past-only and known-future covariates |
| [`candle-timestral`](crates/candle-timestral)       | Timestral inference in Rust with Candle, usable without DataFusion                                                     |

## Quick start

```rust
use std::sync::Arc;
use candle_core::Device;
use datafusion::prelude::SessionContext;
use datafusion_forecast::{candle_timestral::TimestralForecaster, Timestral};

let forecaster = TimestralForecaster::from_dir("path/to/Timestral-checkpoint", &Device::Cpu)?;
let ctx = SessionContext::new();
datafusion_forecast::register(&ctx, Arc::new(Timestral::from(forecaster)));
```

A checkpoint is a directory with the released `config.json` and `model.safetensors` of a version of Timestral. With the
`hub` feature it can be downloaded from the Hugging Face Hub instead:

```sh
# SQL: two stores' hourly sales, with a promotion flag known in advance
cargo run --release -p datafusion-forecast --features hub --example sql -- mistralai/Timestral-5M
# the model alone, without DataFusion
cargo run --release -p candle-timestral --features hub --example forecast -- mistralai/Timestral-5M
```

Both examples also take a local checkpoint directory. Both crates forward Candle's backend features: `metal` (Apple
GPUs), `cuda` and `cudnn` (NVIDIA GPUs), `accelerate` (Apple's Accelerate framework on the CPU) and `mkl` (Intel MKL
on the CPU). For a GPU, pass its device to `TimestralForecaster::from_dir`, e.g. `Device::new_cuda(0)?`.

## `forecast(ts, targets, horizon [, past [, known]])`

Each group is one series.

| argument  |                                                                                                                                                                                                                                                                                                                                                     |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ts`      | timestamp (any unit and time zone), `DATE` or integer time of each row. Rows are sorted by it, so no `ORDER BY` is needed. The grid step is the smallest gap between rows of the group; every row must lie on the grid, and grid slots without a row are missing values. See [Calendar steps and sparse groups](#calendar-steps-and-sparse-groups). |
| `targets` | struct of numeric columns, e.g. `named_struct('sales', sales)`. Its fields are the variates of the series and are forecast jointly. NULL or NaN is a missing value.                                                                                                                                                                                 |
| `horizon` | number of steps to forecast, an integer literal of at most the model's context length (see below).                                                                                                                                                                                                                                                  |
| `past`    | optional struct of numeric covariates known over the context only. Pass `NULL` to skip it.                                                                                                                                                                                                                                                          |
| `known`   | optional struct of numeric covariates known over the context and the horizon. The last `horizon` time steps of each group are then the horizon: those rows carry the known covariates and NULL targets.                                                                                                                                             |

The result is a list of `horizon` steps
`{ts, <target>: {median, quantiles}, ...}` per group. `quantiles` is a fixed-size list at the levels returned
by `forecast_quantile_levels()` (21 levels for Timestral-5M; index 19 is 0.9 there, since SQL lists are 1-based). A
group with fewer than two rows, or with no observed target value, gives NULL.

Only the last steps of each group, up to the context length, are passed to the model. By default this is the
checkpoint's trained context length; see `TimestralForecaster::with_max_context`. Groups whose trimmed contexts have
the same length are forecast together in batches of up to 64 series.

### Calendar steps and sparse groups

The grid step is a fixed length, and it is found for each group on its own. Three cases need a rewrite of `ts`:

- **Calendar months** (28 to 31 days) are not a fixed length, so monthly dates are rejected as irregular.
  Forecast on a month number and turn the result back into dates:

  ```sql
  SELECT store, make_date(CAST(f.ts / 12 AS INT), CAST(f.ts % 12 + 1 AS INT), 1) AS month, f.sales.median
  FROM (
      SELECT store, unnest(forecast(m, named_struct('sales', sales), 6)) AS f
      FROM (
          SELECT store, CAST(date_part('year', d) * 12 + date_part('month', d) - 1 AS BIGINT) AS m, sales
          FROM monthly_sales
      )
      GROUP BY store
  );
  ```

  Quarters and years work the same way.

- **Daily data in a time zone with daylight saving time**, such as local midnights, has 23 and 25 hour gaps.
  Pass `CAST(ts AS DATE)`; for a timestamp with a time zone this is the local date.
- **Sparse groups** get their own step. If a group of daily data only has rows every other day, its step is
  two days and it is forecast in two-day steps, without an error. Join the data onto a full calendar so that
  every slot has a row; rows with NULL targets are missing values. Stop the calendar at the group's last
  row, since padded rows after it become context.

### Options

Timestral's symmetric averaging and non-negative clipping (both on by default) are set on `Timestral`;
`ForecastUdaf` lets you register the aggregate under another name and set the batch size:

```rust
use datafusion::logical_expr::AggregateUDF;
use datafusion_forecast::{ForecastUdaf, Timestral};

let model = Timestral::new(forecaster).with_clip_nonnegative(false);
ctx.register_udaf(AggregateUDF::from(
    ForecastUdaf::new(Arc::new(model)).with_batch_size(16),
));
```

### Limitations

- Timestamps must be on a fixed-length grid, found per group. Calendar months, days across daylight saving
  changes and sparse groups need the rewrites in
  [Calendar steps and sparse groups](#calendar-steps-and-sparse-groups).
- Groups are buffered in memory until the aggregate is evaluated. The forecast runs on the calling thread
  during evaluation.
- DataFusion 55 drops sort keys after a `GROUP BY` key, even after an `unnest`
  ([apache/datafusion#25844](https://github.com/apache/datafusion/issues/25844), fixed after 55.2.0).
  `ORDER BY store, ts` over the unnested result therefore sorts by `store` only; use `ORDER BY ts, store` or
  sort in a separate step.

## Development

```sh
cargo test --workspace
# + parity with the Python reference on any of the checkpoints (absolute paths)
TIMESTRAL_5M_DIR=/abs/path/to/Timestral-5M TIMESTRAL_28M_DIR=/abs/path/to/Timestral-28M \
  TIMESTRAL_123M_DIR=/abs/path/to/Timestral-123M TIMESTRAL_209M_DIR=/abs/path/to/Timestral-209M \
  cargo test --workspace
cargo deny check   # licenses and advisories; requires `cargo install cargo-deny`
```

Without checkpoints the tests run on a tiny random-weight model. The parity tests compare against outputs of the
Python package in `crates/candle-timestral/tests/fixtures`, regenerated with
`crates/candle-timestral/scripts/make_golden.py`; Rust matches them to a relative error below 1e-4.

## Releases

Release notes for each version are on [GitHub Releases](https://github.com/nfnt/datafusion-forecast/releases).

## License

Apache 2.0, see [LICENSE](LICENSE). The code is a port of [mistralai/timestral](https://github.com/mistralai/timestral)
(Apache 2.0). The Timestral weights are distributed separately on the Hugging Face Hub under a Modified MIT license.
