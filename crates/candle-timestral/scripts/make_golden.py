# /// script
# requires-python = "==3.13.*"
# dependencies = [
#     "numpy",
#     "safetensors",
#     "timestral @ git+https://github.com/mistralai/timestral@a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff",
#     "torch",
# ]
# ///
"""Write reference outputs of the Python Timestral package for the Rust parity tests.

Usage (from crates/candle-timestral):

    uv run scripts/make_golden.py <model-dir> tests/fixtures/golden_5m.safetensors

`<model-dir>` holds `config.json` and `model.safetensors` of Timestral).
"""

import math
import sys

import numpy as np
import torch
from safetensors.numpy import save_file

from timestral import TimestralForecaster
from timestral.model import ROLE_KNOWN_COV, ROLE_PAST_COV, ROLE_TARGET

model_dir, out_path = sys.argv[1:3]

torch.set_num_threads(1)
rng = np.random.default_rng(0)
out: dict[str, np.ndarray] = {}


def series(b: int, v: int, t: int, period: float = 24.0, level: float = 0.0) -> np.ndarray:
    steps = np.arange(t, dtype=np.float64)
    x = 10 * np.sin(2 * math.pi * steps / period) + rng.standard_normal((b, v, t)).cumsum(-1) + level
    return x.astype(np.float32)


def put(name: str, arr) -> None:
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    out[name] = np.ascontiguousarray(arr)


fc = TimestralForecaster.from_pretrained(model_dir)
model = fc.model

# (a) univariate, one block of three patches.
x = series(1, 1, 500)
put("a.context", x)
put("a.quantiles", fc.forecast(x, prediction_length=96).quantiles)
put("a.raw", fc.forecast(x, prediction_length=96, symmetric_averaging=False, clip_nonnegative=False).quantiles)

# (b) multivariate with missing values, two decode blocks (median feedback + refine frames).
x = series(2, 3, 1500, period=50.0, level=100.0)
x[0, 1, 100:180] = np.nan
x[1, 0, :300] = np.nan  # late start
x[1, 2, 1490:] = np.nan
put("b.context", x)
put("b.quantiles", fc.forecast(x, prediction_length=1100).quantiles)

# (c) past and known covariates over two decode blocks.
x = series(2, 1, 700)
past = series(2, 2, 700)
past[0, 0, 50:60] = np.nan
known = np.concatenate([series(2, 1, 700 + 1100), (rng.random((2, 1, 1800)) > 0.7).astype(np.float32)], axis=1)
put("c.context", x)
put("c.past", past)
put("c.known", known)
put("c.quantiles", fc.forecast(x, prediction_length=1100, past_covariates=past, known_covariates=known).quantiles)

# (e) non-negative series with a context cap, so blocks drop the oldest steps.
fc_capped = TimestralForecaster(model, max_context=1024)
x = np.abs(series(3, 1, 2000)) + 1.0
put("e.context", x)
put("e.quantiles", fc_capped.forecast(x, prediction_length=1100).quantiles)

# (d) raw predict_block, context-only path.
ps = model.config.patch_size
inputs = torch.from_numpy(series(1, 3, 512)[0])
observed = torch.ones_like(inputs, dtype=torch.bool)
observed[1, :70] = False
observed[2, 200:260] = False
inputs = torch.where(observed, inputs, torch.zeros_like(inputs))
group_ids = torch.tensor([0, 0, 1])
with torch.inference_mode():
    q = model.predict_block(inputs, observed, group_ids, 4)
put("d1.inputs", inputs)
put("d1.observed", observed.to(torch.uint8))
put("d1.group_ids", group_ids)
put("d1.quantiles", q)

# (d2) raw predict_block, covariate window path.
inputs = torch.from_numpy(series(1, 4, 512)[0])
observed = torch.ones_like(inputs, dtype=torch.bool)
observed[0, :40] = False
inputs = torch.where(observed, inputs, torch.zeros_like(inputs))
group_ids = torch.tensor([0, 0, 0, 1])
roles = torch.tensor([ROLE_TARGET, ROLE_PAST_COV, ROLE_KNOWN_COV, ROLE_TARGET])
kf = torch.zeros(4, 4 * ps)
kfm = torch.zeros(4, 4 * ps, dtype=torch.bool)
kf[2] = torch.from_numpy(series(1, 1, 4 * ps)[0, 0])
kfm[2, : 3 * ps + 5] = True
kf = torch.where(kfm, kf, torch.zeros_like(kf))
with torch.inference_mode():
    q = model.predict_block(inputs, observed, group_ids, 4, roles, kf, kfm)
put("d2.inputs", inputs)
put("d2.observed", observed.to(torch.uint8))
put("d2.group_ids", group_ids)
put("d2.roles", roles)
put("d2.known_future", kf)
put("d2.known_future_mask", kfm.to(torch.uint8))
put("d2.quantiles", q)

save_file(out, out_path)
print(f"wrote {len(out)} arrays to {out_path}")
