//! Transformer layers for Timestral inference.
//!
//! Parameter names match the released weights. All tensors use the packed layout: one
//! row per variate of every series in the batch, `(R, N, D)` with `N` patches. Time
//! attention runs per row over patches; variate attention runs per patch over the rows of
//! the same series.

use std::ops::Range;

use candle_core::{D, Device, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder, linear_b};

use crate::config::{AttentionGate, Config, LayerKind};

pub struct RmsNorm {
    scale: Tensor,
    eps: f32,
}

impl RmsNorm {
    pub fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            scale: vb.get(dim, "scale")?,
            eps: eps as f32,
        })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm(&x.contiguous()?, &self.scale, self.eps)
    }
}

/// `output(silu(hidden(x))) + residual(x)`, used by the patch embedding and the head.
pub struct ResidualBlock {
    hidden_layer: Linear,
    output_layer: Linear,
    residual_layer: Linear,
}

impl ResidualBlock {
    pub fn new(
        input: usize,
        hidden: usize,
        output: usize,
        bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            hidden_layer: linear_b(input, hidden, bias, vb.pp("hidden_layer"))?,
            output_layer: linear_b(hidden, output, bias, vb.pp("output_layer"))?,
            residual_layer: linear_b(input, output, bias, vb.pp("residual_layer"))?,
        })
    }
}

impl Module for ResidualBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = self
            .output_layer
            .forward(&candle_nn::ops::silu(&self.hidden_layer.forward(x)?)?)?;
        out + self.residual_layer.forward(x)?
    }
}

/// SwiGLU MLP with one fused input projection; computes `gate * silu(up)`.
pub struct FusedSwiGLU {
    w12: Linear,
    w3: Linear,
    hidden_dim: usize,
}

impl FusedSwiGLU {
    pub fn new(embed_dim: usize, hidden_dim: usize, bias: bool, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            w12: linear_b(embed_dim, 2 * hidden_dim, bias, vb.pp("w12"))?,
            w3: linear_b(hidden_dim, embed_dim, bias, vb.pp("w3"))?,
            hidden_dim,
        })
    }
}

impl Module for FusedSwiGLU {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.w12.forward(x)?;
        let gate = h.narrow(D::Minus1, 0, self.hidden_dim)?;
        let up = h.narrow(D::Minus1, self.hidden_dim, self.hidden_dim)?;
        self.w3.forward(&(gate * candle_nn::ops::silu(&up)?)?)
    }
}

/// Rotary position embedding (sin/cos concatenation), with partial rotation.
///
/// Only the first `rotary_dim` dimensions of each head rotate. The inverse-frequency table
/// is zero-padded past `rotary_dim / 2`, so the remaining dimensions get `cos=1`, `sin=0`,
/// which is the identity rotation.
pub struct Rope {
    inv_freq: Vec<f32>,
}

impl Rope {
    pub fn new(head_dim: usize, base: f64, rotary_dim: usize) -> Self {
        let mut inv_freq = vec![0.0f32; head_dim / 2];
        for (i, f) in inv_freq.iter_mut().take(rotary_dim / 2).enumerate() {
            *f = 1.0 / (base as f32).powf((2 * i) as f32 / rotary_dim as f32);
        }
        Self { inv_freq }
    }

    /// `(cos, sin)` tables, each `(R, 1, S, head_dim)`, for integer positions `(R, S)`.
    pub fn tables(
        &self,
        positions: &[i64],
        rows: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let half = self.inv_freq.len();
        let seq = positions.len() / rows;
        let (mut cos, mut sin) = (
            Vec::with_capacity(positions.len() * 2 * half),
            Vec::with_capacity(positions.len() * 2 * half),
        );
        for &p in positions {
            for _ in 0..2 {
                for &f in &self.inv_freq {
                    let angle = f * p as f32;
                    cos.push(angle.cos());
                    sin.push(angle.sin());
                }
            }
        }
        let shape = (rows, 1, seq, 2 * half);
        Ok((
            Tensor::from_vec(cos, shape, device)?,
            Tensor::from_vec(sin, shape, device)?,
        ))
    }
}

/// Rotate `x` `(B, H, S, head_dim)` with tables broadcasting over heads.
fn apply_rotary(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    let rotated = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
    x.broadcast_mul(cos)? + rotated.broadcast_mul(sin)?
}

/// Scaled dot-product attention on `(B, H, S, head_dim)` with an additive bias broadcasting over heads.
fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let scale = 1.0 / (q.dim(D::Minus1)? as f64).sqrt();
    let att = (q.contiguous()?.matmul(&k.contiguous()?.t()?)? * scale)?;
    let att = match bias {
        Some(b) => att.broadcast_add(b)?,
        None => att,
    };
    candle_nn::ops::softmax_last_dim(&att)?.matmul(&v.contiguous()?)
}

/// Most attention scores, `rows * heads * queries * keys`, materialized at once; time
/// attention runs over blocks of rows or heads to stay below it (16 MiB of f32). Smaller
/// blocks are not slower on CPU, as the scores stay in cache.
const ATTENTION_CHUNK_SCORES: usize = 1 << 22;

/// Additive time-attention bias over `Sq` queries and `Sk` keys, factored as a pattern
/// shared by all rows of one kind plus a per-row key bias (see
/// [`masks::INVALID_KEY_BIAS`](crate::masks::INVALID_KEY_BIAS)). Only one chunk of rows is
/// expanded to `(rows, 1, Sq, Sk)` at a time.
pub struct TimeBias {
    /// `(K, 1, Sq, Sk)` patterns, `0` or `-inf`.
    pub patterns: Tensor,
    /// `(R,)` u32 pattern index per row; None when every row uses pattern 0.
    pub kinds: Option<Tensor>,
    /// `(R, 1, 1, Sk)` key bias.
    pub keys: Tensor,
}

impl TimeBias {
    /// `(rows.len(), 1, Sq, Sk)` bias of a block of rows.
    fn rows(&self, rows: Range<usize>) -> Result<Tensor> {
        let patterns = match &self.kinds {
            Some(kinds) => self
                .patterns
                .index_select(&kinds.narrow(0, rows.start, rows.len())?, 0)?,
            None => self.patterns.clone(),
        };
        patterns.broadcast_add(&self.keys.narrow(0, rows.start, rows.len())?)
    }
}

/// Additive time-attention bias, `0` or `-inf` (or the invalid-key bias).
pub enum TimeMask {
    /// Over the whole `[Enc ; Dec]` sequence.
    Full(TimeBias),
    /// Two query passes: context queries over context keys only (`enc`, `L x L`), and
    /// horizon queries over all keys (`dec`, `(S - L) x S`). This skips the always-empty
    /// context-to-horizon quadrant.
    Split {
        enc: TimeBias,
        dec: TimeBias,
        enc_len: usize,
    },
}

/// Which rows variate attention mixes, per patch slot.
pub enum VariateMask {
    /// Every series has a single row, which reads only itself: the attention output is
    /// its value projection, so queries, keys and scores are skipped.
    SelfOnly,
    /// Series of `size` consecutive rows, each attending within itself, with a
    /// `(S * B, 1, size, size)` additive bias (batch index `slot * B + series`).
    Groups { size: usize, bias: Tensor },
    /// `(S, 1, R, R)` additive bias over all rows, for series whose rows are not
    /// consecutive runs of equal length.
    Dense(Tensor),
}

/// Per-forward inputs shared by every layer.
pub struct AttentionInputs {
    pub time_mask: TimeMask,
    pub variate_mask: VariateMask,
    pub cos: Tensor,
    pub sin: Tensor,
}

/// Multi-head attention over the time axis (`is_time`) or the variate axis.
///
/// Time attention applies RoPE; variate attention has no positional encoding. With the
/// `sdpa_sigmoid` gate, the attention output is multiplied by `sigmoid(x @ W_g)` before the
/// output projection.
pub struct Attention {
    wqkv: Linear,
    /// The value part of `wqkv`, for [`VariateMask::SelfOnly`].
    wv: Option<Linear>,
    wo: Linear,
    gate: Option<Linear>,
    num_heads: usize,
    head_dim: usize,
    is_time: bool,
}

impl Attention {
    pub fn new(
        embed_dim: usize,
        num_heads: usize,
        is_time: bool,
        bias: bool,
        gate: AttentionGate,
        vb: VarBuilder,
    ) -> Result<Self> {
        let gate = match gate {
            AttentionGate::SdpaSigmoid => {
                Some(linear_b(embed_dim, embed_dim, bias, vb.pp("_gate"))?)
            }
            AttentionGate::None => None,
        };
        let wqkv = linear_b(embed_dim, 3 * embed_dim, bias, vb.pp("wQKV"))?;
        // Output features are `(qkv, head_dim, n_heads)`, so the values are the last third.
        let wv = if is_time {
            None
        } else {
            let part = |t: &Tensor| t.narrow(0, 2 * embed_dim, embed_dim)?.contiguous();
            Some(Linear::new(
                part(wqkv.weight())?,
                wqkv.bias().map(part).transpose()?,
            ))
        };
        Ok(Self {
            wqkv,
            wv,
            wo: linear_b(embed_dim, embed_dim, bias, vb.pp("wO"))?,
            gate,
            num_heads,
            head_dim: embed_dim / num_heads,
            is_time,
        })
    }

    pub fn forward(&self, x: &Tensor, inputs: &AttentionInputs) -> Result<Tensor> {
        if self.is_time {
            return self.attend(x, inputs);
        }
        let (rows, slots, dim) = x.dims3()?;
        match &inputs.variate_mask {
            VariateMask::SelfOnly => self.self_only(x),
            // (R, N, D) to (N * B, G, D): each series' rows at one slot attend each other.
            VariateMask::Groups { size, .. } => {
                let xt = x.transpose(0, 1)?.contiguous()?;
                let xt = xt.reshape((slots * rows / size, *size, dim))?;
                let out = self.attend(&xt, inputs)?.reshape((slots, rows, dim))?;
                out.transpose(0, 1)?.contiguous()
            }
            // (R, N, D) to (N, R, D): all rows at one slot attend each other.
            VariateMask::Dense(_) => {
                let out = self.attend(&x.transpose(0, 1)?.contiguous()?, inputs)?;
                out.transpose(0, 1)?.contiguous()
            }
        }
    }

    /// Variate attention of rows that only read themselves: softmax over one key is
    /// exactly 1, so the output is the row's value.
    fn self_only(&self, x: &Tensor) -> Result<Tensor> {
        let (rows, slots, dim) = x.dims3()?;
        let wv = self
            .wv
            .as_ref()
            .expect("variate attention has a value projection");
        // (hd, H) feature order to the (H, hd) order of an attention output.
        let mut out = wv
            .forward(x)?
            .reshape((rows, slots, self.head_dim, self.num_heads))?
            .transpose(2, 3)?
            .reshape((rows, slots, dim))?;
        if let Some(gate) = &self.gate {
            out = (out * candle_nn::ops::sigmoid(&gate.forward(x)?)?)?;
        }
        self.wo.forward(&out)
    }

    /// Attention of `x` `(batch, seq, D)` over `seq`: time attention per row over patches,
    /// or variate attention per slot (and series) over rows.
    fn attend(&self, x: &Tensor, inputs: &AttentionInputs) -> Result<Tensor> {
        let (batch, seq, dim) = x.dims3()?;
        let qkv = self
            .wqkv
            .forward(x)?
            .reshape((batch, seq, 3, self.head_dim, self.num_heads))?;
        let split = |i: usize| qkv.narrow(2, i, 1)?.squeeze(2)?.permute((0, 3, 1, 2)); // (B, H, S, hd)
        let (mut q, mut k, v) = (split(0)?, split(1)?, split(2)?);

        let out = if self.is_time {
            q = apply_rotary(&q, &inputs.cos, &inputs.sin)?;
            k = apply_rotary(&k, &inputs.cos, &inputs.sin)?;
            self.time_attention(&q, &k, &v, &inputs.time_mask)?
        } else {
            let bias = match &inputs.variate_mask {
                VariateMask::Groups { bias, .. } | VariateMask::Dense(bias) => bias,
                VariateMask::SelfOnly => unreachable!("handled in forward"),
            };
            sdpa(&q, &k, &v, Some(bias))?
        };
        let mut out = out.transpose(1, 2)?.reshape((batch, seq, dim))?;
        if let Some(gate) = &self.gate {
            out = (out * candle_nn::ops::sigmoid(&gate.forward(x)?)?)?;
        }
        self.wo.forward(&out)
    }
}

impl Attention {
    /// Time attention of `(R, H, S, hd)` queries, keys and values.
    fn time_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: &TimeMask,
    ) -> Result<Tensor> {
        match mask {
            TimeMask::Full(bias) => chunked_sdpa(q, k, v, bias, ATTENTION_CHUNK_SCORES),
            TimeMask::Split { enc, dec, enc_len } => {
                let (n, seq) = (*enc_len, q.dim(2)?);
                let out_enc = chunked_sdpa(
                    &q.narrow(2, 0, n)?,
                    &k.narrow(2, 0, n)?,
                    &v.narrow(2, 0, n)?,
                    enc,
                    ATTENTION_CHUNK_SCORES,
                )?;
                let out_dec =
                    chunked_sdpa(&q.narrow(2, n, seq - n)?, k, v, dec, ATTENTION_CHUNK_SCORES)?;
                Tensor::cat(&[out_enc, out_dec], 2)
            }
        }
    }
}

/// [`sdpa`] over blocks of rows, or of heads within a row, so that at most `max_scores`
/// scores exist at once.
///
/// Each (row, head) pair is a separate matrix product of the same shape however the pairs
/// are grouped, so blocking does not change the result. Blocking queries would: the matrix
/// kernels pick their summation blocking from the matrix shape.
fn chunked_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    bias: &TimeBias,
    max_scores: usize,
) -> Result<Tensor> {
    let (rows, heads, queries, _) = q.dims4()?;
    let per_head = queries * k.dim(2)?;
    let per_row = heads * per_head;
    let (row_block, head_block) = if per_row <= max_scores {
        ((max_scores / per_row).min(rows), heads)
    } else {
        (1, (max_scores / per_head).clamp(1, heads))
    };
    let mut outs = Vec::with_capacity(rows.div_ceil(row_block));
    for r in (0..rows).step_by(row_block) {
        let r = r..(r + row_block).min(rows);
        let bias = bias.rows(r.clone())?;
        let block = |x: &Tensor, h: &Range<usize>| {
            x.narrow(0, r.start, r.len())?.narrow(1, h.start, h.len())
        };
        let mut head_outs = Vec::with_capacity(heads.div_ceil(head_block));
        for h in (0..heads).step_by(head_block) {
            let h = h..(h + head_block).min(heads);
            head_outs.push(sdpa(
                &block(q, &h)?,
                &block(k, &h)?,
                &block(v, &h)?,
                Some(&bias),
            )?);
        }
        outs.push(Tensor::cat(&head_outs, 1)?);
    }
    Tensor::cat(&outs, 0)
}

/// The attention(s) of one layer kind, in order, each a pre-norm residual, then one MLP.
pub struct TransformerBlock {
    attn_norms: Vec<RmsNorm>,
    attns: Vec<Attention>,
    mlp_norm: RmsNorm,
    mlp: FusedSwiGLU,
}

impl TransformerBlock {
    pub fn new(kind: LayerKind, c: &Config, vb: VarBuilder) -> Result<Self> {
        let axes = kind.axes();
        let attn_norms = (0..axes.len())
            .map(|i| RmsNorm::new(c.d_model, c.norm_eps, vb.pp("attn_norms").pp(i)))
            .collect::<Result<_>>()?;
        let attns = axes
            .iter()
            .enumerate()
            .map(|(i, &is_time)| {
                let gate = if is_time {
                    c.time_attn_gate
                } else {
                    c.variate_attn_gate
                };
                Attention::new(
                    c.d_model,
                    c.num_heads,
                    is_time,
                    c.attn_bias,
                    gate,
                    vb.pp("attns").pp(i),
                )
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            attn_norms,
            attns,
            mlp_norm: RmsNorm::new(c.d_model, c.norm_eps, vb.pp("mlp_norm"))?,
            mlp: FusedSwiGLU::new(c.d_model, c.mlp_hidden_dim, c.mlp_bias, vb.pp("mlp"))?,
        })
    }

    pub fn forward(&self, mut x: Tensor, inputs: &AttentionInputs) -> Result<Tensor> {
        for (norm, attn) in self.attn_norms.iter().zip(&self.attns) {
            x = (&x + attn.forward(&norm.forward(&x)?, inputs)?)?;
        }
        &x + self.mlp.forward(&self.mlp_norm.forward(&x)?)?
    }
}

pub struct Transformer {
    layers: Vec<TransformerBlock>,
    rope: Rope,
    out_norm: RmsNorm,
}

impl Transformer {
    pub fn new(c: &Config, vb: VarBuilder) -> Result<Self> {
        let layers = c
            .block_schedule
            .iter()
            .enumerate()
            .map(|(i, &kind)| TransformerBlock::new(kind, c, vb.pp("layers").pp(i)))
            .collect::<Result<_>>()?;
        Ok(Self {
            layers,
            rope: Rope::new(c.head_dim(), c.rope_theta, c.rope_dim()),
            out_norm: RmsNorm::new(c.d_model, c.norm_eps, vb.pp("out_norm"))?,
        })
    }

    pub fn rope(&self) -> &Rope {
        &self.rope
    }

    /// Run the stack on `x` `(R, N, D)`.
    pub fn forward(&self, mut x: Tensor, inputs: &AttentionInputs) -> Result<Tensor> {
        for layer in &self.layers {
            x = layer.forward(x, inputs)?;
        }
        self.out_norm.forward(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocked_attention_is_exact() {
        let dev = &Device::Cpu;
        let (rows, heads, seq, hd) = (3, 4, 7, 8);
        let x = |seed: f32| {
            let n = rows * heads * seq * hd;
            let v: Vec<f32> = (0..n).map(|i| ((i as f32 + seed) * 0.37).sin()).collect();
            Tensor::from_vec(v, (rows, heads, seq, hd), dev)
        };
        let (q, k, v) = (x(0.0).unwrap(), x(1.0).unwrap(), x(2.0).unwrap());
        // causal pattern for kind 0, full for kind 1; row 2's first key is invalid
        let pattern: Vec<f32> = (0..2 * seq * seq)
            .map(|i| {
                let (kind, i, j) = (i / (seq * seq), i / seq % seq, i % seq);
                if kind == 1 || j <= i {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect();
        let mut keys = vec![0.0f32; rows * seq];
        keys[2 * seq] = crate::masks::INVALID_KEY_BIAS;
        let bias = TimeBias {
            patterns: Tensor::from_vec(pattern, (2, 1, seq, seq), dev).unwrap(),
            kinds: Some(Tensor::new(&[0u32, 1, 0], dev).unwrap()),
            keys: Tensor::from_vec(keys, (rows, 1, 1, seq), dev).unwrap(),
        };
        let full = sdpa(&q, &k, &v, Some(&bias.rows(0..rows).unwrap())).unwrap();
        let full: Vec<f32> = full.flatten_all().unwrap().to_vec1().unwrap();
        // whole, rows of two, single rows, pairs of heads, single heads
        for max_scores in [
            usize::MAX,
            2 * heads * seq * seq,
            heads * seq * seq,
            2 * seq * seq,
            1,
        ] {
            let out = chunked_sdpa(&q, &k, &v, &bias, max_scores).unwrap();
            let out: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
            assert!(
                out.iter()
                    .zip(&full)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{max_scores}"
            );
        }
    }
}
