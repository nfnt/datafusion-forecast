//! Attention masks and time positions for Decoupled Horizon Forecasting (DHF) decoding.
//!
//! The sequence is `[Enc ; Dec]`: encoder tokens embed the observed context patches;
//! decoder tokens are forecast queries for the horizon patches. Masks are boolean,
//! `true` meaning "may attend", stored row-major.

/// A boolean mask of shape `(batch, queries, keys)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Mask {
    pub data: Vec<bool>,
    pub batch: usize,
    pub queries: usize,
    pub keys: usize,
}

impl Mask {
    fn from_fn(
        batch: usize,
        queries: usize,
        keys: usize,
        f: impl Fn(usize, usize, usize) -> bool,
    ) -> Self {
        let mut data = Vec::with_capacity(batch * queries * keys);
        for b in 0..batch {
            for i in 0..queries {
                for j in 0..keys {
                    data.push(f(b, i, j));
                }
            }
        }
        Self {
            data,
            batch,
            queries,
            keys,
        }
    }

    #[cfg(test)]
    pub fn get(&self, b: usize, i: usize, j: usize) -> bool {
        self.data[(b * self.queries + i) * self.keys + j]
    }

    /// Additive attention bias (0 or -inf) for the sub-block `queries[q0..q1] x keys[k0..k1]`.
    pub fn bias(&self, q: std::ops::Range<usize>, k: std::ops::Range<usize>) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.batch * q.len() * k.len());
        for b in 0..self.batch {
            for i in q.clone() {
                let row = &self.data[(b * self.queries + i) * self.keys..][..self.keys];
                out.extend(
                    row[k.clone()]
                        .iter()
                        .map(|&a| if a { 0.0 } else { f32::NEG_INFINITY }),
                );
            }
        }
        out
    }
}

/// Per-patch observation: whether any step of the patch is observed. `(R, N)` from `(R, N * ps)`.
pub fn observed_per_patch(observed: &[bool], patch_size: usize) -> Vec<bool> {
    observed
        .chunks_exact(patch_size)
        .map(|c| c.iter().any(|&o| o))
        .collect()
}

/// `(R, N)` encoder key validity.
///
/// A patch is a valid key if it is observed, or if `attend_missing` is set and the row's
/// series has started at or before it. Leading all-unobserved patches are left-padding
/// and are never valid keys.
pub fn enc_key_validity(obs_per_patch: &[bool], n: usize, attend_missing: bool) -> Vec<bool> {
    if !attend_missing {
        return obs_per_patch.to_vec();
    }
    let mut out = Vec::with_capacity(obs_per_patch.len());
    for row in obs_per_patch.chunks_exact(n) {
        let mut started = false;
        out.extend(row.iter().map(|&o| {
            started |= o;
            started
        }));
    }
    out
}

/// Causal encoder mask over valid keys, with the diagonal always open.
#[cfg(test)]
#[inline]
fn enc_causal(key_valid: &[bool], i: usize, j: usize) -> bool {
    (j <= i && key_valid[j]) || i == j
}

/// `(R, L+h, L+h)` time mask for one block without covariates.
///
/// The encoder is causal; every decoder token reads the whole valid context and its own
/// causal decoder chain; the context never reads the horizon.
/// Materialized reference for [`forecast_time_pattern`] and [`time_key_bias`], used by tests.
#[cfg(test)]
pub fn forecast_time_mask(context_len: usize, horizon: usize, key_valid: &[bool]) -> Mask {
    let ll = context_len;
    let rows = key_valid.len() / ll;
    let s = ll + horizon;
    Mask::from_fn(rows, s, s, |r, i, j| {
        let kv = &key_valid[r * ll..(r + 1) * ll];
        match (i < ll, j < ll) {
            (true, true) => enc_causal(kv, i, j),
            (true, false) => false,
            (false, true) => kv[j],
            (false, false) => j <= i,
        }
    })
}

/// `(R, L+h)` time positions `0..L+h`, anchored at each row's first observed patch.
pub fn forecast_positions(context_len: usize, horizon: usize, first_obs: &[usize]) -> Vec<i64> {
    first_obs
        .iter()
        .flat_map(|&f| (0..(context_len + horizon) as i64).map(move |p| p - f as i64))
        .collect()
}

/// `(R, 2N)` time mask over a full `[Enc ; Dec]` window (the covariate path).
///
/// - Enc to Enc: causal over valid keys; full (non-causal) for `bidirectional_rows`, the
///   known-future covariates.
/// - Enc to Dec: never.
/// - Dec to Enc: a decoder token in the block anchored at `b` reads valid `Enc[j]`, `j < b`.
/// - Dec to Dec: causal within the same block; every decoder token reads itself.
///
/// `min_context` is the first decoded patch of each row; decoder positions before it are
/// inactive.
/// Materialized reference for [`two_stream_time_patterns`] and [`time_key_bias`], used by tests.
#[cfg(test)]
pub fn two_stream_time_mask(
    min_context: &[usize],
    n: usize,
    block_size: usize,
    key_valid: &[bool],
    bidirectional_rows: Option<&[bool]>,
) -> Mask {
    let rows = min_context.len();
    // Per row and decoder position: the anchor of its block, or None if inactive.
    let anchors: Vec<Option<usize>> = min_context
        .iter()
        .flat_map(|&s| {
            (0..n).map(move |p| (p >= s).then(|| s + block_size * ((p - s) / block_size)))
        })
        .collect();
    Mask::from_fn(rows, 2 * n, 2 * n, |r, i, j| {
        let kv = &key_valid[r * n..(r + 1) * n];
        let anchor = &anchors[r * n..(r + 1) * n];
        match (i < n, j < n) {
            (true, true) => {
                if bidirectional_rows.is_some_and(|b| b[r]) {
                    kv[j] || i == j
                } else {
                    enc_causal(kv, i, j)
                }
            }
            (true, false) => false,
            (false, true) => anchor[i - n].is_some_and(|a| j < a) && kv[j],
            (false, false) => {
                let (i, j) = (i - n, j - n);
                let same_block = matches!((anchor[i], anchor[j]), (Some(a), Some(b)) if a == b);
                (same_block && j <= i) || i == j
            }
        }
    })
}

/// `(N_enc + N_dec, R, R)` variate mask, or None for a single row.
///
/// A row reads another row of the same series; on encoder slots the key row must also be
/// a valid key. The firewalls apply on every slot; the diagonal is always open:
/// - a covariate row may not read a target row (covariates are exogenous);
/// - a known-future (bidirectional) row may not read a causal row. Without this, a
///   covariate's view of the future could relay a target's own future back into it.
pub fn two_stream_variate_mask(
    group_ids: &[usize],
    enc_key_valid: &[bool],
    n_enc: usize,
    n_dec: usize,
    target_rows: Option<&[bool]>,
    bidirectional_rows: Option<&[bool]>,
) -> Option<Mask> {
    let r = group_ids.len();
    if r == 1 {
        return None;
    }
    let forbid = |q: usize, k: usize| {
        target_rows.is_some_and(|t| !t[q] && t[k])
            || bidirectional_rows.is_some_and(|b| b[q] && !b[k])
    };
    Some(Mask::from_fn(n_enc + n_dec, r, r, |slot, q, k| {
        let slot_valid = slot >= n_enc || enc_key_valid[k * n_enc + slot];
        (group_ids[q] == group_ids[k] && slot_valid && !forbid(q, k)) || q == k
    }))
}

/// Additive bias of a key that is not a valid key of its row.
///
/// The dense masks above let an invalid encoder token read itself only. Nothing else ever
/// reads an invalid token, so its value cannot reach a forecast; the factored masks below
/// mask invalid keys for every query instead, with a finite bias so that a query whose
/// keys are all invalid stays finite rather than NaN. For every other query `exp`
/// underflows to exactly 0 on these keys, as on `-inf`, so their outputs are unchanged.
pub const INVALID_KEY_BIAS: f32 = -1e30;

/// `(1, L+h, L+h)` time pattern of `forecast_time_mask` without key validity: every row
/// shares it, and [`time_key_bias`] adds the validity of each row's keys.
pub fn forecast_time_pattern(context_len: usize, horizon: usize) -> Mask {
    let ll = context_len;
    let s = ll + horizon;
    Mask::from_fn(1, s, s, |_, i, j| match (i < ll, j < ll) {
        (true, true) | (false, false) => j <= i,
        (true, false) => false,
        (false, true) => true,
    })
}

/// `(2, 2N, 2N)` time patterns of `two_stream_time_mask` without key validity: index 0
/// for causal rows, 1 for bidirectional (known-future covariate) rows.
///
/// Every row decodes from patch `min_context`.
pub fn two_stream_time_patterns(min_context: usize, n: usize, block_size: usize) -> Mask {
    let anchor = |p: usize| {
        (p >= min_context).then(|| min_context + block_size * ((p - min_context) / block_size))
    };
    Mask::from_fn(2, 2 * n, 2 * n, |kind, i, j| match (i < n, j < n) {
        (true, true) => kind == 1 || j <= i,
        (true, false) => false,
        (false, true) => anchor(i - n).is_some_and(|a| j < a),
        (false, false) => {
            let (i, j) = (i - n, j - n);
            let same_block = matches!((anchor(i), anchor(j)), (Some(a), Some(b)) if a == b);
            (same_block && j <= i) || i == j
        }
    })
}

/// `(R, n_enc + n_dec)` additive key bias: [`INVALID_KEY_BIAS`] on invalid encoder keys,
/// 0 elsewhere. Decoder keys are always valid.
pub fn time_key_bias(enc_key_valid: &[bool], n_enc: usize, n_dec: usize) -> Vec<f32> {
    enc_key_valid
        .chunks_exact(n_enc)
        .flat_map(|row| {
            row.iter()
                .map(|&v| if v { 0.0 } else { INVALID_KEY_BIAS })
                .chain(std::iter::repeat_n(0.0, n_dec))
        })
        .collect()
}

/// The number of rows per series if every series is a run of that many consecutive rows.
///
/// Variate attention then only needs to run within each run.
pub fn uniform_groups(group_ids: &[usize]) -> Option<usize> {
    let first = *group_ids.first()?;
    let size = group_ids.iter().take_while(|&&g| g == first).count();
    let mut seen = std::collections::HashSet::new();
    group_ids
        .chunks(size.max(1))
        .all(|run| run.len() == size && run.iter().all(|&g| g == run[0]) && seen.insert(run[0]))
        .then_some(size)
}

/// [`two_stream_variate_mask`] restricted to the series: `(S * B, G, G)` for `B` series of
/// `G` consecutive rows (see [`uniform_groups`]), batch index `slot * B + series`.
pub fn grouped_variate_mask(
    group_size: usize,
    enc_key_valid: &[bool],
    n_enc: usize,
    n_dec: usize,
    target_rows: Option<&[bool]>,
    bidirectional_rows: Option<&[bool]>,
) -> Mask {
    let g = group_size;
    let series = enc_key_valid.len() / n_enc / g;
    let forbid = |q: usize, k: usize| {
        target_rows.is_some_and(|t| !t[q] && t[k])
            || bidirectional_rows.is_some_and(|b| b[q] && !b[k])
    };
    Mask::from_fn((n_enc + n_dec) * series, g, g, |batch, qi, ki| {
        let (slot, s) = (batch / series, batch % series);
        let (q, k) = (s * g + qi, s * g + ki);
        let slot_valid = slot >= n_enc || enc_key_valid[k * n_enc + slot];
        (slot_valid && !forbid(q, k)) || q == k
    })
}

/// `(R, 2N)` positions: encoder and decoder token of the same patch share a time.
pub fn two_stream_positions(n: usize, first_obs: &[usize]) -> Vec<i64> {
    first_obs
        .iter()
        .flat_map(|&f| (0..2 * n as i64).map(move |p| p % n as i64 - f as i64))
        .collect()
}

/// Per row: the first patch any row of its group observes (0 if none).
pub fn group_first_observed_patch(
    obs_per_patch: &[bool],
    n: usize,
    group_ids: &[usize],
) -> Vec<usize> {
    let num_groups = group_ids.iter().max().map_or(0, |&g| g + 1);
    let mut first = vec![usize::MAX; num_groups];
    for (row, &g) in obs_per_patch.chunks_exact(n).zip(group_ids) {
        if let Some(p) = row.iter().position(|&o| o) {
            first[g] = first[g].min(p);
        }
    }
    group_ids
        .iter()
        .map(|&g| if first[g] == usize::MAX { 0 } else { first[g] })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_validity_skips_left_padding() {
        let obs = [false, false, true, false, true, false];
        assert_eq!(
            enc_key_validity(&obs, 3, true),
            [false, false, true, false, true, true]
        );
        assert_eq!(enc_key_validity(&obs, 3, false), obs);
    }

    #[test]
    fn forecast_mask_quadrants() {
        let kv = [false, true, true];
        let m = forecast_time_mask(3, 2, &kv);
        // padding key never read except by itself
        assert!(m.get(0, 0, 0) && !m.get(0, 1, 0) && !m.get(0, 3, 0));
        // causal encoder, encoder never reads decoder
        assert!(m.get(0, 2, 1) && !m.get(0, 1, 2) && !m.get(0, 2, 3));
        // decoder reads all valid context and its causal chain
        assert!(m.get(0, 3, 2) && m.get(0, 4, 3) && !m.get(0, 3, 4));
    }

    #[test]
    fn two_stream_mask_blocks() {
        // n = 6 patches, decoding from patch 2 with block size 2: blocks anchored at 2 and 4.
        let kv = [true; 6];
        let m = two_stream_time_mask(&[2], 6, 2, &kv, None);
        let dec = |p: usize| 6 + p;
        assert!(m.get(0, dec(3), 1) && !m.get(0, dec(3), 2)); // block at 2 reads Enc[<2]
        assert!(m.get(0, dec(5), 3) && !m.get(0, dec(5), 4)); // block at 4 reads Enc[<4]
        assert!(m.get(0, dec(3), dec(2)) && !m.get(0, dec(4), dec(3))); // causal within block only
        assert!(m.get(0, dec(0), dec(0)) && !m.get(0, dec(0), 0)); // inactive: only itself
        let bi = two_stream_time_mask(&[2], 6, 2, &kv, Some(&[true]));
        assert!(bi.get(0, 0, 5) && !m.get(0, 0, 5));
    }

    #[test]
    fn variate_firewall() {
        // rows: target, past cov, known cov (one series) + another series' target
        let groups = [0, 0, 0, 1];
        let targets = [true, false, false, true];
        let known = [false, false, true, false];
        let kv = [true; 4];
        let m = two_stream_variate_mask(&groups, &kv, 1, 0, Some(&targets), Some(&known)).unwrap();
        assert!(m.get(0, 0, 1) && m.get(0, 0, 2)); // target reads covariates
        assert!(!m.get(0, 1, 0) && !m.get(0, 2, 0)); // covariates never read the target
        assert!(!m.get(0, 2, 1) && m.get(0, 1, 2)); // known never reads a causal row
        assert!(!m.get(0, 0, 3)); // other series
    }

    /// Factored masks agree with the dense ones on every query whose own key is valid.
    fn assert_factored(dense: &Mask, pattern: &Mask, kinds: &[usize], keys: &[f32], n_enc: usize) {
        let s = dense.keys;
        for r in 0..dense.batch {
            for i in 0..dense.queries {
                if i < n_enc && keys[r * s + i] != 0.0 {
                    continue; // invalid token: read by nobody else
                }
                for j in 0..s {
                    let factored = pattern.get(kinds[r], i, j) && keys[r * s + j] == 0.0;
                    assert_eq!(dense.get(r, i, j), factored, "row {r} query {i} key {j}");
                }
            }
        }
    }

    #[test]
    fn factored_time_masks_match_dense() {
        // rows: late start, gap in the middle (invalid only without attend_missing), full
        let obs = [
            false, false, true, true, false, true, //
            true, false, false, true, true, true, //
            true, true, true, true, true, true,
        ];
        for attend_missing in [true, false] {
            let kv = enc_key_validity(&obs, 6, attend_missing);
            let dense = forecast_time_mask(6, 3, &kv);
            let keys = time_key_bias(&kv, 6, 3);
            assert_factored(&dense, &forecast_time_pattern(6, 3), &[0; 3], &keys, 6);

            let bidirectional = [false, true, false];
            let dense = two_stream_time_mask(&[4; 3], 6, 2, &kv, Some(&bidirectional));
            let keys = time_key_bias(&kv, 6, 6);
            let patterns = two_stream_time_patterns(4, 6, 2);
            assert_factored(&dense, &patterns, &[0, 1, 0], &keys, 6);
        }
    }

    #[test]
    fn grouped_variate_mask_matches_dense() {
        // two series of three rows: target, past covariate, known covariate
        let groups = [0, 0, 0, 1, 1, 1];
        assert_eq!(uniform_groups(&groups), Some(3));
        let targets = [true, false, false, true, false, false];
        let known = [false, false, true, false, false, true];
        let obs: Vec<bool> = (0..6 * 4).map(|i| i % 5 != 0).collect();
        let kv = enc_key_validity(&obs, 4, false);
        let dense =
            two_stream_variate_mask(&groups, &kv, 4, 2, Some(&targets), Some(&known)).unwrap();
        let grouped = grouped_variate_mask(3, &kv, 4, 2, Some(&targets), Some(&known));
        assert_eq!(grouped.batch, 6 * 2);
        for slot in 0..6 {
            for q in 0..6 {
                for k in 0..6 {
                    let expect = if q / 3 == k / 3 {
                        grouped.get(slot * 2 + q / 3, q % 3, k % 3)
                    } else {
                        false
                    };
                    assert_eq!(dense.get(slot, q, k), expect, "slot {slot} {q}->{k}");
                }
            }
        }
    }

    #[test]
    fn uniform_groups_needs_equal_consecutive_runs() {
        assert_eq!(uniform_groups(&[7]), Some(1));
        assert_eq!(uniform_groups(&[0, 1, 2]), Some(1));
        assert_eq!(uniform_groups(&[0, 0, 1]), None);
        assert_eq!(uniform_groups(&[0, 1, 0, 1]), None);
        assert_eq!(uniform_groups(&[0, 0, 1, 1, 0, 0]), None);
        assert_eq!(uniform_groups(&[]), None);
    }

    #[test]
    fn positions_and_first_obs() {
        let obs = [false, true, true, false, false, true, false, false, false];
        assert_eq!(group_first_observed_patch(&obs, 3, &[0, 0, 1]), [1, 1, 0]);
        assert_eq!(forecast_positions(2, 1, &[1]), [-1, 0, 1]);
        assert_eq!(two_stream_positions(2, &[0]), [0, 1, 0, 1]);
    }
}
