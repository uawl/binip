//! Lookup table definition and multiplicity computation.

use std::collections::HashMap;

use field::{FieldElem, GF2_128, batch_inv};

/// A lookup table: a set of valid field elements.
///
/// The table stores `entries` (the distinct values).
/// Length is padded to a power of two.
#[derive(Debug, Clone, bincode::Encode, bincode::Decode)]
pub struct LookupTable {
  /// Table entries `t_0, t_1, …`. Length must be a power of two.
  pub entries: Vec<GF2_128>,
  /// Number of variables: `entries.len() == 1 << n_vars`.
  pub n_vars: u32,
  /// Blake3 hash of all entry bytes (for fast transcript absorption).
  pub content_hash: [u8; 32],
}

impl LookupTable {
  /// Build a lookup table from raw entries, padded to the next power of two.
  ///
  /// Padding entries are set to `entries[0]` (a valid entry) so that
  /// `β + pad ≠ 0` with overwhelming probability for random β.
  pub fn new(entries: Vec<GF2_128>) -> Self {
    assert!(
      !entries.is_empty(),
      "LookupTable: entries must not be empty"
    );
    let len = entries.len().next_power_of_two();
    let n_vars = len.trailing_zeros();
    let pad_val = entries[0];
    let mut padded = entries;
    padded.resize(len, pad_val);
    let content_hash = Self::hash_entries(&padded);
    Self {
      entries: padded,
      n_vars,
      content_hash,
    }
  }

  fn hash_entries(entries: &[GF2_128]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    for e in entries {
      hasher.update(&e.lo.to_le_bytes());
      hasher.update(&e.hi.to_le_bytes());
    }
    *hasher.finalize().as_bytes()
  }
}

/// Compute the γ-weighted witness fractional evaluations:
///
///   `h_w[j] = γ^j / (β + w_j)` for j = 0, …, N−1
///
/// The γ-weighting prevents char-2 cancellation for repeated witness values.
pub fn witness_fractional_evals(
  witness: &[GF2_128],
  beta: GF2_128,
  gamma: GF2_128,
) -> Vec<GF2_128> {
  let n = witness.len();
  if n == 0 {
    return vec![];
  }

  // ── Single alloc: prefix products → reused as result ─────────────────
  // buf[i] = (β+w_0) * (β+w_1) * … * (β+w_i)
  let mut buf = Vec::with_capacity(n);
  buf.push(beta + witness[0]);
  for i in 1..n {
    buf.push(buf[i - 1] * (beta + witness[i]));
  }

  // ── Single inversion of the full product ─────────────────────────────
  let mut inv_acc = buf[n - 1].inv();

  // ── Backward pass: overwrite buf in-place ────────────────────────────
  // At iteration i (n-1 → 1): read buf[i-1] (untouched), write buf[i].
  for i in (1..n).rev() {
    let inv_i = inv_acc * buf[i - 1]; // 1/(β+w_i)
    inv_acc = inv_acc * (beta + witness[i]);
    buf[i] = inv_i;
  }
  buf[0] = inv_acc; // 1/(β+w_0)

  // ── In-place γ^j weighting: buf[j] *= γ^j ───────────────────────────
  let mut gamma_pow = GF2_128::one();
  for val in buf.iter_mut() {
    *val *= gamma_pow;
    gamma_pow *= gamma;
  }
  buf
}

/// Compute the γ-weighted table fractional evaluations:
///
///   `h_t[i] = M_i / (β + t_i)` for i = 0, …, T−1
///
/// where `M_i = Σ_{j: w_j = t_i} γ^j` is the γ-weighted multiplicity.
pub fn table_fractional_evals(
  table: &LookupTable,
  witness: &[GF2_128],
  beta: GF2_128,
  gamma: GF2_128,
) -> Vec<GF2_128> {
  // Build index: table entry → position.  O(T)
  let index: HashMap<GF2_128, usize> = table
    .entries
    .iter()
    .enumerate()
    .map(|(i, &t)| (t, i))
    .collect();

  // Compute weighted multiplicities: M_i = Σ_{j: w_j = t_i} γ^j.  O(N)
  let mut weighted_mult = vec![GF2_128::zero(); table.entries.len()];
  let mut gamma_pow = GF2_128::one();
  for &w_j in witness {
    let idx = *index
      .get(&w_j)
      .unwrap_or_else(|| panic!("witness element {w_j:?} not found in table"));
    weighted_mult[idx] += gamma_pow;
    gamma_pow *= gamma;
  }

  // Collect non-zero entries for batch inversion.
  let non_zero: Vec<usize> = weighted_mult
    .iter()
    .enumerate()
    .filter(|(_, m)| !m.is_zero())
    .map(|(i, _)| i)
    .collect();

  let denoms: Vec<GF2_128> = non_zero.iter().map(|&i| beta + table.entries[i]).collect();
  let inv_denoms = batch_inv(&denoms);

  // h_t[i] = M_i / (β + t_i)
  let mut result = vec![GF2_128::zero(); table.entries.len()];
  for (k, &i) in non_zero.iter().enumerate() {
    result[i] = weighted_mult[i] * inv_denoms[k];
  }
  result
}
