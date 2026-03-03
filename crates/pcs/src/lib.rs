//! Batch Tensor PCS — polynomial commitment over GF(2^128).
//!
//! ## Protocol
//!
//! For an n-variable MLE `P : {0,1}^n → GF(2^128)`:
//!
//! **Commit**
//! 1. Arrange the `2^n` evaluations as a matrix `M[n_rows][k]` where
//!    `n_rows = 2^m_rows`, `k = 2^m_cols`, `m_cols = n/2`, `m_rows = n - m_cols`.
//!    Layout: `M[row][col] = evals[row * k + col]` (low bits = col index).
//! 2. Encode each row with a rate-1/4 systematic GF(2^128) code → `encoded_rows[n_rows][4k]`.
//! 3. For each encoded column `j`, compute `leaf_j = Blake3(col_j elements)`.
//! 4. Build a Merkle tree over the `4k` leaf hashes → root.
//! 5. Absorb the root into the Fiat-Shamir transcript.
//!
//! **Open at r = (r_hi ‖ r_lo)**  (`r_hi = r[0..m_rows]`, `r_lo = r[m_rows..]`)
//! 1. Compute `t_raw[j] = Σ_i eq(r_hi, i) · M[i][j]` for `j = 0..k`.
//! 2. Absorb `t_raw` into the transcript.
//! 3. Squeeze `n_queries` column indices `q_0 .. q_{Q-1}` from the transcript.
//! 4. For each `q_s`: reveal `encoded_rows[*][q_s]` (the column) + Merkle proof.
//!
//! **Verify**
//! 1. Check inner product claim: `<eq(r_lo), t_raw> == claim`.
//! 2. Absorb `t_raw`, squeeze query indices (mirror prover).
//! 3. Compute `t_enc = encode(t_raw)`.
//! 4. For each queried column `q_s`:
//!    a. Verify Merkle inclusion proof against the root.
//!    b. Check `Σ_i eq(r_hi, i) · col_elem[i] == t_enc[q_s]`
//!       (linear consistency, uses that encode is GF(2^128)-linear).

mod code;
mod eq;
mod merkle;

use field::{FieldElem, GF2_128};
use rayon::prelude::*;
use transcript::Transcript;

pub use merkle::{Hash, MerkleProof};

// ─── params ──────────────────────────────────────────────────────────────────

/// Configuration for the Tensor PCS.
#[derive(Clone, Debug)]
pub struct PcsParams {
  /// Number of variables in the committed MLE.  Must be ≥ 2.
  pub n_vars: u32,
  /// Number of Merkle column queries.
  /// ~40 gives ≥100-bit soundness for a rate-1/4 code.
  pub n_queries: usize,
}

impl PcsParams {
  /// Number of "column" variables (low bits of eval index).
  fn m_cols(&self) -> usize {
    (self.n_vars / 2) as usize
  }
  /// Number of "row" variables (high bits of eval index).
  fn m_rows(&self) -> usize {
    (self.n_vars as usize) - self.m_cols()
  }
  /// Raw columns per row: 2^m_cols.
  fn n_raw_cols(&self) -> usize {
    1 << self.m_cols()
  }
  /// Number of rows: 2^m_rows.
  fn n_rows(&self) -> usize {
    1 << self.m_rows()
  }
  /// Encoded columns per row: 4 * 2^m_cols.
  fn n_enc_cols(&self) -> usize {
    4 * self.n_raw_cols()
  }
}

// ─── public types ────────────────────────────────────────────────────────────

/// The Merkle-root commitment returned by `commit`.
#[derive(Clone, Debug)]
pub struct Commitment {
  pub root: Hash,
  pub n_vars: u32,
  /// Precomputed for the verifier's benefit.
  pub n_enc_cols: usize,
  pub n_rows: usize,
}

/// Internal prover state needed to answer opening queries.
pub struct PcsState {
  pub commitment: Commitment,
  params: PcsParams,
  /// Flat row-major buffer: `raw_flat[i * k + j]` = `M[i][j]`.
  raw_flat: Vec<GF2_128>,
  /// Flat row-major buffer: `enc_flat[i * n_enc + j]` = encoded row `i`, column `j`.
  enc_flat: Vec<GF2_128>,
  tree: merkle::MerkleTree,
}

/// A single column revealed during an opening.
#[derive(Clone, Debug)]
pub struct ColumnOpening {
  pub col_idx: usize,
  /// `encoded_rows[0..n_rows][col_idx]`.
  pub col_elems: Vec<GF2_128>,
  pub proof: MerkleProof,
}

/// The full opening proof produced by `open`.
#[derive(Clone, Debug)]
pub struct OpenProof {
  /// Mixed row: `t_raw[j] = Σ_i eq(r_hi, i) · M[i][j]`.  Length = k.
  pub t_raw: Vec<GF2_128>,
  pub column_openings: Vec<ColumnOpening>,
}

// ─── commit ───────────────────────────────────────────────────────────────────

/// Commit to an n-variable MLE given as its `2^n` evaluations in lex order.
///
/// Absorbs the commitment root into `transcript`.
pub fn commit<T: Transcript>(
  evals: &[GF2_128],
  params: &PcsParams,
  transcript: &mut T,
) -> (Commitment, PcsState) {
  let n_rows = params.n_rows();
  let k = params.n_raw_cols();
  let n_enc = params.n_enc_cols();
  assert_eq!(evals.len(), n_rows * k, "evals.len() must equal 2^n_vars");

  // Keep raw evals as flat buffer (already row-major).
  let raw_flat = evals.to_vec();

  // Encode every row into a flat buffer.
  let code = code::LinearCode::new(k);
  let mut enc_flat = vec![GF2_128::zero(); n_rows * n_enc];
  for i in 0..n_rows {
    code.encode_into(
      &raw_flat[i * k..(i + 1) * k],
      &mut enc_flat[i * n_enc..(i + 1) * n_enc],
    );
  }

  // Hash each column to form Merkle leaves
  let leaf_hashes: Vec<Hash> = (0..n_enc)
    .map(|j| {
      let col: Vec<GF2_128> = (0..n_rows).map(|i| enc_flat[i * n_enc + j]).collect();
      hash_column(&col)
    })
    .collect();

  let tree = merkle::MerkleTree::build(&leaf_hashes);
  let root = tree.root();

  transcript.absorb_bytes(&root);

  let commitment = Commitment {
    root,
    n_vars: params.n_vars,
    n_enc_cols: n_enc,
    n_rows,
  };

  let state = PcsState {
    commitment: commitment.clone(),
    params: params.clone(),
    raw_flat,
    enc_flat,
    tree,
  };

  (commitment, state)
}

/// Parallel CPU variant of [`commit`].
///
/// Parallelises row encoding (heaviest: O(n_rows × k × 3k) field muls)
/// and Merkle leaf hashing (O(n_enc × n_rows) blake3 calls).
pub fn commit_par<T: Transcript>(
  evals: &[GF2_128],
  params: &PcsParams,
  transcript: &mut T,
) -> (Commitment, PcsState) {
  let n_rows = params.n_rows();
  let k = params.n_raw_cols();
  let n_enc = params.n_enc_cols();
  assert_eq!(evals.len(), n_rows * k, "evals.len() must equal 2^n_vars");

  // Keep raw evals as flat buffer (already row-major).
  let raw_flat = evals.to_vec();

  // Encode every row into a single flat buffer — PARALLEL
  let code = code::LinearCode::new(k);
  let mut enc_flat = vec![GF2_128::zero(); n_rows * n_enc];
  enc_flat
    .par_chunks_mut(n_enc)
    .enumerate()
    .for_each(|(i, dest)| {
      code.encode_into(&raw_flat[i * k..(i + 1) * k], dest);
    });

  // Hash each column to form Merkle leaves — PARALLEL
  let leaf_hashes: Vec<Hash> = (0..n_enc)
    .into_par_iter()
    .map(|j| {
      let col: Vec<GF2_128> = (0..n_rows).map(|i| enc_flat[i * n_enc + j]).collect();
      hash_column(&col)
    })
    .collect();

  let tree = merkle::MerkleTree::build(&leaf_hashes);
  let root = tree.root();

  transcript.absorb_bytes(&root);

  let commitment = Commitment {
    root,
    n_vars: params.n_vars,
    n_enc_cols: n_enc,
    n_rows,
  };

  let state = PcsState {
    commitment: commitment.clone(),
    params: params.clone(),
    raw_flat,
    enc_flat,
    tree,
  };

  (commitment, state)
}

// ─── open ─────────────────────────────────────────────────────────────────────

/// Produce an opening proof for evaluation at `r`.
///
/// Returns `(claimed_eval, proof)`.
/// Precondition: `transcript` is in the state left by `commit`.
pub fn open<T: Transcript>(
  state: &PcsState,
  r: &[GF2_128],
  transcript: &mut T,
) -> (GF2_128, OpenProof) {
  let params = &state.params;
  let m_rows = params.m_rows();
  let n_rows = params.n_rows();
  let k = params.n_raw_cols();
  let n_enc = params.n_enc_cols();
  assert_eq!(r.len(), params.n_vars as usize);

  // r[0..m_rows] = challenges for high bits (row variables).
  // r[m_rows..]  = challenges for low bits (col variables).
  // eq_evals(v)[i]: v[0] = highest bit of i, matching mle_eval convention.
  let eq_hi = eq::eq_evals(&r[..m_rows]);
  let eq_lo = eq::eq_evals(&r[m_rows..]);

  // t_raw[j] = Σ_i eq_hi[i] · M[i][j]  (flat buffer indexed)
  let t_raw: Vec<GF2_128> = (0..k)
    .map(|j| {
      (0..n_rows).fold(GF2_128::zero(), |acc, i| {
        acc + eq_hi[i] * state.raw_flat[i * k + j]
      })
    })
    .collect();

  // claimed eval = <eq_lo, t_raw>
  let claim = eq::inner_product(&eq_lo, &t_raw);

  // Absorb t_raw into transcript
  for &v in &t_raw {
    transcript.absorb_field(v);
  }

  // Squeeze query column indices from Fiat-Shamir
  let queries: Vec<usize> = (0..params.n_queries)
    .map(|_| (transcript.squeeze_challenge().lo as usize) % n_enc)
    .collect();

  // Build column openings  (strided access into flat enc buffer)
  let column_openings = queries
    .into_iter()
    .map(|j| {
      let col_elems: Vec<GF2_128> = (0..n_rows).map(|i| state.enc_flat[i * n_enc + j]).collect();
      let proof = state.tree.prove(j);
      ColumnOpening {
        col_idx: j,
        col_elems,
        proof,
      }
    })
    .collect();

  (
    claim,
    OpenProof {
      t_raw,
      column_openings,
    },
  )
}

// ─── verify ───────────────────────────────────────────────────────────────────

/// Verify an opening proof.
///
/// Precondition: `transcript` is in the state produced by re-absorbing the
/// commitment root (i.e., `transcript.absorb_bytes(&commitment.root)`).
pub fn verify<T: Transcript>(
  commitment: &Commitment,
  r: &[GF2_128],
  claim: GF2_128,
  proof: &OpenProof,
  params: &PcsParams,
  transcript: &mut T,
) -> bool {
  let m_rows = params.m_rows();
  let k = params.n_raw_cols();
  let n_enc = params.n_enc_cols();
  let n_rows = params.n_rows();
  assert_eq!(r.len(), params.n_vars as usize);

  // Same split as open()
  let eq_lo = eq::eq_evals(&r[m_rows..]);
  let eq_hi = eq::eq_evals(&r[..m_rows]);

  // 1. Inner-product claim
  if eq::inner_product(&eq_lo, &proof.t_raw) != claim {
    return false;
  }

  // 2. Re-absorb t_raw (mirrors prover)
  for &v in &proof.t_raw {
    transcript.absorb_field(v);
  }

  // 3. Re-squeeze same query indices
  let queries: Vec<usize> = (0..params.n_queries)
    .map(|_| (transcript.squeeze_challenge().lo as usize) % n_enc)
    .collect();

  // 4. Encode t_raw (same code as prover)
  let code = code::LinearCode::new(k);
  let t_enc = code.encode(&proof.t_raw);

  // 5. Check each column opening
  if proof.column_openings.len() != params.n_queries {
    return false;
  }
  for (opening, expected_j) in proof.column_openings.iter().zip(queries.iter()) {
    if opening.col_idx != *expected_j {
      return false;
    }
    if opening.col_elems.len() != n_rows {
      return false;
    }

    // Merkle inclusion
    let leaf = hash_column(&opening.col_elems);
    if !opening
      .proof
      .verify(leaf, commitment.root, commitment.n_enc_cols)
    {
      return false;
    }

    // Linear consistency: Σ_i eq_hi[i] · col_elems[i]  ==  t_enc[q]
    let lc = (0..n_rows).fold(GF2_128::zero(), |acc, i| {
      acc + eq_hi[i] * opening.col_elems[i]
    });
    if lc != t_enc[opening.col_idx] {
      return false;
    }
  }

  true
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn hash_column(col: &[GF2_128]) -> Hash {
  let mut h = blake3::Hasher::new();
  h.update(b"binip:pcs:col:");
  // GF2_128 is #[repr(C)] { lo: u64, hi: u64 } — on little-endian this is
  // exactly the 16-byte wire format.  A single bulk update lets blake3 use
  // its SIMD multi-chunk path without per-element function-call overhead.
  #[cfg(target_endian = "little")]
  {
    let bytes: &[u8] = unsafe {
      std::slice::from_raw_parts(
        col.as_ptr() as *const u8,
        col.len() * std::mem::size_of::<GF2_128>(),
      )
    };
    h.update(bytes);
  }
  #[cfg(not(target_endian = "little"))]
  {
    for elem in col {
      h.update(&elem.lo.to_le_bytes());
      h.update(&elem.hi.to_le_bytes());
    }
  }
  *h.finalize().as_bytes()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Batch Tensor PCS
// ═══════════════════════════════════════════════════════════════════════════════
//
// Commits multiple MLEs (all with the same `n_vars`) into a single Merkle tree.
// `batch_open` opens several (entry, point) queries at once, sharing one set of
// column queries — this amortises Merkle proof cost across all openings.
//
// ## Matrix layout
//
// Entry `e`, row `i` occupies global row `e * n_rows + i`.  Every global row is
// encoded with the same rate-1/4 systematic code, so encoded column `j` spans
// `n_entries * n_rows` elements.

// ─── batch types ─────────────────────────────────────────────────────────────

/// Batch PCS builder — accumulate MLEs then call [`BatchTensorPCS::commit`].
pub struct BatchTensorPCS {
  params: PcsParams,
  entries: Vec<Vec<GF2_128>>,
}

/// Merkle-root commitment for a batch of MLEs.
#[derive(Clone, Debug)]
pub struct BatchCommitment {
  pub root: Hash,
  pub n_vars: u32,
  pub n_enc_cols: usize,
  pub n_rows_per_entry: usize,
  pub n_entries: usize,
}

/// Internal prover state for answering batch opening queries.
pub struct BatchPcsState {
  pub commitment: BatchCommitment,
  params: PcsParams,
  /// `raw_rows[entry][row_within_entry]` — each inner vec has `k` elements.
  raw_rows: Vec<Vec<Vec<GF2_128>>>,
  /// All encoded rows in global order.  `encoded_rows[e*n_rows + i]` has `4k`
  /// elements.
  encoded_rows: Vec<Vec<GF2_128>>,
  tree: merkle::MerkleTree,
}

/// A single opening query within a batch.
#[derive(Clone, Debug)]
pub struct BatchOpenQuery {
  /// Which polynomial (entry index) to open.
  pub entry: usize,
  /// Evaluation point (length = `n_vars`).
  pub point: Vec<GF2_128>,
}

/// Per-query data inside a [`BatchOpenProof`].
#[derive(Clone, Debug)]
pub struct BatchQueryOpen {
  /// Mixed row: `t_raw[j] = Σ_i eq(r_hi, i) · entry_rows[i][j]`.
  pub t_raw: Vec<GF2_128>,
}

/// Batch opening proof with shared column queries.
#[derive(Clone, Debug)]
pub struct BatchOpenProof {
  /// One per query — the mixed row and claim data.
  pub query_opens: Vec<BatchQueryOpen>,
  /// Shared column openings (one set for all queries).
  pub column_openings: Vec<ColumnOpening>,
}

// ─── batch commit ────────────────────────────────────────────────────────────

impl BatchTensorPCS {
  /// Create a new batch PCS.  All added polynomials must have `2^n_vars`
  /// evaluations.
  pub fn new(params: PcsParams) -> Self {
    Self {
      params,
      entries: Vec::new(),
    }
  }

  /// Add an MLE by its `2^n_vars` evaluations in lex order.  Returns the
  /// entry index.
  pub fn add_poly(&mut self, evals: Vec<GF2_128>) -> usize {
    let expected = 1usize << self.params.n_vars;
    assert_eq!(evals.len(), expected, "evals length must be 2^n_vars");
    let idx = self.entries.len();
    self.entries.push(evals);
    idx
  }

  /// Commit all accumulated MLEs.  Absorbs the root into `transcript`.
  pub fn commit<T: Transcript>(&self, transcript: &mut T) -> (BatchCommitment, BatchPcsState) {
    assert!(!self.entries.is_empty(), "must add at least one polynomial");

    let n_entries = self.entries.len();
    let n_rows = self.params.n_rows();
    let k = self.params.n_raw_cols();
    let n_enc = self.params.n_enc_cols();

    // Split each entry into rows
    let raw_rows: Vec<Vec<Vec<GF2_128>>> = self
      .entries
      .iter()
      .map(|evals| {
        (0..n_rows)
          .map(|i| evals[i * k..(i + 1) * k].to_vec())
          .collect()
      })
      .collect();

    // Encode every global row
    let code = code::LinearCode::new(k);
    let encoded_rows: Vec<Vec<GF2_128>> = raw_rows
      .iter()
      .flat_map(|entry_rows| entry_rows.iter().map(|row| code.encode(row)))
      .collect();

    // Hash each column (spans n_entries * n_rows elements)
    let leaf_hashes: Vec<Hash> = (0..n_enc)
      .map(|j| {
        let col: Vec<GF2_128> = encoded_rows.iter().map(|row| row[j]).collect();
        hash_column(&col)
      })
      .collect();

    let tree = merkle::MerkleTree::build(&leaf_hashes);
    let root = tree.root();

    transcript.absorb_bytes(&root);

    let commitment = BatchCommitment {
      root,
      n_vars: self.params.n_vars,
      n_enc_cols: n_enc,
      n_rows_per_entry: n_rows,
      n_entries,
    };

    let state = BatchPcsState {
      commitment: commitment.clone(),
      params: self.params.clone(),
      raw_rows,
      encoded_rows,
      tree,
    };

    (commitment, state)
  }
}

// ─── batch open ──────────────────────────────────────────────────────────────

/// Open multiple (entry, point) queries against the same batch commitment.
///
/// Returns `(claims, proof)` where `claims[i]` is the evaluation of entry
/// `queries[i].entry` at `queries[i].point`.
///
/// Precondition: `transcript` is in the state left by
/// [`BatchTensorPCS::commit`].
pub fn batch_open<T: Transcript>(
  state: &BatchPcsState,
  queries: &[BatchOpenQuery],
  transcript: &mut T,
) -> (Vec<GF2_128>, BatchOpenProof) {
  let params = &state.params;
  let m_rows = params.m_rows();
  let n_rows = params.n_rows();
  let k = params.n_raw_cols();
  let n_enc = params.n_enc_cols();

  let mut claims = Vec::with_capacity(queries.len());
  let mut query_opens = Vec::with_capacity(queries.len());

  for q in queries {
    assert!(q.entry < state.raw_rows.len(), "entry index out of bounds");
    assert_eq!(q.point.len(), params.n_vars as usize);

    let eq_hi = eq::eq_evals(&q.point[..m_rows]);
    let eq_lo = eq::eq_evals(&q.point[m_rows..]);

    let entry_rows = &state.raw_rows[q.entry];

    // t_raw[j] = Σ_i eq_hi[i] · entry_rows[i][j]
    let t_raw: Vec<GF2_128> = (0..k)
      .map(|j| (0..n_rows).fold(GF2_128::zero(), |acc, i| acc + eq_hi[i] * entry_rows[i][j]))
      .collect();

    let claim = eq::inner_product(&eq_lo, &t_raw);
    claims.push(claim);

    // Absorb this query's t_raw
    for &v in &t_raw {
      transcript.absorb_field(v);
    }

    query_opens.push(BatchQueryOpen { t_raw });
  }

  // Squeeze one shared set of column queries
  let col_queries: Vec<usize> = (0..params.n_queries)
    .map(|_| (transcript.squeeze_challenge().lo as usize) % n_enc)
    .collect();

  // Build shared column openings (columns span all entries)
  let column_openings = col_queries
    .into_iter()
    .map(|j| {
      let col_elems: Vec<GF2_128> = state.encoded_rows.iter().map(|row| row[j]).collect();
      let proof = state.tree.prove(j);
      ColumnOpening {
        col_idx: j,
        col_elems,
        proof,
      }
    })
    .collect();

  (
    claims,
    BatchOpenProof {
      query_opens,
      column_openings,
    },
  )
}

// ─── batch verify ────────────────────────────────────────────────────────────

/// Verify a batch opening proof.
///
/// Precondition: `transcript` is in the state produced by re-absorbing the
/// batch commitment root.
pub fn batch_verify<T: Transcript>(
  commitment: &BatchCommitment,
  queries: &[BatchOpenQuery],
  claims: &[GF2_128],
  proof: &BatchOpenProof,
  params: &PcsParams,
  transcript: &mut T,
) -> bool {
  if queries.len() != claims.len() {
    return false;
  }
  if queries.len() != proof.query_opens.len() {
    return false;
  }

  let m_rows = params.m_rows();
  let k = params.n_raw_cols();
  let n_enc = params.n_enc_cols();
  let n_rows = params.n_rows();
  let total_rows = commitment.n_entries * n_rows;

  // 1. Validate each query's inner-product claim and absorb t_raw.
  let mut eq_his: Vec<Vec<GF2_128>> = Vec::with_capacity(queries.len());

  for (i, q) in queries.iter().enumerate() {
    if q.point.len() != params.n_vars as usize {
      return false;
    }
    if q.entry >= commitment.n_entries {
      return false;
    }

    let eq_lo = eq::eq_evals(&q.point[m_rows..]);
    let eq_hi = eq::eq_evals(&q.point[..m_rows]);

    let t_raw = &proof.query_opens[i].t_raw;
    if t_raw.len() != k {
      return false;
    }

    if eq::inner_product(&eq_lo, t_raw) != claims[i] {
      return false;
    }

    for &v in t_raw {
      transcript.absorb_field(v);
    }

    eq_his.push(eq_hi);
  }

  // 2. Re-squeeze shared column queries.
  let col_queries: Vec<usize> = (0..params.n_queries)
    .map(|_| (transcript.squeeze_challenge().lo as usize) % n_enc)
    .collect();

  // 3. Encode each query's t_raw.
  let code = code::LinearCode::new(k);
  let t_encs: Vec<Vec<GF2_128>> = proof
    .query_opens
    .iter()
    .map(|qo| code.encode(&qo.t_raw))
    .collect();

  // 4. Check shared column openings.
  if proof.column_openings.len() != params.n_queries {
    return false;
  }

  for (opening, &expected_j) in proof.column_openings.iter().zip(col_queries.iter()) {
    if opening.col_idx != expected_j {
      return false;
    }
    if opening.col_elems.len() != total_rows {
      return false;
    }

    // Merkle inclusion
    let leaf = hash_column(&opening.col_elems);
    if !opening
      .proof
      .verify(leaf, commitment.root, commitment.n_enc_cols)
    {
      return false;
    }

    // Linear consistency for every query
    for (qi, q) in queries.iter().enumerate() {
      let row_off = q.entry * n_rows;
      let lc = (0..n_rows).fold(GF2_128::zero(), |acc, i| {
        acc + eq_his[qi][i] * opening.col_elems[row_off + i]
      });
      if lc != t_encs[qi][opening.col_idx] {
        return false;
      }
    }
  }

  true
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use super::*;
  use field::FieldElem;
  use transcript::VisionTranscript;

  fn fixed_evals(n_vars: u32) -> Vec<GF2_128> {
    // Deterministic non-trivial evaluations
    (0u64..1u64 << n_vars)
      .map(|i| GF2_128::from(i.wrapping_mul(6364136223846793005).wrapping_add(1)))
      .collect()
  }

  fn fixed_point(n_vars: u32, seed: u64) -> Vec<GF2_128> {
    (0..n_vars as u64)
      .map(|i| GF2_128::from(seed.wrapping_add(i + 1).wrapping_mul(2654435761)))
      .collect()
  }

  /// Direct MLE evaluation (for comparison).
  fn mle_eval(evals: &[GF2_128], r: &[GF2_128]) -> GF2_128 {
    let mut table = evals.to_vec();
    for &ri in r {
      let one_minus_r = GF2_128::one() + ri;
      let half = table.len() / 2;
      for i in 0..half {
        table[i] = table[i] * one_minus_r + table[i + half] * ri;
      }
      table.truncate(half);
    }
    table[0]
  }

  #[test]
  fn commit_open_verify_n4() {
    let n_vars = 4u32;
    let params = PcsParams {
      n_vars,
      n_queries: 8,
    };
    let evals = fixed_evals(n_vars);
    let r = fixed_point(n_vars, 42);

    let mut tp = VisionTranscript::new();
    let (commitment, state) = commit(&evals, &params, &mut tp);
    let (claim, proof) = open(&state, &r, &mut tp);

    // claim must match direct MLE evaluation
    let direct = mle_eval(&evals, &r);
    assert_eq!(claim, direct, "claim must equal direct MLE evaluation");

    // mirror prover transcript for verifier
    let mut tv = VisionTranscript::new();
    tv.absorb_bytes(&commitment.root);
    assert!(verify(&commitment, &r, claim, &proof, &params, &mut tv));
  }

  #[test]
  fn commit_open_verify_n6() {
    let n_vars = 6u32;
    let params = PcsParams {
      n_vars,
      n_queries: 10,
    };
    let evals = fixed_evals(n_vars);
    let r = fixed_point(n_vars, 137);

    let mut tp = VisionTranscript::new();
    let (commitment, state) = commit(&evals, &params, &mut tp);
    let (claim, proof) = open(&state, &r, &mut tp);

    let direct = mle_eval(&evals, &r);
    assert_eq!(claim, direct);

    let mut tv = VisionTranscript::new();
    tv.absorb_bytes(&commitment.root);
    assert!(verify(&commitment, &r, claim, &proof, &params, &mut tv));
  }

  #[test]
  fn verify_rejects_wrong_claim() {
    let n_vars = 4u32;
    let params = PcsParams {
      n_vars,
      n_queries: 8,
    };
    let evals = fixed_evals(n_vars);
    let r = fixed_point(n_vars, 99);

    let mut tp = VisionTranscript::new();
    let (commitment, state) = commit(&evals, &params, &mut tp);
    let (claim, proof) = open(&state, &r, &mut tp);

    let wrong_claim = claim + GF2_128::one(); // flip a bit

    let mut tv = VisionTranscript::new();
    tv.absorb_bytes(&commitment.root);
    assert!(!verify(
      &commitment,
      &r,
      wrong_claim,
      &proof,
      &params,
      &mut tv
    ));
  }

  #[test]
  fn verify_rejects_tampered_t_raw() {
    let n_vars = 4u32;
    let params = PcsParams {
      n_vars,
      n_queries: 8,
    };
    let evals = fixed_evals(n_vars);
    let r = fixed_point(n_vars, 7);

    let mut tp = VisionTranscript::new();
    let (commitment, state) = commit(&evals, &params, &mut tp);
    let (claim, mut proof) = open(&state, &r, &mut tp);

    // Tamper with t_raw[0]
    proof.t_raw[0] = proof.t_raw[0] + GF2_128::one();

    let mut tv = VisionTranscript::new();
    tv.absorb_bytes(&commitment.root);
    assert!(!verify(&commitment, &r, claim, &proof, &params, &mut tv));
  }

  #[test]
  fn verify_rejects_tampered_column() {
    let n_vars = 4u32;
    let params = PcsParams {
      n_vars,
      n_queries: 8,
    };
    let evals = fixed_evals(n_vars);
    let r = fixed_point(n_vars, 23);

    let mut tp = VisionTranscript::new();
    let (commitment, state) = commit(&evals, &params, &mut tp);
    let (claim, mut proof) = open(&state, &r, &mut tp);

    // Tamper with the first revealed column
    if let Some(col) = proof.column_openings.first_mut() {
      if !col.col_elems.is_empty() {
        col.col_elems[0] = col.col_elems[0] + GF2_128::one();
      }
    }

    let mut tv = VisionTranscript::new();
    tv.absorb_bytes(&commitment.root);
    assert!(!verify(&commitment, &r, claim, &proof, &params, &mut tv));
  }

  // ── batch PCS tests ──────────────────────────────────────────────────

  #[test]
  fn batch_single_entry_roundtrip() {
    let n_vars = 4u32;
    let params = PcsParams {
      n_vars,
      n_queries: 8,
    };
    let evals = fixed_evals(n_vars);
    let r = fixed_point(n_vars, 42);

    let mut pcs = BatchTensorPCS::new(params.clone());
    pcs.add_poly(evals.clone());

    let mut tp = VisionTranscript::new();
    let (bc, state) = pcs.commit(&mut tp);

    let queries = vec![BatchOpenQuery {
      entry: 0,
      point: r.clone(),
    }];
    let (claims, proof) = batch_open(&state, &queries, &mut tp);

    let direct = mle_eval(&evals, &r);
    assert_eq!(claims[0], direct);

    let mut tv = VisionTranscript::new();
    tv.absorb_bytes(&bc.root);
    assert!(batch_verify(
      &bc, &queries, &claims, &proof, &params, &mut tv
    ));
  }

  #[test]
  fn batch_multi_entry_roundtrip() {
    let n_vars = 4u32;
    let params = PcsParams {
      n_vars,
      n_queries: 10,
    };

    let evals_a = fixed_evals(n_vars);
    let evals_b: Vec<GF2_128> = (0u64..1u64 << n_vars)
      .map(|i| GF2_128::from(i.wrapping_mul(1442695040888963407).wrapping_add(3)))
      .collect();
    let evals_c: Vec<GF2_128> = (0u64..1u64 << n_vars)
      .map(|i| GF2_128::from(i ^ 0xCAFE))
      .collect();

    let mut pcs = BatchTensorPCS::new(params.clone());
    pcs.add_poly(evals_a.clone());
    pcs.add_poly(evals_b.clone());
    pcs.add_poly(evals_c.clone());

    let mut tp = VisionTranscript::new();
    let (bc, state) = pcs.commit(&mut tp);

    let r_a = fixed_point(n_vars, 10);
    let r_b = fixed_point(n_vars, 20);
    let r_c = fixed_point(n_vars, 30);

    let queries = vec![
      BatchOpenQuery {
        entry: 0,
        point: r_a.clone(),
      },
      BatchOpenQuery {
        entry: 1,
        point: r_b.clone(),
      },
      BatchOpenQuery {
        entry: 2,
        point: r_c.clone(),
      },
    ];
    let (claims, proof) = batch_open(&state, &queries, &mut tp);

    assert_eq!(claims[0], mle_eval(&evals_a, &r_a));
    assert_eq!(claims[1], mle_eval(&evals_b, &r_b));
    assert_eq!(claims[2], mle_eval(&evals_c, &r_c));

    let mut tv = VisionTranscript::new();
    tv.absorb_bytes(&bc.root);
    assert!(batch_verify(
      &bc, &queries, &claims, &proof, &params, &mut tv
    ));
  }

  #[test]
  fn batch_same_entry_two_points() {
    let n_vars = 4u32;
    let params = PcsParams {
      n_vars,
      n_queries: 8,
    };
    let evals = fixed_evals(n_vars);

    let mut pcs = BatchTensorPCS::new(params.clone());
    pcs.add_poly(evals.clone());

    let mut tp = VisionTranscript::new();
    let (bc, state) = pcs.commit(&mut tp);

    let r1 = fixed_point(n_vars, 77);
    let r2 = fixed_point(n_vars, 88);

    let queries = vec![
      BatchOpenQuery {
        entry: 0,
        point: r1.clone(),
      },
      BatchOpenQuery {
        entry: 0,
        point: r2.clone(),
      },
    ];
    let (claims, proof) = batch_open(&state, &queries, &mut tp);

    assert_eq!(claims[0], mle_eval(&evals, &r1));
    assert_eq!(claims[1], mle_eval(&evals, &r2));

    let mut tv = VisionTranscript::new();
    tv.absorb_bytes(&bc.root);
    assert!(batch_verify(
      &bc, &queries, &claims, &proof, &params, &mut tv
    ));
  }

  #[test]
  fn batch_rejects_wrong_claim() {
    let n_vars = 4u32;
    let params = PcsParams {
      n_vars,
      n_queries: 8,
    };

    let mut pcs = BatchTensorPCS::new(params.clone());
    pcs.add_poly(fixed_evals(n_vars));
    pcs.add_poly(fixed_evals(n_vars)); // duplicate is fine

    let mut tp = VisionTranscript::new();
    let (bc, state) = pcs.commit(&mut tp);

    let queries = vec![
      BatchOpenQuery {
        entry: 0,
        point: fixed_point(n_vars, 1),
      },
      BatchOpenQuery {
        entry: 1,
        point: fixed_point(n_vars, 2),
      },
    ];
    let (mut claims, proof) = batch_open(&state, &queries, &mut tp);
    claims[1] = claims[1] + GF2_128::one();

    let mut tv = VisionTranscript::new();
    tv.absorb_bytes(&bc.root);
    assert!(!batch_verify(
      &bc, &queries, &claims, &proof, &params, &mut tv
    ));
  }

  #[test]
  fn batch_rejects_tampered_column() {
    let n_vars = 4u32;
    let params = PcsParams {
      n_vars,
      n_queries: 8,
    };

    let mut pcs = BatchTensorPCS::new(params.clone());
    pcs.add_poly(fixed_evals(n_vars));

    let mut tp = VisionTranscript::new();
    let (bc, state) = pcs.commit(&mut tp);

    let queries = vec![BatchOpenQuery {
      entry: 0,
      point: fixed_point(n_vars, 55),
    }];
    let (claims, mut proof) = batch_open(&state, &queries, &mut tp);

    if let Some(col) = proof.column_openings.first_mut() {
      if !col.col_elems.is_empty() {
        col.col_elems[0] = col.col_elems[0] + GF2_128::one();
      }
    }

    let mut tv = VisionTranscript::new();
    tv.absorb_bytes(&bc.root);
    assert!(!batch_verify(
      &bc, &queries, &claims, &proof, &params, &mut tv
    ));
  }

  #[test]
  fn batch_rejects_tampered_t_raw() {
    let n_vars = 4u32;
    let params = PcsParams {
      n_vars,
      n_queries: 8,
    };

    let mut pcs = BatchTensorPCS::new(params.clone());
    pcs.add_poly(fixed_evals(n_vars));
    pcs.add_poly(fixed_evals(n_vars));

    let mut tp = VisionTranscript::new();
    let (bc, state) = pcs.commit(&mut tp);

    let queries = vec![
      BatchOpenQuery {
        entry: 0,
        point: fixed_point(n_vars, 3),
      },
      BatchOpenQuery {
        entry: 1,
        point: fixed_point(n_vars, 5),
      },
    ];
    let (claims, mut proof) = batch_open(&state, &queries, &mut tp);

    // Tamper with second query's t_raw
    proof.query_opens[1].t_raw[0] = proof.query_opens[1].t_raw[0] + GF2_128::one();

    let mut tv = VisionTranscript::new();
    tv.absorb_bytes(&bc.root);
    assert!(!batch_verify(
      &bc, &queries, &claims, &proof, &params, &mut tv
    ));
  }

  #[test]
  fn batch_n6_five_entries() {
    let n_vars = 6u32;
    let params = PcsParams {
      n_vars,
      n_queries: 12,
    };

    let mut pcs = BatchTensorPCS::new(params.clone());
    let all_evals: Vec<Vec<GF2_128>> = (0..5u64)
      .map(|seed| {
        (0u64..1u64 << n_vars)
          .map(|i| {
            GF2_128::from(i.wrapping_mul(seed.wrapping_add(1).wrapping_mul(6364136223846793005)))
          })
          .collect()
      })
      .collect();

    for e in &all_evals {
      pcs.add_poly(e.clone());
    }

    let mut tp = VisionTranscript::new();
    let (bc, state) = pcs.commit(&mut tp);

    let queries: Vec<BatchOpenQuery> = (0..5)
      .map(|i| BatchOpenQuery {
        entry: i,
        point: fixed_point(n_vars, (i as u64 + 1) * 100),
      })
      .collect();
    let (claims, proof) = batch_open(&state, &queries, &mut tp);

    for (i, q) in queries.iter().enumerate() {
      assert_eq!(claims[i], mle_eval(&all_evals[q.entry], &q.point));
    }

    let mut tv = VisionTranscript::new();
    tv.absorb_bytes(&bc.root);
    assert!(batch_verify(
      &bc, &queries, &claims, &proof, &params, &mut tv
    ));
  }
}
