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
//! **Open at r = (r_lo ‖ r_hi)**  (`r_lo = r[0..m_cols]`, `r_hi = r[m_cols..]`)
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

use field::{gf2_128::GF2_128, FieldElem};
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
    fn m_cols(&self) -> usize { (self.n_vars / 2) as usize }
    /// Number of "row" variables (high bits of eval index).
    fn m_rows(&self) -> usize { (self.n_vars as usize) - self.m_cols() }
    /// Raw columns per row: 2^m_cols.
    fn n_raw_cols(&self) -> usize { 1 << self.m_cols() }
    /// Number of rows: 2^m_rows.
    fn n_rows(&self) -> usize { 1 << self.m_rows() }
    /// Encoded columns per row: 4 * 2^m_cols.
    fn n_enc_cols(&self) -> usize { 4 * self.n_raw_cols() }
}

// ─── public types ────────────────────────────────────────────────────────────

/// The Merkle-root commitment returned by `commit`.
#[derive(Clone, Debug)]
pub struct Commitment {
    pub root:      Hash,
    pub n_vars:    u32,
    /// Precomputed for the verifier's benefit.
    pub n_enc_cols: usize,
    pub n_rows:    usize,
}

/// Internal prover state needed to answer opening queries.
pub struct PcsState {
    pub commitment: Commitment,
    params:         PcsParams,
    raw_rows:       Vec<Vec<GF2_128>>,      // [n_rows][k]
    encoded_rows:   Vec<Vec<GF2_128>>,      // [n_rows][4k]
    tree:           merkle::MerkleTree,
}

/// A single column revealed during an opening.
#[derive(Clone, Debug)]
pub struct ColumnOpening {
    pub col_idx:   usize,
    /// `encoded_rows[0..n_rows][col_idx]`.
    pub col_elems: Vec<GF2_128>,
    pub proof:     MerkleProof,
}

/// The full opening proof produced by `open`.
#[derive(Clone, Debug)]
pub struct OpenProof {
    /// Mixed row: `t_raw[j] = Σ_i eq(r_hi, i) · M[i][j]`.  Length = k.
    pub t_raw:           Vec<GF2_128>,
    pub column_openings: Vec<ColumnOpening>,
}

// ─── commit ───────────────────────────────────────────────────────────────────

/// Commit to an n-variable MLE given as its `2^n` evaluations in lex order.
///
/// Absorbs the commitment root into `transcript`.
pub fn commit<T: Transcript>(
    evals:      &[GF2_128],
    params:     &PcsParams,
    transcript: &mut T,
) -> (Commitment, PcsState) {
    let n_rows = params.n_rows();
    let k      = params.n_raw_cols();
    let n_enc  = params.n_enc_cols();
    assert_eq!(evals.len(), n_rows * k, "evals.len() must equal 2^n_vars");

    // Split into rows: row i = evals[i*k .. (i+1)*k]
    let raw_rows: Vec<Vec<GF2_128>> = (0..n_rows)
        .map(|i| evals[i * k..(i + 1) * k].to_vec())
        .collect();

    // Encode every row
    let code = code::LinearCode::new(k);
    let encoded_rows: Vec<Vec<GF2_128>> = raw_rows.iter()
        .map(|row| code.encode(row))
        .collect();

    // Hash each column to form Merkle leaves
    let leaf_hashes: Vec<Hash> = (0..n_enc)
        .map(|j| {
            let col: Vec<GF2_128> = encoded_rows.iter().map(|row| row[j]).collect();
            hash_column(&col)
        })
        .collect();

    let tree = merkle::MerkleTree::build(&leaf_hashes);
    let root = tree.root();

    transcript.absorb_bytes(&root);

    let commitment = Commitment {
        root,
        n_vars:     params.n_vars,
        n_enc_cols: n_enc,
        n_rows,
    };

    let state = PcsState {
        commitment: commitment.clone(),
        params:     params.clone(),
        raw_rows,
        encoded_rows,
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
    state:      &PcsState,
    r:          &[GF2_128],
    transcript: &mut T,
) -> (GF2_128, OpenProof) {
    let params  = &state.params;
    let m_rows  = params.m_rows();
    let n_rows  = params.n_rows();
    let k       = params.n_raw_cols();
    let n_enc   = params.n_enc_cols();
    assert_eq!(r.len(), params.n_vars as usize);

    // r[0..m_rows] = challenges for high bits (row variables).
    // r[m_rows..]  = challenges for low bits (col variables).
    // eq_evals(v)[i]: v[0] = highest bit of i, matching mle_eval convention.
    let eq_hi = eq::eq_evals(&r[..m_rows]);
    let eq_lo = eq::eq_evals(&r[m_rows..]);

    // t_raw[j] = Σ_i eq_hi[i] · M[i][j]
    let t_raw: Vec<GF2_128> = (0..k)
        .map(|j| {
            (0..n_rows).fold(GF2_128::zero(), |acc, i| {
                acc + eq_hi[i] * state.raw_rows[i][j]
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
        .map(|_| (transcript.squeeze_challenge().lo.0 as usize) % n_enc)
        .collect();

    // Build column openings
    let column_openings = queries
        .into_iter()
        .map(|j| {
            let col_elems: Vec<GF2_128> = state.encoded_rows.iter().map(|row| row[j]).collect();
            let proof = state.tree.prove(j);
            ColumnOpening { col_idx: j, col_elems, proof }
        })
        .collect();

    (claim, OpenProof { t_raw, column_openings })
}

// ─── verify ───────────────────────────────────────────────────────────────────

/// Verify an opening proof.
///
/// Precondition: `transcript` is in the state produced by re-absorbing the
/// commitment root (i.e., `transcript.absorb_bytes(&commitment.root)`).
pub fn verify<T: Transcript>(
    commitment: &Commitment,
    r:          &[GF2_128],
    claim:      GF2_128,
    proof:      &OpenProof,
    params:     &PcsParams,
    transcript: &mut T,
) -> bool {
    let m_rows = params.m_rows();
    let k      = params.n_raw_cols();
    let n_enc  = params.n_enc_cols();
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
        .map(|_| (transcript.squeeze_challenge().lo.0 as usize) % n_enc)
        .collect();

    // 4. Encode t_raw (same code as prover)
    let code  = code::LinearCode::new(k);
    let t_enc = code.encode(&proof.t_raw);

    // 5. Check each column opening
    if proof.column_openings.len() != params.n_queries {
        return false;
    }
    for (opening, expected_j) in proof.column_openings.iter().zip(queries.iter()) {
        if opening.col_idx != *expected_j { return false; }
        if opening.col_elems.len() != n_rows { return false; }

        // Merkle inclusion
        let leaf = hash_column(&opening.col_elems);
        if !opening.proof.verify(leaf, commitment.root, commitment.n_enc_cols) {
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
    for elem in col {
        h.update(&elem.lo.0.to_le_bytes());
        h.update(&elem.hi.0.to_le_bytes());
    }
    *h.finalize().as_bytes()
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
        let params = PcsParams { n_vars, n_queries: 8 };
        let evals  = fixed_evals(n_vars);
        let r      = fixed_point(n_vars, 42);

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
        let params = PcsParams { n_vars, n_queries: 10 };
        let evals  = fixed_evals(n_vars);
        let r      = fixed_point(n_vars, 137);

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
        let params = PcsParams { n_vars, n_queries: 8 };
        let evals  = fixed_evals(n_vars);
        let r      = fixed_point(n_vars, 99);

        let mut tp = VisionTranscript::new();
        let (commitment, state) = commit(&evals, &params, &mut tp);
        let (claim, proof) = open(&state, &r, &mut tp);

        let wrong_claim = claim + GF2_128::one(); // flip a bit

        let mut tv = VisionTranscript::new();
        tv.absorb_bytes(&commitment.root);
        assert!(!verify(&commitment, &r, wrong_claim, &proof, &params, &mut tv));
    }

    #[test]
    fn verify_rejects_tampered_t_raw() {
        let n_vars = 4u32;
        let params = PcsParams { n_vars, n_queries: 8 };
        let evals  = fixed_evals(n_vars);
        let r      = fixed_point(n_vars, 7);

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
        let params = PcsParams { n_vars, n_queries: 8 };
        let evals  = fixed_evals(n_vars);
        let r      = fixed_point(n_vars, 23);

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

}