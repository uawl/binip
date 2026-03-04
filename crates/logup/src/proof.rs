//! LogUp proof types.

use field::GF2_128;

/// PCS evaluation binding for committed logup proofs.
///
/// Depending on whether h_w and h_t share the same n_vars, the prover
/// chooses between a shared [`Batch`] commit (shared Merkle tree + column
/// queries) and separate [`Individual`] commits.
#[derive(Debug, Clone)]
pub enum PcsBinding {
  /// h_w and h_t committed separately (different n_vars).
  Individual {
    h_w_commit: pcs::Commitment,
    h_t_commit: pcs::Commitment,
    h_w_opening: (GF2_128, pcs::OpenProof),
    h_t_opening: (GF2_128, pcs::OpenProof),
  },
  /// h_w and h_t batched into a single commitment (same n_vars).
  /// Entry 0 = h_w, entry 1 = h_t.
  Batch {
    n_vars: u32,
    commit: pcs::BatchCommitment,
    open_proof: pcs::BatchOpenProof,
  },
}

/// A single LogUp relation proof: witness values ⊂ table entries.
///
/// The γ-weighted LogUp protocol proves
///   `Σ_j γ^j/(β + w_j) = Σ_i M_i/(β + t_i)`
/// where `M_i = Σ_{j: w_j=t_i} γ^j` is the γ-weighted multiplicity.
///
/// The γ-weighting prevents characteristic-2 cancellation: even when
/// `w_j = w_k`, the terms `γ^j/(β+w_j)` and `γ^k/(β+w_k)` are distinct.
#[derive(Debug, Clone)]
pub struct LogUpProof {
  /// Random challenge β used to form the logarithmic derivatives.
  pub beta: GF2_128,

  /// Random challenge γ used to weight each witness lookup.
  /// Prevents char-2 cancellation for repeated lookups.
  pub gamma: GF2_128,

  /// Claimed sum: `Σ h_w(x) = Σ h_t(x)`.
  pub claimed_sum: GF2_128,

  // ── Witness-side sumcheck ─────────────────────────────────────────────
  /// Sumcheck proof for `Σ_{x ∈ {0,1}^n} h_w(x)`.
  /// Here `h_w[j] = γ^j / (β + w_j)`.
  pub witness_sumcheck: sumcheck::SumcheckProof,

  // ── Table-side sumcheck ───────────────────────────────────────────────
  /// Sumcheck proof for `Σ_{x ∈ {0,1}^t} h_t(x)`.
  /// Here `h_t[i] = M_i / (β + t_i)`.
  pub table_sumcheck: sumcheck::SumcheckProof,

  // ── PCS evaluation binding (committed path only) ──────────────────────
  /// Present only in the committed proof variant.
  pub pcs_binding: Option<PcsBinding>,
}
