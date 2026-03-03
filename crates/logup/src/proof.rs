//! LogUp proof types.

use field::GF2_128;

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
  /// PCS commitment to the witness-side fractional MLE h_w.
  /// Present only in the committed proof variant.
  pub h_w_commit: Option<pcs::Commitment>,

  /// PCS commitment to the table-side fractional MLE h_t.
  pub h_t_commit: Option<pcs::Commitment>,

  /// PCS opening proof for h_w at the witness sumcheck challenge point.
  /// `(evaluation, opening_proof)` — evaluation must equal `witness_sumcheck.final_eval`.
  pub h_w_opening: Option<(GF2_128, pcs::OpenProof)>,

  /// PCS opening proof for h_t at the table sumcheck challenge point.
  /// `(evaluation, opening_proof)` — evaluation must equal `table_sumcheck.final_eval`.
  pub h_t_opening: Option<(GF2_128, pcs::OpenProof)>,
}
