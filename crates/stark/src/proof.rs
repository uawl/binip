//! Top-level proof types for the GF(2) ZK-STARK system.
//!
//! The [`Proof`] struct bundles every artefact the verifier needs:
//!
//! | Component           | Purpose                                  |
//! |---------------------|------------------------------------------|
//! | `type_cert`         | Structural type-derivation certificate   |
//! | `batch_commit`      | PCS Merkle root over all witness MLEs    |
//! | `constraint_sum`    | Claimed sum of the constraint MLE (== 0) |
//! | `shard_batch`       | Shard-level sumcheck proofs              |
//! | `recursive_proof`   | Recursive aggregation proof              |
//! | `pcs_open`          | PCS opening proof at the challenge point |

use evm_types::TypeCert;
use field::GF2_128;
use pcs::{Commitment, OpenProof};
use recursive::RecursiveProof;
use shard::{RecursiveConfig, ShardProofBatch};

/// Top-level ZK-STARK proof.
#[derive(Debug, Clone)]
pub struct Proof {
  // ── structural layer ──────────────────────────────────────────────────
  /// Type-derivation certificate (Proof Tree shape commitment).
  pub type_cert: TypeCert,

  // ── PCS layer ─────────────────────────────────────────────────────────
  /// PCS commitment over the constraint MLE evaluations.
  pub batch_commit: Commitment,

  // ── sumcheck layer ────────────────────────────────────────────────────
  /// Batching challenge used to combine per-row constraints.
  pub beta: GF2_128,
  /// Claimed sum of the constraint MLE (must be zero for a valid proof).
  pub constraint_sum: GF2_128,

  // ── shard layer ───────────────────────────────────────────────────────
  /// Shard-level proofs (independent sumcheck per shard).
  pub shard_batch: ShardProofBatch,

  // ── recursive layer ───────────────────────────────────────────────────
  /// Recursive aggregation proof (levels of fan-in sumcheck proofs).
  pub recursive_proof: RecursiveProof,

  // ── PCS opening ───────────────────────────────────────────────────────
  /// PCS opening at the evaluation point derived from sumcheck challenges.
  pub pcs_open: OpenProof,
  /// The evaluation point `r` used for PCS opening (= sumcheck challenges).
  pub open_point: Vec<GF2_128>,
  /// Claimed evaluation of the constraint MLE at `open_point`.
  pub open_eval: GF2_128,

  // ── config ────────────────────────────────────────────────────────────
  /// The recursion configuration used during proving.
  pub config: RecursiveConfig,
}

/// Parameters for proof generation and verification.
#[derive(Debug, Clone)]
pub struct StarkParams {
  /// Recursion / sharding configuration.
  pub config: RecursiveConfig,
  /// PCS parameters (n_vars, n_queries).
  pub pcs_params: pcs::PcsParams,
}

impl StarkParams {
  /// Sensible defaults for a given trace size.
  ///
  /// `n_vars` is `ceil(log2(n_rows))` — the number of MLE variables
  /// needed to represent the trace.
  pub fn for_n_vars(n_vars: u32) -> Self {
    let config = RecursiveConfig::for_n_vars(n_vars);
    Self {
      config,
      pcs_params: pcs::PcsParams {
        n_vars,
        n_queries: 40,
      },
    }
  }
}
