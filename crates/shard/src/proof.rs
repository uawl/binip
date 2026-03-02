//! Shard proof types.

use field::GF2_128;
use sumcheck::SumcheckProof;

/// Proof for a single shard — a sumcheck proof over the shard's sub-MLE.
///
/// The shard prover runs a standard sumcheck on the `shard_vars`-variable
/// sub-MLE using a forked transcript `fork("shard", shard_idx)`.
#[derive(Debug, Clone)]
pub struct ShardProof {
  /// Which shard this proof covers (0-indexed).
  pub shard_idx: u32,
  /// The shard's sumcheck proof.
  pub sumcheck: SumcheckProof,
}

/// Aggregated result of proving all shards.
#[derive(Debug, Clone)]
pub struct ShardProofBatch {
  /// Per-shard proofs, ordered by `shard_idx`.
  pub shard_proofs: Vec<ShardProof>,
  /// The sum of all shard claimed sums (should equal the full MLE sum).
  pub total_sum: GF2_128,
}
