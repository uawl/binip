//! Shard proof types.

use field::GF2_128;
use sumcheck::SumcheckProof;

/// A 32-byte Blake3 hash used for shard boundary commitments.
pub type Hash = [u8; 32];

/// Hash a shard's evaluations with domain separation.
///
/// Produces `Blake3("binip:shard:evals:" ‖ shard_idx_le ‖ n_evals_le ‖ evals_bytes)`.
pub fn hash_shard_evals(shard_idx: u32, evals: &[GF2_128]) -> Hash {
  let mut h = blake3::Hasher::new();
  h.update(b"binip:shard:evals:");
  h.update(&shard_idx.to_le_bytes());
  h.update(&(evals.len() as u32).to_le_bytes());
  for e in evals {
    h.update(&e.lo.to_le_bytes());
    h.update(&e.hi.to_le_bytes());
  }
  *h.finalize().as_bytes()
}

/// Hash all shard commitments into a single partition root.
///
/// Produces `Blake3("binip:shard:partition:" ‖ count_le ‖ commitment_0 ‖ … ‖ commitment_{n-1})`.
pub fn hash_partition(shard_commitments: &[Hash]) -> Hash {
  let mut h = blake3::Hasher::new();
  h.update(b"binip:shard:partition:");
  h.update(&(shard_commitments.len() as u32).to_le_bytes());
  for c in shard_commitments {
    h.update(c);
  }
  *h.finalize().as_bytes()
}

/// Proof for a single shard — a sumcheck proof over the shard's sub-MLE.
///
/// The shard prover runs a standard sumcheck on the `shard_vars`-variable
/// sub-MLE using a forked transcript `fork("shard", shard_idx)`.
#[derive(Debug, Clone, bincode::Encode, bincode::Decode)]
pub struct ShardProof {
  /// Which shard this proof covers (0-indexed).
  pub shard_idx: u32,
  /// Blake3 commitment to the shard's evaluations (boundary binding).
  pub shard_commitment: Hash,
  /// The shard's sumcheck proof.
  pub sumcheck: SumcheckProof,
}

/// Aggregated result of proving all shards.
#[derive(Debug, Clone, bincode::Encode, bincode::Decode)]
pub struct ShardProofBatch {
  /// Per-shard proofs, ordered by `shard_idx`.
  pub shard_proofs: Vec<ShardProof>,
  /// The sum of all shard claimed sums (should equal the full MLE sum).
  pub total_sum: GF2_128,
  /// Blake3 hash over all shard commitments, binding the partition order.
  pub partition_root: Hash,
}
