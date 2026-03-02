//! Shard prover — splits an MLE into sub-MLEs and proves each independently.
//!
//! # Sharding scheme
//!
//! A `total_vars`-variable MLE has `2^total_vars` evaluations stored in a
//! flat vector.  We partition it into `n_shards = 2^(total_vars - shard_vars)`
//! contiguous chunks of `2^shard_vars` elements each.
//!
//! Each shard `i` runs a standard sumcheck on its sub-MLE using a
//! *forked* Blake3 transcript:  `root_transcript.fork("shard", i)`.
//! This gives cryptographic independence between shards.
//!
//! The verifier checks each shard independently, then confirms:
//!   `Σ_i shard_claimed_sum_i == full_mle.sum()`

use field::{FieldElem, GF2_128};
use poly::MlePoly;
use transcript::Blake3Transcript;

use crate::config::RecursiveConfig;
use crate::proof::{ShardProof, ShardProofBatch};

/// Split an MLE into `n_shards` sub-MLEs.
///
/// Each sub-MLE has `shard_vars` variables and `2^shard_vars` evaluations.
///
/// # Panics
///
/// Panics if `poly.n_vars != config.total_vars`.
pub fn split_mle(poly: &MlePoly, config: &RecursiveConfig) -> Vec<MlePoly> {
  assert_eq!(
    poly.n_vars, config.total_vars,
    "MLE has {} vars but config.total_vars = {}",
    poly.n_vars, config.total_vars
  );

  let shard_size = 1usize << config.shard_vars;
  let n_shards = config.n_shards() as usize;

  (0..n_shards)
    .map(|i| {
      let start = i * shard_size;
      let chunk = poly.evals[start..start + shard_size].to_vec();
      MlePoly::new(chunk)
    })
    .collect()
}

/// Prove a single shard: run sumcheck on the sub-MLE with a forked transcript.
pub fn prove_shard(
  shard_idx: u32,
  sub_mle: &MlePoly,
  root_transcript: &Blake3Transcript,
) -> ShardProof {
  let mut t = root_transcript.fork("shard", shard_idx);
  let sumcheck = sumcheck::prove(sub_mle, &mut t);
  ShardProof { shard_idx, sumcheck }
}

/// Prove all shards: split the full MLE and prove each independently.
///
/// Returns a [`ShardProofBatch`] containing all shard proofs and
/// the total claimed sum.
pub fn prove_all(
  poly: &MlePoly,
  config: &RecursiveConfig,
  root_transcript: &Blake3Transcript,
) -> ShardProofBatch {
  let sub_mles = split_mle(poly, config);
  let n = sub_mles.len();

  let mut shard_proofs = Vec::with_capacity(n);
  let mut total_sum = GF2_128::zero();

  for (i, sub) in sub_mles.iter().enumerate() {
    let proof = prove_shard(i as u32, sub, root_transcript);
    total_sum = total_sum + proof.sumcheck.claimed_sum;
    shard_proofs.push(proof);
  }

  ShardProofBatch { shard_proofs, total_sum }
}

/// GPU-accelerated variant of [`prove_all`].
///
/// Dispatches all shards to the GPU in parallel using
/// [`sumcheck::prove_shards_gpu`].
pub fn prove_all_gpu(
  poly: &MlePoly,
  config: &RecursiveConfig,
  root_transcript: &Blake3Transcript,
  ctx: &gpu::GpuContext,
  cache: &mut gpu::PipelineCache,
) -> ShardProofBatch {
  let n_shards = config.n_shards() as usize;

  // Build per-shard forked transcripts
  let mut transcripts: Vec<_> = (0..n_shards)
    .map(|i| root_transcript.fork("shard", i as u32))
    .collect();

  let proofs = sumcheck::prove_shards_gpu(
    poly,
    &mut transcripts,
    config.shard_vars,
    ctx,
    cache,
  );

  let mut shard_proofs = Vec::with_capacity(n_shards);
  let mut total_sum = GF2_128::zero();

  for (i, sumcheck) in proofs.into_iter().enumerate() {
    total_sum = total_sum + sumcheck.claimed_sum;
    shard_proofs.push(ShardProof { shard_idx: i as u32, sumcheck });
  }

  ShardProofBatch { shard_proofs, total_sum }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn g(v: u64) -> GF2_128 {
    GF2_128::from(v)
  }

  fn test_config() -> RecursiveConfig {
    // 4-var MLE split into 4 shards of 2 vars each
    RecursiveConfig { total_vars: 4, shard_vars: 2, fan_in: 2 }
  }

  #[test]
  fn split_mle_correct_sizes() {
    let evals: Vec<GF2_128> = (1u64..=16).map(g).collect();
    let poly = MlePoly::new(evals);
    let cfg = test_config();
    let shards = split_mle(&poly, &cfg);
    assert_eq!(shards.len(), 4);
    for s in &shards {
      assert_eq!(s.n_vars, 2);
      assert_eq!(s.evals.len(), 4);
    }
  }

  #[test]
  fn split_preserves_evaluations() {
    let evals: Vec<GF2_128> = (1u64..=16).map(g).collect();
    let poly = MlePoly::new(evals.clone());
    let cfg = test_config();
    let shards = split_mle(&poly, &cfg);
    // Concatenating shards should reproduce the original
    let reassembled: Vec<GF2_128> = shards.iter().flat_map(|s| s.evals.iter().copied()).collect();
    assert_eq!(reassembled, evals);
  }

  #[test]
  fn shard_sums_equal_total() {
    let evals: Vec<GF2_128> = (1u64..=16).map(g).collect();
    let poly = MlePoly::new(evals);
    let cfg = test_config();
    let shards = split_mle(&poly, &cfg);
    let shard_sum: GF2_128 = shards.iter().map(|s| s.sum()).fold(GF2_128::zero(), |a, b| a + b);
    assert_eq!(shard_sum, poly.sum());
  }

  #[test]
  fn prove_shard_produces_valid_proof() {
    let evals: Vec<GF2_128> = (1u64..=4).map(g).collect();
    let sub = MlePoly::new(evals);
    let root_t = Blake3Transcript::new();
    let proof = prove_shard(0, &sub, &root_t);
    assert_eq!(proof.shard_idx, 0);
    assert_eq!(proof.sumcheck.claimed_sum, sub.sum());
    assert_eq!(proof.sumcheck.round_polys.len(), 2); // 2 vars
  }

  #[test]
  fn prove_all_total_sum_matches() {
    let evals: Vec<GF2_128> = (1u64..=16).map(g).collect();
    let poly = MlePoly::new(evals);
    let cfg = test_config();
    let root_t = Blake3Transcript::new();
    let batch = prove_all(&poly, &cfg, &root_t);
    assert_eq!(batch.shard_proofs.len(), 4);
    assert_eq!(batch.total_sum, poly.sum());
  }

  #[test]
  fn different_shards_have_different_proofs() {
    let evals: Vec<GF2_128> = (1u64..=16).map(g).collect();
    let poly = MlePoly::new(evals);
    let cfg = test_config();
    let root_t = Blake3Transcript::new();
    let batch = prove_all(&poly, &cfg, &root_t);
    // Shard 0 and Shard 1 should differ (different data, different transcripts)
    assert_ne!(
      batch.shard_proofs[0].sumcheck.claimed_sum,
      batch.shard_proofs[1].sumcheck.claimed_sum
    );
  }

  #[test]
  fn single_shard_is_full_proof() {
    // total_vars == shard_vars → 1 shard = full MLE
    let evals: Vec<GF2_128> = (1u64..=8).map(g).collect();
    let poly = MlePoly::new(evals);
    let cfg = RecursiveConfig { total_vars: 3, shard_vars: 3, fan_in: 2 };
    let root_t = Blake3Transcript::new();
    let batch = prove_all(&poly, &cfg, &root_t);
    assert_eq!(batch.shard_proofs.len(), 1);
    assert_eq!(batch.total_sum, poly.sum());
  }
}
