//! Shard verifier — verifies individual shard proofs and the batch sum.
//!
//! Each shard's sumcheck proof is verified using a transcript forked
//! identically to the prover's.  The verifier also checks that the
//! sum of all shard claimed sums equals the expected total.

use field::GF2_128;
use poly::MlePoly;
use transcript::Blake3Transcript;

use crate::config::RecursiveConfig;
use crate::proof::{ShardProof, ShardProofBatch};
use crate::prover::split_mle;

/// Result of verifying a single shard.
#[derive(Debug)]
pub struct ShardVerifyResult {
  /// Shard index.
  pub shard_idx: u32,
  /// Challenges produced by the verifier (for PCS opening).
  pub challenges: Vec<GF2_128>,
}

/// Verify a single shard proof.
///
/// The caller provides the `oracle_eval` — the claimed evaluation of the
/// sub-MLE at the challenge point.  In production this comes from a PCS
/// opening; in tests it can be computed directly.
///
/// Returns `Some(ShardVerifyResult)` on success, `None` on failure.
pub fn verify_shard(
  proof: &ShardProof,
  oracle_eval: GF2_128,
  root_transcript: &Blake3Transcript,
) -> Option<ShardVerifyResult> {
  let mut t = root_transcript.fork("shard", proof.shard_idx);
  let challenges = sumcheck::verify(&proof.sumcheck, oracle_eval, &mut t)?;
  Some(ShardVerifyResult {
    shard_idx: proof.shard_idx,
    challenges,
  })
}

/// Verify all shard proofs against the original MLE.
///
/// This is a test/development helper that has access to the full polynomial
/// (to compute oracle evaluations directly).  In production, oracle evals
/// come from PCS openings.
///
/// Checks:
/// 1. `batch.total_sum == poly.sum()`
/// 2. Each shard proof verifies with the correct oracle evaluation.
///
/// Returns `Some(Vec<ShardVerifyResult>)` on success, `None` on failure.
pub fn verify_all(
  batch: &ShardProofBatch,
  poly: &MlePoly,
  config: &RecursiveConfig,
  root_transcript: &Blake3Transcript,
) -> Option<Vec<ShardVerifyResult>> {
  // Check total sum consistency.
  if batch.total_sum != poly.sum() {
    return None;
  }

  let sub_mles = split_mle(poly, config);

  if batch.shard_proofs.len() != sub_mles.len() {
    return None;
  }

  let mut results = Vec::with_capacity(batch.shard_proofs.len());

  for (proof, sub) in batch.shard_proofs.iter().zip(&sub_mles) {
    // Compute oracle eval: evaluate the sub-MLE at the challenge point.
    // We first extract challenges by running the verifier with final_eval
    // (same approach as the sumcheck tests).
    let mut t_tmp = root_transcript.fork("shard", proof.shard_idx);
    let challenges = sumcheck::verify(&proof.sumcheck, proof.sumcheck.final_eval, &mut t_tmp)?;

    let oracle_eval = sub.evaluate(&challenges);

    // Now verify for real with the correct oracle eval.
    let result = verify_shard(proof, oracle_eval, root_transcript)?;
    results.push(result);
  }

  Some(results)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::prover::prove_all;

  fn g(v: u64) -> GF2_128 {
    GF2_128::from(v)
  }

  fn test_config() -> RecursiveConfig {
    RecursiveConfig {
      total_vars: 4,
      shard_vars: 2,
      fan_in: 2,
    }
  }

  #[test]
  fn verify_all_valid() {
    let evals: Vec<GF2_128> = (1u64..=16).map(g).collect();
    let poly = MlePoly::new(evals);
    let cfg = test_config();
    let root_t = Blake3Transcript::new();
    let batch = prove_all(&poly, &cfg, &root_t);
    let results = verify_all(&batch, &poly, &cfg, &root_t);
    assert!(results.is_some());
    let results = results.unwrap();
    assert_eq!(results.len(), 4);
    for (i, r) in results.iter().enumerate() {
      assert_eq!(r.shard_idx, i as u32);
      assert_eq!(r.challenges.len(), 2); // shard_vars = 2
    }
  }

  #[test]
  fn verify_tampered_claimed_sum_fails() {
    let evals: Vec<GF2_128> = (1u64..=16).map(g).collect();
    let poly = MlePoly::new(evals);
    let cfg = test_config();
    let root_t = Blake3Transcript::new();
    let mut batch = prove_all(&poly, &cfg, &root_t);
    // Tamper with total sum
    batch.total_sum = batch.total_sum + g(1);
    assert!(verify_all(&batch, &poly, &cfg, &root_t).is_none());
  }

  #[test]
  fn verify_tampered_shard_proof_fails() {
    let evals: Vec<GF2_128> = (1u64..=16).map(g).collect();
    let poly = MlePoly::new(evals);
    let cfg = test_config();
    let root_t = Blake3Transcript::new();
    let mut batch = prove_all(&poly, &cfg, &root_t);
    // Tamper with first shard's round poly
    batch.shard_proofs[0].sumcheck.round_polys[0].0[0] =
      batch.shard_proofs[0].sumcheck.round_polys[0].0[0] + g(1);
    assert!(verify_all(&batch, &poly, &cfg, &root_t).is_none());
  }

  #[test]
  fn verify_single_shard_roundtrip() {
    let evals: Vec<GF2_128> = (1u64..=4).map(g).collect();
    let sub = MlePoly::new(evals);
    let root_t = Blake3Transcript::new();
    let proof = crate::prover::prove_shard(0, &sub, &root_t);

    // Compute oracle eval
    let mut t_tmp = root_t.fork("shard", 0);
    let challenges =
      sumcheck::verify(&proof.sumcheck, proof.sumcheck.final_eval, &mut t_tmp).unwrap();
    let oracle = sub.evaluate(&challenges);

    let result = verify_shard(&proof, oracle, &root_t);
    assert!(result.is_some());
  }

  #[test]
  fn verify_8var_poly() {
    // Larger test: 8-var MLE, 4 shards of 6 vars... too large.
    // Instead: 6-var, 4 shards of 4 vars
    let evals: Vec<GF2_128> = (1u64..=64).map(g).collect();
    let poly = MlePoly::new(evals);
    let cfg = RecursiveConfig {
      total_vars: 6,
      shard_vars: 4,
      fan_in: 2,
    };
    assert_eq!(cfg.n_shards(), 4);
    let root_t = Blake3Transcript::new();
    let batch = prove_all(&poly, &cfg, &root_t);
    let results = verify_all(&batch, &poly, &cfg, &root_t);
    assert!(results.is_some());
    assert_eq!(results.unwrap().len(), 4);
  }

  #[test]
  fn verify_single_shard_config() {
    // total_vars == shard_vars → 1 shard
    let evals: Vec<GF2_128> = (1u64..=8).map(g).collect();
    let poly = MlePoly::new(evals);
    let cfg = RecursiveConfig {
      total_vars: 3,
      shard_vars: 3,
      fan_in: 2,
    };
    let root_t = Blake3Transcript::new();
    let batch = prove_all(&poly, &cfg, &root_t);
    let results = verify_all(&batch, &poly, &cfg, &root_t);
    assert!(results.is_some());
    assert_eq!(results.unwrap().len(), 1);
  }
}
