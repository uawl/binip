//! Recursive verifier — checks all aggregation levels bottom-up.
//!
//! For each level and node the verifier:
//! 1. Rebuilds the aggregation MLE from the node's `child_claims`.
//! 2. Runs the sumcheck verifier with `final_eval` as the oracle claim.
//! 3. Evaluates the aggregation MLE at the returned challenges and checks
//!    that it matches `final_eval`.
//! 4. Checks that the node's claimed sum equals the sum of its children.
//!
//! Between levels the verifier checks that the aggregate claims produced
//! at level *l* equal the child claims consumed at level *l + 1*.

use field::{FieldElem, GF2_128};
use shard::{RecursiveConfig, ShardProofBatch};
use transcript::{Blake3Transcript, Transcript};

use crate::circuit::build_aggregation_mle;
use crate::proof::RecursiveProof;

/// Error type for recursive verification failures.
#[derive(Debug, thiserror::Error)]
pub enum RecursiveVerifyError {
  #[error("level {level}, node {node}: sumcheck verification failed")]
  SumcheckFailed { level: u32, node: u32 },

  #[error("level {level}, node {node}: oracle eval mismatch")]
  OracleEvalMismatch { level: u32, node: u32 },

  #[error("level {level}, node {node}: claimed sum != sum-of-children")]
  ClaimedSumMismatch { level: u32, node: u32 },

  #[error("level {level}: expected {expected} nodes, got {got}")]
  NodeCountMismatch { level: u32, expected: usize, got: usize },

  #[error("inter-level claim mismatch at level {level}, position {pos}")]
  InterLevelMismatch { level: u32, pos: usize },

  #[error("root claim mismatch: expected {expected:?}, got {got:?}")]
  RootClaimMismatch { expected: GF2_128, got: GF2_128 },

  #[error("expected {expected} levels, got {got}")]
  DepthMismatch { expected: u32, got: usize },
}

/// Verify the full recursive aggregation proof.
///
/// `shard_batch` provides the ground-truth shard claimed sums (level-0 inputs).
/// The transcript must be in the same initial state as was used during proving.
pub fn verify_recursive(
  proof: &RecursiveProof,
  shard_batch: &ShardProofBatch,
  config: &RecursiveConfig,
  root_transcript: &Blake3Transcript,
) -> Result<(), RecursiveVerifyError> {
  let depth = config.depth();

  if proof.levels.len() != depth as usize {
    return Err(RecursiveVerifyError::DepthMismatch {
      expected: depth,
      got: proof.levels.len(),
    });
  }

  // Collect initial claims from shard proofs.
  let mut current_claims: Vec<GF2_128> = shard_batch
    .shard_proofs
    .iter()
    .map(|sp| sp.sumcheck.claimed_sum)
    .collect();

  for (level_idx, level_proofs) in proof.levels.iter().enumerate() {
    let level = level_idx as u32;
    let fan = config.fan_in as usize;
    let expected_nodes = (current_claims.len() + fan - 1) / fan;

    if level_proofs.len() != expected_nodes {
      return Err(RecursiveVerifyError::NodeCountMismatch {
        level,
        expected: expected_nodes,
        got: level_proofs.len(),
      });
    }

    let mut next_claims = Vec::with_capacity(expected_nodes);

    for (node_idx, lp) in level_proofs.iter().enumerate() {
      // 1. Check child_claims match current_claims slice.
      let start = node_idx * fan;
      let end = (start + fan).min(current_claims.len());
      let expected_children = &current_claims[start..end];

      if lp.child_claims.len() != expected_children.len() {
        return Err(RecursiveVerifyError::InterLevelMismatch {
          level,
          pos: node_idx,
        });
      }
      for (i, (&got, &exp)) in lp.child_claims.iter().zip(expected_children).enumerate() {
        if got != exp {
          return Err(RecursiveVerifyError::InterLevelMismatch {
            level,
            pos: start + i,
          });
        }
      }

      // 2. Build aggregation MLE from child claims.
      let agg_mle = build_aggregation_mle(&lp.child_claims);

      // 3. Check claimed_sum == sum of children.
      let children_sum: GF2_128 = lp
        .child_claims
        .iter()
        .fold(GF2_128::zero(), |acc, &c| acc + c);
      if lp.sumcheck.claimed_sum != children_sum {
        return Err(RecursiveVerifyError::ClaimedSumMismatch {
          level,
          node: node_idx as u32,
        });
      }

      // 4. Verify sumcheck with final_eval as oracle eval.
      let mut t = root_transcript.fork("recursive", level * 0x1_0000 + node_idx as u32);
      t.absorb_bytes(&level.to_le_bytes());
      t.absorb_bytes(&(node_idx as u32).to_le_bytes());

      let challenges = sumcheck::verify(&lp.sumcheck, lp.sumcheck.final_eval, &mut t);
      let challenges = challenges.ok_or(RecursiveVerifyError::SumcheckFailed {
        level,
        node: node_idx as u32,
      })?;

      // 5. Check that MLE(challenges) == final_eval.
      let actual_eval = agg_mle.evaluate(&challenges);
      if actual_eval != lp.sumcheck.final_eval {
        return Err(RecursiveVerifyError::OracleEvalMismatch {
          level,
          node: node_idx as u32,
        });
      }

      next_claims.push(children_sum);
    }

    current_claims = next_claims;
  }

  // After all levels, one claim remains — it must equal root_claim.
  if depth == 0 {
    // No recursion levels: root_claim should equal the total shard sum.
    let expected = shard_batch.total_sum;
    if proof.root_claim != expected {
      return Err(RecursiveVerifyError::RootClaimMismatch {
        expected,
        got: proof.root_claim,
      });
    }
  } else {
    assert_eq!(current_claims.len(), 1);
    if proof.root_claim != current_claims[0] {
      return Err(RecursiveVerifyError::RootClaimMismatch {
        expected: current_claims[0],
        got: proof.root_claim,
      });
    }
  }

  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::prover::prove_recursive;
  use poly::MlePoly;

  fn g(v: u64) -> GF2_128 {
    GF2_128::from(v)
  }

  fn make_and_prove(
    total_vars: u32,
    shard_vars: u32,
    fan_in: u32,
  ) -> (RecursiveConfig, Blake3Transcript, ShardProofBatch, RecursiveProof) {
    let n = 1usize << total_vars;
    let evals: Vec<GF2_128> = (1..=(n as u64)).map(g).collect();
    let poly = MlePoly::new(evals);
    let cfg = RecursiveConfig {
      total_vars,
      shard_vars,
      fan_in,
    };
    let t = Blake3Transcript::new();
    let batch = shard::prove_all(&poly, &cfg, &t);
    let rproof = prove_recursive(&batch, &cfg, &t);
    (cfg, t, batch, rproof)
  }

  #[test]
  fn verify_basic() {
    let (cfg, t, batch, rproof) = make_and_prove(4, 2, 2);
    verify_recursive(&rproof, &batch, &cfg, &t).unwrap();
  }

  #[test]
  fn verify_fan4() {
    let (cfg, t, batch, rproof) = make_and_prove(4, 2, 4);
    verify_recursive(&rproof, &batch, &cfg, &t).unwrap();
  }

  #[test]
  fn verify_single_shard() {
    let (cfg, t, batch, rproof) = make_and_prove(3, 3, 2);
    verify_recursive(&rproof, &batch, &cfg, &t).unwrap();
  }

  #[test]
  fn verify_larger() {
    let (cfg, t, batch, rproof) = make_and_prove(6, 4, 2);
    verify_recursive(&rproof, &batch, &cfg, &t).unwrap();
  }

  #[test]
  fn verify_rejects_tampered_root_claim() {
    let (cfg, t, batch, mut rproof) = make_and_prove(4, 2, 2);
    rproof.root_claim = rproof.root_claim + g(1);
    let err = verify_recursive(&rproof, &batch, &cfg, &t).unwrap_err();
    assert!(matches!(err, RecursiveVerifyError::RootClaimMismatch { .. }));
  }

  #[test]
  fn verify_rejects_tampered_child_claim() {
    let (cfg, t, batch, mut rproof) = make_and_prove(4, 2, 2);
    // Tamper with a child claim in level 0
    rproof.levels[0][0].child_claims[0] = g(999);
    let err = verify_recursive(&rproof, &batch, &cfg, &t).unwrap_err();
    assert!(matches!(err, RecursiveVerifyError::InterLevelMismatch { .. }));
  }
}
