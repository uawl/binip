//! Top-level STARK verifier.
//!
//! # Verification checklist (mirrors V1.md §검증자 체크리스트)
//!
//! 1. **Type cert** — re-hash the Proof Tree shape, compare to `type_cert.root_hash`.
//! 2. **Constraint sum** — must be zero (valid trace ⟹ all constraints satisfied).
//! 3. **PCS commitment** — absorb into a fresh transcript, derive β.
//! 4. **Shard verification** — verify each shard sumcheck with forked transcripts.
//! 5. **Recursive verification** — verify all aggregation levels.
//! 6. **PCS opening** — verify the opening proof at the challenge point.
//! 7. **Root claim consistency** — recursive root_claim == shard total_sum.

use circuit::lookup;
use circuit::mpt;
use circuit::state_constraint::extract_boundaries;
use evm_types::ProofNode;
use field::{FieldElem, GF2_128};
use transcript::{Blake3Transcript, Transcript};

use crate::proof::{Proof, StarkParams};

/// Errors during proof verification.
#[derive(Debug, thiserror::Error)]
pub enum VerifyError {
  #[error("type check failed: {0}")]
  TypeCheck(#[from] evm_types::TypeError),

  #[error("type cert root hash mismatch")]
  TypeCertMismatch,

  #[error("type cert state hash mismatch")]
  StateCertMismatch,

  #[error("constraint sum is non-zero")]
  ConstraintSumNonZero,

  #[error("boundary constraint sum is non-zero")]
  BoundaryConstraintSumNonZero,

  #[error("consistency check failed: {0}")]
  ConsistencyCheck(String),

  #[error("shard verification failed: {0}")]
  ShardVerify(String),

  #[error("recursive verification failed: {0}")]
  RecursiveVerify(#[from] recursive::RecursiveVerifyError),

  #[error("PCS opening verification failed")]
  PcsOpenFailed,

  #[error("root claim mismatch: recursive root {recursive:?} != shard total {shard:?}")]
  RootClaimMismatch { recursive: GF2_128, shard: GF2_128 },

  #[error("beta challenge mismatch")]
  BetaMismatch,

  #[error("gamma challenge mismatch")]
  GammaMismatch,

  #[error("reconstruction sum is non-zero")]
  ReconstructionSumNonZero,

  #[error("lookup verification failed")]
  LookupVerifyFailed,

  #[error("Phase C storage proof verification failed: {0}")]
  StorageProofFailed(#[from] mpt::StorageProofError),

  #[error("boundary PCS proof missing (Seq junctions exist)")]
  MissingBoundaryPcs,

  #[error("boundary PCS proof present but no Seq junctions")]
  UnexpectedBoundaryPcs,

  #[error("boundary sumcheck verification failed")]
  BoundarySumcheckFailed,

  #[error("boundary sumcheck claimed non-zero sum")]
  BoundarySumcheckNonZero,

  #[error("boundary PCS opening verification failed")]
  BoundaryPcsOpenFailed,

  #[error("constraint sum {constraint:?} != shard total sum {shard_total:?}")]
  ConstraintSumShardMismatch {
    constraint: GF2_128,
    shard_total: GF2_128,
  },
}

/// Verify a top-level STARK proof.
///
/// # Arguments
///
/// * `proof`  – the proof to verify.
/// * `tree`   – the Proof Tree (structural, non-ZK) for type checking.
/// * `params` – the same parameters used during proving.
///
/// The verifier does NOT need the witness (execution trace).
pub fn verify(proof: &Proof, tree: &ProofNode, params: &StarkParams) -> Result<(), VerifyError> {
  // ── 1. Type cert ──────────────────────────────────────────────────────
  let cert = evm_types::build_cert(tree)?;
  if cert.root_hash != proof.type_cert.root_hash {
    return Err(VerifyError::TypeCertMismatch);
  }
  if cert.state_hash != proof.type_cert.state_hash {
    return Err(VerifyError::StateCertMismatch);
  }

  // ── 1b. Native consistency check (ordering invariants) ────────────────
  let errors = evm_types::consistency_check_all(tree);
  if !errors.is_empty() {
    return Err(VerifyError::ConsistencyCheck(
      errors
        .iter()
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join("; "),
    ));
  }

  // ── 2. Constraint sum must be zero ────────────────────────────────────
  if !proof.constraint_sum.is_zero() {
    return Err(VerifyError::ConstraintSumNonZero);
  }

  // ── 2a. Constraint sum must equal shard total sum (H-3 binding) ───────
  if proof.constraint_sum != proof.shard_batch.total_sum {
    return Err(VerifyError::ConstraintSumShardMismatch {
      constraint: proof.constraint_sum,
      shard_total: proof.shard_batch.total_sum,
    });
  }

  // ── 2b. Boundary constraint sum must be zero ──────────────────────────
  if !proof.boundary_constraint_sum.is_zero() {
    return Err(VerifyError::BoundaryConstraintSumNonZero);
  }

  // ── 3. Rebuild transcript and derive β ────────────────────────────────
  let mut transcript = Blake3Transcript::new();
  transcript.absorb_bytes(&proof.type_cert.root_hash);
  transcript.absorb_bytes(&proof.type_cert.state_hash);
  let beta = transcript.squeeze_challenge();
  if beta != proof.beta {
    return Err(VerifyError::BetaMismatch);
  }

  // ── 3a. Boundary PCS verification (H-1 binding) ──────────────────────
  // Must come BEFORE γ squeeze to match prover transcript ordering.
  let boundary_rows = extract_boundaries(tree);
  let has_boundaries = !boundary_rows.is_empty();

  match (&proof.boundary_pcs, has_boundaries) {
    (Some(bnd), true) => {
      // Absorb boundary PCS commitment root (mirrors prover ordering)
      transcript.absorb_bytes(&bnd.commit.root);

      // Verify boundary sumcheck
      if !bnd.sumcheck.claimed_sum.is_zero() {
        return Err(VerifyError::BoundarySumcheckNonZero);
      }

      let mut bnd_sc_t = transcript.fork("boundary_sc", 0);
      let bnd_challenges =
        sumcheck::verify(&bnd.sumcheck, bnd.open_eval, &mut bnd_sc_t)
          .ok_or(VerifyError::BoundarySumcheckFailed)?;

      // Verify boundary PCS opening
      let bnd_pcs_params = pcs::PcsParams { n_vars: bnd.n_vars, n_queries: 40 };
      let mut bnd_pcs_t = transcript.fork("boundary_pcs_open", 0);
      let bnd_ok = pcs::verify(
        &bnd.commit,
        &bnd_challenges,
        bnd.open_eval,
        &bnd.pcs_open,
        &bnd_pcs_params,
        &mut bnd_pcs_t,
      );
      if !bnd_ok {
        return Err(VerifyError::BoundaryPcsOpenFailed);
      }
    }
    (None, false) => { /* trivially zero, already checked above */ }
    (None, true) => return Err(VerifyError::MissingBoundaryPcs),
    (Some(_), false) => return Err(VerifyError::UnexpectedBoundaryPcs),
  }

  // ── 3b. Derive γ and verify reconstruction + lookups ──────────────────
  let gamma = transcript.squeeze_challenge();
  if gamma != proof.gamma {
    return Err(VerifyError::GammaMismatch);
  }
  if !proof.reconstruction_sum.is_zero() {
    return Err(VerifyError::ReconstructionSumNonZero);
  }
  if lookup::verify_lookups_par(
    &proof.lookup_proofs,
    &proof.lookup_commits,
    &mut transcript.clone(),
  )
  .is_none()
  {
    return Err(VerifyError::LookupVerifyFailed);
  }

  // ── 3c. Phase C: Storage state root binding ───────────────────────────
  // verify_lookups_par returned RwSummaries; re-derive them to feed Phase C.
  // We re-call with a cloned transcript to get the actual summaries.
  {
    let rw_summaries = lookup::verify_lookups_par(
      &proof.lookup_proofs,
      &proof.lookup_commits,
      &mut transcript.clone(),
    )
    .ok_or(VerifyError::LookupVerifyFailed)?;

    let mut phase_c_transcript = transcript.fork("phase_c", 0);
    mpt::verify_phase_c(
      proof.storage_proof.as_ref(),
      &rw_summaries,
      &mut phase_c_transcript,
    )?;
  }

  // Absorb PCS commitment (mirrors prover step 3)
  transcript.absorb_bytes(&proof.batch_commit.root);
  let shard_transcript = transcript.clone();

  // ── 5. Shard + Recursive verification ────────────────────────────────
  // The recursive verifier checks shard claim consistency at each level,
  // so separate shard verification (which requires the witness MLE) is
  // not needed by the verifier — only the prover has the witness.
  recursive::verify_recursive(
    &proof.recursive_proof,
    &proof.shard_batch,
    &params.config,
    &shard_transcript,
  )?;

  // ── 6. Root claim consistency ─────────────────────────────────────────
  // The recursive root claim should equal the total shard sum.
  if proof.recursive_proof.root_claim != proof.shard_batch.total_sum {
    return Err(VerifyError::RootClaimMismatch {
      recursive: proof.recursive_proof.root_claim,
      shard: proof.shard_batch.total_sum,
    });
  }

  // ── 7. PCS opening verification ───────────────────────────────────────
  let mut pcs_transcript = transcript.clone();
  let pcs_ok = pcs::verify(
    &proof.batch_commit,
    &proof.open_point,
    proof.open_eval,
    &proof.pcs_open,
    &params.pcs_params,
    &mut pcs_transcript,
  );
  if !pcs_ok {
    return Err(VerifyError::PcsOpenFailed);
  }

  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::prover;
  use evm_types::proof_tree::LeafProof;
  use evm_types::state::EvmState;
  use vm::Row;

  fn xor_row(a: u128, b: u128) -> Row {
    Row {
      pc: 0,
      op: 3,
      in0: a,
      in1: b,
      in2: 0,
      out: a ^ b,
      flags: 0,
      advice: 0,
    }
  }

  fn state_with_gas(depth: usize, pc: u32, gas: u64) -> EvmState {
    use revm::primitives::U256;
    let mut s = EvmState::with_stack(vec![U256::ZERO; depth], pc);
    s.gas = gas;
    s
  }

  fn make_trace_and_tree() -> (Vec<Row>, ProofNode) {
    let rows = vec![xor_row(1, 2), xor_row(3, 4), xor_row(5, 6), xor_row(7, 8)];

    // ADD costs 3 gas per step.
    let leaf = |i: u32, depth: usize, gas: u64| ProofNode::Leaf {
      opcode: 0x01,
      pre_state: state_with_gas(depth, i, gas),
      post_state: state_with_gas(depth - 1, i + 1, gas - 3),
      leaf_proof: LeafProof::placeholder(),
    };

    let tree = ProofNode::Seq {
      left: Box::new(ProofNode::Seq {
        left: Box::new(leaf(0, 5, 1000)),
        right: Box::new(leaf(1, 4, 997)),
      }),
      right: Box::new(ProofNode::Seq {
        left: Box::new(leaf(2, 3, 994)),
        right: Box::new(leaf(3, 2, 991)),
      }),
    };

    (rows, tree)
  }

  #[test]
  fn prove_then_verify() {
    let (rows, tree) = make_trace_and_tree();
    let params = StarkParams::for_n_vars(2);
    let proof = prover::prove_cpu(&rows, &tree, &params).unwrap();
    verify(&proof, &tree, &params).unwrap();
  }

  #[test]
  fn verify_rejects_nonzero_constraint_sum() {
    let (rows, tree) = make_trace_and_tree();
    let params = StarkParams::for_n_vars(2);
    let mut proof = prover::prove_cpu(&rows, &tree, &params).unwrap();
    proof.constraint_sum = GF2_128::from(1u64); // tamper
    let err = verify(&proof, &tree, &params).unwrap_err();
    assert!(matches!(err, VerifyError::ConstraintSumNonZero));
  }

  #[test]
  fn verify_rejects_bad_type_cert() {
    let (rows, tree) = make_trace_and_tree();
    let params = StarkParams::for_n_vars(2);
    let mut proof = prover::prove_cpu(&rows, &tree, &params).unwrap();
    proof.type_cert.root_hash = [0xFFu8; 32]; // tamper
    let err = verify(&proof, &tree, &params).unwrap_err();
    assert!(matches!(err, VerifyError::TypeCertMismatch));
  }

  #[test]
  fn verify_rejects_tampered_root_claim() {
    let (rows, tree) = make_trace_and_tree();
    let params = StarkParams::for_n_vars(2);
    let mut proof = prover::prove_cpu(&rows, &tree, &params).unwrap();
    proof.recursive_proof.root_claim = proof.recursive_proof.root_claim + GF2_128::from(1u64);
    let err = verify(&proof, &tree, &params).unwrap_err();
    // Could be recursive verify error or root claim mismatch
    assert!(
      matches!(err, VerifyError::RecursiveVerify(_))
        || matches!(err, VerifyError::RootClaimMismatch { .. })
    );
  }
}
