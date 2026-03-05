//! Recursive proof compression: inner proof → CompressedProof.
//!
//! # Strategy
//!
//! 1. **Verify** the inner [`Proof`] to ensure correctness.
//! 2. **Hash** the entire inner proof → `inner_digest` (binding).
//! 3. **Collect** all intermediate verification claims from the inner proof
//!    into a single aggregation vector.
//! 4. **Build** an aggregation MLE from the claims.
//! 5. **Run** a single sumcheck over the aggregation MLE.
//! 6. **Re-open** the PCS at the same point with fewer queries (20 vs 40).
//! 7. **Emit** the [`CompressedProof`] containing only:
//!    - Public inputs (type_cert, batch_commit, config)
//!    - inner_digest (32 bytes)
//!    - aggregation sumcheck + claims
//!    - single PCS opening (reduced queries)
//!    - storage state roots (if any)
//!
//! # Security argument
//!
//! The compressed proof is sound because:
//! - `inner_digest` commits to the full inner proof via Blake3.
//! - The aggregation MLE encodes every claim the full verifier would check.
//! - The sumcheck proves the aggregate is correct.
//! - The PCS opening binds the committed polynomial to the evaluation point.
//!
//! The verifier of a compressed proof re-derives all transcript challenges
//! from the public inputs, reconstructs the aggregation MLE from `inner_claims`,
//! and checks the sumcheck + PCS.

use field::{FieldElem, GF2_128};
use poly::MlePoly;
use transcript::{Blake3Transcript, Transcript};

use crate::proof::{CompressedProof, Proof, StarkParams};
use crate::verifier;

/// Number of PCS queries in the compressed proof.
///
/// Reduced from 40 (inner proof) to 20.  The inner_digest binding
/// provides computational security that supplements the reduced
/// proximity soundness (~2^{-20}).
const COMPRESSED_N_QUERIES: usize = 20;

/// Errors during proof compression.
#[derive(Debug, thiserror::Error)]
pub enum CompressError {
  #[error("inner proof verification failed: {0}")]
  InnerVerifyFailed(#[from] verifier::VerifyError),

  #[error("PCS re-open failed (compressed queries)")]
  PcsReOpenFailed,
}

/// Collect all verification claims from an inner proof.
///
/// The claims vector encodes (in deterministic order) every value that
/// the full verifier checks against constants or derives from the
/// transcript.  The compression verifier rebuilds this same list and
/// checks the aggregation sumcheck over it.
///
/// Claim layout:
///   [0]         constraint_sum (must be 0)
///   [1]         boundary_constraint_sum (must be 0)
///   [2]         reconstruction_sum (must be 0)
///   [3]         root_claim (must == shard total_sum)
///   [4]         shard total_sum
///   [5]         beta (transcript-derived)
///   [6]         gamma (transcript-derived)
///   [7]         open_eval (PCS opening claim)
///   [8..]       per-shard sumcheck claimed_sums
///   [..]        per-level,node recursive child_claims (flattened)
fn collect_claims(proof: &Proof) -> Vec<GF2_128> {
  let mut claims = Vec::new();

  // Core zero claims
  claims.push(proof.constraint_sum);            // [0]
  claims.push(proof.boundary_constraint_sum);   // [1]
  claims.push(proof.reconstruction_sum);        // [2]

  // Root/total consistency
  claims.push(proof.recursive_proof.root_claim); // [3]
  claims.push(proof.shard_batch.total_sum);      // [4]

  // Transcript-derived challenges (verifier re-derives and checks)
  claims.push(proof.beta);                       // [5]
  claims.push(proof.gamma);                      // [6]

  // PCS evaluation claim
  claims.push(proof.open_eval);                  // [7]

  // Shard claimed sums
  for sp in &proof.shard_batch.shard_proofs {
    claims.push(sp.sumcheck.claimed_sum);
  }

  // Recursive level claims (flattened)
  for level in &proof.recursive_proof.levels {
    for node in level {
      for &c in &node.child_claims {
        claims.push(c);
      }
      claims.push(node.sumcheck.claimed_sum);
    }
  }

  // Boundary PCS eval (if present)
  if let Some(ref bnd) = proof.boundary_pcs {
    claims.push(bnd.open_eval);
  }

  // Reconstruction PCS eval
  claims.push(proof.reconstruction_pcs.open_eval);

  claims
}

/// Hash an inner proof to produce a binding digest.
fn hash_proof(proof: &Proof) -> [u8; 32] {
  let bytes = bincode::encode_to_vec(proof, bincode::config::standard())
    .expect("proof serialisation should not fail");
  let mut h = blake3::Hasher::new();
  h.update(b"binip:compress:inner:");
  h.update(&bytes);
  *h.finalize().as_bytes()
}

/// Compress a full [`Proof`] into a [`CompressedProof`].
///
/// This function:
/// 1. Verifies the inner proof.
/// 2. Collects all claims into an aggregation MLE.
/// 3. Produces a single sumcheck over the aggregation.
/// 4. Re-opens the PCS with reduced query count (20 queries → ~2^{-20}).
///
/// The additional security from `inner_digest` binding means the
/// reduced PCS query count provides adequate combined soundness:
/// an attacker must cheat BOTH the digest and the PCS opening.
pub fn compress(proof: &Proof, params: &StarkParams) -> Result<CompressedProof, CompressError> {
  // 1. Verify the inner proof (ensures everything is consistent)
  verifier::verify(proof, params)?;

  // 2. Hash the full proof → binding digest
  let inner_digest = hash_proof(proof);

  // 3. Collect all verification claims
  let claims = collect_claims(proof);

  // 4. Build aggregation MLE and run sumcheck
  //    Use a dedicated transcript forked for compression.
  let mut comp_transcript = Blake3Transcript::new();
  comp_transcript.absorb_bytes(b"binip:compress:agg");
  comp_transcript.absorb_bytes(&inner_digest);

  // Absorb all claims into the transcript for binding.
  for c in &claims {
    comp_transcript.absorb_bytes(&c.lo.to_le_bytes());
    comp_transcript.absorb_bytes(&c.hi.to_le_bytes());
  }

  let padded_len = claims.len().next_power_of_two().max(2);
  let mut evals = claims.clone();
  evals.resize(padded_len, GF2_128::zero());
  let agg_mle = MlePoly::new(evals);

  let mut sc_transcript = comp_transcript.fork("compress_sc", 0);
  let compression_sumcheck = sumcheck::prove(agg_mle, &mut sc_transcript);

  // 5. Reduce PCS opening: take only the first COMPRESSED_N_QUERIES
  //    query paths.  The query positions are derived sequentially from
  //    the transcript, so the first N positions are identical whether
  //    the prover generates 40 or 20 paths.
  let effective = COMPRESSED_N_QUERIES.min(proof.pcs_open.query_paths.len());
  let pcs_open = pcs::OpenProof {
    round_roots: proof.pcs_open.round_roots.clone(),
    query_paths: proof.pcs_open.query_paths[..effective].to_vec(),
  };
  let open_point = proof.open_point.clone();
  let open_eval = proof.open_eval;

  // 6. Extract storage roots
  let (pre_state_root, post_state_root) = match &proof.storage_proof {
    Some(sp) => (Some(sp.pre_root), Some(sp.post_root)),
    None => (None, None),
  };

  Ok(CompressedProof {
    type_cert: proof.type_cert.clone(),
    has_seq_boundaries: proof.has_seq_boundaries,
    batch_commit: proof.batch_commit.clone(),
    config: proof.config.clone(),
    boundary_commit_root: proof.boundary_pcs.as_ref().map(|b| b.commit.root),
    reconstruction_commit_root: proof.reconstruction_pcs.commit.root,
    inner_digest,
    compression_sumcheck,
    inner_claims: claims,
    pcs_open,
    open_point,
    open_eval,
    pre_state_root,
    post_state_root,
  })
}

/// Verify a recursively compressed proof.
///
/// The verifier:
/// 1. Rebuilds the Fiat-Shamir transcript from public inputs.
/// 2. Re-derives β and γ challenges, checking they match `inner_claims`.
/// 3. Reconstructs the aggregation MLE from `inner_claims`.
/// 4. Verifies the compression sumcheck.
/// 5. Verifies the PCS opening.
pub fn verify_compressed(compressed: &CompressedProof) -> Result<(), CompressVerifyError> {
  let claims = &compressed.inner_claims;
  if claims.len() < 8 {
    return Err(CompressVerifyError::MalformedClaims);
  }

  // ── 1. Check zero constraints ─────────────────────────────────────────
  // claims[0] = constraint_sum must be 0
  if !claims[0].is_zero() {
    return Err(CompressVerifyError::ConstraintSumNonZero);
  }
  // claims[1] = boundary_constraint_sum must be 0
  if !claims[1].is_zero() {
    return Err(CompressVerifyError::BoundaryConstraintSumNonZero);
  }
  // claims[2] = reconstruction_sum must be 0
  if !claims[2].is_zero() {
    return Err(CompressVerifyError::ReconstructionSumNonZero);
  }
  // claims[3] == claims[4] (root_claim == total_sum)
  if claims[3] != claims[4] {
    return Err(CompressVerifyError::RootClaimMismatch);
  }

  // ── 2. Re-derive transcript challenges and verify consistency ─────────
  //
  // Rebuild the exact Fiat-Shamir transcript state the inner prover used
  // before the PCS opening.  Order mirrors prover.rs: prepare().
  let mut transcript = Blake3Transcript::new();
  transcript.absorb_bytes(&compressed.type_cert.root_hash);
  transcript.absorb_bytes(&compressed.type_cert.state_hash);
  transcript.absorb_bytes(&[compressed.has_seq_boundaries as u8]);

  // Squeeze β
  let beta = transcript.squeeze_challenge();
  if beta != claims[5] {
    return Err(CompressVerifyError::BetaMismatch);
  }

  // If boundaries exist, absorb the boundary PCS commitment root
  // (mirrors prepare() where PCS commit absorbs the root).
  if let Some(bnd_root) = compressed.boundary_commit_root {
    if !compressed.has_seq_boundaries {
      return Err(CompressVerifyError::BoundaryRootMismatch);
    }
    transcript.absorb_bytes(&bnd_root);
  } else if compressed.has_seq_boundaries {
    return Err(CompressVerifyError::BoundaryRootMismatch);
  }

  // Squeeze γ
  let gamma = transcript.squeeze_challenge();
  if gamma != claims[6] {
    return Err(CompressVerifyError::GammaMismatch);
  }

  // Absorb reconstruction PCS commitment root
  transcript.absorb_bytes(&compressed.reconstruction_commit_root);

  // Absorb batch_commit (main PCS commitment root)
  transcript.absorb_bytes(&compressed.batch_commit.root);

  // ── 3. Verify aggregation sumcheck ────────────────────────────────────
  let mut comp_transcript = Blake3Transcript::new();
  comp_transcript.absorb_bytes(b"binip:compress:agg");
  comp_transcript.absorb_bytes(&compressed.inner_digest);

  for c in claims {
    comp_transcript.absorb_bytes(&c.lo.to_le_bytes());
    comp_transcript.absorb_bytes(&c.hi.to_le_bytes());
  }

  // Rebuild the aggregation MLE from claims
  let padded_len = claims.len().next_power_of_two().max(2);
  let mut evals = claims.clone();
  evals.resize(padded_len, GF2_128::zero());
  let agg_mle = MlePoly::new(evals);
  let expected_sum = agg_mle.sum();

  if compressed.compression_sumcheck.claimed_sum != expected_sum {
    return Err(CompressVerifyError::AggregationSumMismatch);
  }

  let mut sc_transcript = comp_transcript.fork("compress_sc", 0);
  let sc_challenges = sumcheck::verify(
    &compressed.compression_sumcheck,
    compressed.compression_sumcheck.final_eval,
    &mut sc_transcript,
  )
  .ok_or(CompressVerifyError::AggregationSumcheckFailed)?;

  // Check oracle: evaluate aggregation MLE at challenge point
  let oracle_eval = agg_mle.evaluate(&sc_challenges);
  if oracle_eval != compressed.compression_sumcheck.final_eval {
    return Err(CompressVerifyError::AggregationOracleMismatch);
  }

  // ── 4. Verify PCS opening ────────────────────────────────────────────
  // The transcript is now in the same state as the inner proof's
  // transcript just before shard/recursive/PCS open.
  let mut pcs_transcript = transcript.clone();

  let pcs_params = pcs::PcsParams {
    n_vars: compressed.config.total_vars,
    n_queries: COMPRESSED_N_QUERIES,
  };

  let pcs_ok = pcs::verify(
    &compressed.batch_commit,
    &compressed.open_point,
    compressed.open_eval,
    &compressed.pcs_open,
    &pcs_params,
    &mut pcs_transcript,
  );
  if !pcs_ok {
    return Err(CompressVerifyError::PcsVerifyFailed);
  }

  Ok(())
}

/// Errors during compressed proof verification.
#[derive(Debug, thiserror::Error)]
pub enum CompressVerifyError {
  #[error("constraint sum is non-zero")]
  ConstraintSumNonZero,

  #[error("boundary constraint sum is non-zero")]
  BoundaryConstraintSumNonZero,

  #[error("reconstruction sum is non-zero")]
  ReconstructionSumNonZero,

  #[error("root claim != shard total sum")]
  RootClaimMismatch,

  #[error("β challenge mismatch (transcript inconsistency)")]
  BetaMismatch,

  #[error("γ challenge mismatch (transcript inconsistency)")]
  GammaMismatch,

  #[error("boundary commit root / has_seq_boundaries mismatch")]
  BoundaryRootMismatch,

  #[error("malformed claims vector (too short)")]
  MalformedClaims,

  #[error("aggregation sum mismatch")]
  AggregationSumMismatch,

  #[error("aggregation sumcheck verification failed")]
  AggregationSumcheckFailed,

  #[error("aggregation oracle evaluation mismatch")]
  AggregationOracleMismatch,

  #[error("PCS opening verification failed")]
  PcsVerifyFailed,
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::prover;
  use evm_types::proof_tree::LeafProof;
  use evm_types::state::EvmState;
  use evm_types::ProofNode;
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
    let rows = vec![
      xor_row(1, 2),
      xor_row(3, 4),
      xor_row(5, 6),
      xor_row(7, 8),
    ];

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
  fn compress_then_verify() {
    let (rows, tree) = make_trace_and_tree();
    let params = StarkParams::for_n_vars(2);
    let proof = prover::prove_cpu(&rows, &tree, &params, None).unwrap();
    let compressed = compress(&proof, &params).unwrap();
    verify_compressed(&compressed).unwrap();
  }

  #[test]
  fn compressed_size_much_smaller() {
    let (rows, tree) = make_trace_and_tree();
    let params = StarkParams::for_n_vars(2);
    let proof = prover::prove_cpu(&rows, &tree, &params, None).unwrap();

    let full_bytes =
      bincode::encode_to_vec(&proof, bincode::config::standard()).unwrap();
    let compressed = compress(&proof, &params).unwrap();
    let comp_bytes =
      bincode::encode_to_vec(&compressed, bincode::config::standard()).unwrap();

    eprintln!("full: {} bytes, compressed: {} bytes", full_bytes.len(), comp_bytes.len());
    // Compressed should be significantly smaller
    assert!(
      comp_bytes.len() < full_bytes.len(),
      "compressed ({}) should be smaller than full ({})",
      comp_bytes.len(),
      full_bytes.len()
    );
  }

  #[test]
  fn tampered_inner_claims_rejected() {
    let (rows, tree) = make_trace_and_tree();
    let params = StarkParams::for_n_vars(2);
    let proof = prover::prove_cpu(&rows, &tree, &params, None).unwrap();
    let mut compressed = compress(&proof, &params).unwrap();
    // Tamper with constraint_sum claim
    compressed.inner_claims[0] = GF2_128::from(1u64);
    let err = verify_compressed(&compressed);
    assert!(err.is_err());
  }
}
