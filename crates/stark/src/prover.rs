//! Top-level STARK prover — orchestrates the full proof pipeline.
//!
//! # Pipeline
//!
//! 1. **Type-check** the Proof Tree → [`TypeCert`].
//! 2. **Encode** the VM trace into a column-major table → constraint MLE.
//! 3. **PCS commit** the constraint MLE evaluations.
//! 4. **Shard** the constraint MLE and prove each shard independently.
//! 5. **Recursive aggregation** of shard proofs.
//! 6. **PCS open** at the sumcheck challenge point.
//! 7. Assemble into a [`Proof`].

use circuit::TraceTable;
use circuit::lookup;
use circuit::state_constraint::{BoundaryTraceTable, extract_boundaries};
use evm_types::{ProofNode, TypeCert, build_cert};
use field::{FieldElem, GF2_128};
use poly::MlePoly;
use rand::rng;
use transcript::{Blake3Transcript, Transcript};
use vm::Row;

use crate::proof::{Proof, StarkParams};

/// Errors during proof generation.
#[derive(Debug, thiserror::Error)]
pub enum ProveError {
  #[error("type check failed: {0}")]
  TypeCheck(#[from] evm_types::TypeError),

  #[error("constraint encoding failed: {0}")]
  Constraint(#[from] circuit::ConstraintError),

  #[error("trace is empty")]
  EmptyTrace,
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Intermediate state after steps 1–3 of the prove pipeline.
struct Prepared {
  type_cert: TypeCert,
  beta: GF2_128,
  gamma: GF2_128,
  blinded_mle: MlePoly,
  constraint_sum: GF2_128,
  boundary_constraint_sum: GF2_128,
  reconstruction_sum: GF2_128,
  lookup_proofs: lookup::LookupProofs,
  lookup_witness: lookup::LookupWitness,
  batch_commit: pcs::Commitment,
  pcs_state: pcs::PcsState,
  transcript: Blake3Transcript,
}

/// Steps 1–3: type-check, encode, blind, PCS commit.
fn prepare(rows: &[Row], tree: &ProofNode, params: &StarkParams) -> Result<Prepared, ProveError> {
  if rows.is_empty() {
    return Err(ProveError::EmptyTrace);
  }

  // 1. Type-check the Proof Tree
  let type_cert: TypeCert = build_cert(tree)?;

  // 2. Encode trace → constraint MLE
  let mut transcript = Blake3Transcript::new();
  transcript.absorb_bytes(&type_cert.root_hash);
  transcript.absorb_bytes(&type_cert.state_hash);
  let beta: GF2_128 = transcript.squeeze_challenge();

  let table = TraceTable::from_rows(rows);
  let constraint_mle: MlePoly = table.constraint_mle(beta)?;
  let constraint_sum = constraint_mle.sum();

  // 2a. Boundary constraint MLE (state continuity at Seq junctions)
  let boundary_rows = extract_boundaries(tree);
  let boundary_table = BoundaryTraceTable::from_rows(&boundary_rows);
  let boundary_mle = boundary_table.constraint_mle(beta);
  let boundary_constraint_sum = boundary_mle.sum();

  // 2b. Reconstruction constraint (STARK ↔ LUT binding)
  let gamma: GF2_128 = transcript.squeeze_challenge();
  let reconstruction_mle = table.reconstruction_mle(gamma)?;
  let reconstruction_sum = reconstruction_mle.sum();

  // 2c. Lookup proofs (byte-level LUT arguments)
  let mut lookup_witness = lookup::collect_witnesses(rows);
  let mut lookup_transcript = transcript.clone();
  let lookup_proofs = lookup::prove_lookups_par(&mut lookup_witness, &mut lookup_transcript);

  // 2d. ZK blinding
  let blinded_mle = constraint_mle.blind(&mut rng());

  // 3. PCS commit
  let (batch_commit, pcs_state) =
    pcs::commit(&blinded_mle.evals, &params.pcs_params, &mut transcript);

  Ok(Prepared {
    type_cert,
    beta,
    gamma,
    blinded_mle,
    constraint_sum,
    boundary_constraint_sum,
    reconstruction_sum,
    lookup_proofs,
    lookup_witness,
    batch_commit,
    pcs_state,
    transcript,
  })
}

/// Steps 1–3 (parallel CPU): type-check, encode ∥, blind ∥, PCS commit ∥.
fn prepare_par(
  rows: &[Row],
  tree: &ProofNode,
  params: &StarkParams,
) -> Result<Prepared, ProveError> {
  if rows.is_empty() {
    return Err(ProveError::EmptyTrace);
  }

  // 1. Type-check the Proof Tree
  let type_cert: TypeCert = build_cert(tree)?;

  // 2. Encode trace → constraint MLE (rayon parallel rows)
  let mut transcript = Blake3Transcript::new();
  transcript.absorb_bytes(&type_cert.root_hash);
  transcript.absorb_bytes(&type_cert.state_hash);
  let beta: GF2_128 = transcript.squeeze_challenge();

  let table = TraceTable::from_rows_par(rows);
  let constraint_mle: MlePoly = table.constraint_mle_par(beta)?;
  let constraint_sum = constraint_mle.sum();

  // 2a. Boundary constraint MLE (state continuity at Seq junctions)
  let boundary_rows = extract_boundaries(tree);
  let boundary_table = BoundaryTraceTable::from_rows(&boundary_rows);
  let boundary_mle = boundary_table.constraint_mle(beta);
  let boundary_constraint_sum = boundary_mle.sum();

  // 2b. Reconstruction constraint (STARK ↔ LUT binding)
  let gamma: GF2_128 = transcript.squeeze_challenge();
  let reconstruction_mle = table.reconstruction_mle_par(gamma)?;
  let reconstruction_sum = reconstruction_mle.sum();

  // 2c. Lookup proofs (byte-level LUT arguments)
  let mut lookup_witness = lookup::collect_witnesses_par(rows);
  let mut lookup_transcript = transcript.clone();
  let lookup_proofs = lookup::prove_lookups_par(&mut lookup_witness, &mut lookup_transcript);

  // 2d. ZK blinding (rayon parallel pointwise)
  let blinded_mle = constraint_mle.blind_par(&mut rng());

  // 3. PCS commit (rayon parallel row encoding + leaf hashing)
  let (batch_commit, pcs_state) =
    pcs::commit_par(&blinded_mle.evals, &params.pcs_params, &mut transcript);

  Ok(Prepared {
    type_cert,
    beta,
    gamma,
    blinded_mle,
    constraint_sum,
    boundary_constraint_sum,
    reconstruction_sum,
    lookup_proofs,
    lookup_witness,
    batch_commit,
    pcs_state,
    transcript,
  })
}

/// Steps 5–7: recursive aggregation, PCS open, assemble proof.
fn assemble(
  prep: Prepared,
  shard_batch: shard::ShardProofBatch,
  params: &StarkParams,
  shard_transcript: Blake3Transcript,
) -> Result<Proof, ProveError> {
  let Prepared {
    type_cert,
    beta,
    gamma,
    blinded_mle: _,
    constraint_sum,
    boundary_constraint_sum,
    reconstruction_sum,
    lookup_proofs,
    lookup_witness,
    batch_commit,
    pcs_state,
    mut transcript,
  } = prep;

  // 5. Recursive aggregation
  let recursive_proof = recursive::prove_recursive(&shard_batch, &params.config, &shard_transcript);

  // 6. PCS open
  let open_point = derive_open_point(
    &shard_batch,
    &recursive_proof,
    &params.config,
    &shard_transcript,
  );
  let (open_eval, pcs_open) = pcs::open(&pcs_state, &open_point, &mut transcript);

  // 7. Assemble
  Ok(Proof {
    type_cert,
    batch_commit,
    beta,
    constraint_sum,
    boundary_constraint_sum,
    gamma,
    reconstruction_sum,
    lookup_proofs,
    lookup_witness,
    shard_batch,
    recursive_proof,
    pcs_open,
    open_point,
    open_eval,
    config: params.config.clone(),
  })
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a full ZK-STARK proof using the **CPU** shard prover.
///
/// # Arguments
///
/// * `rows`   – the execution trace (one [`Row`] per micro-op step).
/// * `tree`   – the structural Proof Tree for the EVM execution.
/// * `params` – tuning knobs (shard/recursion config, PCS queries).
pub fn prove_cpu(
  rows: &[Row],
  tree: &ProofNode,
  params: &StarkParams,
) -> Result<Proof, ProveError> {
  let prep = prepare(rows, tree, params)?;

  // 4. Shard proving (CPU)
  let shard_transcript = prep.transcript.clone();
  let shard_batch = shard::prove_all(&prep.blinded_mle, &params.config, &shard_transcript);

  assemble(prep, shard_batch, params, shard_transcript)
}

/// Generate a full ZK-STARK proof using **parallel CPU** proving.
///
/// Every phase is parallelised via rayon:
///
/// ```text
///   prepare_par (constraint_mle ∥, blind ∥, PCS commit ∥)
///       │
///       ▼
///   ┌─ shard_0 ─┐
///   ├─ shard_1 ──┤  ← rayon par_iter (no split_mle copy)
///   ├─ ...      ─┤
///   └─ shard_N ──┘
///       │
///       ▼
///   recursive DAG — each node starts as soon as its
///   children complete (no level-wide barrier)
///       │
///       ▼
///   derive_open_point → PCS open → assemble (sequential)
/// ```
///
/// # Arguments
///
/// * `rows`   – the execution trace (one [`Row`] per micro-op step).
/// * `tree`   – the structural Proof Tree for the EVM execution.
/// * `params` – tuning knobs (shard/recursion config, PCS queries).
pub fn prove_cpu_par(
  rows: &[Row],
  tree: &ProofNode,
  params: &StarkParams,
) -> Result<Proof, ProveError> {
  let prep = prepare_par(rows, tree, params)?;
  let shard_transcript: Blake3Transcript = prep.transcript.clone();

  // ── Phase 1: parallel shard proving ────────────────────────────────────
  let shard_batch = shard::prove_all_par(
    &prep.blinded_mle,
    &params.config,
    &shard_transcript,
  );

  // ── Phase 2..D+1: parallel recursive aggregation (per-level) ──────────
  let recursive_proof = recursive::prove_recursive_par(
    &shard_batch,
    &params.config,
    &shard_transcript,
  );

  // ── Phase D+2: PCS open + assemble (sequential) ───────────────────────
  let Prepared {
    type_cert,
    beta,
    gamma,
    blinded_mle: _,
    constraint_sum,
    boundary_constraint_sum,
    reconstruction_sum,
    lookup_proofs,
    lookup_witness,
    batch_commit,
    pcs_state,
    mut transcript,
  } = prep;

  let open_point = derive_open_point(
    &shard_batch,
    &recursive_proof,
    &params.config,
    &shard_transcript,
  );
  let (open_eval, pcs_open) = pcs::open(&pcs_state, &open_point, &mut transcript);

  Ok(Proof {
    type_cert,
    batch_commit,
    beta,
    constraint_sum,
    boundary_constraint_sum,
    gamma,
    reconstruction_sum,
    lookup_proofs,
    lookup_witness,
    shard_batch,
    recursive_proof,
    pcs_open,
    open_point,
    open_eval,
    config: params.config.clone(),
  })
}

/// Derive the PCS opening point from sumcheck challenges.
///
/// The full MLE has `total_vars` variables, decomposed as:
///   `r = [high_vars... | low_vars...]`
///
/// - **Low variables** (`shard_vars` dimensions): come from shard-0's sumcheck
///   challenges — these fix the position within a shard.
/// - **High variables** (`total_vars - shard_vars` dimensions): come from the
///   recursive aggregation.  We follow the node-0 path from the bottom level
///   to the root: at each level, node 0 aggregates the first `fan_in` children,
///   so shard 0 is always covered.  The sumcheck challenges from each level's
///   node 0, concatenated bottom-up, form the high-variable challenges.
///
/// When `fan_in` is a power of two, `depth * log2(fan_in) == high_vars` exactly.
fn derive_open_point(
  shard_batch: &shard::ShardProofBatch,
  recursive_proof: &recursive::RecursiveProof,
  config: &shard::RecursiveConfig,
  shard_transcript: &Blake3Transcript,
) -> Vec<GF2_128> {
  let total_vars = config.total_vars as usize;
  let shard_vars = config.shard_vars as usize;
  let high_vars = total_vars - shard_vars;

  // ── Low variables: shard-0 sumcheck challenges ────────────────────────
  let shard0 = &shard_batch.shard_proofs[0].sumcheck;
  let low_challenges: Vec<GF2_128> = {
    let mut t = shard_transcript.fork("shard", 0);
    sumcheck::verify(shard0, shard0.final_eval, &mut t)
      .expect("shard 0 should be internally consistent")
  };

  // ── High variables: recursive node-0 challenges at each level ─────────
  let mut high_challenges = Vec::with_capacity(high_vars);

  for level_proofs in &recursive_proof.levels {
    // Node 0 at this level covers shard 0's ancestry.
    if let Some(node0) = level_proofs.first() {
      let mut t = shard_transcript.fork("recursive", node0.level * 0x1_0000 + node0.node_idx);
      t.absorb_bytes(&node0.level.to_le_bytes());
      t.absorb_bytes(&node0.node_idx.to_le_bytes());

      if let Some(challenges) = sumcheck::verify(&node0.sumcheck, node0.sumcheck.final_eval, &mut t)
      {
        high_challenges.extend_from_slice(&challenges);
      }
    }
  }

  // ── Compose: high ∥ low, truncate/pad to total_vars ───────────────────
  let mut point = Vec::with_capacity(total_vars);
  point.extend_from_slice(&high_challenges);
  point.extend_from_slice(&low_challenges);
  point.truncate(total_vars);
  point.resize(total_vars, GF2_128::zero());
  point
}

#[cfg(test)]
mod tests {
  use super::*;
  use evm_types::proof_tree::LeafProof;
  use evm_types::state::EvmState;
  use vm::Row;

  fn xor_row(a: u128, b: u128) -> Row {
    Row {
      pc: 0,
      op: 3, // Xor128
      in0: a,
      in1: b,
      in2: 0,
      out: a ^ b,
      flags: 0,
      advice: 0,
    }
  }

  fn add_leaf(opcode: u8, pre: EvmState, post: EvmState) -> ProofNode {
    ProofNode::Leaf {
      opcode,
      pre_state: pre,
      post_state: post,
      leaf_proof: LeafProof::placeholder(),
    }
  }

  /// Build an EvmState with `n` zero-valued U256 stack items.
  fn state_with_depth(depth: usize, pc: u32) -> EvmState {
    use revm::primitives::U256;
    EvmState::with_stack(vec![U256::ZERO; depth], pc)
  }

  fn make_simple_trace_and_tree() -> (Vec<Row>, ProofNode) {
    let rows = vec![xor_row(1, 2), xor_row(3, 4), xor_row(5, 6), xor_row(7, 8)];

    // 4 ADD leaves chained: depth 5→4→3→2→1
    // Each leaf: pre.pc = i, post.pc = i+1
    let leaf = |i: u32, depth: usize| {
      add_leaf(
        0x01,
        state_with_depth(depth, i),
        state_with_depth(depth - 1, i + 1),
      )
    };

    let tree = ProofNode::Seq {
      left: Box::new(ProofNode::Seq {
        left: Box::new(leaf(0, 5)),
        right: Box::new(leaf(1, 4)),
      }),
      right: Box::new(ProofNode::Seq {
        left: Box::new(leaf(2, 3)),
        right: Box::new(leaf(3, 2)),
      }),
    };

    (rows, tree)
  }

  #[test]
  fn prove_produces_valid_proof() {
    let (rows, tree) = make_simple_trace_and_tree();
    let params = StarkParams::for_n_vars(2);
    let proof = prove_cpu(&rows, &tree, &params).unwrap();
    assert!(proof.type_cert.leaf_count > 0);
    assert!(
      proof.constraint_sum.is_zero(),
      "valid trace should have zero constraint sum"
    );
  }

  #[test]
  fn prove_rejects_empty_trace() {
    let (_, tree) = make_simple_trace_and_tree();
    let params = StarkParams::for_n_vars(2);
    let err = prove_cpu(&[], &tree, &params).unwrap_err();
    assert!(matches!(err, ProveError::EmptyTrace));
  }

  #[test]
  fn prove_rejects_type_error() {
    let rows = vec![xor_row(1, 2)];
    let pre = state_with_depth(2, 0);
    let post = state_with_depth(5, 1); // Wrong depth for ADD
    let bad_tree = add_leaf(0x01, pre, post);
    let params = StarkParams::for_n_vars(1);
    let err = prove_cpu(&rows, &bad_tree, &params).unwrap_err();
    assert!(matches!(err, ProveError::TypeCheck(_)));
  }

  #[test]
  fn proof_has_correct_structure() {
    let (rows, tree) = make_simple_trace_and_tree();
    let params = StarkParams::for_n_vars(2);
    let proof = prove_cpu(&rows, &tree, &params).unwrap();
    assert_eq!(proof.batch_commit.n_vars, 2);
    assert_eq!(proof.open_point.len(), params.config.total_vars as usize);
    assert_eq!(
      proof.recursive_proof.levels.len(),
      params.config.depth() as usize
    );
  }

  #[test]
  fn larger_trace_proof() {
    let rows: Vec<Row> = (0..8u128).map(|i| xor_row(i, i + 100)).collect();

    // Chain 8 ADD leaves: depth 9→8→...→1, pc 0→1→...→8
    let leaves: Vec<ProofNode> = (0..8)
      .map(|i| {
        add_leaf(
          0x01,
          state_with_depth(9 - i, i as u32),
          state_with_depth(8 - i, i as u32 + 1),
        )
      })
      .collect();
    let tree = leaves
      .into_iter()
      .reduce(|left, right| ProofNode::Seq {
        left: Box::new(left),
        right: Box::new(right),
      })
      .unwrap();

    let params = StarkParams::for_n_vars(3);
    let proof = prove_cpu(&rows, &tree, &params).unwrap();
    assert!(proof.constraint_sum.is_zero());
  }
}
