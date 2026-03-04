//! End-to-end library: **revm execution → ZK proof generation → verification**.
//!
//! Connects the revm EVM interpreter to binip's ZK-STARK pipeline:
//!
//! 1. Execute EVM bytecode under a [`TracingInspector`] that records every
//!    opcode step with pre/post stack snapshots.
//! 2. Convert captured steps into the binip witness: [`vm::Row`] trace +
//!    [`evm_types::ProofNode`] tree + advice tape.
//! 3. Feed the witness into [`stark::prove_cpu`] / [`stark::verify`].
//!
//! # Quick start
//!
//! ```ignore
//! use e2e::{execute, build_witness, prove_cpu, verify};
//!
//! let steps = execute(&bytecode, &input)?;
//! let witness = build_witness(&steps)?;
//! let proof = prove_cpu(&witness)?;
//! verify(&proof, &witness.tree)?;
//! ```

mod inspector;
mod witness;

pub use inspector::{EvmStep, TracingInspector};
pub use witness::{Witness, WitnessError, build_witness};

use evm_types::ProofNode;
use shard::RecursiveConfig;
use stark::{Proof, StarkParams};

/// Errors from the E2E pipeline.
#[derive(Debug, thiserror::Error)]
pub enum E2eError {
  #[error("witness construction failed: {0}")]
  Witness(#[from] WitnessError),

  #[error("proof generation failed: {0}")]
  Prove(#[from] stark::ProveError),

  #[error("proof verification failed: {0}")]
  Verify(#[from] stark::VerifyError),
}

/// Optional tuning knobs exposed to callers.
///
/// All fields default to `None`, which falls back to the built-in
/// heuristics in [`StarkParams::for_n_vars`].
#[derive(Debug, Clone, Default)]
pub struct ProveConfig {
  /// Override the number of MLE variables per shard (`shard_vars`).
  ///
  /// Default heuristic: `total_vars / 2`.  Smaller values create more
  /// (and cheaper) shards; larger values create fewer but heavier shards.
  /// Must satisfy `1 ≤ shard_vars ≤ total_vars`.
  pub shard_vars: Option<u32>,
}

/// Compute [`StarkParams`] for a witness.
fn params_for(witness: &Witness) -> StarkParams {
  params_for_with_config(witness, &ProveConfig::default())
}

/// Compute [`StarkParams`] for a witness, applying [`ProveConfig`] overrides.
fn params_for_with_config(witness: &Witness, cfg: &ProveConfig) -> StarkParams {
  let n_rows = witness.rows.len().max(1);
  let n_vars = (usize::BITS - (n_rows - 1).max(1).leading_zeros()) as u32;
  let mut params = StarkParams::for_n_vars(n_vars);
  if let Some(sv) = cfg.shard_vars {
    params.config = RecursiveConfig {
      shard_vars: sv.max(1).min(n_vars),
      ..params.config
    };
  }
  params
}

/// Generate a ZK-STARK proof from a witness (CPU path).
pub fn prove_cpu(witness: &Witness) -> Result<(Proof, StarkParams), E2eError> {
  let params = params_for(witness);
  let proof = stark::prove_cpu(&witness.rows, &witness.tree, &params)?;
  Ok((proof, params))
}

/// Like [`prove_cpu`] but with explicit [`ProveConfig`] overrides.
pub fn prove_cpu_with(
  witness: &Witness,
  cfg: &ProveConfig,
) -> Result<(Proof, StarkParams), E2eError> {
  let params = params_for_with_config(witness, cfg);
  let proof = stark::prove_cpu(&witness.rows, &witness.tree, &params)?;
  Ok((proof, params))
}

/// Generate a ZK-STARK proof from a witness (parallel CPU path).
///
/// Shards are proven concurrently via rayon.  Same result as [`prove_cpu`]
/// but scales across CPU cores.
pub fn prove_cpu_par(witness: &Witness) -> Result<(Proof, StarkParams), E2eError> {
  let params = params_for(witness);
  let proof = stark::prove_cpu_par(&witness.rows, &witness.tree, &params)?;
  Ok((proof, params))
}

/// Like [`prove_cpu_par`] but with explicit [`ProveConfig`] overrides.
pub fn prove_cpu_par_with(
  witness: &Witness,
  cfg: &ProveConfig,
) -> Result<(Proof, StarkParams), E2eError> {
  let params = params_for_with_config(witness, cfg);
  let proof = stark::prove_cpu_par(&witness.rows, &witness.tree, &params)?;
  Ok((proof, params))
}

/// Verify a ZK-STARK proof against a Proof Tree.
pub fn verify(proof: &Proof, tree: &ProofNode, params: &StarkParams) -> Result<(), E2eError> {
  stark::verify(proof, tree, params)?;
  Ok(())
}

/// Execute EVM bytecode, generate witness, prove (CPU), and verify — all in one call.
pub fn prove_and_verify(steps: &[EvmStep]) -> Result<Proof, E2eError> {
  let witness = build_witness(steps)?;
  let (proof, params) = prove_cpu(&witness)?;
  verify(&proof, &witness.tree, &params)?;
  Ok(proof)
}

/// Like [`prove_and_verify`] but with explicit [`ProveConfig`] overrides.
pub fn prove_and_verify_with(
  steps: &[EvmStep],
  cfg: &ProveConfig,
) -> Result<Proof, E2eError> {
  let witness = build_witness(steps)?;
  let (proof, params) = prove_cpu_with(&witness, cfg)?;
  verify(&proof, &witness.tree, &params)?;
  Ok(proof)
}

#[cfg(test)]
mod tests {
  use super::*;
  use revm::primitives::U256;

  /// Helper: create a synthetic EvmStep with correct static gas consumption.
  fn step(opcode: u8, pc: u32, pre: &[U256], post: &[U256]) -> EvmStep {
    let cost = evm_types::opcode::gas_cost(opcode)
      .map(|g| g.static_gas)
      .unwrap_or(0);
    EvmStep {
      opcode,
      pc,
      gas_before: 100_000,
      gas_after: 100_000 - cost,
      pre_stack: pre.to_vec(),
      post_stack: post.to_vec(),
    }
  }

  // ── Witness builder tests ─────────────────────────────────────────────

  #[test]
  fn witness_empty_returns_error() {
    let result = build_witness(&[]);
    assert!(result.is_err());
  }

  #[test]
  fn witness_single_add() {
    use revm::bytecode::opcode::ADD;
    let a = U256::from(3u64);
    let b = U256::from(5u64);
    let result = U256::from(8u64);

    let steps = vec![step(ADD, 0, &[a, b], &[result])];
    let witness = build_witness(&steps).unwrap();

    assert_eq!(witness.n_steps, 1);
    // ADD compiles to 2 Add128 micro-ops (one per limb).
    assert_eq!(witness.rows.len(), 2);
    // Tree should be a single Leaf.
    assert!(matches!(witness.tree, ProofNode::Leaf { .. }));
  }

  #[test]
  fn witness_single_sub() {
    use revm::bytecode::opcode::SUB;
    let a = U256::from(10u64);
    let b = U256::from(3u64);
    let result = U256::from(7u64);

    let steps = vec![step(SUB, 1, &[a, b], &[result])];
    let witness = build_witness(&steps).unwrap();

    assert_eq!(witness.n_steps, 1);
    assert!(!witness.rows.is_empty());
  }

  #[test]
  fn witness_bitwise_and() {
    use revm::bytecode::opcode::AND;
    let a = U256::from(0xFF00u64);
    let b = U256::from(0x0FF0u64);
    let result = U256::from(0x0F00u64);

    let steps = vec![step(AND, 2, &[a, b], &[result])];
    let witness = build_witness(&steps).unwrap();

    assert_eq!(witness.n_steps, 1);
    // AND compiles to 2 And128 micro-ops.
    assert_eq!(witness.rows.len(), 2);
  }

  #[test]
  fn witness_push_zero() {
    use revm::bytecode::opcode::PUSH0;
    let steps = vec![step(PUSH0, 0, &[], &[U256::ZERO])];
    let witness = build_witness(&steps).unwrap();

    assert_eq!(witness.n_steps, 1);
    // PUSH0 compiles to 2 Const ops.
    assert_eq!(witness.rows.len(), 2);
  }

  #[test]
  fn witness_noop_opcodes() {
    use revm::bytecode::opcode::{JUMPDEST, POP};
    // POP and JUMPDEST compile to noop (0 ops).
    let steps = vec![
      step(POP, 0, &[U256::from(42u64)], &[]),
      step(JUMPDEST, 1, &[], &[]),
    ];
    let witness = build_witness(&steps).unwrap();

    assert_eq!(witness.n_steps, 2);
    // Both are noop → 0 rows each. But we still build the tree.
    assert_eq!(witness.rows.len(), 0);
  }

  #[test]
  fn witness_advice_bool_lt() {
    use revm::bytecode::opcode::LT;
    // 3 < 5 = true → 1
    let a = U256::from(3u64);
    let b = U256::from(5u64);
    let result = U256::from(1u64);

    let steps = vec![step(LT, 0, &[a, b], &[result])];
    let witness = build_witness(&steps).unwrap();

    assert_eq!(witness.n_steps, 1);
    assert!(!witness.rows.is_empty());
  }

  #[test]
  fn witness_advice_u256_exp() {
    use revm::bytecode::opcode::EXP;
    // 2^10 = 1024
    let base = U256::from(2u64);
    let exp = U256::from(10u64);
    let result = U256::from(1024u64);

    let steps = vec![step(EXP, 0, &[base, exp], &[result])];
    let witness = build_witness(&steps).unwrap();

    assert_eq!(witness.n_steps, 1);
    // EXP uses advice_u256: 1 Advice2 + 2 Mov = 3 ops.
    assert_eq!(witness.rows.len(), 3);
  }

  #[test]
  fn witness_div_advice() {
    use revm::bytecode::opcode::DIV;
    // 100 / 7 = 14
    let a = U256::from(100u64);
    let b = U256::from(7u64);
    let result = U256::from(14u64);

    let steps = vec![step(DIV, 0, &[a, b], &[result])];
    let witness = build_witness(&steps).unwrap();

    assert_eq!(witness.n_steps, 1);
    // DIV: 1 Advice4(quot+rem) + 1 CheckDiv + 2 Mov = 4
    assert_eq!(witness.rows.len(), 4);
  }

  #[test]
  fn witness_stop() {
    use revm::bytecode::opcode::STOP;
    let steps = vec![step(STOP, 0, &[], &[])];
    let witness = build_witness(&steps).unwrap();

    assert_eq!(witness.n_steps, 1);
    // STOP compiles to Done → 1 row.
    assert_eq!(witness.rows.len(), 1);
  }

  #[test]
  fn witness_mul_with_advice() {
    use revm::bytecode::opcode::MUL;
    let a = U256::from(6u64);
    let b = U256::from(7u64);
    let result = U256::from(42u64);

    let steps = vec![step(MUL, 0, &[a, b], &[result])];
    let witness = build_witness(&steps).unwrap();

    assert_eq!(witness.n_steps, 1);
    assert!(!witness.rows.is_empty());
  }

  // ── Multi-step / tree folding tests ───────────────────────────────────

  #[test]
  fn witness_two_steps_seq_tree() {
    use revm::bytecode::opcode::{ADD, STOP};
    let steps = vec![
      step(
        ADD,
        0,
        &[U256::from(1u64), U256::from(2u64)],
        &[U256::from(3u64)],
      ),
      step(STOP, 1, &[], &[]),
    ];
    let witness = build_witness(&steps).unwrap();

    assert_eq!(witness.n_steps, 2);
    // Tree should be Seq(Leaf, Leaf).
    assert!(matches!(witness.tree, ProofNode::Seq { .. }));
  }

  #[test]
  fn witness_three_steps_balanced_tree() {
    use revm::bytecode::opcode::{ADD, STOP, SUB};
    let steps = vec![
      step(
        ADD,
        0,
        &[U256::from(3u64), U256::from(2u64)],
        &[U256::from(5u64)],
      ),
      step(
        SUB,
        1,
        &[U256::from(5u64), U256::from(1u64)],
        &[U256::from(4u64)],
      ),
      step(STOP, 2, &[], &[]),
    ];
    let witness = build_witness(&steps).unwrap();

    assert_eq!(witness.n_steps, 3);
    // 3 leaves → Seq(Seq(Leaf, Leaf), Leaf)
    match &witness.tree {
      ProofNode::Seq { left, right } => {
        assert!(matches!(**left, ProofNode::Seq { .. }));
        assert!(matches!(**right, ProofNode::Leaf { .. }));
      }
      _ => panic!("expected Seq at root"),
    }
  }

  #[test]
  fn witness_unsupported_opcode() {
    // 0xFE is INVALID but mapped to compile_stop. Use something truly unknown.
    let steps = vec![step(0xC0, 0, &[], &[])];
    let result = build_witness(&steps);
    assert!(result.is_err());
  }

  // ── Inspector tests ──────────────────────────────────────────────────

  #[test]
  fn inspector_starts_empty() {
    let insp = TracingInspector::new();
    assert!(insp.steps.is_empty());
  }

  // ── Proof pipeline tests ─────────────────────────────────────────────

  #[test]
  fn prove_and_verify_simple_add() {
    use revm::bytecode::opcode::ADD;
    let steps = vec![step(
      ADD,
      0,
      &[U256::from(1u64), U256::from(2u64)],
      &[U256::from(3u64)],
    )];
    let witness = build_witness(&steps).unwrap();
    let (proof, params) = prove_cpu(&witness).unwrap();
    verify(&proof, &witness.tree, &params).unwrap();
  }

  #[test]
  fn prove_and_verify_simple_mul() {
    use revm::bytecode::opcode::MUL;
    let a = U256::from(6u64);
    let b = U256::from(7u64);
    let result = U256::from(42u64);
    let steps = vec![step(MUL, 0, &[a, b], &[result])];
    let witness = build_witness(&steps).unwrap();
    let (proof, params) = prove_cpu(&witness).unwrap();
    verify(&proof, &witness.tree, &params).unwrap();
  }

  #[test]
  fn prove_and_verify_simple_exp() {
    use revm::bytecode::opcode::EXP;
    // 0^7 = 0
    let a = U256::from(0u64);
    let b = U256::from(7u64);
    let result = U256::from(0u64);
    let steps = vec![step(EXP, 0, &[a, b], &[result])];
    let witness = build_witness(&steps).unwrap();
    let (proof, params) = prove_cpu(&witness).unwrap();
    verify(&proof, &witness.tree, &params).unwrap();
  }

  #[test]
  fn prove_and_verify_push0_mul() {
    use revm::bytecode::opcode::{MUL, PUSH0};
    let a = U256::from(7u64);
    let b = U256::from(3u64);
    let push0_cost = evm_types::opcode::gas_cost(PUSH0).unwrap().static_gas;
    let mul_cost = evm_types::opcode::gas_cost(MUL).unwrap().static_gas;
    let gas0 = 100_000u64;
    let gas1 = gas0 - push0_cost;
    let gas2 = gas1 - mul_cost;
    let steps = vec![
      EvmStep {
        opcode: PUSH0,
        pc: 0,
        gas_before: gas0,
        gas_after: gas1,
        pre_stack: vec![a, b],
        post_stack: vec![U256::ZERO, a, b],
      },
      EvmStep {
        opcode: MUL,
        pc: 1,
        gas_before: gas1,
        gas_after: gas2,
        pre_stack: vec![U256::ZERO, a, b],
        post_stack: vec![U256::ZERO, b],
      },
    ];
    prove_and_verify(&steps).unwrap();
  }

  #[test]
  fn prove_and_verify_push0_div() {
    use revm::bytecode::opcode::{DIV, PUSH0};
    let a = U256::from(7u64);
    let b = U256::from(3u64);
    let push0_cost = evm_types::opcode::gas_cost(PUSH0).unwrap().static_gas;
    let div_cost = evm_types::opcode::gas_cost(DIV).unwrap().static_gas;
    let gas0 = 100_000u64;
    let gas1 = gas0 - push0_cost;
    let gas2 = gas1 - div_cost;
    let steps = vec![
      EvmStep {
        opcode: PUSH0,
        pc: 0,
        gas_before: gas0,
        gas_after: gas1,
        pre_stack: vec![a, b],
        post_stack: vec![U256::ZERO, a, b],
      },
      EvmStep {
        opcode: DIV,
        pc: 1,
        gas_before: gas1,
        gas_after: gas2,
        pre_stack: vec![U256::ZERO, a, b],
        post_stack: vec![U256::ZERO, b],
      },
    ];
    prove_and_verify(&steps).unwrap();
  }

  #[test]
  fn prove_and_verify_simple_div() {
    use revm::bytecode::opcode::DIV;
    let a = U256::from(100u64);
    let b = U256::from(7u64);
    let result = U256::from(14u64);
    let steps = vec![step(DIV, 0, &[a, b], &[result])];
    let witness = build_witness(&steps).unwrap();
    let (proof, params) = prove_cpu(&witness).unwrap();
    verify(&proof, &witness.tree, &params).unwrap();
  }

  #[test]
  fn prove_and_verify_multi_step() {
    use revm::bytecode::opcode::{ADD, STOP};
    // ADD costs 3, STOP costs 0.
    // Gas must flow: ADD.gas_after == STOP.gas_before
    let steps = vec![
      EvmStep {
        opcode: ADD,
        pc: 0,
        gas_before: 100_000,
        gas_after: 99_997,
        pre_stack: vec![U256::from(10u64), U256::from(20u64)],
        post_stack: vec![U256::from(30u64)],
      },
      EvmStep {
        opcode: STOP,
        pc: 1,
        gas_before: 99_997,
        gas_after: 99_997,
        pre_stack: vec![U256::from(30u64)],
        post_stack: vec![U256::from(30u64)],
      },
    ];
    let result = prove_and_verify(&steps);
    assert!(
      result.is_ok(),
      "prove_and_verify failed: {:?}",
      result.err()
    );
  }

  // ── L-1 indirect coverage: comparison ops prove & verify ──────────
  //
  // These tests verify that correct comparison results pass the full
  // prove→verify pipeline.  The circuit's AdviceLoad + RangeCheck
  // only ensures the result is 0 or 1; actual correctness is verified
  // by `consistency_check` in the verifier.

  #[test]
  fn prove_and_verify_lt_true() {
    use revm::bytecode::opcode::LT;
    // 3 < 5 = 1
    let steps = vec![step(LT, 0, &[U256::from(3u64), U256::from(5u64)], &[U256::from(1u64)])];
    prove_and_verify(&steps).unwrap();
  }

  #[test]
  fn prove_and_verify_lt_false() {
    use revm::bytecode::opcode::LT;
    // 5 < 3 = 0
    let steps = vec![step(LT, 0, &[U256::from(5u64), U256::from(3u64)], &[U256::ZERO])];
    prove_and_verify(&steps).unwrap();
  }

  #[test]
  fn prove_and_verify_gt_true() {
    use revm::bytecode::opcode::GT;
    // 5 > 3 = 1
    let steps = vec![step(GT, 0, &[U256::from(5u64), U256::from(3u64)], &[U256::from(1u64)])];
    prove_and_verify(&steps).unwrap();
  }

  #[test]
  fn prove_and_verify_eq_true() {
    use revm::bytecode::opcode::EQ;
    // 7 == 7 = 1
    let steps = vec![step(EQ, 0, &[U256::from(7u64), U256::from(7u64)], &[U256::from(1u64)])];
    prove_and_verify(&steps).unwrap();
  }

  #[test]
  fn prove_and_verify_eq_false() {
    use revm::bytecode::opcode::EQ;
    // 7 == 8 = 0
    let steps = vec![step(EQ, 0, &[U256::from(7u64), U256::from(8u64)], &[U256::ZERO])];
    prove_and_verify(&steps).unwrap();
  }

  #[test]
  fn prove_and_verify_iszero_true() {
    use revm::bytecode::opcode::ISZERO;
    // ISZERO(0) = 1
    let steps = vec![step(ISZERO, 0, &[U256::ZERO], &[U256::from(1u64)])];
    prove_and_verify(&steps).unwrap();
  }

  #[test]
  fn prove_and_verify_iszero_false() {
    use revm::bytecode::opcode::ISZERO;
    // ISZERO(42) = 0
    let steps = vec![step(ISZERO, 0, &[U256::from(42u64)], &[U256::ZERO])];
    prove_and_verify(&steps).unwrap();
  }
}
