//! End-to-end library: **revm execution → ZK proof generation → verification**.
//!
//! Connects the revm EVM interpreter to binip's ZK-STARK pipeline:
//!
//! 1. Execute EVM bytecode under a [`TracingInspector`] that records every
//!    opcode step with pre/post stack snapshots.
//! 2. Convert captured steps into the binip witness: [`vm::Row`] trace +
//!    [`evm_types::ProofNode`] tree + advice tape.
//! 3. Feed the witness into [`stark::prove_cpu`] / [`stark::prove_gpu`] / [`stark::verify`].
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

use stark::{Proof, StarkParams};
use evm_types::ProofNode;

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

/// Compute [`StarkParams`] for a witness.
fn params_for(witness: &Witness) -> StarkParams {
    let n_rows = witness.rows.len();
    let n_vars = (n_rows as f64).log2().ceil() as u32;
    let n_vars = n_vars.max(2);
    StarkParams::for_n_vars(n_vars)
}

/// Generate a ZK-STARK proof from a witness (CPU path).
pub fn prove_cpu(witness: &Witness) -> Result<(Proof, StarkParams), E2eError> {
    let params = params_for(witness);
    let proof = stark::prove_cpu(&witness.rows, &witness.tree, &params)?;
    Ok((proof, params))
}

/// Generate a ZK-STARK proof from a witness (GPU path).
pub fn prove_gpu(
    witness: &Witness,
    ctx: &gpu::GpuContext,
    cache: &mut gpu::PipelineCache,
) -> Result<(Proof, StarkParams), E2eError> {
    let params = params_for(witness);
    let proof = stark::prove_gpu(&witness.rows, &witness.tree, &params, ctx, cache)?;
    Ok((proof, params))
}

/// Verify a ZK-STARK proof against a Proof Tree.
pub fn verify(
    proof: &Proof,
    tree: &ProofNode,
    params: &StarkParams,
) -> Result<(), E2eError> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use revm::primitives::U256;

    /// Helper: create a synthetic EvmStep (no real revm execution).
    fn step(opcode: u8, pc: u32, pre: &[U256], post: &[U256]) -> EvmStep {
        EvmStep {
            opcode,
            pc,
            gas_before: 100_000,
            gas_after: 99_000,
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
        // ADD compiles to 8 Add32 micro-ops (one per limb).
        assert_eq!(witness.rows.len(), 8);
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
        // AND compiles to 8 And32 micro-ops.
        assert_eq!(witness.rows.len(), 8);
    }

    #[test]
    fn witness_push_zero() {
        use revm::bytecode::opcode::PUSH0;
        let steps = vec![step(PUSH0, 0, &[], &[U256::ZERO])];
        let witness = build_witness(&steps).unwrap();

        assert_eq!(witness.n_steps, 1);
        // PUSH0 compiles to 8 Const ops.
        assert_eq!(witness.rows.len(), 8);
    }

    #[test]
    fn witness_noop_opcodes() {
        use revm::bytecode::opcode::{POP, JUMPDEST};
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
        // EXP uses advice_u256: 8 AdviceLoad + 8 Mov = 16 ops.
        assert_eq!(witness.rows.len(), 16);
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
        // DIV: 8 AdviceLoad(quot) + 8 AdviceLoad(rem) + 1 CheckDiv + 8 Mov = 25
        assert_eq!(witness.rows.len(), 25);
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
            step(ADD, 0, &[U256::from(1u64), U256::from(2u64)], &[U256::from(3u64)]),
            step(STOP, 1, &[], &[]),
        ];
        let witness = build_witness(&steps).unwrap();

        assert_eq!(witness.n_steps, 2);
        // Tree should be Seq(Leaf, Leaf).
        assert!(matches!(witness.tree, ProofNode::Seq { .. }));
    }

    #[test]
    fn witness_three_steps_balanced_tree() {
        use revm::bytecode::opcode::{ADD, SUB, STOP};
        let steps = vec![
            step(ADD, 0, &[U256::from(3u64), U256::from(2u64)], &[U256::from(5u64)]),
            step(SUB, 1, &[U256::from(5u64), U256::from(1u64)], &[U256::from(4u64)]),
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
        let steps = vec![
            step(ADD, 0, &[U256::from(1u64), U256::from(2u64)], &[U256::from(3u64)]),
        ];
        let witness = build_witness(&steps).unwrap();
        let (proof, params) = prove_cpu(&witness).unwrap();
        verify(&proof, &witness.tree, &params).unwrap();
    }

    #[test]
    fn prove_and_verify_multi_step() {
        use revm::bytecode::opcode::{ADD, AND, STOP};
        // Stack depths must be consistent across sequential steps:
        // ADD: pre=[10, 20] → post=[30]  (depth 2 → 1)
        // AND: pre=[30, 0x0F] → post=[0x0F]  (depth 2 → ... wait, that's 2→1)
        // Actually the type checker requires left.post depth == right.pre depth.
        // So: ADD.post.len()==1, AND.pre.len() must ==1.
        // Let's use consistent depths:
        // Step 1: ADD pre=[10,20], post=[30]  — depth 2→1
        // Step 2: STOP pre=[30], post=[30]   — depth 1→1
        let steps = vec![
            step(ADD, 0, &[U256::from(10u64), U256::from(20u64)], &[U256::from(30u64)]),
            step(STOP, 1, &[U256::from(30u64)], &[U256::from(30u64)]),
        ];
        let result = prove_and_verify(&steps);
        assert!(result.is_ok(), "prove_and_verify failed: {:?}", result.err());
    }
}
