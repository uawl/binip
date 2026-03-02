//! Witness builder — converts [`EvmStep`] traces into the binip witness.
//!
//! The witness consists of:
//! - A flat [`vm::Row`] trace (fed into the STARK prover).
//! - A [`ProofNode`] tree (used for type checking).

use evm_types::proof_tree::{LeafProof, ProofNode};
use evm_types::state::EvmState;
use revm::primitives::U256;
use vm::{AdviceTape, Compiled, Row, Vm};

use crate::inspector::EvmStep;

/// Number of 32-bit limbs in a U256 word.
const LIMBS: usize = 8;

/// Output of the witness builder.
#[derive(Debug)]
pub struct Witness {
    /// Flat execution trace rows (micro-op level).
    pub rows: Vec<Row>,
    /// Structural proof tree for the EVM execution.
    pub tree: ProofNode,
    /// Number of EVM-level steps.
    pub n_steps: usize,
}

/// Errors during witness construction.
#[derive(Debug, thiserror::Error)]
pub enum WitnessError {
    #[error("unsupported opcode 0x{0:02x}")]
    UnsupportedOpcode(u8),

    #[error("VM execution failed: {0}")]
    VmExec(#[from] vm::VmError),

    #[error("empty step trace")]
    Empty,
}

/// Build a full witness from captured EVM steps.
///
/// For each [`EvmStep`]:
/// 1. Compute advice u32 limbs from pre/post stack values.
/// 2. Compile the opcode into micro-ops via [`vm::compile`].
/// 3. Load pre-stack values into VM registers, run micro-ops, collect rows.
/// 4. Build [`EvmState`] pre/post and a [`ProofNode::Leaf`].
///
/// Finally, fold all leaves into a balanced binary tree of `Seq` nodes.
pub fn build_witness(steps: &[EvmStep]) -> Result<Witness, WitnessError> {
    if steps.is_empty() {
        return Err(WitnessError::Empty);
    }

    let mut all_rows: Vec<Row> = Vec::new();
    let mut leaves: Vec<ProofNode> = Vec::new();

    for step in steps {
        let compiled = vm::compile(step.opcode)
            .ok_or(WitnessError::UnsupportedOpcode(step.opcode))?;

        // Compute advice limbs the prover must supply.
        let advice_limbs = compute_advice(step, &compiled);

        // Set up a fresh VM with the advice tape.
        let mut vm = Vm::new(AdviceTape::new(advice_limbs));

        // Load pre-stack U256 values into register slots.
        load_stack_into_regs(&mut vm, &step.pre_stack);

        // Execute micro-ops.
        vm.run(&compiled.ops)?;

        // Collect rows.
        all_rows.extend_from_slice(&vm.trace);

        // Build EvmState pre/post.
        let pre_state = EvmState {
            stack: step.pre_stack.clone(),
            memory: Vec::new(),
            pc: step.pc,
            gas: step.gas_before,
            storage: Default::default(),
            transient_storage: Default::default(),
            jumpdest_table: Default::default(),
        };

        let post_state = EvmState {
            stack: step.post_stack.clone(),
            memory: Vec::new(),
            pc: step.pc + 1, // simplified; PUSH advances by data len
            gas: step.gas_after,
            storage: Default::default(),
            transient_storage: Default::default(),
            jumpdest_table: Default::default(),
        };

        leaves.push(ProofNode::Leaf {
            opcode: step.opcode,
            pre_state,
            post_state,
            leaf_proof: LeafProof::placeholder(),
        });
    }

    let tree = fold_leaves(leaves);

    Ok(Witness {
        rows: all_rows,
        tree,
        n_steps: steps.len(),
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Advice computation
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the advice u32 limbs a prover must supply for one opcode step.
///
/// The advice pattern depends on the opcode's `advice_count`:
/// - 0: no advice needed (ADD, SUB, AND, XOR, etc.)
/// - 1: boolean result (LT, GT, SLT, SGT, ISZERO, EQ, BYTE)
/// - 8 (LIMBS): full U256 result (EXP, SDIV, SHL, SHR, SAR, MUL, env queries, etc.)
/// - 16 (LIMBS*2): quotient + remainder (DIV, MOD, ADDMOD, MULMOD)
fn compute_advice(step: &EvmStep, compiled: &Compiled) -> Vec<u32> {
    let count = compiled.advice_count;
    if count == 0 {
        return Vec::new();
    }

    // The result U256: for most opcodes it's post_stack[0].
    let result = step.post_stack.first().copied().unwrap_or(U256::ZERO);

    match count {
        // Boolean result (1 limb): LT, GT, SLT, SGT, ISZERO, EQ, BYTE
        1 => {
            let limb0 = result.as_limbs()[0] as u32;
            vec![limb0]
        }

        // Full U256 result (8 limbs): EXP, SDIV, MUL, SHL, SHR, SAR, env, etc.
        c if c == LIMBS => {
            u256_to_limbs(&result)
        }

        // Quotient + remainder (16 limbs): DIV, MOD, ADDMOD, MULMOD
        c if c == LIMBS * 2 => {
            compute_div_mod_advice(step)
        }

        // Fallback: zero-fill
        _ => vec![0u32; count],
    }
}

/// Convert a U256 to 8 u32 limbs (little-endian: limb 0 = lowest 32 bits).
fn u256_to_limbs(val: &U256) -> Vec<u32> {
    let le_limbs = val.as_limbs(); // [u64; 4], limbs[0] = lowest
    let mut out = Vec::with_capacity(LIMBS);
    for &w in le_limbs.iter() {
        out.push(w as u32);
        out.push((w >> 32) as u32);
    }
    out
}

/// Compute quotient + remainder advice for DIV/MOD/ADDMOD/MULMOD.
///
/// The advice tape layout is: `[quot_limb0..quot_limb7, rem_limb0..rem_limb7]`.
///
/// For DIV/MOD: `dividend = pre_stack[0]`, `divisor = pre_stack[1]`.
/// For ADDMOD: `dividend = pre_stack[0] + pre_stack[1]`, `divisor = pre_stack[2]`.
/// For MULMOD: `dividend = pre_stack[0] * pre_stack[1]`, `divisor = pre_stack[2]`.
fn compute_div_mod_advice(step: &EvmStep) -> Vec<u32> {
    use revm::bytecode::opcode::*;

    let a = step.pre_stack.first().copied().unwrap_or(U256::ZERO);
    let b = step.pre_stack.get(1).copied().unwrap_or(U256::ZERO);

    let (quot, rem) = match step.opcode {
        DIV => {
            if b.is_zero() {
                (U256::ZERO, U256::ZERO)
            } else {
                (a / b, a % b)
            }
        }
        MOD => {
            if b.is_zero() {
                (U256::ZERO, U256::ZERO)
            } else {
                (a / b, a % b)
            }
        }
        ADDMOD => {
            let n = step.pre_stack.get(2).copied().unwrap_or(U256::ZERO);
            if n.is_zero() {
                (U256::ZERO, U256::ZERO)
            } else {
                let sum = a.overflowing_add(b).0;
                (sum / n, sum % n)
            }
        }
        MULMOD => {
            let n = step.pre_stack.get(2).copied().unwrap_or(U256::ZERO);
            if n.is_zero() {
                (U256::ZERO, U256::ZERO)
            } else {
                let prod = a.overflowing_mul(b).0;
                (prod / n, prod % n)
            }
        }
        _ => {
            // MUL also uses LIMBS advice but with advice_count == LIMBS (8),
            // handled separately. This path shouldn't be reached for MUL.
            (U256::ZERO, U256::ZERO)
        }
    };

    let mut limbs = u256_to_limbs(&quot);
    limbs.extend(u256_to_limbs(&rem));
    limbs
}

// ─────────────────────────────────────────────────────────────────────────────
// Register loading
// ─────────────────────────────────────────────────────────────────────────────

/// Load EVM stack values into VM register slots.
///
/// Stack slot `s` maps to base register `s * 8`. Each U256 occupies 8
/// consecutive registers in little-endian limb order.  We load up to 5
/// stack slots (the maximum any single opcode reads).
fn load_stack_into_regs(vm: &mut Vm, stack: &[U256]) {
    let max_slots = stack.len().min(5);
    for s in 0..max_slots {
        let base = (s * LIMBS) as u8;
        let limbs = stack[s].as_limbs(); // [u64; 4] little-endian
        for (i, &w) in limbs.iter().enumerate() {
            vm.regs.write(base + (i * 2) as u8, w as u32);
            vm.regs.write(base + (i * 2 + 1) as u8, (w >> 32) as u32);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tree folding
// ─────────────────────────────────────────────────────────────────────────────

/// Fold a list of leaf nodes into a balanced binary tree of `Seq` nodes.
///
/// ```text
/// [A, B, C, D] →  Seq(Seq(A, B), Seq(C, D))
/// [A, B, C]    →  Seq(Seq(A, B), C)
/// [A]          →  A
/// ```
fn fold_leaves(mut leaves: Vec<ProofNode>) -> ProofNode {
    assert!(!leaves.is_empty());
    while leaves.len() > 1 {
        let mut next = Vec::with_capacity((leaves.len() + 1) / 2);
        let mut iter = leaves.into_iter();
        while let Some(left) = iter.next() {
            if let Some(right) = iter.next() {
                next.push(ProofNode::Seq {
                    left: Box::new(left),
                    right: Box::new(right),
                });
            } else {
                next.push(left);
            }
        }
        leaves = next;
    }
    leaves.into_iter().next().unwrap()
}
