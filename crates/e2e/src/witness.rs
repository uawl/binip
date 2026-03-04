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

/// Number of 128-bit limbs in a U256 word.
const LIMBS: usize = 2;

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
    let compiled = vm::compile(step.opcode, &step.pre_stack)
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

/// Compute the advice u128 limbs a prover must supply for one opcode step.
///
/// The advice pattern depends on the opcode's `advice_count`:
/// - 0: no advice needed (ADD, SUB, AND, XOR, etc.)
/// - 2 (special): ISZERO, EQ → [boolean, GF(2^128) inverse of accumulator]
/// - 2 (special): LT, GT, SLT, SGT → [eq_hi_boolean, GF(2^128) inverse]
/// - 2 (LIMBS): full U256 result (EXP, SDIV, SHL, SHR, SAR, MUL, env queries, etc.)
/// - 4 (LIMBS*2): quotient + remainder (DIV, MOD, ADDMOD, MULMOD)
fn compute_advice(step: &EvmStep, compiled: &Compiled) -> Vec<u128> {
  use revm::bytecode::opcode::{EQ, EXP, GT, ISZERO, LT, SDIV, SGT, SLT, SMOD};

  let count = compiled.advice_count;
  if count == 0 {
    return Vec::new();
  }

  // ISZERO and EQ: [boolean, GF(2^128) inverse of accumulator].
  if step.opcode == ISZERO {
    return compute_zero_check_advice(&step.pre_stack, true);
  }
  if step.opcode == EQ {
    return compute_zero_check_advice(&step.pre_stack, false);
  }

  // LT, GT, SLT, SGT: [eq_hi_boolean, GF(2^128) inverse for zero-check].
  if matches!(step.opcode, LT | GT | SLT | SGT) {
    return compute_cmp_advice(step);
  }

  // SDIV/SMOD: quotient + remainder (signed division).
  if matches!(step.opcode, SDIV | SMOD) {
    return compute_sdiv_smod_advice(step);
  }

  // EXP: binary exponentiation intermediate chain.
  if step.opcode == EXP {
    return compute_exp_advice(step);
  }

  // The result U256: for most opcodes it's post_stack[0].
  let result = step.post_stack.first().copied().unwrap_or(U256::ZERO);

  match count {
    // Boolean result (1 limb): BYTE
    1 => {
      let limb0 = result.as_limbs()[0] as u128;
      vec![limb0]
    }

    // Full U256 result (2 limbs): EXP, SDIV, MUL, SHL, SHR, SAR, env, etc.
    c if c == LIMBS => u256_to_limbs(&result),

    // Quotient + remainder (4 limbs): DIV, MOD, ADDMOD, MULMOD
    c if c == LIMBS * 2 => compute_div_mod_advice(step),

    // Fallback: zero-fill
    _ => vec![0u128; count],
  }
}

/// Convert a U256 to 2 u128 limbs (little-endian: limb 0 = lowest 128 bits).
fn u256_to_limbs(val: &U256) -> Vec<u128> {
  let le = val.as_limbs(); // [u64; 4], limbs[0] = lowest
  vec![
    (le[0] as u128) | ((le[1] as u128) << 64),
    (le[2] as u128) | ((le[3] as u128) << 64),
  ]
}

/// Compute advice for ISZERO / EQ zero-check: `[boolean, GF(2^128) inverse]`.
///
/// - `is_iszero = true`:  accumulator = OR of all limbs of `pre_stack[0]`.
/// - `is_iszero = false` (EQ): accumulator = OR of XOR of limb pairs.
fn compute_zero_check_advice(pre_stack: &[U256], is_iszero: bool) -> Vec<u128> {
  use field::{FieldElem, GF2_128};

  let a = pre_stack.first().copied().unwrap_or(U256::ZERO);
  let limbs_a = u256_to_limbs(&a);

  let acc = if is_iszero {
    // ISZERO: accumulator = limb0 | limb1
    limbs_a[0] | limbs_a[1]
  } else {
    // EQ: accumulator = (a_limb0 ^ b_limb0) | (a_limb1 ^ b_limb1)
    let b = pre_stack.get(1).copied().unwrap_or(U256::ZERO);
    let limbs_b = u256_to_limbs(&b);
    (limbs_a[0] ^ limbs_b[0]) | (limbs_a[1] ^ limbs_b[1])
  };

  let boolean = if acc == 0 { 1u128 } else { 0u128 };
  let inv = if acc == 0 {
    0u128 // arbitrary; constraint satisfied vacuously when acc = 0
  } else {
    let gf_acc = GF2_128::new(acc as u64, (acc >> 64) as u64);
    let gf_inv = gf_acc.inv();
    (gf_inv.lo as u128) | ((gf_inv.hi as u128) << 64)
  };

  vec![boolean, inv]
}

/// Compute advice for LT/GT/SLT/SGT comparison opcodes.
///
/// The comparison opcode uses two CmpLt micro-ops (one per limb) plus an
/// equality check on the high limbs.  CmpLt computes pad internally, so
/// the only advice needed is `[eq_hi_boolean, GF(2^128) inverse]` for the
/// CheckZeroInv/CheckZeroMul pair on the high-limb XOR accumulator.
fn compute_cmp_advice(step: &EvmStep) -> Vec<u128> {
  use field::{FieldElem, GF2_128};
  use revm::bytecode::opcode::*;

  let a = step.pre_stack.first().copied().unwrap_or(U256::ZERO);
  let b = step.pre_stack.get(1).copied().unwrap_or(U256::ZERO);
  let limbs_a = u256_to_limbs(&a);
  let limbs_b = u256_to_limbs(&b);

  // For SLT/SGT: we flip the sign bit (bit 127 of the high limb).
  let sign_mask: u128 = 1u128 << 127;
  let (a_hi, b_hi) = match step.opcode {
    SLT | SGT => (limbs_a[1] ^ sign_mask, limbs_b[1] ^ sign_mask),
    _ => (limbs_a[1], limbs_b[1]),
  };

  // The eq_hi check operates on a_hi XOR b_hi.
  let acc = a_hi ^ b_hi;

  let eq_bool = if acc == 0 { 1u128 } else { 0u128 };
  let inv = if acc == 0 {
    0u128
  } else {
    let gf_acc = GF2_128::new(acc as u64, (acc >> 64) as u64);
    let gf_inv = gf_acc.inv();
    (gf_inv.lo as u128) | ((gf_inv.hi as u128) << 64)
  };

  vec![eq_bool, inv]
}

/// Compute quotient + remainder advice for DIV/MOD/ADDMOD/MULMOD.
///
/// The advice tape layout is: `[quot_lo, quot_hi, rem_lo, rem_hi]`.
///
/// For DIV/MOD: `dividend = pre_stack[0]`, `divisor = pre_stack[1]`.
/// For ADDMOD: `dividend = pre_stack[0] + pre_stack[1]`, `divisor = pre_stack[2]`.
/// For MULMOD: `dividend = pre_stack[0] * pre_stack[1]`, `divisor = pre_stack[2]`.
fn compute_div_mod_advice(step: &EvmStep) -> Vec<u128> {
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
      // MUL also uses LIMBS advice but with advice_count == LIMBS (2),
      // handled separately. This path shouldn't be reached for MUL.
      (U256::ZERO, U256::ZERO)
    }
  };

  let mut limbs = u256_to_limbs(&quot);
  limbs.extend(u256_to_limbs(&rem));
  limbs
}

/// Compute quotient + remainder advice for SDIV/SMOD (signed division).
///
/// The advice tape layout is: `[quot_lo, quot_hi, rem_lo, rem_hi]`.
/// The quotient and remainder satisfy `a ≡ q * b + r (mod 2^256)`.
fn compute_sdiv_smod_advice(step: &EvmStep) -> Vec<u128> {
  let a = step.pre_stack.first().copied().unwrap_or(U256::ZERO);
  let b = step.pre_stack.get(1).copied().unwrap_or(U256::ZERO);

  if b.is_zero() {
    return vec![0u128; 4];
  }

  // Signed division: work with absolute values, then apply signs.
  let a_neg = (a >> 255) == U256::from(1);
  let b_neg = (b >> 255) == U256::from(1);

  let abs_a = if a_neg {
    (!a).overflowing_add(U256::from(1)).0
  } else {
    a
  };
  let abs_b = if b_neg {
    (!b).overflowing_add(U256::from(1)).0
  } else {
    b
  };

  let abs_q = abs_a / abs_b;
  let abs_r = abs_a % abs_b;

  // SDIV quotient sign: negative iff exactly one operand is negative.
  let q = if a_neg != b_neg && !abs_q.is_zero() {
    (!abs_q).overflowing_add(U256::from(1)).0
  } else {
    abs_q
  };

  // SMOD remainder sign: same sign as dividend (or zero).
  let r = if a_neg && !abs_r.is_zero() {
    (!abs_r).overflowing_add(U256::from(1)).0
  } else {
    abs_r
  };

  let mut limbs = u256_to_limbs(&q);
  limbs.extend(u256_to_limbs(&r));
  limbs
}

/// Compute advice for EXP (binary exponentiation).
///
/// For each step of the square-and-multiply chain the prover supplies
/// 3 u128 limbs: `[result_lo, result_hi, mul_hi]`.
///
/// - `result_lo`, `result_hi`: the U256 product mod 2^256 of the step.
/// - `mul_hi`: the high 128 bits of the 128×128 widening multiplication
///   of the two low limbs (needed for CheckMul verification).
fn compute_exp_advice(step: &EvmStep) -> Vec<u128> {
  let base = step.pre_stack.first().copied().unwrap_or(U256::ZERO);
  let exponent = step.pre_stack.get(1).copied().unwrap_or(U256::ZERO);

  if exponent <= U256::from(1) || base.is_zero() || base == U256::from(1) {
    return vec![];
  }

  let n_bits = 256 - exponent.leading_zeros() as u32;

  let mut limbs = Vec::new();
  let mut acc = base;

  for bit_idx in (0..n_bits - 1).rev() {
    let bit_set = (exponent >> bit_idx) & U256::from(1) == U256::from(1);

    // Square: acc = acc * acc mod 2^256
    let squared = acc.overflowing_mul(acc).0;
    let acc_lo = u256_lo128(&acc);
    limbs.extend(u256_to_limbs(&squared));
    limbs.push(widening_mul_hi(acc_lo, acc_lo));
    acc = squared;

    if bit_set {
      // Multiply by base: acc = acc * base mod 2^256
      let product = acc.overflowing_mul(base).0;
      let cur_lo = u256_lo128(&acc);
      let base_lo = u256_lo128(&base);
      limbs.extend(u256_to_limbs(&product));
      limbs.push(widening_mul_hi(cur_lo, base_lo));
      acc = product;
    }
  }

  limbs
}

/// Low 128 bits of a U256.
fn u256_lo128(v: &U256) -> u128 {
  let le = v.as_limbs();
  (le[0] as u128) | ((le[1] as u128) << 64)
}

/// High 128 bits of the 128×128 widening multiplication `a * b`.
fn widening_mul_hi(a: u128, b: u128) -> u128 {
  let product = U256::from(a).overflowing_mul(U256::from(b)).0;
  let le = product.as_limbs();
  (le[2] as u128) | ((le[3] as u128) << 64)
}

// ─────────────────────────────────────────────────────────────────────────────
// Register loading
// ─────────────────────────────────────────────────────────────────────────────

/// Load EVM stack values into VM register slots.
///
/// Stack slot `s` maps to base register `s * 2`. Each U256 occupies 2
/// consecutive registers in little-endian limb order.  We load up to 5
/// stack slots (the maximum any single opcode reads).
fn load_stack_into_regs(vm: &mut Vm, stack: &[U256]) {
  let max_slots = stack.len().min(5);
  for s in 0..max_slots {
    let base = (s * LIMBS) as u8;
    let le = stack[s].as_limbs(); // [u64; 4] little-endian
    let lo = (le[0] as u128) | ((le[1] as u128) << 64);
    let hi = (le[2] as u128) | ((le[3] as u128) << 64);
    vm.regs.write(base, lo);
    vm.regs.write(base + 1, hi);
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
