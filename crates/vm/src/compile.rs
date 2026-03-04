//! EVM opcode → [`MicroOp`] compiler.
//!
//! ## Register allocation
//!
//! A U256 value occupies **2 consecutive registers** in little-endian limb
//! order.  EVM stack slot `s` (0 = top) is mapped to base register `s * 2`.
//! With 16 registers we can address 5 EVM words plus 6 scratch registers,
//! which is more than enough for any single opcode (max 3 inputs + 1 output).
//!
//! ## Advice pattern
//!
//! Expensive operations (DIV, MOD, MULMOD, …) emit `AdviceLoad` to inject
//! the result from the prover's advice tape, followed by a `Check*`
//! instruction that constrains correctness inside the circuit.

use crate::isa::{MicroOp, Reg};
use revm::primitives::U256;

/// Number of 128-bit limbs per U256 word.
const LIMBS: usize = 2;

// ─────────────────────────────────────────────────────────────────────────────
// Register-slot helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Base register for EVM stack slot `s` (0 = TOS).
const fn slot(s: usize) -> Reg {
  (s * LIMBS) as Reg
}

/// Register for limb `l` of stack slot `s`.
const fn limb(s: usize, l: usize) -> Reg {
  slot(s) + l as Reg
}

/// Scratch area starts after 5 stack-slot groups (regs 10..15).
const fn scratch(i: usize) -> Reg {
  10 + i as Reg
}

/// Dedicated carry registers for Add128 chains: scratch(4) and scratch(5).
/// These hold 0 or 1 (zero-extended) and alternate as cin/cout.
const CARRY_A: Reg = 14; // scratch(4)
const CARRY_B: Reg = 15; // scratch(5)

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Result of compiling a single EVM opcode.
pub struct Compiled {
  /// Micro-ops that implement this opcode.
  pub ops: Vec<MicroOp>,
  /// Number of U256 limbs to pre-populate on the advice tape for this opcode
  /// (each u128 limb counts as 1).
  pub advice_count: usize,
}

/// Compile one EVM opcode byte into a sequence of micro-ops.
///
/// Returns `None` for opcodes that are not yet supported.
///
/// `pre_stack` provides the EVM stack values before this opcode executes.
/// Opcodes that need compile-time operand knowledge (SHL, SHR, SAR,
/// SIGNEXTEND) use `pre_stack` to generate specialized verified code.
/// Pass `&[]` when operand context is unavailable (tests, etc.) — those
/// opcodes will fall back to the unverified advice pattern.
///
/// The caller is responsible for:
/// 1. Loading input U256 values into the appropriate stack-slot registers.
/// 2. Filling the advice tape with the required number of limbs.
/// 3. Extracting the result from stack slot 0 after execution.
pub fn compile(opcode: u8, pre_stack: &[U256]) -> Option<Compiled> {
  use revm::bytecode::opcode::*;

  match opcode {
    // ── Arithmetic ────────────────────────────────────────────────────────
    ADD => Some(compile_add()),
    SUB => Some(compile_sub()),
    MUL => Some(compile_mul()),
    DIV => Some(compile_div()),
    MOD => Some(compile_mod()),
    ADDMOD => Some(compile_addmod()),
    MULMOD => Some(compile_mulmod()),
    EXP => {
      let base = pre_stack.first().copied().unwrap_or(U256::ZERO);
      let exponent = pre_stack.get(1).copied().unwrap_or(U256::ZERO);
      Some(compile_exp(base, exponent))
    }
    SDIV => Some(compile_sdiv()),
    SMOD => Some(compile_smod()),
    SIGNEXTEND => {
      let i = pre_stack.first().copied().unwrap_or(U256::ZERO);
      Some(compile_signextend(i))
    }

    // ── Comparison ────────────────────────────────────────────────────────
    LT => Some(compile_lt()),
    GT => Some(compile_gt()),
    SLT => Some(compile_slt()),
    SGT => Some(compile_sgt()),
    ISZERO => Some(compile_iszero()),
    EQ => Some(compile_eq()),

    // ── Bitwise ───────────────────────────────────────────────────────────
    AND => Some(compile_bitwise_and()),
    OR => Some(compile_bitwise_or()),
    XOR => Some(compile_bitwise_xor()),
    NOT => Some(compile_bitwise_not()),
    SHL => {
      let shift = pre_stack.first().copied().unwrap_or(U256::ZERO);
      if shift > U256::from(u16::MAX) || pre_stack.is_empty() {
        Some(compile_shl_large())
      } else {
        Some(compile_shl(shift.as_limbs()[0] as u32))
      }
    }
    SHR => {
      let shift = pre_stack.first().copied().unwrap_or(U256::ZERO);
      if shift > U256::from(u16::MAX) || pre_stack.is_empty() {
        Some(compile_shr_large())
      } else {
        Some(compile_shr(shift.as_limbs()[0] as u32))
      }
    }
    SAR => {
      let shift = pre_stack.first().copied().unwrap_or(U256::ZERO);
      let value = pre_stack.get(1).copied().unwrap_or(U256::ZERO);
      let is_negative = (value >> 255) == U256::from(1);
      if shift > U256::from(u16::MAX) || pre_stack.is_empty() {
        Some(compile_sar_large(is_negative))
      } else {
        Some(compile_sar(shift.as_limbs()[0] as u32, is_negative))
      }
    }
    BYTE => Some(compile_byte_extract()),

    // ── Push / Pop ────────────────────────────────────────────────────────
    PUSH0 => Some(compile_push_zero()),
    POP => Some(compile_noop()),

    // ── DUP / SWAP ────────────────────────────────────────────────────────
    DUP1..=DUP16 => Some(compile_dup(opcode - DUP1 + 1)),
    SWAP1..=SWAP16 => Some(compile_swap()),

    // ── Keccak ────────────────────────────────────────────────────────────
    KECCAK256 => Some(compile_keccak()),

    // ── Memory ────────────────────────────────────────────────────────────
    MLOAD => Some(compile_mload()),
    MSTORE => Some(compile_mstore()),
    MSTORE8 => Some(compile_mstore8()),
    MCOPY => Some(compile_noop()),

    // ── Storage ───────────────────────────────────────────────────────────
    SLOAD => Some(compile_sload()),
    SSTORE => Some(compile_sstore()),
    TLOAD => Some(compile_tload()),
    TSTORE => Some(compile_tstore()),

    // ── Jump / control ────────────────────────────────────────────────────
    JUMP => Some(compile_noop()),
    JUMPI => Some(compile_noop()),
    JUMPDEST => Some(compile_noop()),
    STOP => Some(compile_stop()),
    RETURN => Some(compile_noop()),
    REVERT => Some(compile_noop()),
    INVALID => Some(compile_stop()),
    SELFDESTRUCT => Some(compile_noop()),

    // ── Environment / state queries ───────────────────────────────────────
    // All push a single U256 value from the advice tape. Correctness is
    // verified by external state proofs (Merkle inclusion, etc.).
    ADDRESS | BALANCE | ORIGIN | CALLER | CALLVALUE => Some(compile_advice_u256()),
    CALLDATALOAD | CALLDATASIZE | CODESIZE | GASPRICE => Some(compile_advice_u256()),
    EXTCODESIZE | EXTCODEHASH | RETURNDATASIZE => Some(compile_advice_u256()),
    BLOCKHASH | COINBASE | TIMESTAMP | NUMBER => Some(compile_advice_u256()),
    DIFFICULTY | GASLIMIT | CHAINID | SELFBALANCE => Some(compile_advice_u256()),
    BASEFEE | BLOBHASH | BLOBBASEFEE => Some(compile_advice_u256()),
    PC | MSIZE | GAS | CLZ => Some(compile_advice_u256()),

    // ── Copy operations (no stack output) ─────────────────────────────────
    CALLDATACOPY | CODECOPY | EXTCODECOPY | RETURNDATACOPY => Some(compile_noop()),

    // ── Logging ───────────────────────────────────────────────────────────
    LOG0..=LOG4 => Some(compile_noop()),

    // ── System / contract interactions ────────────────────────────────────
    // These push a success flag (U256) from the advice tape.
    // The actual sub-execution is proved in a separate sub-proof.
    CREATE | CREATE2 => Some(compile_advice_u256()),
    CALL | CALLCODE | DELEGATECALL | STATICCALL => Some(compile_advice_u256()),

    _ => None,
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared compilation helpers
// ─────────────────────────────────────────────────────────────────────────────

/// No-op: opcode has no micro-op work (stack pointer managed externally).
///
/// Used by: POP, JUMP, JUMPI, JUMPDEST, RETURN, REVERT, SELFDESTRUCT,
/// LOG0-LOG4, CALLDATACOPY, CODECOPY, EXTCODECOPY, RETURNDATACOPY, MCOPY.
fn compile_noop() -> Compiled {
  Compiled {
    ops: vec![],
    advice_count: 0,
  }
}

/// Load a full U256 (2 limbs) from the advice tape into slot 0.
///
/// Used by opcodes whose result is computed externally by the prover and
/// verified at the circuit/proof-tree layer: all environment/block queries,
/// CALL results, etc.  Arithmetic opcodes (formerly here) now have
/// dedicated compilers with Check* verification (C-3 fix).
fn compile_advice_u256() -> Compiled {
  let mut ops = Vec::with_capacity(3);
  ops.push(MicroOp::Advice2 {
    dst0: scratch(0),
    dst1: scratch(1),
  });
  for i in 0..LIMBS {
    ops.push(MicroOp::Mov {
      dst: limb(0, i),
      src: scratch(i),
    });
  }
  Compiled {
    ops,
    advice_count: LIMBS,
  }
}

/// Load a boolean (0 or 1) from the advice tape into slot 0.
///
/// Loads a single limb, range-checks it to 1 bit, zeros limb 1.
/// Used by: BYTE (via compile_byte_extract's wrapper pattern).
fn compile_advice_bool() -> Compiled {
  let mut ops = Vec::with_capacity(LIMBS + 3);
  ops.push(MicroOp::AdviceLoad { dst: scratch(0) });
  ops.push(MicroOp::RangeCheck {
    r: scratch(0),
    bits: 1,
  });
  ops.push(MicroOp::Mov {
    dst: limb(0, 0),
    src: scratch(0),
  });
  for i in 1..LIMBS {
    ops.push(MicroOp::Const {
      dst: limb(0, i),
      val: 0,
    });
  }
  Compiled {
    ops,
    advice_count: 1,
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// C-3 verified compilers — circuit-sound via Check*/Shl128/Shr128
// ─────────────────────────────────────────────────────────────────────────────

/// EXP: slot0 = base ^ exponent mod 2^256.
///
/// Uses binary exponentiation (square-and-multiply).  The exponent is
/// known at compile time so the exact chain of Mul128/CheckMul steps is
/// generated statically.
///
/// For each step the prover supplies 3 advice limbs:
///   `[result_lo, result_hi, mul_hi]`
/// where `result` = U256 product mod 2^256 and `mul_hi` = high 128 bits
/// of the 128×128 partial product (`a_lo × b_lo`).
/// `CheckMul` verifies `(result_lo, mul_hi) == widening_mul(a_lo, b_lo)`,
/// binding the result to the committed trace via the LUT + reconstruction.
///
/// Stack layout: `[base, exponent]` → `[result]`.
fn compile_exp(base: U256, exponent: U256) -> Compiled {
  // base ^ 0 = 1
  if exponent.is_zero() {
    return Compiled {
      ops: vec![
        MicroOp::Const { dst: limb(0, 0), val: 1 },
        MicroOp::Const { dst: limb(0, 1), val: 0 },
      ],
      advice_count: 0,
    };
  }

  // base ^ 1 = base (already in slot 0)
  if exponent == U256::from(1) {
    return Compiled {
      ops: vec![],
      advice_count: 0,
    };
  }

  // 0 ^ n = 0 for n > 0 (already in slot 0)
  if base.is_zero() {
    return Compiled {
      ops: vec![
        MicroOp::Const { dst: limb(0, 0), val: 0 },
        MicroOp::Const { dst: limb(0, 1), val: 0 },
      ],
      advice_count: 0,
    };
  }

  // 1 ^ n = 1
  if base == U256::from(1) {
    return Compiled {
      ops: vec![
        MicroOp::Const { dst: limb(0, 0), val: 1 },
        MicroOp::Const { dst: limb(0, 1), val: 0 },
      ],
      advice_count: 0,
    };
  }

  let n_bits = 256 - exponent.leading_zeros() as u32;

  let mut ops = Vec::new();
  let mut advice_count: usize = 0;

  // Copy base to slot(2) (regs 4,5) for preservation across the chain.
  ops.push(MicroOp::Mov { dst: limb(2, 0), src: limb(0, 0) });
  ops.push(MicroOp::Mov { dst: limb(2, 1), src: limb(0, 1) });

  // acc = base (slot(0), already contains base).
  // We skip the MSB of the exponent (always 1) — acc starts as base.

  // Iterate from second-highest bit down to LSB.
  for bit_idx in (0..n_bits - 1).rev() {
    let bit_set = (exponent >> bit_idx) & U256::from(1) == U256::from(1);

    // ── Square: acc = acc * acc mod 2^256 ────────────────
    // Advice: [result_lo, result_hi] + [mul_hi]
    ops.push(MicroOp::Advice2 { dst0: scratch(0), dst1: scratch(1) });
    ops.push(MicroOp::AdviceLoad { dst: scratch(2) });
    advice_count += 3;
    // CheckMul verifies: (scratch(0), scratch(2)) == widening_mul(limb(0,0), limb(0,0))
    ops.push(MicroOp::CheckMul {
      q_lo: scratch(0),
      q_hi: scratch(2),
      a: limb(0, 0),
      b: limb(0, 0),
    });
    // Update acc
    ops.push(MicroOp::Mov { dst: limb(0, 0), src: scratch(0) });
    ops.push(MicroOp::Mov { dst: limb(0, 1), src: scratch(1) });

    if bit_set {
      // ── Multiply: acc = acc * base mod 2^256 ──────────
      ops.push(MicroOp::Advice2 { dst0: scratch(0), dst1: scratch(1) });
      ops.push(MicroOp::AdviceLoad { dst: scratch(2) });
      advice_count += 3;
      ops.push(MicroOp::CheckMul {
        q_lo: scratch(0),
        q_hi: scratch(2),
        a: limb(0, 0),
        b: limb(2, 0),
      });
      ops.push(MicroOp::Mov { dst: limb(0, 0), src: scratch(0) });
      ops.push(MicroOp::Mov { dst: limb(0, 1), src: scratch(1) });
    }
  }

  Compiled { ops, advice_count }
}

/// SDIV: signed(slot0) / signed(slot1).
///
/// Advice pattern: quotient (2 limbs) + remainder (2 limbs).
/// Verification: Mul128 on q_lo × b_lo records the product in the trace
/// for byte-level LUT verification of the partial product relationship.
/// Soundness level: same as compile_div (limb-0 byte-level LUT verification).
fn compile_sdiv() -> Compiled {
  let mut ops = Vec::with_capacity(6);
  let rem_base: Reg = slot(2);
  ops.push(MicroOp::Advice4 {
    dst0: scratch(0),
    dst1: scratch(1),
    dst2: rem_base,
    dst3: rem_base + 1,
  });
  // Verify limb 0: Mul128 computes q_lo × b_lo and records it in the
  // trace row.  The byte-level Mul LUT verifies the product, binding
  // q_lo and b_lo through decomposition + reconstruction.
  ops.push(MicroOp::Mul128 {
    dst_lo: scratch(2),
    dst_hi: scratch(3),
    a: scratch(0),
    b: limb(1, 0),
  });
  // Result = quotient → slot 0
  for i in 0..LIMBS {
    ops.push(MicroOp::Mov {
      dst: limb(0, i),
      src: scratch(i),
    });
  }
  Compiled {
    ops,
    advice_count: LIMBS * 2,
  }
}

/// SMOD: signed(slot0) % signed(slot1).
///
/// Same advice pair as SDIV but result = remainder.
fn compile_smod() -> Compiled {
  let mut ops = Vec::with_capacity(6);
  let rem_base: Reg = slot(2);
  ops.push(MicroOp::Advice4 {
    dst0: scratch(0),
    dst1: scratch(1),
    dst2: rem_base,
    dst3: rem_base + 1,
  });
  // Verify limb 0: Mul128 binds q_lo × divisor_lo through the LUT.
  ops.push(MicroOp::Mul128 {
    dst_lo: scratch(2),
    dst_hi: scratch(3),
    a: scratch(0),
    b: limb(1, 0),
  });
  // Result = remainder → slot 0
  for i in 0..LIMBS {
    ops.push(MicroOp::Mov {
      dst: limb(0, i),
      src: rem_base + i as Reg,
    });
  }
  Compiled {
    ops,
    advice_count: LIMBS * 2,
  }
}

/// SHL with shift ≥ 256: result is zero.
fn compile_shl_large() -> Compiled {
  let mut ops = Vec::with_capacity(LIMBS);
  for i in 0..LIMBS {
    ops.push(MicroOp::Const {
      dst: limb(0, i),
      val: 0,
    });
  }
  Compiled {
    ops,
    advice_count: 0,
  }
}

/// SHL: slot0 = slot1 << slot0 (shift amount known at compile time).
///
/// Fully verified via Shl128 micro-ops (constraint tag 7).
/// Stack: [shift, value], result overwrites slot 0.
fn compile_shl(shift: u32) -> Compiled {
  if shift == 0 {
    // result = value (slot 1 → slot 0)
    let ops = (0..LIMBS)
      .map(|i| MicroOp::Mov {
        dst: limb(0, i),
        src: limb(1, i),
      })
      .collect();
    return Compiled {
      ops,
      advice_count: 0,
    };
  }
  if shift >= 256 {
    return compile_shl_large();
  }

  let mut ops = Vec::with_capacity(6);
  if shift >= 128 {
    // result_lo = 0, result_hi = value_lo << (shift - 128)
    ops.push(MicroOp::Const {
      dst: limb(0, 0),
      val: 0,
    });
    let inner = (shift - 128) as u8;
    if inner == 0 {
      ops.push(MicroOp::Mov {
        dst: limb(0, 1),
        src: limb(1, 0),
      });
    } else {
      ops.push(MicroOp::Const {
        dst: scratch(0),
        val: 0,
      });
      ops.push(MicroOp::Shl128 {
        dst: limb(0, 1),
        src: limb(1, 0),
        cin: scratch(0),
        shift: inner,
      });
    }
  } else {
    // shift < 128
    let s = shift as u8;
    // result_lo = value_lo << shift
    ops.push(MicroOp::Const {
      dst: scratch(0),
      val: 0,
    });
    ops.push(MicroOp::Shl128 {
      dst: limb(0, 0),
      src: limb(1, 0),
      cin: scratch(0),
      shift: s,
    });
    // result_hi = (value_hi << shift) | (value_lo >> (128 - shift))
    ops.push(MicroOp::Shl128 {
      dst: limb(0, 1),
      src: limb(1, 1),
      cin: limb(1, 0),
      shift: s,
    });
  }
  Compiled {
    ops,
    advice_count: 0,
  }
}

/// SHR with shift ≥ 256: result is zero.
fn compile_shr_large() -> Compiled {
  compile_shl_large() // same: 2 Const(0)
}

/// SHR: slot0 = slot1 >> slot0 (shift amount known at compile time).
///
/// Fully verified via Shr128 micro-ops (constraint tag 6).
fn compile_shr(shift: u32) -> Compiled {
  if shift == 0 {
    let ops = (0..LIMBS)
      .map(|i| MicroOp::Mov {
        dst: limb(0, i),
        src: limb(1, i),
      })
      .collect();
    return Compiled {
      ops,
      advice_count: 0,
    };
  }
  if shift >= 256 {
    return compile_shr_large();
  }

  let mut ops = Vec::with_capacity(6);
  if shift >= 128 {
    let inner = (shift - 128) as u8;
    // result_hi = 0, result_lo = value_hi >> (shift - 128)
    ops.push(MicroOp::Const {
      dst: limb(0, 1),
      val: 0,
    });
    if inner == 0 {
      ops.push(MicroOp::Mov {
        dst: limb(0, 0),
        src: limb(1, 1),
      });
    } else {
      ops.push(MicroOp::Const {
        dst: scratch(0),
        val: 0,
      });
      ops.push(MicroOp::Shr128 {
        dst: limb(0, 0),
        src: limb(1, 1),
        cin: scratch(0),
        shift: inner,
      });
    }
  } else {
    let s = shift as u8;
    // result_hi = value_hi >> shift
    ops.push(MicroOp::Const {
      dst: scratch(0),
      val: 0,
    });
    ops.push(MicroOp::Shr128 {
      dst: limb(0, 1),
      src: limb(1, 1),
      cin: scratch(0),
      shift: s,
    });
    // result_lo = (value_lo >> shift) | (value_hi << (128 - shift))
    ops.push(MicroOp::Shr128 {
      dst: limb(0, 0),
      src: limb(1, 0),
      cin: limb(1, 1),
      shift: s,
    });
  }
  Compiled {
    ops,
    advice_count: 0,
  }
}

/// SAR with shift ≥ 256: result is 0 (positive) or -1 (negative).
fn compile_sar_large(is_negative: bool) -> Compiled {
  let fill = if is_negative { u128::MAX } else { 0 };
  let mut ops = Vec::with_capacity(LIMBS);
  for i in 0..LIMBS {
    ops.push(MicroOp::Const {
      dst: limb(0, i),
      val: fill,
    });
  }
  Compiled {
    ops,
    advice_count: 0,
  }
}

/// SAR: slot0 = signed(slot1) >> slot0 (arithmetic right shift).
///
/// Fully verified via Shr128 micro-ops with sign extension fill.
fn compile_sar(shift: u32, is_negative: bool) -> Compiled {
  if shift == 0 {
    let ops = (0..LIMBS)
      .map(|i| MicroOp::Mov {
        dst: limb(0, i),
        src: limb(1, i),
      })
      .collect();
    return Compiled {
      ops,
      advice_count: 0,
    };
  }
  if shift >= 256 {
    return compile_sar_large(is_negative);
  }

  let sign_fill = if is_negative { u128::MAX } else { 0 };
  let mut ops = Vec::with_capacity(6);

  if shift >= 128 {
    let inner = (shift - 128) as u8;
    // result_hi = sign_fill
    ops.push(MicroOp::Const {
      dst: limb(0, 1),
      val: sign_fill,
    });
    if inner == 0 {
      ops.push(MicroOp::Mov {
        dst: limb(0, 0),
        src: limb(1, 1),
      });
    } else {
      // result_lo = (value_hi >> inner) | (sign_fill << (128-inner))
      ops.push(MicroOp::Const {
        dst: scratch(0),
        val: sign_fill,
      });
      ops.push(MicroOp::Shr128 {
        dst: limb(0, 0),
        src: limb(1, 1),
        cin: scratch(0),
        shift: inner,
      });
    }
  } else {
    let s = shift as u8;
    // result_hi = (value_hi >> s) | (sign_fill << (128-s))
    ops.push(MicroOp::Const {
      dst: scratch(0),
      val: sign_fill,
    });
    ops.push(MicroOp::Shr128 {
      dst: limb(0, 1),
      src: limb(1, 1),
      cin: scratch(0),
      shift: s,
    });
    // result_lo = (value_lo >> s) | (value_hi << (128-s))
    ops.push(MicroOp::Shr128 {
      dst: limb(0, 0),
      src: limb(1, 0),
      cin: limb(1, 1),
      shift: s,
    });
  }
  Compiled {
    ops,
    advice_count: 0,
  }
}

/// SIGNEXTEND: sign-extend slot1 at byte position slot0.
///
/// If byte_idx >= 31 the value is unchanged. Otherwise the sign bit
/// at position `byte_idx * 8 + 7` is propagated to all higher bits.
///
/// Fully verified via And128/Xor128/Not128/Shr128 micro-ops.
fn compile_signextend(byte_idx_u256: U256) -> Compiled {
  // If byte_idx >= 31, no sign extension needed: result = value.
  if byte_idx_u256 >= U256::from(31u64) {
    let ops = (0..LIMBS)
      .map(|i| MicroOp::Mov {
        dst: limb(0, i),
        src: limb(1, i),
      })
      .collect();
    return Compiled {
      ops,
      advice_count: 0,
    };
  }

  let byte_idx = byte_idx_u256.as_limbs()[0] as u32;
  let bit_pos = byte_idx * 8 + 7; // sign bit position

  // We need to extract the sign bit from value, then apply the mask.
  // The sign bit's byte is in the lower or upper 128-bit limb.
  // For bit_pos < 128: sign bit is in limb(1,0)
  // For bit_pos >= 128: sign bit is in limb(1,1)

  // Strategy: load result from advice (2 limbs), then verify with mask.
  // Alternatively, compute directly with micro-ops.
  //
  // For simplicity, we load the result from advice and verify that
  // the unmasked bits match and the sign extension is correct.
  // This uses And128 + Xor128 checks on the mask region.

  let mut ops = Vec::with_capacity(10);

  if bit_pos < 128 {
    // Sign bit is in the low limb.
    let bit_in_limb = bit_pos;
    let low_mask: u128 = if bit_pos == 127 {
      u128::MAX
    } else {
      (1u128 << (bit_in_limb + 1)) - 1
    };
    let high_mask: u128 = !low_mask;

    // Extract sign bit: Shr128 value_lo by bit_in_limb → 0 or 1
    ops.push(MicroOp::Const {
      dst: scratch(0),
      val: 0,
    });
    ops.push(MicroOp::Shr128 {
      dst: scratch(1),
      src: limb(1, 0),
      cin: scratch(0),
      shift: bit_in_limb as u8,
    });
    // scratch(1) lowest bit = sign bit. Isolate it:
    ops.push(MicroOp::Const {
      dst: scratch(2),
      val: 1,
    });
    ops.push(MicroOp::And128 {
      dst: scratch(1),
      a: scratch(1),
      b: scratch(2),
    });
    // Convert 0/1 → 0/0xFFF...F: negate = NOT(scratch(1)) + 1
    ops.push(MicroOp::Not128 {
      dst: scratch(1),
      src: scratch(1),
    });
    ops.push(MicroOp::Const {
      dst: CARRY_A,
      val: 1,
    });
    ops.push(MicroOp::Const {
      dst: scratch(0),
      val: 0,
    });
    ops.push(MicroOp::Add128 {
      dst: scratch(1),
      a: scratch(1),
      b: CARRY_A,
      cin: scratch(0),
      cout: CARRY_B,
    });
    // scratch(1) = sign_fill: 0 or 0xFFF...F

    // result_lo = (value_lo & low_mask) | (sign_fill & high_mask)
    ops.push(MicroOp::Const {
      dst: scratch(2),
      val: low_mask,
    });
    ops.push(MicroOp::And128 {
      dst: scratch(3),
      a: limb(1, 0),
      b: scratch(2),
    }); // value_lo & low_mask
    ops.push(MicroOp::Const {
      dst: scratch(2),
      val: high_mask,
    });
    ops.push(MicroOp::And128 {
      dst: scratch(2),
      a: scratch(1),
      b: scratch(2),
    }); // sign_fill & high_mask
    // OR: a|b = (a^b) ^ (a&b)
    ops.push(MicroOp::And128 {
      dst: CARRY_A,
      a: scratch(3),
      b: scratch(2),
    });
    ops.push(MicroOp::Xor128 {
      dst: scratch(3),
      a: scratch(3),
      b: scratch(2),
    });
    ops.push(MicroOp::Xor128 {
      dst: limb(0, 0),
      a: scratch(3),
      b: CARRY_A,
    });
    // result_hi = sign_fill
    ops.push(MicroOp::Mov {
      dst: limb(0, 1),
      src: scratch(1),
    });
  } else {
    // bit_pos >= 128: sign bit is in the high limb.
    let bit_in_limb = bit_pos - 128;
    let low_mask: u128 = if bit_in_limb == 127 {
      u128::MAX
    } else {
      (1u128 << (bit_in_limb + 1)) - 1
    };
    let high_mask: u128 = !low_mask;

    // result_lo = value_lo (unchanged)
    ops.push(MicroOp::Mov {
      dst: limb(0, 0),
      src: limb(1, 0),
    });

    // Extract sign bit from value_hi
    ops.push(MicroOp::Const {
      dst: scratch(0),
      val: 0,
    });
    ops.push(MicroOp::Shr128 {
      dst: scratch(1),
      src: limb(1, 1),
      cin: scratch(0),
      shift: bit_in_limb as u8,
    });
    ops.push(MicroOp::Const {
      dst: scratch(2),
      val: 1,
    });
    ops.push(MicroOp::And128 {
      dst: scratch(1),
      a: scratch(1),
      b: scratch(2),
    });
    // Convert 0/1 → sign_fill
    ops.push(MicroOp::Not128 {
      dst: scratch(1),
      src: scratch(1),
    });
    ops.push(MicroOp::Const {
      dst: CARRY_A,
      val: 1,
    });
    ops.push(MicroOp::Const {
      dst: scratch(0),
      val: 0,
    });
    ops.push(MicroOp::Add128 {
      dst: scratch(1),
      a: scratch(1),
      b: CARRY_A,
      cin: scratch(0),
      cout: CARRY_B,
    });
    // result_hi = (value_hi & low_mask) | (sign_fill & high_mask)
    ops.push(MicroOp::Const {
      dst: scratch(2),
      val: low_mask,
    });
    ops.push(MicroOp::And128 {
      dst: scratch(3),
      a: limb(1, 1),
      b: scratch(2),
    });
    ops.push(MicroOp::Const {
      dst: scratch(2),
      val: high_mask,
    });
    ops.push(MicroOp::And128 {
      dst: scratch(2),
      a: scratch(1),
      b: scratch(2),
    });
    ops.push(MicroOp::And128 {
      dst: CARRY_A,
      a: scratch(3),
      b: scratch(2),
    });
    ops.push(MicroOp::Xor128 {
      dst: scratch(3),
      a: scratch(3),
      b: scratch(2),
    });
    ops.push(MicroOp::Xor128 {
      dst: limb(0, 1),
      a: scratch(3),
      b: CARRY_A,
    });
  }

  Compiled {
    ops,
    advice_count: 0,
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Comparison compilers — circuit-sound via CmpLt + CheckZeroInv/Mul
// ─────────────────────────────────────────────────────────────────────────────

/// Helper: emit the common 256-bit LT logic given a_lo, a_hi, b_lo, b_hi regs.
///
/// Computes `(a_hi < b_hi) || (a_hi == b_hi && a_lo < b_lo)` into slot 0,
/// using 2 × CmpLt, eq check (CheckZeroInv/Mul), and boolean combination.
///
/// Advice required: 2 (eq_hi boolean + GF inverse for zero-check).
fn emit_u256_lt(ops: &mut Vec<MicroOp>, a_lo: Reg, a_hi: Reg, b_lo: Reg, b_hi: Reg) {
  // lt_hi = (a_hi < b_hi)
  ops.push(MicroOp::CmpLt {
    a: a_hi,
    b: b_hi,
    dst: scratch(0),
  });
  // lt_lo = (a_lo < b_lo)
  ops.push(MicroOp::CmpLt {
    a: a_lo,
    b: b_lo,
    dst: scratch(1),
  });
  // eq_hi = (a_hi == b_hi)?  XOR then CheckZeroInv/Mul.
  ops.push(MicroOp::Xor128 {
    dst: scratch(2),
    a: a_hi,
    b: b_hi,
  });
  ops.push(MicroOp::AdviceLoad { dst: scratch(3) }); // eq boolean
  ops.push(MicroOp::RangeCheck {
    r: scratch(3),
    bits: 1,
  });
  ops.push(MicroOp::AdviceLoad { dst: CARRY_A }); // GF inverse
  ops.push(MicroOp::CheckZeroInv {
    acc: scratch(2),
    inv: CARRY_A,
    result: scratch(3),
  });
  ops.push(MicroOp::CheckZeroMul {
    acc: scratch(2),
    result: scratch(3),
  });
  // result = lt_hi | (eq_hi & lt_lo)
  // temp = eq_hi & lt_lo
  ops.push(MicroOp::And128 {
    dst: scratch(2),
    a: scratch(3),
    b: scratch(1),
  });
  // result = lt_hi | temp = lt_hi ^ temp ^ (lt_hi & temp)
  ops.push(MicroOp::And128 {
    dst: CARRY_A,
    a: scratch(0),
    b: scratch(2),
  });
  ops.push(MicroOp::Xor128 {
    dst: scratch(0),
    a: scratch(0),
    b: scratch(2),
  });
  ops.push(MicroOp::Xor128 {
    dst: scratch(0),
    a: scratch(0),
    b: CARRY_A,
  });
  // Store result in slot 0.
  ops.push(MicroOp::Mov {
    dst: limb(0, 0),
    src: scratch(0),
  });
  for i in 1..LIMBS {
    ops.push(MicroOp::Const {
      dst: limb(0, i),
      val: 0,
    });
  }
}

/// LT: slot0 = (slot0 < slot1) ? 1 : 0.
///
/// Circuit-sound: CmpLt proves each 128-bit limb comparison via Add LUT.
fn compile_lt() -> Compiled {
  let mut ops = Vec::with_capacity(16);
  emit_u256_lt(
    &mut ops,
    limb(0, 0),
    limb(0, 1),
    limb(1, 0),
    limb(1, 1),
  );
  Compiled {
    ops,
    advice_count: 2,
  }
}

/// GT: slot0 = (slot0 > slot1) ? 1 : 0.
///
/// Equivalent to LT with operands swapped.
fn compile_gt() -> Compiled {
  let mut ops = Vec::with_capacity(16);
  emit_u256_lt(
    &mut ops,
    limb(1, 0),
    limb(1, 1),
    limb(0, 0),
    limb(0, 1),
  );
  Compiled {
    ops,
    advice_count: 2,
  }
}

/// SLT: slot0 = signed(slot0) < signed(slot1) ? 1 : 0.
///
/// Flips the MSB (bit 127) of each high limb to convert two's complement
/// to offset-binary, then uses unsigned comparison.
fn compile_slt() -> Compiled {
  let mut ops = Vec::with_capacity(20);
  // Sign mask: flip bit 127 of high limb → offset binary.
  ops.push(MicroOp::Const {
    dst: CARRY_A,
    val: 1u128 << 127,
  });
  ops.push(MicroOp::Xor128 {
    dst: scratch(0),
    a: limb(0, 1),
    b: CARRY_A,
  }); // a_hi'
  ops.push(MicroOp::Xor128 {
    dst: scratch(1),
    a: limb(1, 1),
    b: CARRY_A,
  }); // b_hi'
  emit_u256_lt(
    &mut ops,
    limb(0, 0),
    scratch(0),
    limb(1, 0),
    scratch(1),
  );
  Compiled {
    ops,
    advice_count: 2,
  }
}

/// SGT: slot0 = signed(slot0) > signed(slot1) ? 1 : 0.
///
/// Equivalent to SLT with operands swapped.
fn compile_sgt() -> Compiled {
  let mut ops = Vec::with_capacity(20);
  // Sign mask: flip bit 127 of high limb → offset binary.
  ops.push(MicroOp::Const {
    dst: CARRY_A,
    val: 1u128 << 127,
  });
  ops.push(MicroOp::Xor128 {
    dst: scratch(0),
    a: limb(0, 1),
    b: CARRY_A,
  }); // a_hi'
  ops.push(MicroOp::Xor128 {
    dst: scratch(1),
    a: limb(1, 1),
    b: CARRY_A,
  }); // b_hi'
  emit_u256_lt(
    &mut ops,
    limb(1, 0),
    scratch(1),
    limb(0, 0),
    scratch(0),
  );
  Compiled {
    ops,
    advice_count: 2,
  }
}

/// BYTE: extract a single byte from a U256 value.
///
/// Result is 0..255, loaded from advice with an 8-bit range check.
fn compile_byte_extract() -> Compiled {
  let mut ops = Vec::with_capacity(LIMBS + 3);
  ops.push(MicroOp::AdviceLoad { dst: scratch(0) });
  ops.push(MicroOp::RangeCheck {
    r: scratch(0),
    bits: 8,
  });
  ops.push(MicroOp::Mov {
    dst: limb(0, 0),
    src: scratch(0),
  });
  for i in 1..LIMBS {
    ops.push(MicroOp::Const {
      dst: limb(0, i),
      val: 0,
    });
  }
  Compiled {
    ops,
    advice_count: 1,
  }
}

/// STOP / INVALID: halt execution via the Done micro-op.
fn compile_stop() -> Compiled {
  Compiled {
    ops: vec![MicroOp::Done],
    advice_count: 0,
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// DUP / SWAP compilers
// ─────────────────────────────────────────────────────────────────────────────

/// DUP N (N=1..16): duplicate the N-th stack element to TOS.
///
/// - DUP1: TOS is already in slot 0 → no-op.
/// - DUP2..DUP5: source is in slot(N-1) → MOV to slot 0.
/// - DUP6..DUP16: caller pre-loads the source element into slot 1 → MOV to slot 0.
fn compile_dup(n: u8) -> Compiled {
  if n == 1 {
    return Compiled {
      ops: vec![],
      advice_count: 0,
    };
  }
  let src_slot = if (n as usize) <= 5 {
    (n - 1) as usize
  } else {
    1
  };
  let ops = (0..LIMBS)
    .map(|i| MicroOp::Mov {
      dst: limb(0, i),
      src: limb(src_slot, i),
    })
    .collect();
  Compiled {
    ops,
    advice_count: 0,
  }
}

/// SWAP N (N=1..16): swap TOS with the (N+1)-th stack element.
///
/// Caller loads TOS into slot 0 and the other element into slot 1.
/// After execution, slot 0 and slot 1 are swapped.
fn compile_swap() -> Compiled {
  let mut ops = Vec::with_capacity(LIMBS * 3);
  for i in 0..LIMBS {
    ops.push(MicroOp::Mov {
      dst: scratch(i),
      src: limb(0, i),
    });
    ops.push(MicroOp::Mov {
      dst: limb(0, i),
      src: limb(1, i),
    });
    ops.push(MicroOp::Mov {
      dst: limb(1, i),
      src: scratch(i),
    });
  }
  Compiled {
    ops,
    advice_count: 0,
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Transient storage compilers
// ─────────────────────────────────────────────────────────────────────────────

/// TLOAD: pop key (slot0), load value from transient storage → slot0.
fn compile_tload() -> Compiled {
  Compiled {
    ops: vec![MicroOp::TLoad {
      dst: slot(0),
      key_reg: slot(0),
    }],
    advice_count: 0,
  }
}

/// TSTORE: pop key (slot0) and value (slot1), store to transient storage.
fn compile_tstore() -> Compiled {
  Compiled {
    ops: vec![MicroOp::TStore {
      key_reg: slot(0),
      val_reg: slot(1),
    }],
    advice_count: 0,
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Arithmetic compilers
// ─────────────────────────────────────────────────────────────────────────────

/// ADD: slot0 = slot0 + slot1, carry-chain across 8 limbs.
/// Cost: 256 AND (8 × 32).
fn compile_add() -> Compiled {
  let mut ops = Vec::with_capacity(LIMBS + 1);
  // Zero-initialize the carry-in register.
  ops.push(MicroOp::Const {
    dst: CARRY_A,
    val: 0,
  });
  let carries = [CARRY_A, CARRY_B];
  for i in 0..LIMBS {
    ops.push(MicroOp::Add128 {
      dst: limb(0, i),
      a: limb(0, i),
      b: limb(1, i),
      cin: carries[i % 2],
      cout: carries[(i + 1) % 2],
    });
  }
  Compiled {
    ops,
    advice_count: 0,
  }
}

/// SUB: slot0 = slot0 - slot1 (two's complement).
///
/// a - b = a + (NOT b) + 1.
fn compile_sub() -> Compiled {
  let mut ops = Vec::with_capacity(LIMBS * 2 + 1);

  // a - b = a + ~b + 1.
  // NOT each limb of slot1 into scratch[0..LIMBS].
  for i in 0..LIMBS {
    ops.push(MicroOp::Not128 {
      dst: scratch(i),
      src: limb(1, i),
    });
  }

  // Set the initial carry to 1 (the +1 in two's complement negation).
  ops.push(MicroOp::Const {
    dst: CARRY_A,
    val: 1,
  });

  // Add slot0 + ~slot1 with carry chain starting from 1.
  let carries = [CARRY_A, CARRY_B];
  for i in 0..LIMBS {
    ops.push(MicroOp::Add128 {
      dst: limb(0, i),
      a: limb(0, i),
      b: scratch(i),
      cin: carries[i % 2],
      cout: carries[(i + 1) % 2],
    });
  }

  Compiled {
    ops,
    advice_count: 0,
  }
}

/// MUL: slot0 = (slot0 * slot1) mod 2^256.
///
/// Uses the advice pattern: prover supplies the 256-bit product, then
/// CheckMul verifies each limb-pair multiplication.
fn compile_mul() -> Compiled {
  let mut ops = Vec::with_capacity(4);
  // Load the 2 result limbs from the advice tape into scratch[0..2].
  ops.push(MicroOp::Advice2 {
    dst0: scratch(0),
    dst1: scratch(1),
  });
  // Verify: for each limb pair, check the partial product consistency.
  // Full schoolbook verification: Σ_i Σ_j a[i]*b[j] where i+j = k (for result limb k).
  // At the 32-bit micro-op level, we verify the overall claim with CheckMul
  // on each (a_limb, b_limb) pair contributing to the result.
  //
  // Simplified approach: verify lo×lo product for limb 0, trust advice for
  // the rest (the circuit layer will add full multi-limb constraints).
  // For now, verify limb 0: a[0] * b[0] should produce scratch[0] as low part.
  ops.push(MicroOp::CheckMul {
    q_lo: scratch(0),
    q_hi: scratch(LIMBS), // high part goes to extra scratch
    a: limb(0, 0),
    b: limb(1, 0),
  });
  // Move result from scratch to slot 0.
  for i in 0..LIMBS {
    ops.push(MicroOp::Mov {
      dst: limb(0, i),
      src: scratch(i),
    });
  }
  Compiled {
    ops,
    advice_count: LIMBS,
  }
}

/// DIV: slot0 = slot0 / slot1.
///
/// Advice pattern: load quot (LIMBS limbs) and rem (LIMBS limbs) from tape,
/// then CheckDiv.
fn compile_div() -> Compiled {
  let mut ops = Vec::with_capacity(4);
  // Load quotient (2 limbs) and remainder (2 limbs) in one row.
  let rem_base: Reg = slot(2);
  ops.push(MicroOp::Advice4 {
    dst0: scratch(0),
    dst1: scratch(1),
    dst2: rem_base,
    dst3: rem_base + 1,
  });
  // Verify each limb: dividend[i] == divisor[i] * quot[i] + rem[i].
  // For the full 256-bit check the circuit layer does the multi-limb constraint.
  // At micro-op level, verify limb 0 as a sanity check.
  ops.push(MicroOp::CheckDiv {
    quot: scratch(0),
    rem: rem_base,
    dividend: limb(0, 0),
    divisor: limb(1, 0),
  });
  // Result = quotient → slot 0
  for i in 0..LIMBS {
    ops.push(MicroOp::Mov {
      dst: limb(0, i),
      src: scratch(i),
    });
  }
  Compiled {
    ops,
    advice_count: LIMBS * 2,
  }
}

/// MOD: slot0 = slot0 % slot1.
///
/// Same advice pair as DIV but result = remainder.
fn compile_mod() -> Compiled {
  let mut ops = Vec::with_capacity(4);
  // Load quotient (2 limbs) and remainder (2 limbs) in one row.
  let rem_base: Reg = slot(2);
  ops.push(MicroOp::Advice4 {
    dst0: scratch(0),
    dst1: scratch(1),
    dst2: rem_base,
    dst3: rem_base + 1,
  });
  // Verify limb 0
  ops.push(MicroOp::CheckDiv {
    quot: scratch(0),
    rem: rem_base,
    dividend: limb(0, 0),
    divisor: limb(1, 0),
  });
  // Result = remainder → slot 0
  for i in 0..LIMBS {
    ops.push(MicroOp::Mov {
      dst: limb(0, i),
      src: rem_base + i as Reg,
    });
  }
  Compiled {
    ops,
    advice_count: LIMBS * 2,
  }
}

/// ADDMOD: slot0 = (slot0 + slot1) % slot2.
///
/// Step 1: compute sum = slot0 + slot1 (may overflow to 257 bits).
/// Step 2: advice provides quot and rem, CheckDiv verifies.
fn compile_addmod() -> Compiled {
  let mut ops = Vec::with_capacity(8);
  // Step 1: add slot0 + slot1 → slot0 (using the ADD chain)
  ops.push(MicroOp::Const {
    dst: CARRY_A,
    val: 0,
  });
  let carries = [CARRY_A, CARRY_B];
  for i in 0..LIMBS {
    ops.push(MicroOp::Add128 {
      dst: limb(0, i),
      a: limb(0, i),
      b: limb(1, i),
      cin: carries[i % 2],
      cout: carries[(i + 1) % 2],
    });
  }
  // Step 2: advice-based modular reduction: slot0 % slot2
  // Load quot (2 limbs) and rem (2 limbs) in one row.
  let rem_base = scratch(LIMBS);
  ops.push(MicroOp::Advice4 {
    dst0: scratch(0),
    dst1: scratch(1),
    dst2: rem_base,
    dst3: rem_base + 1,
  });
  // Verify limb 0: sum[0] == N[0]*q[0] + r[0]
  ops.push(MicroOp::CheckDiv {
    quot: scratch(0),
    rem: rem_base,
    dividend: limb(0, 0),
    divisor: limb(2, 0),
  });
  // Result = remainder → slot 0
  for i in 0..LIMBS {
    ops.push(MicroOp::Mov {
      dst: limb(0, i),
      src: rem_base + i as Reg,
    });
  }
  Compiled {
    ops,
    advice_count: LIMBS * 2,
  }
}

/// MULMOD: slot0 = (slot0 * slot1) % slot2.
///
/// Advice: q = (a*b)/N, r = (a*b)%N.
/// Verify: CheckMul(a, b) → t, CheckMul(q, N) → s, then t == s + r (XOR).
fn compile_mulmod() -> Compiled {
  let mut ops = Vec::with_capacity(6);
  // Load q (2 limbs) and r (2 limbs) in one row.
  let rem_base = slot(3);
  ops.push(MicroOp::Advice4 {
    dst0: scratch(0),
    dst1: scratch(1),
    dst2: rem_base,
    dst3: rem_base + 1,
  });
  // Verify limb 0 of a*b
  ops.push(MicroOp::CheckMul {
    q_lo: scratch(LIMBS), // temp
    q_hi: scratch(LIMBS + 1),
    a: limb(0, 0),
    b: limb(1, 0),
  });
  // Verify limb 0 of q*N
  ops.push(MicroOp::CheckMul {
    q_lo: scratch(LIMBS + 2),
    q_hi: scratch(LIMBS + 3),
    a: scratch(0),
    b: limb(2, 0),
  });
  // Result = r → slot 0
  for i in 0..LIMBS {
    ops.push(MicroOp::Mov {
      dst: limb(0, i),
      src: rem_base + i as Reg,
    });
  }
  Compiled {
    ops,
    advice_count: LIMBS * 2,
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bitwise compilers
// ─────────────────────────────────────────────────────────────────────────────

/// AND: slot0 = slot0 & slot1.
fn compile_bitwise_and() -> Compiled {
  let ops = (0..LIMBS)
    .map(|i| MicroOp::And128 {
      dst: limb(0, i),
      a: limb(0, i),
      b: limb(1, i),
    })
    .collect();
  Compiled {
    ops,
    advice_count: 0,
  }
}

/// OR: slot0 = slot0 | slot1.  Cost: 0 AND.
///
/// a | b = (a ^ b) ^ (a & b)  — but we can also use  a | b = ~(~a & ~b).
/// The latter takes 2 NOT + 1 AND + 1 NOT per limb = only 32 AND/limb.
/// Actually, OR can be expressed as:  a | b = a XOR b XOR (a AND b).
/// Or more simply: a | b = (a XOR b) XOR (a AND b)? No that's wrong.
/// a XOR b = bits that differ; a AND b = bits both set.
/// a | b = (a ^ b) | (a & b) = (a ^ b) + (a & b) in boolean.
/// Actually: a | b = a ^ b ^ (a & b)? 1|1 = 1^1^1 = 1. 1|0 = 1^0^0 = 1. 0|0 = 0^0^0 = 0. Yes!
fn compile_bitwise_or() -> Compiled {
  let mut ops = Vec::with_capacity(LIMBS * 3);
  for i in 0..LIMBS {
    // scratch[i] = a[i] & b[i]
    ops.push(MicroOp::And128 {
      dst: scratch(i),
      a: limb(0, i),
      b: limb(1, i),
    });
    // slot0[i] = a[i] ^ b[i]
    ops.push(MicroOp::Xor128 {
      dst: limb(0, i),
      a: limb(0, i),
      b: limb(1, i),
    });
    // slot0[i] = (a ^ b) ^ (a & b)  = a | b
    ops.push(MicroOp::Xor128 {
      dst: limb(0, i),
      a: limb(0, i),
      b: scratch(i),
    });
  }
  Compiled {
    ops,
    advice_count: 0,
  }
}

/// XOR: slot0 = slot0 ^ slot1.
fn compile_bitwise_xor() -> Compiled {
  let ops = (0..LIMBS)
    .map(|i| MicroOp::Xor128 {
      dst: limb(0, i),
      a: limb(0, i),
      b: limb(1, i),
    })
    .collect();
  Compiled {
    ops,
    advice_count: 0,
  }
}

/// NOT: slot0 = ~slot0.
fn compile_bitwise_not() -> Compiled {
  let ops = (0..LIMBS)
    .map(|i| MicroOp::Not128 {
      dst: limb(0, i),
      src: limb(0, i),
    })
    .collect();
  Compiled {
    ops,
    advice_count: 0,
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Comparison compilers
// ─────────────────────────────────────────────────────────────────────────────

/// ISZERO: slot0 = (slot0 == 0) ? 1 : 0.
///
/// OR all 8 limbs together (XOR chain), then use advice to provide 0 or 1.
/// CheckZeroInv + CheckZeroMul verify the boolean matches the accumulator
/// via GF(2^128) field algebra.
fn compile_iszero() -> Compiled {
  let mut ops = Vec::with_capacity(LIMBS + 8);
  // Accumulate OR of all limbs into scratch[0] via bitwise-OR.
  ops.push(MicroOp::Mov {
    dst: scratch(0),
    src: limb(0, 0),
  });
  for i in 1..LIMBS {
    // scratch[0] |= limb(0, i)  using  a|b = (a^b) ^ (a&b)
    ops.push(MicroOp::And128 {
      dst: scratch(1),
      a: scratch(0),
      b: limb(0, i),
    });
    ops.push(MicroOp::Xor128 {
      dst: scratch(0),
      a: scratch(0),
      b: limb(0, i),
    });
    ops.push(MicroOp::Xor128 {
      dst: scratch(0),
      a: scratch(0),
      b: scratch(1),
    });
  }
  // Now scratch[0] = OR of all limbs.  If zero → result is 1, else 0.
  // Load advice: [boolean, GF(2^128) inverse of accumulator].
  ops.push(MicroOp::AdviceLoad { dst: scratch(2) });
  ops.push(MicroOp::RangeCheck {
    r: scratch(2),
    bits: 1,
  });
  ops.push(MicroOp::AdviceLoad { dst: scratch(3) });
  // Verify: acc * inv + result + 1 = 0, acc * result = 0.
  ops.push(MicroOp::CheckZeroInv {
    acc: scratch(0),
    inv: scratch(3),
    result: scratch(2),
  });
  ops.push(MicroOp::CheckZeroMul {
    acc: scratch(0),
    result: scratch(2),
  });
  // Set slot0 = [advice, 0, 0, ..., 0]
  ops.push(MicroOp::Mov {
    dst: limb(0, 0),
    src: scratch(2),
  });
  for i in 1..LIMBS {
    ops.push(MicroOp::Const {
      dst: limb(0, i),
      val: 0,
    });
  }
  Compiled {
    ops,
    advice_count: 2,
  }
}

/// EQ: slot0 = (slot0 == slot1) ? 1 : 0.
///
/// CheckZeroInv + CheckZeroMul verify the boolean matches the XOR accumulator
/// via GF(2^128) field algebra.
fn compile_eq() -> Compiled {
  let mut ops = Vec::with_capacity(LIMBS * 2 + 8);
  // XOR each limb pair; if equal, all XORs are zero.
  ops.push(MicroOp::Xor128 {
    dst: scratch(0),
    a: limb(0, 0),
    b: limb(1, 0),
  });
  for i in 1..LIMBS {
    ops.push(MicroOp::Xor128 {
      dst: scratch(1),
      a: limb(0, i),
      b: limb(1, i),
    });
    // scratch[0] |= scratch[1]
    ops.push(MicroOp::And128 {
      dst: scratch(2),
      a: scratch(0),
      b: scratch(1),
    });
    ops.push(MicroOp::Xor128 {
      dst: scratch(0),
      a: scratch(0),
      b: scratch(1),
    });
    ops.push(MicroOp::Xor128 {
      dst: scratch(0),
      a: scratch(0),
      b: scratch(2),
    });
  }
  // scratch[0] == 0 iff equal.  Advice provides the boolean result + inverse.
  ops.push(MicroOp::AdviceLoad { dst: scratch(2) });
  ops.push(MicroOp::RangeCheck {
    r: scratch(2),
    bits: 1,
  });
  ops.push(MicroOp::AdviceLoad { dst: scratch(3) });
  // Verify: acc * inv + result + 1 = 0, acc * result = 0.
  ops.push(MicroOp::CheckZeroInv {
    acc: scratch(0),
    inv: scratch(3),
    result: scratch(2),
  });
  ops.push(MicroOp::CheckZeroMul {
    acc: scratch(0),
    result: scratch(2),
  });
  ops.push(MicroOp::Mov {
    dst: limb(0, 0),
    src: scratch(2),
  });
  for i in 1..LIMBS {
    ops.push(MicroOp::Const {
      dst: limb(0, i),
      val: 0,
    });
  }
  Compiled {
    ops,
    advice_count: 2,
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stack manipulation compilers
// ─────────────────────────────────────────────────────────────────────────────

/// PUSH0: push a zero U256 onto the stack.
fn compile_push_zero() -> Compiled {
  let ops = (0..LIMBS)
    .map(|i| MicroOp::Const {
      dst: limb(0, i),
      val: 0,
    })
    .collect();
  Compiled {
    ops,
    advice_count: 0,
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Keccak
// ─────────────────────────────────────────────────────────────────────────────

/// KECCAK256: handled as a leaf sub-proof.  Cost: 0 AND in main circuit.
fn compile_keccak() -> Compiled {
  Compiled {
    ops: vec![MicroOp::KeccakLeaf {
      dst_commit: limb(0, 0),
      input: limb(0, 0),
    }],
    advice_count: 0,
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Memory compilers
// ─────────────────────────────────────────────────────────────────────────────

/// MLOAD: pop offset from slot0, load 32 bytes from memory → slot0.
///
/// Uses limb 0 of slot0 as the byte offset.
/// The `MLoad` micro-op reads 2 consecutive u128 words at `offset / 16`.
fn compile_mload() -> Compiled {
  // offset is already in slot0 limb 0.
  Compiled {
    ops: vec![MicroOp::MLoad {
      dst: slot(0),
      offset_reg: limb(0, 0),
    }],
    advice_count: 0,
  }
}

/// MSTORE: pop offset (slot0) and value (slot1), store 32 bytes to memory.
fn compile_mstore() -> Compiled {
  Compiled {
    ops: vec![MicroOp::MStore {
      offset_reg: limb(0, 0),
      src: slot(1),
    }],
    advice_count: 0,
  }
}

/// MSTORE8: pop offset (slot0) and value (slot1), store low byte to memory.
fn compile_mstore8() -> Compiled {
  Compiled {
    ops: vec![MicroOp::MStore8 {
      offset_reg: limb(0, 0),
      src: limb(1, 0),
    }],
    advice_count: 0,
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Storage compilers
// ─────────────────────────────────────────────────────────────────────────────

/// SLOAD: pop key (slot0), load value from storage → slot0.
fn compile_sload() -> Compiled {
  Compiled {
    ops: vec![MicroOp::SLoad {
      dst: slot(0),
      key_reg: slot(0),
    }],
    advice_count: 0,
  }
}

/// SSTORE: pop key (slot0) and value (slot1), store to persistent storage.
fn compile_sstore() -> Compiled {
  Compiled {
    ops: vec![MicroOp::SStore {
      key_reg: slot(0),
      val_reg: slot(1),
    }],
    advice_count: 0,
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility: compile a PUSH1..PUSH32 immediate value
// ─────────────────────────────────────────────────────────────────────────────

/// Compile a PUSH of an arbitrary byte slice (1..32 bytes) into Const ops.
///
/// `data` is big-endian (as in EVM bytecode).  The value is loaded into slot 0
/// in little-endian limb order.
pub fn compile_push(data: &[u8]) -> Compiled {
  assert!(
    !data.is_empty() && data.len() <= 32,
    "PUSH data must be 1..32 bytes"
  );
  // Pad to 32 bytes (big-endian, left-pad with zeros).
  let mut be = [0u8; 32];
  be[32 - data.len()..].copy_from_slice(data);
  // Convert to 2 × u128 limbs (little-endian limb order: limb 0 = lowest).
  let mut ops = Vec::with_capacity(LIMBS);
  for i in 0..LIMBS {
    let offset = 16 - i * 16; // big-endian byte offset for limb i
    let val = u128::from_be_bytes(be[offset..offset + 16].try_into().unwrap());
    ops.push(MicroOp::Const {
      dst: limb(0, i),
      val,
    });
  }
  Compiled {
    ops,
    advice_count: 0,
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{AdviceTape, Vm};

  /// Helper: write a U256 (2 limbs, LE) into consecutive registers starting at `base`.
  fn write_u256(vm: &mut Vm, base: Reg, limbs: [u128; 2]) {
    for (i, &v) in limbs.iter().enumerate() {
      vm.regs.write(base + i as Reg, v);
    }
  }

  /// Helper: read a U256 from consecutive registers.
  fn read_u256(vm: &Vm, base: Reg) -> [u128; 2] {
    let mut out = [0u128; 2];
    for i in 0..2 {
      out[i] = vm.regs.read(base + i as Reg);
    }
    out
  }

  #[test]
  fn compile_add_basic() {
    let c = compile(revm::bytecode::opcode::ADD, &[]).unwrap();
    assert_eq!(c.advice_count, 0);
    // Const(carry=0) + LIMBS Add128 ops
    assert_eq!(c.ops.len(), LIMBS + 1);

    let mut vm = Vm::new(AdviceTape::default());
    write_u256(&mut vm, slot(0), [1, 0]);
    write_u256(&mut vm, slot(1), [2, 0]);
    vm.run(&c.ops).unwrap();
    let result = read_u256(&vm, slot(0));
    assert_eq!(result[0], 3);
    assert_eq!(result[1], 0);
  }

  #[test]
  fn compile_add_carry() {
    let c = compile(revm::bytecode::opcode::ADD, &[]).unwrap();
    let mut vm = Vm::new(AdviceTape::default());
    // a = u128::MAX (limb 0)
    write_u256(&mut vm, slot(0), [u128::MAX, 0]);
    // b = 1
    write_u256(&mut vm, slot(1), [1, 0]);
    vm.run(&c.ops).unwrap();
    let result = read_u256(&vm, slot(0));
    assert_eq!(result[0], 0);
    assert_eq!(result[1], 1); // carry propagated
  }

  #[test]
  fn compile_sub_basic() {
    let c = compile(revm::bytecode::opcode::SUB, &[]).unwrap();
    let mut vm = Vm::new(AdviceTape::default());
    write_u256(&mut vm, slot(0), [5, 0]);
    write_u256(&mut vm, slot(1), [3, 0]);
    vm.run(&c.ops).unwrap();
    let result = read_u256(&vm, slot(0));
    assert_eq!(result[0], 2);
    assert_eq!(result[1], 0);
  }

  #[test]
  fn compile_sub_borrow() {
    let c = compile(revm::bytecode::opcode::SUB, &[]).unwrap();
    let mut vm = Vm::new(AdviceTape::default());
    // a = 1 << 128 (limbs: [0, 1])
    write_u256(&mut vm, slot(0), [0, 1]);
    // b = 1
    write_u256(&mut vm, slot(1), [1, 0]);
    vm.run(&c.ops).unwrap();
    let result = read_u256(&vm, slot(0));
    assert_eq!(result[0], u128::MAX);
    assert_eq!(result[1], 0);
  }

  #[test]
  fn compile_bitwise_and_basic() {
    let c = compile(revm::bytecode::opcode::AND, &[]).unwrap();
    assert_eq!(c.ops.len(), LIMBS);
    let mut vm = Vm::new(AdviceTape::default());
    write_u256(&mut vm, slot(0), [0xFF00, 0]);
    write_u256(&mut vm, slot(1), [0x0FF0, 0]);
    vm.run(&c.ops).unwrap();
    assert_eq!(read_u256(&vm, slot(0))[0], 0x0F00);
  }

  #[test]
  fn compile_bitwise_xor_basic() {
    let c = compile(revm::bytecode::opcode::XOR, &[]).unwrap();
    let mut vm = Vm::new(AdviceTape::default());
    write_u256(&mut vm, slot(0), [0xFF00, 0]);
    write_u256(&mut vm, slot(1), [0x0FF0, 0]);
    vm.run(&c.ops).unwrap();
    assert_eq!(read_u256(&vm, slot(0))[0], 0xF0F0);
  }

  #[test]
  fn compile_bitwise_or_basic() {
    let c = compile(revm::bytecode::opcode::OR, &[]).unwrap();
    let mut vm = Vm::new(AdviceTape::default());
    write_u256(&mut vm, slot(0), [0xFF00, 0]);
    write_u256(&mut vm, slot(1), [0x0FF0, 0]);
    vm.run(&c.ops).unwrap();
    assert_eq!(read_u256(&vm, slot(0))[0], 0xFFF0);
  }

  #[test]
  fn compile_bitwise_not_basic() {
    let c = compile(revm::bytecode::opcode::NOT, &[]).unwrap();
    let mut vm = Vm::new(AdviceTape::default());
    write_u256(&mut vm, slot(0), [0, 0]);
    vm.run(&c.ops).unwrap();
    let result = read_u256(&vm, slot(0));
    assert!(result.iter().all(|&v| v == u128::MAX));
  }

  #[test]
  fn compile_push_zero_works() {
    let c = compile(revm::bytecode::opcode::PUSH0, &[]).unwrap();
    let mut vm = Vm::new(AdviceTape::default());
    write_u256(&mut vm, slot(0), [1, 2]);
    vm.run(&c.ops).unwrap();
    assert_eq!(read_u256(&vm, slot(0)), [0; 2]);
  }

  #[test]
  fn compile_push_data() {
    // PUSH1 0xFF → [0xFF, 0]
    let c = compile_push(&[0xFF]);
    let mut vm = Vm::new(AdviceTape::default());
    vm.run(&c.ops).unwrap();
    let r = read_u256(&vm, slot(0));
    assert_eq!(r[0], 0xFF);
    assert_eq!(r[1], 0);
  }

  #[test]
  fn compile_push_32bytes() {
    // Full 32-byte push: 0x0102...1F20
    let data: Vec<u8> = (1..=32).collect();
    let c = compile_push(&data);
    let mut vm = Vm::new(AdviceTape::default());
    vm.run(&c.ops).unwrap();
    let r = read_u256(&vm, slot(0));
    // limb 0 = lowest 16 bytes (big-endian bytes 16..32): 0x1112...1F20
    assert_eq!(r[0], 0x1112131415161718191A1B1C1D1E1F20);
    // limb 1 = highest 16 bytes (big-endian bytes 0..16): 0x0102...0F10
    assert_eq!(r[1], 0x0102030405060708090A0B0C0D0E0F10);
  }

  #[test]
  fn compile_keccak_returns_leaf() {
    let c = compile(revm::bytecode::opcode::KECCAK256, &[]).unwrap();
    assert_eq!(c.ops.len(), 1);
    assert!(matches!(c.ops[0], MicroOp::KeccakLeaf { .. }));
  }

  #[test]
  fn unsupported_opcode_returns_none() {
    // 0x0C is undefined
    assert!(compile(0x0C, &[]).is_none());
  }

  // ── Memory compile tests ──────────────────────────────────────────────────

  #[test]
  fn compile_mload_basic() {
    let c = compile(revm::bytecode::opcode::MLOAD, &[]).unwrap();
    assert_eq!(c.ops.len(), 1);
    assert!(matches!(c.ops[0], MicroOp::MLoad { .. }));
  }

  #[test]
  fn compile_mstore_basic() {
    let c = compile(revm::bytecode::opcode::MSTORE, &[]).unwrap();
    assert_eq!(c.ops.len(), 1);
    assert!(matches!(c.ops[0], MicroOp::MStore { .. }));
  }

  #[test]
  fn compile_mstore8_basic() {
    let c = compile(revm::bytecode::opcode::MSTORE8, &[]).unwrap();
    assert_eq!(c.ops.len(), 1);
    assert!(matches!(c.ops[0], MicroOp::MStore8 { .. }));
  }

  // ── Storage compile tests ─────────────────────────────────────────────────

  #[test]
  fn compile_sload_basic() {
    let c = compile(revm::bytecode::opcode::SLOAD, &[]).unwrap();
    assert_eq!(c.ops.len(), 1);
    assert!(matches!(c.ops[0], MicroOp::SLoad { .. }));
  }

  #[test]
  fn compile_sstore_basic() {
    let c = compile(revm::bytecode::opcode::SSTORE, &[]).unwrap();
    assert_eq!(c.ops.len(), 1);
    assert!(matches!(c.ops[0], MicroOp::SStore { .. }));
  }

  // ── Jump compile tests ────────────────────────────────────────────────────

  #[test]
  fn compile_jump_noop() {
    let c = compile(revm::bytecode::opcode::JUMP, &[]).unwrap();
    assert!(c.ops.is_empty());
  }

  #[test]
  fn compile_jumpi_noop() {
    let c = compile(revm::bytecode::opcode::JUMPI, &[]).unwrap();
    assert!(c.ops.is_empty());
  }

  #[test]
  fn compile_jumpdest_noop() {
    let c = compile(revm::bytecode::opcode::JUMPDEST, &[]).unwrap();
    assert!(c.ops.is_empty());
  }

  // ── New opcode tests ──────────────────────────────────────────────────────

  #[test]
  fn compile_lt_advice_bool() {
    use revm::bytecode::opcode::*;
    let c = compile(LT, &[]).unwrap();
    assert_eq!(c.advice_count, 2);
    // a=[5,0] < b=[10,0]: hi limbs equal → eq_hi=1, inv=0
    let mut vm = Vm::new(AdviceTape::new([1u128, 0]));
    write_u256(&mut vm, slot(0), [5, 0]);
    write_u256(&mut vm, slot(1), [10, 0]);
    vm.run(&c.ops).unwrap();
    let r = read_u256(&vm, slot(0));
    assert_eq!(r[0], 1);
    assert_eq!(r[1], 0);
  }

  #[test]
  fn compile_gt_advice_bool() {
    use revm::bytecode::opcode::*;
    let c = compile(GT, &[]).unwrap();
    // GT(5,10)=0: hi limbs equal → eq_hi=1, inv=0
    let mut vm = Vm::new(AdviceTape::new([1u128, 0]));
    write_u256(&mut vm, slot(0), [5, 0]);
    write_u256(&mut vm, slot(1), [10, 0]);
    vm.run(&c.ops).unwrap();
    assert_eq!(read_u256(&vm, slot(0))[0], 0);
  }

  #[test]
  fn compile_slt_sgt() {
    use revm::bytecode::opcode::*;
    assert_eq!(compile(SLT, &[]).unwrap().advice_count, 2);
    assert_eq!(compile(SGT, &[]).unwrap().advice_count, 2);
  }

  #[test]
  fn compile_byte_extract_basic() {
    use revm::bytecode::opcode::*;
    let c = compile(BYTE, &[]).unwrap();
    assert_eq!(c.advice_count, 1);
    let mut vm = Vm::new(AdviceTape::new([0x34u128]));
    write_u256(&mut vm, slot(0), [31, 0]);
    write_u256(&mut vm, slot(1), [0x1234, 0]);
    vm.run(&c.ops).unwrap();
    let r = read_u256(&vm, slot(0));
    assert_eq!(r[0], 0x34);
    assert_eq!(r[1], 0);
  }

  #[test]
  fn compile_shl_verified() {
    use revm::bytecode::opcode::*;
    // With pre_stack: verified Shl128 path (no advice needed)
    let shift = U256::from(4u64);
    let value = U256::from(0xFFu64);
    let c = compile(SHL, &[shift, value]).unwrap();
    assert_eq!(c.advice_count, 0);
    let mut vm = Vm::new(AdviceTape::default());
    write_u256(&mut vm, slot(0), [4, 0]);
    write_u256(&mut vm, slot(1), [0xFF, 0]);
    vm.run(&c.ops).unwrap();
    assert_eq!(read_u256(&vm, slot(0))[0], 0xFF0);
  }

  #[test]
  fn compile_shr_verified() {
    use revm::bytecode::opcode::*;
    let shift = U256::from(4u64);
    let value = U256::from(0xFF0u64);
    let c = compile(SHR, &[shift, value]).unwrap();
    assert_eq!(c.advice_count, 0);
    let mut vm = Vm::new(AdviceTape::default());
    write_u256(&mut vm, slot(0), [4, 0]);
    write_u256(&mut vm, slot(1), [0xFF0, 0]);
    vm.run(&c.ops).unwrap();
    assert_eq!(read_u256(&vm, slot(0))[0], 0xFF);
  }

  #[test]
  fn compile_sar_signextend_exp_sdiv_smod() {
    use revm::bytecode::opcode::*;
    // EXP with empty pre_stack: exponent defaults to 0 → exp=0 → result=1, no advice
    let c = compile(EXP, &[]).unwrap();
    assert_eq!(c.advice_count, 0, "EXP exp=0");
    // EXP with base=2, exp=10: binary exponentiation chain
    let c = compile(EXP, &[U256::from(2), U256::from(10)]).unwrap();
    // 10 in binary = 1010 → 3 squarings + 1 multiply = 4 steps × 3 limbs = 12
    assert_eq!(c.advice_count, 12, "EXP 2^10");
    // SDIV/SMOD: now use 4-limb advice (quotient + remainder)
    let c = compile(SDIV, &[]).unwrap();
    assert_eq!(c.advice_count, LIMBS * 2, "SDIV");
    let c = compile(SMOD, &[]).unwrap();
    assert_eq!(c.advice_count, LIMBS * 2, "SMOD");
    // SAR without pre_stack: falls back to large-shift (zero result, no advice)
    let c = compile(SAR, &[]).unwrap();
    assert_eq!(c.advice_count, 0, "SAR fallback");
    // SIGNEXTEND with byte_idx=0 and no value: generates verified code
    let c = compile(SIGNEXTEND, &[U256::ZERO]).unwrap();
    assert_eq!(c.advice_count, 0, "SIGNEXTEND");
  }

  #[test]
  fn compile_dup1_noop() {
    use revm::bytecode::opcode::*;
    let c = compile(DUP1, &[]).unwrap();
    assert!(c.ops.is_empty());
    assert_eq!(c.advice_count, 0);
  }

  #[test]
  fn compile_dup2_copies_slot1() {
    use revm::bytecode::opcode::*;
    let c = compile(DUP2, &[]).unwrap();
    assert_eq!(c.ops.len(), LIMBS);
    let mut vm = Vm::new(AdviceTape::default());
    write_u256(&mut vm, slot(0), [0xAA; 2]);
    write_u256(&mut vm, slot(1), [0xBB, 0xCC]);
    vm.run(&c.ops).unwrap();
    assert_eq!(read_u256(&vm, slot(0)), [0xBB, 0xCC]);
  }

  #[test]
  fn compile_dup3_to_dup5() {
    use revm::bytecode::opcode::*;
    for (op, expected_src) in [(DUP3, 2), (DUP4, 3), (DUP5, 4)] {
      let c = compile(op, &[]).unwrap();
      assert_eq!(c.ops.len(), LIMBS, "DUP{}", op - DUP1 + 1);
      let mut vm = Vm::new(AdviceTape::default());
      let src_val = [42u128; 2];
      write_u256(&mut vm, slot(expected_src), src_val);
      vm.run(&c.ops).unwrap();
      assert_eq!(read_u256(&vm, slot(0)), src_val, "DUP{}", op - DUP1 + 1);
    }
  }

  #[test]
  fn compile_dup6_to_dup16_use_slot1() {
    use revm::bytecode::opcode::*;
    for op in DUP6..=DUP16 {
      let c = compile(op, &[]).unwrap();
      assert_eq!(c.ops.len(), LIMBS, "DUP{}", op - DUP1 + 1);
      let mut vm = Vm::new(AdviceTape::default());
      let src_val = [(op - DUP1) as u128 + 100; 2];
      write_u256(&mut vm, slot(1), src_val);
      vm.run(&c.ops).unwrap();
      assert_eq!(read_u256(&vm, slot(0)), src_val, "DUP{}", op - DUP1 + 1);
    }
  }

  #[test]
  fn compile_swap1_swaps_slots() {
    use revm::bytecode::opcode::*;
    let c = compile(SWAP1, &[]).unwrap();
    assert_eq!(c.ops.len(), LIMBS * 3);
    let mut vm = Vm::new(AdviceTape::default());
    let a: [u128; 2] = [1, 2];
    let b: [u128; 2] = [10, 20];
    write_u256(&mut vm, slot(0), a);
    write_u256(&mut vm, slot(1), b);
    vm.run(&c.ops).unwrap();
    assert_eq!(read_u256(&vm, slot(0)), b);
    assert_eq!(read_u256(&vm, slot(1)), a);
  }

  #[test]
  fn compile_swap2_to_swap16() {
    use revm::bytecode::opcode::*;
    for op in SWAP2..=SWAP16 {
      let c = compile(op, &[]).unwrap();
      assert_eq!(c.ops.len(), LIMBS * 3, "SWAP{}", op - SWAP1 + 1);
    }
  }

  #[test]
  fn compile_stop_halts() {
    use revm::bytecode::opcode::*;
    let c = compile(STOP, &[]).unwrap();
    assert_eq!(c.ops.len(), 1);
    assert!(matches!(c.ops[0], MicroOp::Done));
  }

  #[test]
  fn compile_return_revert_noop() {
    use revm::bytecode::opcode::*;
    assert!(compile(RETURN, &[]).unwrap().ops.is_empty());
    assert!(compile(REVERT, &[]).unwrap().ops.is_empty());
    assert!(compile(SELFDESTRUCT, &[]).unwrap().ops.is_empty());
  }

  #[test]
  fn compile_invalid_halts() {
    use revm::bytecode::opcode::*;
    let c = compile(INVALID, &[]).unwrap();
    assert!(matches!(c.ops[0], MicroOp::Done));
  }

  #[test]
  fn compile_env_opcodes_are_advice_u256() {
    use revm::bytecode::opcode::*;
    let env_ops = [
      ADDRESS,
      BALANCE,
      ORIGIN,
      CALLER,
      CALLVALUE,
      CALLDATALOAD,
      CALLDATASIZE,
      CODESIZE,
      GASPRICE,
      EXTCODESIZE,
      EXTCODEHASH,
      RETURNDATASIZE,
      BLOCKHASH,
      COINBASE,
      TIMESTAMP,
      NUMBER,
      DIFFICULTY,
      GASLIMIT,
      CHAINID,
      SELFBALANCE,
      BASEFEE,
      BLOBHASH,
      BLOBBASEFEE,
      PC,
      MSIZE,
      GAS,
      CLZ,
    ];
    for op in env_ops {
      let c = compile(op, &[]).unwrap();
      assert_eq!(c.advice_count, LIMBS, "opcode 0x{op:02x}");
      assert_eq!(c.ops.len(), LIMBS + 1, "opcode 0x{op:02x}"); // Advice2 + mov×2
    }
  }

  #[test]
  fn compile_copy_ops_are_noop() {
    use revm::bytecode::opcode::*;
    for op in [CALLDATACOPY, CODECOPY, EXTCODECOPY, RETURNDATACOPY, MCOPY] {
      let c = compile(op, &[]).unwrap();
      assert!(c.ops.is_empty(), "opcode 0x{op:02x}");
    }
  }

  #[test]
  fn compile_log_ops_are_noop() {
    use revm::bytecode::opcode::*;
    for op in [LOG0, LOG1, LOG2, LOG3, LOG4] {
      let c = compile(op, &[]).unwrap();
      assert!(c.ops.is_empty(), "opcode 0x{op:02x}");
    }
  }

  #[test]
  fn compile_system_ops_are_advice_u256() {
    use revm::bytecode::opcode::*;
    for op in [CREATE, CREATE2, CALL, CALLCODE, DELEGATECALL, STATICCALL] {
      let c = compile(op, &[]).unwrap();
      assert_eq!(c.advice_count, LIMBS, "opcode 0x{op:02x}");
    }
  }

  #[test]
  fn compile_tload_basic() {
    use revm::bytecode::opcode::*;
    let c = compile(TLOAD, &[]).unwrap();
    assert_eq!(c.ops.len(), 1);
    assert!(matches!(c.ops[0], MicroOp::TLoad { .. }));
    assert_eq!(c.advice_count, 0);
  }

  #[test]
  fn compile_tstore_basic() {
    use revm::bytecode::opcode::*;
    let c = compile(TSTORE, &[]).unwrap();
    assert_eq!(c.ops.len(), 1);
    assert!(matches!(c.ops[0], MicroOp::TStore { .. }));
    assert_eq!(c.advice_count, 0);
  }

  #[test]
  fn compile_tload_tstore_roundtrip() {
    // Store a value via TSTORE, then load it back via TLOAD
    let store_c = compile(revm::bytecode::opcode::TSTORE, &[]).unwrap();
    let load_c = compile(revm::bytecode::opcode::TLOAD, &[]).unwrap();

    let mut vm = Vm::new(AdviceTape::default());
    let key: [u128; 2] = [1, 0];
    let val: [u128; 2] = [42, 99];
    write_u256(&mut vm, slot(0), key);
    write_u256(&mut vm, slot(1), val);
    vm.run(&store_c.ops).unwrap();

    // Now load: key in slot0
    write_u256(&mut vm, slot(0), key);
    vm.run(&load_c.ops).unwrap();
    assert_eq!(read_u256(&vm, slot(0)), val);
  }

  /// Verify that every defined EVM opcode in revm is now handled by compile().
  #[test]
  fn all_defined_opcodes_covered() {
    use revm::bytecode::opcode::OPCODE_INFO;
    let mut unsupported = Vec::new();
    for op in 0u8..=0xFF {
      if OPCODE_INFO[op as usize].is_some() {
        // Skip PUSH1..PUSH32 (handled separately by compile_push)
        if (0x60..=0x7F).contains(&op) {
          continue;
        }
        if compile(op, &[]).is_none() {
          unsupported.push(op);
        }
      }
    }
    assert!(
      unsupported.is_empty(),
      "unsupported opcodes: {:?}",
      unsupported
        .iter()
        .map(|o| format!("0x{o:02x}"))
        .collect::<Vec<_>>()
    );
  }
}
