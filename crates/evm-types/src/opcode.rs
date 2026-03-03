//! EVM opcode utilities built on top of revm's [`OpCode`] / [`OPCODE_INFO`].
//!
//! All opcode byte constants and stack I/O data come directly from revm so
//! there is a single source of truth. This module only adds the
//! [`StackEffect`] wrapper and the [`stack_effect`] helper used by the
//! type-checker.

// Re-export every opcode constant (ADD, MUL, KECCAK256, …) from revm.
// All constants are defined in revm-bytecode; no manual list is needed.
pub use revm::bytecode::opcode::*;

// ─────────────────────────────────────────────────────────────────────────────

/// Stack depth effect of a single EVM opcode.
///
/// Values are derived directly from [`revm::bytecode::opcode::OPCODE_INFO`]:
/// - `pops`      = `OpcodeInfo::inputs()`  — items consumed (= min depth required)
/// - `pushes`    = `OpcodeInfo::outputs()` — items produced
/// - `min_stack` = `pops` (always equal under revm's model)
///
/// **revm encoding for DUP / SWAP:**
/// - `DUP1`  → inputs=1, outputs=2   (delta = +1, min_stack = 1)
/// - `SWAP1` → inputs=2, outputs=2   (delta =  0, min_stack = 2)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StackEffect {
  /// Items consumed from the stack (= minimum required depth).
  pub pops: u16,
  /// Items produced onto the stack.
  pub pushes: u16,
  /// Minimum stack depth required before the opcode executes.
  /// Always equal to `pops` under revm's model.
  pub min_stack: u16,
}

impl StackEffect {
  /// Net change in stack depth: positive = grow, negative = shrink.
  #[inline]
  pub const fn delta(self) -> i32 {
    self.pushes as i32 - self.pops as i32
  }
}

/// Returns the [`StackEffect`] of `op`, or `None` for undefined / reserved bytes.
///
/// Delegates entirely to [`revm::bytecode::opcode::OPCODE_INFO`]; no separate
/// match table is needed.  DUP/SWAP `min_stack` is correct because revm encodes
/// them as `stack_io(inputs, outputs)` where `inputs` equals the minimum depth.
pub fn stack_effect(op: u8) -> Option<StackEffect> {
  use revm::bytecode::opcode::OPCODE_INFO;
  let info = OPCODE_INFO[op as usize]?;
  let pops = info.inputs() as u16;
  Some(StackEffect {
    pops,
    pushes: info.outputs() as u16,
    min_stack: pops,
  })
}

/// Gas-cost information for a single EVM opcode.
///
/// - For opcodes with a fixed ("static") gas cost, `static_gas` is `Some(cost)`.
/// - For opcodes whose cost depends on runtime context (memory expansion,
///   storage warmth, etc.), `static_gas` is the known base component and
///   `dynamic` is `true`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GasCostInfo {
  /// Fixed gas component. All opcodes cost *at least* this much.
  pub static_gas: u64,
  /// `true` if additional dynamic gas may be charged on top of `static_gas`.
  pub dynamic: bool,
}

/// Returns the [`GasCostInfo`] for `op`, or `None` for undefined opcodes.
///
/// Gas values follow the Ethereum Yellow Paper / EIP-2929 (Berlin+) schedule.
/// Dynamic costs (memory expansion, cold/warm storage, etc.) are NOT included
/// — only the guaranteed static base.
pub fn gas_cost(op: u8) -> Option<GasCostInfo> {
  // Yellow Paper gas tiers.
  const ZERO: u64 = 0;
  const BASE: u64 = 2;
  const VERYLOW: u64 = 3;
  const LOW: u64 = 5;
  const MID: u64 = 8;
  const HIGH: u64 = 10;

  let (static_gas, dynamic) = match op {
    // ── W_zero: 0 gas ──────────────────────────────────────────────────
    STOP | RETURN | REVERT => (ZERO, false),

    // ── W_base: 2 gas ──────────────────────────────────────────────────
    ADDRESS | ORIGIN | CALLER | CALLVALUE | CALLDATASIZE | CODESIZE | GASPRICE | COINBASE
    | TIMESTAMP | NUMBER | DIFFICULTY | GASLIMIT | CHAINID | SELFBALANCE | BASEFEE | POP | PC
    | MSIZE | GAS | RETURNDATASIZE | PUSH0 | BLOBBASEFEE => (BASE, false),

    // ── W_verylow: 3 gas ───────────────────────────────────────────────
    ADD | SUB | NOT | LT | GT | SLT | SGT | EQ | ISZERO | AND | OR | XOR | BYTE | SHL | SHR
    | SAR | CALLDATALOAD | SIGNEXTEND => (VERYLOW, false),

    // MLOAD/MSTORE/MSTORE8: verylow base but may expand memory (dynamic)
    MLOAD | MSTORE | MSTORE8 => (VERYLOW, true),

    // PUSH1..PUSH32: verylow
    op if (0x60..=0x7f).contains(&op) => (VERYLOW, false),
    // DUP1..DUP16: verylow
    op if (0x80..=0x8f).contains(&op) => (VERYLOW, false),
    // SWAP1..SWAP16: verylow
    op if (0x90..=0x9f).contains(&op) => (VERYLOW, false),

    // ── W_low: 5 gas ───────────────────────────────────────────────────
    MUL | DIV | SDIV | MOD | SMOD => (LOW, false),

    // ── W_mid: 8 gas ───────────────────────────────────────────────────
    ADDMOD | MULMOD | JUMP => (MID, false),

    // ── W_high: 10 gas ─────────────────────────────────────────────────
    JUMPI => (HIGH, false),

    // ── Fixed special ──────────────────────────────────────────────────
    JUMPDEST => (1, false),

    // ── Dynamic opcodes (base + memory/context) ────────────────────────
    EXP => (10, true),       // 10 + 50 * byte_len(exponent)
    KECCAK256 => (30, true), // 30 + 6 * ceil(size/32)
    BLOCKHASH => (20, false),

    // Memory copy opcodes: base 3 + dynamic
    CALLDATACOPY | CODECOPY | RETURNDATACOPY => (VERYLOW, true),
    MCOPY => (VERYLOW, true),

    // ── Storage (EIP-2929 warm/cold) ───────────────────────────────────
    SLOAD => (0, true),    // 100 warm, 2100 cold — always dynamic
    SSTORE => (0, true),   // complex EIP-2200/2929 rules
    TLOAD => (100, false), // EIP-1153
    TSTORE => (100, false),

    // ── External account access ────────────────────────────────────────
    BALANCE | EXTCODESIZE | EXTCODEHASH => (0, true), // 100 warm, 2600 cold
    EXTCODECOPY => (0, true),

    // ── LOG ────────────────────────────────────────────────────────────
    op if (0xa0..=0xa4).contains(&op) => {
      let topics = (op - 0xa0) as u64;
      (375 + 375 * topics, true) // + 8 * data_size + memory
    }

    // ── System ops (dynamic, context-dependent) ────────────────────────
    CREATE => (32000, true),
    CREATE2 => (32000, true),
    CALL | CALLCODE | DELEGATECALL | STATICCALL => (0, true),
    SELFDESTRUCT => (5000, true),

    // ── Anything else: unknown/undefined ───────────────────────────────
    _ => return None,
  };

  Some(GasCostInfo {
    static_gas,
    dynamic,
  })
}

#[cfg(test)]
mod tests {
  use super::*;

  // ── Binary ops: pops=2, pushes=1 ──────────────────────────────────────────

  #[test]
  fn binary_ops_pop2_push1() {
    for op in [ADD, MUL, SUB, DIV, KECCAK256, AND, OR, XOR] {
      let e = stack_effect(op).unwrap();
      assert_eq!((e.pops, e.pushes), (2, 1), "opcode 0x{op:02x}");
    }
  }

  // ── DUP: revm model = stack_io(n, n+1) ───────────────────────────────────

  #[test]
  fn dup_effects() {
    for i in 0u8..16 {
      let op = 0x80 + i;
      let n = (i + 1) as u16; // DUP1→1, DUP16→16
      let e = stack_effect(op).unwrap();
      assert_eq!(e.pops, n, "DUP{} pops", i + 1);
      assert_eq!(e.pushes, n + 1, "DUP{} pushes", i + 1);
      assert_eq!(e.delta(), 1, "DUP{} delta", i + 1);
      assert_eq!(e.min_stack, n, "DUP{} min_stack", i + 1);
    }
  }

  // ── SWAP: revm model = stack_io(n+1, n+1) ────────────────────────────────

  #[test]
  fn swap_effects() {
    for i in 0u8..16 {
      let op = 0x90 + i;
      let n = (i + 2) as u16; // SWAP1→2, SWAP16→17
      let e = stack_effect(op).unwrap();
      assert_eq!(e.pops, n, "SWAP{} pops", i + 1);
      assert_eq!(e.pushes, n, "SWAP{} pushes", i + 1);
      assert_eq!(e.delta(), 0, "SWAP{} delta", i + 1);
      assert_eq!(e.min_stack, n, "SWAP{} min_stack", i + 1);
    }
  }

  // ── LOG: pops = 2 + topics ────────────────────────────────────────────────

  #[test]
  fn log_effects() {
    for i in 0u8..=4 {
      let e = stack_effect(0xa0 + i).unwrap();
      assert_eq!(e.pops, 2 + i as u16, "LOG{i} pops");
      assert_eq!(e.pushes, 0, "LOG{i} pushes");
    }
  }

  // ── Undefined bytes return None ───────────────────────────────────────────

  #[test]
  fn undefined_opcode_returns_none() {
    assert!(stack_effect(0x0c).is_none()); // gap in arithmetic range
    assert!(stack_effect(0x4b).is_none()); // gap between BLOBBASEFEE and POP
    assert!(stack_effect(0xfb).is_none()); // gap in system range
  }

  // ── Ternary arithmetic ────────────────────────────────────────────────────

  #[test]
  fn ternary_ops_pop3_push1() {
    for op in [ADDMOD, MULMOD] {
      let e = stack_effect(op).unwrap();
      assert_eq!((e.pops, e.pushes), (3, 1), "opcode 0x{op:02x}");
      assert_eq!(e.delta(), -2);
    }
  }

  // ── PUSH0 ─────────────────────────────────────────────────────────────────

  #[test]
  fn push0_pushes_one() {
    let e = stack_effect(PUSH0).unwrap();
    assert_eq!((e.pops, e.pushes), (0, 1));
  }
}
