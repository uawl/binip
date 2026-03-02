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
