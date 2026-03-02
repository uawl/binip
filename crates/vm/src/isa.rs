//! Micro-op ISA for the register-based Meta-VM.
//!
//! All opcodes operate on **32-bit slices** of the 256-bit register file;
//! the "Cost" annotation is the number of AND2 gates required in GF(2)
//! (AND is the "expensive" gate; XOR/NOT/rotation are free wire substitutions).
//!
//! A full 256-bit EVM operation is assembled by chaining eight 32-bit micro-ops
//! (one per limb), passing carry flags between them.

/// Register index — selects one of the 64 × u32 registers.
pub type Reg = u8;

/// Flag-register index — selects one of the 8 carry/boolean flags.
pub type FlagReg = u8;

// ─────────────────────────────────────────────────────────────────────────────

/// Single instruction in the Meta-VM ISA.
///
/// Variants whose name ends in `32` operate on the **low 32 bits** of a
/// register.  The result is stored zero-extended in the destination register.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MicroOp {
  // ── 32-bit ALU ────────────────────────────────────────────────────────────
  /// 32-bit full adder: `dst = (a + b + flags[cin]) & 0xFFFF_FFFF`.
  /// Carry-out written to `flags[cout]`.  Cost: 32 AND.
  Add32 {
    dst: Reg,
    a: Reg,
    b: Reg,
    cin: FlagReg,
    cout: FlagReg,
  },

  /// 32×32 → 64-bit multiply: low 32 bits → `dst_lo`, high 32 bits → `dst_hi`.
  /// Cost: 32 AND (Karatsuba).
  Mul32 {
    dst_lo: Reg,
    dst_hi: Reg,
    a: Reg,
    b: Reg,
  },

  /// Bitwise AND on low 32 bits.  Cost: 32 AND.
  And32 { dst: Reg, a: Reg, b: Reg },

  /// Bitwise XOR on low 32 bits.  Cost: 0 AND.
  Xor32 { dst: Reg, a: Reg, b: Reg },

  /// Bitwise NOT on low 32 bits.  Cost: 0 AND.
  Not32 { dst: Reg, src: Reg },

  /// Rotate-right on low 32 bits by `shift` positions (wire substitution).
  /// Cost: 0 AND.
  Rot32 { dst: Reg, src: Reg, shift: u8 },

  /// Shift-right on low 32 bits by `shift` positions, with `regs[cin]`
  /// providing the bits shifted in from the left.  Used to chain multi-limb
  /// right shifts.  Cost: 0 AND.
  Shr32 {
    dst: Reg,
    src: Reg,
    cin: Reg,
    shift: u8,
  },

  /// Shift-left on low 32 bits by `shift` positions, with `regs[cin]`
  /// providing the bits shifted in from the right.  Mirror of [`Shl32`];
  /// used to chain multi-limb left shifts (e.g. EVM SHL).  Cost: 0 AND.
  ///
  /// `dst = (src << shift) | (cin >> (32 - shift))`
  Shl32 {
    dst: Reg,
    src: Reg,
    cin: Reg,
    shift: u8,
  },

  // ── Keccak χ step ─────────────────────────────────────────────────────────
  /// χ: `dst = a XOR ((NOT b) AND c)` on low 32 bits.  Cost: 32 AND.
  Chi32 { dst: Reg, a: Reg, b: Reg, c: Reg },

  // ── Register / immediate ──────────────────────────────────────────────────
  /// Load a 32-bit immediate into a register.  Cost: 0 AND.
  /// To load a 256-bit constant, use 8 consecutive `Const` ops.
  Const { dst: Reg, val: u32 },

  /// Copy a full U256 register.  Cost: 0 AND.
  Mov { dst: Reg, src: Reg },

  // ── Advice witness ────────────────────────────────────────────────────────
  /// Pop the next value from the advice tape and write it to `dst`.
  /// The circuit verifies correctness through a paired `CheckDiv` / other
  /// `Check*` row; the tape itself is never seen by the verifier.
  AdviceLoad { dst: Reg },

  /// Assert `regs[dividend] == regs[divisor] * regs[quot] + regs[rem]`
  /// and `regs[rem] < regs[divisor]`.
  /// The prover pre-loads `quot` and `rem` via `AdviceLoad`.  Cost: ~64 AND.
  CheckDiv {
    quot: Reg,
    rem: Reg,
    dividend: Reg,
    divisor: Reg,
  },

  // ── Proof-tree / EVM structural ───────────────────────────────────────────
  /// Sequential composition: `dst` receives a handle to `Seq(left, right)`.
  /// At VM level this is a structural annotation; the circuit enforces typing.
  Compose { dst: Reg, left: Reg, right: Reg },

  /// EVM opcode structural type-check annotation.
  /// `pre` / `post` registers hold opaque `EvmStateType` handles.
  TypeCheck { opcode: u8, pre: Reg, post: Reg },
}

impl MicroOp {
  /// Numeric tag for the [`crate::row::Row::op`] column.
  pub fn tag(&self) -> u32 {
    match self {
      Self::Add32 { .. } => 0,
      Self::Mul32 { .. } => 1,
      Self::And32 { .. } => 2,
      Self::Xor32 { .. } => 3,
      Self::Not32 { .. } => 4,
      Self::Rot32 { .. } => 5,
      Self::Shr32 { .. } => 6,
      Self::Shl32 { .. } => 7,
      Self::Chi32 { .. } => 8,
      Self::Const { .. } => 9,
      Self::Mov { .. } => 10,
      Self::AdviceLoad { .. } => 11,
      Self::CheckDiv { .. } => 12,
      Self::Compose { .. } => 13,
      Self::TypeCheck { .. } => 14,
    }
  }
}
