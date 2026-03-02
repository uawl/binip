//! Micro-op ISA for the register-based Meta-VM.
//!
//! All opcodes operate on **128-bit slices** of the 256-bit register file;
//! the "Cost" annotation is the number of AND2 gates required in GF(2)
//! (AND is the "expensive" gate; XOR/NOT/rotation are free wire substitutions).
//!
//! A full 256-bit EVM operation is assembled by chaining two 128-bit micro-ops
//! (one per limb), passing carry flags between them.

/// Register index — selects one of the 16 × u128 registers.
pub type Reg = u8;

/// Flag-register index — selects one of the 8 carry/boolean flags.
pub type FlagReg = u8;

// ─────────────────────────────────────────────────────────────────────────────

/// Single instruction in the Meta-VM ISA.
///
/// Variants whose name ends in `128` operate on the full 128 bits of a
/// register.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MicroOp {
  // ── 128-bit ALU ───────────────────────────────────────────────────────────
  /// 128-bit full adder: `dst = (a + b + flags[cin]) mod 2^128`.
  /// Carry-out written to `flags[cout]`.
  Add128 {
    dst: Reg,
    a: Reg,
    b: Reg,
    cin: FlagReg,
    cout: FlagReg,
  },

  /// 128×128 → 256-bit multiply: low 128 bits → `dst_lo`, high 128 bits → `dst_hi`.
  Mul128 {
    dst_lo: Reg,
    dst_hi: Reg,
    a: Reg,
    b: Reg,
  },

  /// Bitwise AND on 128 bits.
  And128 { dst: Reg, a: Reg, b: Reg },

  /// Bitwise XOR on 128 bits.  Cost: 0 AND.
  Xor128 { dst: Reg, a: Reg, b: Reg },

  /// Bitwise NOT on 128 bits.  Cost: 0 AND.
  Not128 { dst: Reg, src: Reg },

  /// Rotate-right on 128 bits by `shift` positions.
  /// Cost: 0 AND.
  Rot128 { dst: Reg, src: Reg, shift: u8 },

  /// Shift-right on 128 bits by `shift` positions, with `regs[cin]`
  /// providing the bits shifted in from the left.  Used to chain multi-limb
  /// right shifts.  Cost: 0 AND.
  Shr128 {
    dst: Reg,
    src: Reg,
    cin: Reg,
    shift: u8,
  },

  /// Shift-left on 128 bits by `shift` positions, with `regs[cin]`
  /// providing the bits shifted in from the right.
  ///
  /// `dst = (src << shift) | (cin >> (128 - shift))`
  Shl128 {
    dst: Reg,
    src: Reg,
    cin: Reg,
    shift: u8,
  },

  // ── Keccak χ step ─────────────────────────────────────────────────────────
  /// χ: `dst = a XOR ((NOT b) AND c)` on 128 bits.
  Chi128 { dst: Reg, a: Reg, b: Reg, c: Reg },

  // ── Register / immediate ──────────────────────────────────────────────────
  /// Load a 128-bit immediate into a register.  Cost: 0 AND.
  /// To load a 256-bit constant, use 2 consecutive `Const` ops.
  Const { dst: Reg, val: u128 },

  /// Copy a register.  Cost: 0 AND.
  Mov { dst: Reg, src: Reg },

  // ── Advice witness ────────────────────────────────────────────────────────
  /// Pop the next value from the advice tape and write it to `dst`.
  AdviceLoad { dst: Reg },

  /// Assert `regs[dividend] == regs[divisor] * regs[quot] + regs[rem]`
  /// and `regs[rem] < regs[divisor]` (128-bit integer arithmetic).
  CheckDiv {
    quot: Reg,
    rem: Reg,
    dividend: Reg,
    divisor: Reg,
  },

  /// Assert `(a as u256) * (b as u256) == ((q_hi as u256) << 128) | (q_lo as u256)`.
  CheckMul {
    q_lo: Reg,
    q_hi: Reg,
    a: Reg,
    b: Reg,
  },

  /// Assert `regs[a] * regs[a_inv] ≡ 1 (mod 2^128)`.
  CheckInv { a: Reg, a_inv: Reg },

  /// Assert `regs[r] < 2^bits`.
  RangeCheck { r: Reg, bits: u8 },

  // ── Memory ────────────────────────────────────────────────────────────────
  /// Load a 128-bit word from VM memory at `addr` into `dst`.
  Load { dst: Reg, addr: u32 },

  /// Store the 128-bit value in `src` to VM memory at `addr`.
  Store { addr: u32, src: Reg },

  /// Load a U256 (2 consecutive u128 words) from VM memory at the byte
  /// offset in `regs[offset_reg] / 16` into `dst..dst+1`.
  MLoad { dst: Reg, offset_reg: Reg },

  /// Store a U256 (2 words from `src..src+1`) to VM memory at the byte
  /// offset in `regs[offset_reg] / 16`.
  MStore { offset_reg: Reg, src: Reg },

  /// Store the low byte of `regs[src]` to VM memory at the byte offset
  /// in `regs[offset_reg]`.
  MStore8 { offset_reg: Reg, src: Reg },

  // ── Storage ────────────────────────────────────────────────────────────────
  /// Load a U256 value from persistent storage by key.
  /// Key = 2 limbs at `key_reg..key_reg+1`, result written to `dst..dst+1`.
  SLoad { dst: Reg, key_reg: Reg },

  /// Store a U256 value to persistent storage.
  /// Key = 2 limbs at `key_reg..key_reg+1`, value = 2 limbs at `val_reg..val_reg+1`.
  SStore { key_reg: Reg, val_reg: Reg },

  // ── Transient storage (EIP-1153) ──────────────────────────────────────────
  /// Load a U256 value from transient storage by key.
  /// Key = 2 limbs at `key_reg..key_reg+1`, result written to `dst..dst+1`.
  TLoad { dst: Reg, key_reg: Reg },

  /// Store a U256 value to transient storage.
  /// Key = 2 limbs at `key_reg..key_reg+1`, value = 2 limbs at `val_reg..val_reg+1`.
  TStore { key_reg: Reg, val_reg: Reg },

  // ── Keccak sub-proof ──────────────────────────────────────────────────────
  /// Absorb a Keccak sub-proof commitment.
  KeccakLeaf { dst_commit: Reg, input: Reg },

  // ── Proof-tree / EVM structural ───────────────────────────────────────────
  /// Sequential composition.
  Compose { dst: Reg, left: Reg, right: Reg },

  /// EVM opcode structural type-check annotation.
  TypeCheck { opcode: u8, pre: Reg, post: Reg },

  // ── Control ───────────────────────────────────────────────────────────────
  /// Halt execution.
  Done,
}

impl MicroOp {
  /// Numeric tag for the [`crate::row::Row::op`] column.
  pub fn tag(&self) -> u32 {
    match self {
      Self::Add128 { .. } => 0,
      Self::Mul128 { .. } => 1,
      Self::And128 { .. } => 2,
      Self::Xor128 { .. } => 3,
      Self::Not128 { .. } => 4,
      Self::Rot128 { .. } => 5,
      Self::Shr128 { .. } => 6,
      Self::Shl128 { .. } => 7,
      Self::Chi128 { .. } => 8,
      Self::Const { .. } => 9,
      Self::Mov { .. } => 10,
      Self::AdviceLoad { .. } => 11,
      Self::CheckDiv { .. } => 12,
      Self::CheckMul { .. } => 13,
      Self::CheckInv { .. } => 14,
      Self::RangeCheck { .. } => 15,
      Self::Load { .. } => 16,
      Self::Store { .. } => 17,
      Self::KeccakLeaf { .. } => 18,
      Self::Compose { .. } => 19,
      Self::TypeCheck { .. } => 20,
      Self::Done => 21,
      Self::MLoad { .. } => 22,
      Self::MStore { .. } => 23,
      Self::MStore8 { .. } => 24,
      Self::SLoad { .. } => 25,
      Self::SStore { .. } => 26,
      Self::TLoad { .. } => 27,
      Self::TStore { .. } => 28,
    }
  }
}
