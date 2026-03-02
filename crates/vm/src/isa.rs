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

  /// Assert `(a as u64) * (b as u64) == ((q_hi as u64) << 32) | (q_lo as u64)`.
  /// Used for multi-limb U256 multiplication verification.  Cost: ~64 AND.
  CheckMul {
    q_lo: Reg,
    q_hi: Reg,
    a: Reg,
    b: Reg,
  },

  /// Assert `regs[a] * regs[a_inv] ≡ 1 (mod 2^32)`.
  /// Used for modular inverse verification.  Cost: ~64 AND.
  CheckInv {
    a: Reg,
    a_inv: Reg,
  },

  /// Assert `regs[r] < 2^bits`.  Cost: ~`bits` AND.
  RangeCheck {
    r: Reg,
    bits: u8,
  },

  // ── Memory ────────────────────────────────────────────────────────────────
  /// Load a 32-bit word from VM memory at `addr` into `dst`.
  Load { dst: Reg, addr: u32 },

  /// Store the 32-bit value in `src` to VM memory at `addr`.
  Store { addr: u32, src: Reg },

  /// Load a U256 (8 consecutive u32 words) from VM memory at the byte
  /// offset in `regs[offset_reg] / 4` into `dst..dst+7`.
  MLoad { dst: Reg, offset_reg: Reg },

  /// Store a U256 (8 words from `src..src+7`) to VM memory at the byte
  /// offset in `regs[offset_reg] / 4`.
  MStore { offset_reg: Reg, src: Reg },

  /// Store the low byte of `regs[src]` to VM memory at the byte offset
  /// in `regs[offset_reg]`.  The surrounding u32 word is read-modify-written.
  MStore8 { offset_reg: Reg, src: Reg },

  // ── Storage ────────────────────────────────────────────────────────────────
  /// Load a U256 value from persistent storage by key.
  /// Key = 8 limbs at `key_reg..key_reg+7`, result written to `dst..dst+7`.
  SLoad { dst: Reg, key_reg: Reg },

  /// Store a U256 value to persistent storage.
  /// Key = 8 limbs at `key_reg..key_reg+7`, value = 8 limbs at `val_reg..val_reg+7`.
  SStore { key_reg: Reg, val_reg: Reg },

  // ── Transient storage (EIP-1153) ──────────────────────────────────────────
  /// Load a U256 value from transient storage by key.
  /// Key = 8 limbs at `key_reg..key_reg+7`, result written to `dst..dst+7`.
  /// Transient storage is reset at the end of each transaction.
  TLoad { dst: Reg, key_reg: Reg },

  /// Store a U256 value to transient storage.
  /// Key = 8 limbs at `key_reg..key_reg+7`, value = 8 limbs at `val_reg..val_reg+7`.
  TStore { key_reg: Reg, val_reg: Reg },

  // ── Keccak sub-proof ──────────────────────────────────────────────────────
  /// Absorb a Keccak sub-proof commitment.  The Keccak circuit is proved
  /// separately; the main circuit only sees the commitment.  Cost: 0 AND.
  KeccakLeaf { dst_commit: Reg, input: Reg },

  // ── Proof-tree / EVM structural ───────────────────────────────────────────
  /// Sequential composition: `dst` receives a handle to `Seq(left, right)`.
  /// At VM level this is a structural annotation; the circuit enforces typing.
  Compose { dst: Reg, left: Reg, right: Reg },

  /// EVM opcode structural type-check annotation.
  /// `pre` / `post` registers hold opaque `EvmStateType` handles.
  TypeCheck { opcode: u8, pre: Reg, post: Reg },

  // ── Control ───────────────────────────────────────────────────────────────
  /// Halt execution.  Any remaining program ops are skipped.
  Done,
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
