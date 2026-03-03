//! VM interpreter: executes a [`MicroOp`] program and produces a trace.
//!
//! Registers and trace fields are all native `u128`.  A 256-bit EVM value
//! is represented as 2 consecutive registers (little-endian limb order).

use std::collections::{BTreeSet, HashMap};

use thiserror::Error;

use crate::{advice::AdviceTape, isa::MicroOp, reg::RegisterFile, row::Row};

// ─────────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum VmError {
  #[error("advice tape exhausted at pc={0}")]
  AdviceExhausted(u32),

  #[error("division by zero at pc={0}")]
  DivisionByZero(u32),

  #[error("advice check failed at pc={0}: {1}")]
  AdviceCheckFailed(u32, &'static str),

  #[error("memory access out of bounds at pc={0}, addr={1}")]
  MemoryOutOfBounds(u32, u32),

  #[error("storage key not found at pc={0}")]
  StorageKeyNotFound(u32),

  #[error("invalid jump destination at pc={0}, target={1}")]
  InvalidJumpDest(u32, u32),
}

// ─────────────────────────────────────────────────────────────────────────────────

/// Register-based Meta-VM.
pub struct Vm {
  pub regs: RegisterFile,
  pub advice: AdviceTape,
  pub trace: Vec<Row>,
  pub memory: Vec<u128>,
  /// Persistent storage: U256 key (2 limbs) → U256 value (2 limbs).
  pub storage: HashMap<[u128; 2], [u128; 2]>,
  /// Transient storage (EIP-1153): reset at end of each transaction.
  pub transient_storage: HashMap<[u128; 2], [u128; 2]>,
  /// Valid JUMPDEST byte offsets in the EVM bytecode.
  pub jumpdest_table: BTreeSet<u32>,
  pub pc: u32,
  pub halted: bool,
}

impl Vm {
  pub fn new(advice: AdviceTape) -> Self {
    Self {
      regs: RegisterFile::new(),
      advice,
      trace: Vec::new(),
      memory: Vec::new(),
      storage: HashMap::new(),
      transient_storage: HashMap::new(),
      jumpdest_table: BTreeSet::new(),
      pc: 0,
      halted: false,
    }
  }

  /// Create a VM with pre-allocated memory of `n_words` zero-initialized u128 slots.
  pub fn with_memory(advice: AdviceTape, n_words: usize) -> Self {
    Self {
      regs: RegisterFile::new(),
      advice,
      trace: Vec::new(),
      memory: vec![0u128; n_words],
      storage: HashMap::new(),
      transient_storage: HashMap::new(),
      jumpdest_table: BTreeSet::new(),
      pc: 0,
      halted: false,
    }
  }

  /// Create a VM with a pre-populated jumpdest table.
  pub fn with_jumpdests(advice: AdviceTape, jumpdests: BTreeSet<u32>) -> Self {
    Self {
      regs: RegisterFile::new(),
      advice,
      trace: Vec::new(),
      memory: Vec::new(),
      storage: HashMap::new(),
      transient_storage: HashMap::new(),
      jumpdest_table: jumpdests,
      pc: 0,
      halted: false,
    }
  }

  pub fn run(&mut self, program: &[MicroOp]) -> Result<(), VmError> {
    for op in program {
      if self.halted {
        break;
      }
      self.step(op)?;
      self.pc += 1;
    }
    Ok(())
  }

  fn step(&mut self, op: &MicroOp) -> Result<(), VmError> {
    let pc = self.pc as u128;

    let row = match op {
      // ── Add128 ────────────────────────────────────────────────────────────
      MicroOp::Add128 {
        dst,
        a,
        b,
        cin,
        cout,
      } => {
        let va = self.regs.read(*a);
        let vb = self.regs.read(*b);
        let c_in = self.regs.flag(*cin) as u128;
        let (s1, c1) = va.overflowing_add(vb);
        let (result, c2) = s1.overflowing_add(c_in);
        self.regs.write(*dst, result);
        self.regs.set_flag(*cout, c1 | c2);
        Row {
          pc,
          op: op.tag() as u128,
          in0: va,
          in1: vb,
          in2: 0,
          out: result,
          flags: (c1 | c2) as u128,
          advice: 0,
        }
      }

      // ── Mul128 ────────────────────────────────────────────────────────────
      MicroOp::Mul128 {
        dst_lo,
        dst_hi,
        a,
        b,
      } => {
        let va = self.regs.read(*a);
        let vb = self.regs.read(*b);
        let (lo, hi) = widening_mul_u128(va, vb);
        self.regs.write(*dst_lo, lo);
        self.regs.write(*dst_hi, hi);
        Row {
          pc,
          op: op.tag() as u128,
          in0: va,
          in1: vb,
          in2: 0,
          out: lo,
          flags: 0,
          advice: 0,
        }
      }

      // ── And128 ────────────────────────────────────────────────────────────
      MicroOp::And128 { dst, a, b } => {
        let va = self.regs.read(*a);
        let vb = self.regs.read(*b);
        let result = va & vb;
        self.regs.write(*dst, result);
        Row {
          pc,
          op: op.tag() as u128,
          in0: va,
          in1: vb,
          in2: 0,
          out: result,
          flags: 0,
          advice: 0,
        }
      }

      // ── Xor128 ────────────────────────────────────────────────────────────
      MicroOp::Xor128 { dst, a, b } => {
        let va = self.regs.read(*a);
        let vb = self.regs.read(*b);
        let result = va ^ vb;
        self.regs.write(*dst, result);
        Row {
          pc,
          op: op.tag() as u128,
          in0: va,
          in1: vb,
          in2: 0,
          out: result,
          flags: 0,
          advice: 0,
        }
      }

      // ── Not128 ────────────────────────────────────────────────────────────
      MicroOp::Not128 { dst, src } => {
        let vs = self.regs.read(*src);
        let result = !vs;
        self.regs.write(*dst, result);
        Row {
          pc,
          op: op.tag() as u128,
          in0: vs,
          in1: 0,
          in2: 0,
          out: result,
          flags: 0,
          advice: 0,
        }
      }

      // ── Rot128 ────────────────────────────────────────────────────────────
      MicroOp::Rot128 { dst, src, shift } => {
        let vs = self.regs.read(*src);
        let result = vs.rotate_right(*shift as u32);
        self.regs.write(*dst, result);
        Row {
          pc,
          op: op.tag() as u128,
          in0: vs,
          in1: 0,
          in2: 0,
          out: result,
          flags: 0,
          advice: 0,
        }
      }

      // ── Shr128 ────────────────────────────────────────────────────────────
      MicroOp::Shr128 {
        dst,
        src,
        cin,
        shift,
      } => {
        let vs = self.regs.read(*src);
        let vc = self.regs.read(*cin);
        let s = *shift as u32;
        let result = if s == 0 {
          vs
        } else if s >= 128 {
          vc
        } else {
          (vs >> s) | (vc << (128 - s))
        };
        self.regs.write(*dst, result);
        Row {
          pc,
          op: op.tag() as u128,
          in0: vs,
          in1: vc,
          in2: 0,
          out: result,
          flags: 0,
          advice: 0,
        }
      }

      // ── Shl128 ────────────────────────────────────────────────────────────
      MicroOp::Shl128 {
        dst,
        src,
        cin,
        shift,
      } => {
        let vs = self.regs.read(*src);
        let vc = self.regs.read(*cin);
        let s = *shift as u32;
        let result = if s == 0 {
          vs
        } else if s >= 128 {
          vc
        } else {
          (vs << s) | (vc >> (128 - s))
        };
        self.regs.write(*dst, result);
        Row {
          pc,
          op: op.tag() as u128,
          in0: vs,
          in1: vc,
          in2: 0,
          out: result,
          flags: 0,
          advice: 0,
        }
      }

      // ── Chi128 ────────────────────────────────────────────────────────────
      MicroOp::Chi128 { dst, a, b, c } => {
        let va = self.regs.read(*a);
        let vb = self.regs.read(*b);
        let vc = self.regs.read(*c);
        let result = va ^ ((!vb) & vc);
        self.regs.write(*dst, result);
        Row {
          pc,
          op: op.tag() as u128,
          in0: va,
          in1: vb,
          in2: vc,
          out: result,
          flags: 0,
          advice: 0,
        }
      }

      // ── Const ─────────────────────────────────────────────────────────────
      MicroOp::Const { dst, val } => {
        self.regs.write(*dst, *val);
        Row {
          pc,
          op: op.tag() as u128,
          in0: 0,
          in1: 0,
          in2: 0,
          out: *val,
          flags: 0,
          advice: 0,
        }
      }

      // ── Mov ───────────────────────────────────────────────────────────────
      MicroOp::Mov { dst, src } => {
        let vs = self.regs.read(*src);
        self.regs.write(*dst, vs);
        Row {
          pc,
          op: op.tag() as u128,
          in0: vs,
          in1: 0,
          in2: 0,
          out: vs,
          flags: 0,
          advice: 0,
        }
      }

      // ── AdviceLoad ──────────────────────────────────────────────────────────
      MicroOp::AdviceLoad { dst } => {
        let val = self
          .advice
          .next()
          .ok_or(VmError::AdviceExhausted(self.pc))?;
        self.regs.write(*dst, val);
        Row {
          pc,
          op: op.tag() as u128,
          in0: 0,
          in1: 0,
          in2: 0,
          out: val,
          flags: 0,
          advice: val,
        }
      }

      // ── CheckDiv ──────────────────────────────────────────────────────────
      MicroOp::CheckDiv {
        quot,
        rem,
        dividend,
        divisor,
      } => {
        let p = self.regs.read(*dividend);
        let d = self.regs.read(*divisor);
        let q = self.regs.read(*quot);
        let r = self.regs.read(*rem);
        if d == 0 {
          return Err(VmError::DivisionByZero(self.pc));
        }
        if d.wrapping_mul(q).wrapping_add(r) != p {
          return Err(VmError::AdviceCheckFailed(
            self.pc,
            "dividend ≠ divisor×quot + rem",
          ));
        }
        if r >= d {
          return Err(VmError::AdviceCheckFailed(self.pc, "remainder ≥ divisor"));
        }
        Row {
          pc,
          op: op.tag() as u128,
          in0: p,
          in1: d,
          in2: q,
          out: r,
          flags: 0,
          advice: 0,
        }
      }

      // ── CheckMul ──────────────────────────────────────────────────────────
      MicroOp::CheckMul { q_lo, q_hi, a, b } => {
        let va = self.regs.read(*a);
        let vb = self.regs.read(*b);
        let ql = self.regs.read(*q_lo);
        let qh = self.regs.read(*q_hi);
        let (expected_lo, expected_hi) = widening_mul_u128(va, vb);
        if ql != expected_lo || qh != expected_hi {
          return Err(VmError::AdviceCheckFailed(self.pc, "a*b ≠ q_hi:q_lo"));
        }
        Row {
          pc,
          op: op.tag() as u128,
          in0: va,
          in1: vb,
          in2: ql,
          out: qh,
          flags: 0,
          advice: 0,
        }
      }

      // ── CheckInv ──────────────────────────────────────────────────────────
      MicroOp::CheckInv { a, a_inv } => {
        let va = self.regs.read(*a);
        let vi = self.regs.read(*a_inv);
        if va.wrapping_mul(vi) != 1 {
          return Err(VmError::AdviceCheckFailed(
            self.pc,
            "a * a_inv ≠ 1 (mod 2^128)",
          ));
        }
        Row {
          pc,
          op: op.tag() as u128,
          in0: va,
          in1: vi,
          in2: 0,
          out: 1,
          flags: 0,
          advice: 0,
        }
      }

      // ── RangeCheck ────────────────────────────────────────────────────────
      MicroOp::RangeCheck { r, bits } => {
        let vr = self.regs.read(*r);
        let b = *bits;
        if b < 128 && vr >= (1u128 << b) {
          return Err(VmError::AdviceCheckFailed(self.pc, "value out of range"));
        }
        // b >= 128 always passes for u128
        Row {
          pc,
          op: op.tag() as u128,
          in0: vr,
          in1: b as u128,
          in2: 0,
          out: 0,
          flags: 0,
          advice: 0,
        }
      }

      // ── Load ──────────────────────────────────────────────────────────────
      MicroOp::Load { dst, addr } => {
        let a = *addr as usize;
        if a >= self.memory.len() {
          return Err(VmError::MemoryOutOfBounds(self.pc, *addr));
        }
        let val = self.memory[a];
        self.regs.write(*dst, val);
        Row {
          pc,
          op: op.tag() as u128,
          in0: *addr as u128,
          in1: 0,
          in2: 0,
          out: val,
          flags: 0,
          advice: 0,
        }
      }

      // ── Store ─────────────────────────────────────────────────────────────
      MicroOp::Store { addr, src } => {
        let a = *addr as usize;
        let val = self.regs.read(*src);
        // Auto-extend memory up to the addressed location.
        if a >= self.memory.len() {
          self.memory.resize(a + 1, 0);
        }
        self.memory[a] = val;
        Row {
          pc,
          op: op.tag() as u128,
          in0: *addr as u128,
          in1: val,
          in2: 0,
          out: 0,
          flags: 0,
          advice: 0,
        }
      }

      // ── KeccakLeaf ────────────────────────────────────────────────────────
      MicroOp::KeccakLeaf { dst_commit, input } => {
        let vi = self.regs.read(*input);
        // At VM level, the commitment is a placeholder handle; the real
        // Keccak sub-proof is handled outside the main circuit.
        let commit_handle = vi.wrapping_mul(0x9E37_79B9); // non-cryptographic tag
        self.regs.write(*dst_commit, commit_handle);
        Row {
          pc,
          op: op.tag() as u128,
          in0: vi,
          in1: 0,
          in2: 0,
          out: commit_handle,
          flags: 0,
          advice: 0,
        }
      }

      // ── Compose ────────────────────────────────────────────────────────────
      MicroOp::Compose { dst, left, right } => {
        let vl = self.regs.read(*left);
        let vr = self.regs.read(*right);
        let handle = vl ^ vr;
        self.regs.write(*dst, handle);
        Row {
          pc,
          op: op.tag() as u128,
          in0: vl,
          in1: vr,
          in2: 0,
          out: handle,
          flags: 0,
          advice: 0,
        }
      }

      // ── TypeCheck ───────────────────────────────────────────────────────────
      MicroOp::TypeCheck { opcode, pre, post } => {
        let vp = self.regs.read(*pre);
        let vq = self.regs.read(*post);
        Row {
          pc,
          op: op.tag() as u128,
          in0: vp,
          in1: vq,
          in2: *opcode as u128,
          out: 0,
          flags: 0,
          advice: 0,
        }
      }

      // ── MLoad ─────────────────────────────────────────────────────────────
      MicroOp::MLoad { dst, offset_reg } => {
        let byte_off = self.regs.read(*offset_reg) as usize;
        let word_base = byte_off / 16;
        let end = word_base + 2;
        if end > self.memory.len() {
          return Err(VmError::MemoryOutOfBounds(self.pc, byte_off as u32));
        }
        for i in 0..2u8 {
          // EVM is big-endian: memory word at lowest address → highest limb.
          let val = self.memory[word_base + i as usize];
          self.regs.write(*dst + (1 - i), val);
        }
        Row {
          pc,
          op: op.tag() as u128,
          in0: byte_off as u128,
          in1: 0,
          in2: 0,
          out: self.regs.read(*dst),
          flags: 0,
          advice: 0,
        }
      }

      // ── MStore ────────────────────────────────────────────────────────────
      MicroOp::MStore { offset_reg, src } => {
        let byte_off = self.regs.read(*offset_reg) as usize;
        let word_base = byte_off / 16;
        let end = word_base + 2;
        // Auto-extend memory.
        if end > self.memory.len() {
          self.memory.resize(end, 0);
        }
        for i in 0..2u8 {
          let val = self.regs.read(*src + (1 - i));
          self.memory[word_base + i as usize] = val;
        }
        Row {
          pc,
          op: op.tag() as u128,
          in0: byte_off as u128,
          in1: self.regs.read(*src),
          in2: 0,
          out: 0,
          flags: 0,
          advice: 0,
        }
      }

      // ── MStore8 ───────────────────────────────────────────────────────────
      MicroOp::MStore8 { offset_reg, src } => {
        let byte_off = self.regs.read(*offset_reg) as usize;
        let word_idx = byte_off / 16;
        let byte_pos = byte_off % 16;
        if word_idx >= self.memory.len() {
          self.memory.resize(word_idx + 1, 0);
        }
        let byte_val = (self.regs.read(*src) & 0xFF) as u8;
        // Big-endian byte ordering within each u128 word.
        let shift = (15 - byte_pos) * 8;
        let mask: u128 = !(0xFFu128 << shift);
        self.memory[word_idx] = (self.memory[word_idx] & mask) | ((byte_val as u128) << shift);
        Row {
          pc,
          op: op.tag() as u128,
          in0: byte_off as u128,
          in1: byte_val as u128,
          in2: 0,
          out: 0,
          flags: 0,
          advice: 0,
        }
      }

      // ── SLoad ─────────────────────────────────────────────────────────────
      MicroOp::SLoad { dst, key_reg } => {
        let mut key = [0u128; 2];
        for i in 0..2u8 {
          key[i as usize] = self.regs.read(*key_reg + i);
        }
        let val = self.storage.get(&key).copied().unwrap_or([0u128; 2]);
        for i in 0..2u8 {
          self.regs.write(*dst + i, val[i as usize]);
        }
        Row {
          pc,
          op: op.tag() as u128,
          in0: key[0],
          in1: 0,
          in2: 0,
          out: val[0],
          flags: 0,
          advice: 0,
        }
      }

      // ── SStore ────────────────────────────────────────────────────────────
      MicroOp::SStore { key_reg, val_reg } => {
        let mut key = [0u128; 2];
        let mut val = [0u128; 2];
        for i in 0..2u8 {
          key[i as usize] = self.regs.read(*key_reg + i);
          val[i as usize] = self.regs.read(*val_reg + i);
        }
        self.storage.insert(key, val);
        Row {
          pc,
          op: op.tag() as u128,
          in0: key[0],
          in1: val[0],
          in2: 0,
          out: 0,
          flags: 0,
          advice: 0,
        }
      }

      // ── TLoad ─────────────────────────────────────────────────────────────
      MicroOp::TLoad { dst, key_reg } => {
        let mut key = [0u128; 2];
        for i in 0..2u8 {
          key[i as usize] = self.regs.read(*key_reg + i);
        }
        let val = self
          .transient_storage
          .get(&key)
          .copied()
          .unwrap_or([0u128; 2]);
        for i in 0..2u8 {
          self.regs.write(*dst + i, val[i as usize]);
        }
        Row {
          pc,
          op: op.tag() as u128,
          in0: key[0],
          in1: 0,
          in2: 0,
          out: val[0],
          flags: 0,
          advice: 0,
        }
      }

      // ── TStore ────────────────────────────────────────────────────────────
      MicroOp::TStore { key_reg, val_reg } => {
        let mut key = [0u128; 2];
        let mut val = [0u128; 2];
        for i in 0..2u8 {
          key[i as usize] = self.regs.read(*key_reg + i);
          val[i as usize] = self.regs.read(*val_reg + i);
        }
        self.transient_storage.insert(key, val);
        Row {
          pc,
          op: op.tag() as u128,
          in0: key[0],
          in1: val[0],
          in2: 0,
          out: 0,
          flags: 0,
          advice: 0,
        }
      }

      // ── Done ──────────────────────────────────────────────────────────────
      MicroOp::Done => {
        self.halted = true;
        Row {
          pc,
          op: op.tag() as u128,
          in0: 0,
          in1: 0,
          in2: 0,
          out: 0,
          flags: 0,
          advice: 0,
        }
      }
    };

    self.trace.push(row);
    Ok(())
  }
}

/// 128×128 → 256-bit widening multiply via schoolbook on u64 halves.
/// Returns `(lo, hi)` where the full product is `hi << 128 | lo`.
fn widening_mul_u128(a: u128, b: u128) -> (u128, u128) {
  let a_lo = a as u64 as u128;
  let a_hi = (a >> 64) as u64 as u128;
  let b_lo = b as u64 as u128;
  let b_hi = (b >> 64) as u64 as u128;

  let ll = a_lo * b_lo;
  let lh = a_lo * b_hi;
  let hl = a_hi * b_lo;
  let hh = a_hi * b_hi;

  let (mid_sum, carry1) = lh.overflowing_add(hl);
  let lo = ll.wrapping_add(mid_sum << 64);
  let carry2 = if lo < ll { 1u128 } else { 0 };
  let hi = hh + (mid_sum >> 64) + (if carry1 { 1u128 << 64 } else { 0 }) + carry2;

  (lo, hi)
}
