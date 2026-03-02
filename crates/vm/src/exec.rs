//! VM interpreter: executes a [`MicroOp`] program and produces a trace.
//!
//! Registers and trace fields are all native `u32`.  A 256-bit EVM value
//! is represented as 8 consecutive registers (little-endian limb order).

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
  pub regs:   RegisterFile,
  pub advice: AdviceTape,
  pub trace:  Vec<Row>,
  pub memory: Vec<u32>,
  /// Persistent storage: U256 key (8 limbs) → U256 value (8 limbs).
  pub storage: HashMap<[u32; 8], [u32; 8]>,
  /// Transient storage (EIP-1153): reset at end of each transaction.
  pub transient_storage: HashMap<[u32; 8], [u32; 8]>,
  /// Valid JUMPDEST byte offsets in the EVM bytecode.
  pub jumpdest_table: BTreeSet<u32>,
  pub pc:     u32,
  pub halted: bool,
}

impl Vm {
  pub fn new(advice: AdviceTape) -> Self {
    Self {
      regs: RegisterFile::new(), advice, trace: Vec::new(),
      memory: Vec::new(), storage: HashMap::new(),
      transient_storage: HashMap::new(),
      jumpdest_table: BTreeSet::new(), pc: 0, halted: false,
    }
  }

  /// Create a VM with pre-allocated memory of `n_words` zero-initialized u32 slots.
  pub fn with_memory(advice: AdviceTape, n_words: usize) -> Self {
    Self {
      regs: RegisterFile::new(), advice, trace: Vec::new(),
      memory: vec![0u32; n_words], storage: HashMap::new(),
      transient_storage: HashMap::new(),
      jumpdest_table: BTreeSet::new(), pc: 0, halted: false,
    }
  }

  /// Create a VM with a pre-populated jumpdest table.
  pub fn with_jumpdests(advice: AdviceTape, jumpdests: BTreeSet<u32>) -> Self {
    Self {
      regs: RegisterFile::new(), advice, trace: Vec::new(),
      memory: Vec::new(), storage: HashMap::new(),
      transient_storage: HashMap::new(),
      jumpdest_table: jumpdests, pc: 0, halted: false,
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
    let pc = self.pc;

    let row = match op {
      // ── Add32 ─────────────────────────────────────────────────────────────
      MicroOp::Add32 { dst, a, b, cin, cout } => {
        let va = self.regs.read(*a);
        let vb = self.regs.read(*b);
        let c_in = self.regs.flag(*cin) as u32;
        let sum = (va as u64) + (vb as u64) + (c_in as u64);
        let result = sum as u32;
        self.regs.write(*dst, result);
        self.regs.set_flag(*cout, (sum >> 32) != 0);
        Row { pc, op: op.tag(), in0: va, in1: vb, in2: 0, out: result, flags: (sum >> 32) as u32, advice: 0 }
      }

      // ── Mul32 ─────────────────────────────────────────────────────────────
      MicroOp::Mul32 { dst_lo, dst_hi, a, b } => {
        let va = self.regs.read(*a);
        let vb = self.regs.read(*b);
        let product = (va as u64) * (vb as u64);
        let lo = product as u32;
        let hi = (product >> 32) as u32;
        self.regs.write(*dst_lo, lo);
        self.regs.write(*dst_hi, hi);
        Row { pc, op: op.tag(), in0: va, in1: vb, in2: 0, out: lo, flags: 0, advice: 0 }
      }

      // ── And32 ─────────────────────────────────────────────────────────────
      MicroOp::And32 { dst, a, b } => {
        let va = self.regs.read(*a);
        let vb = self.regs.read(*b);
        let result = va & vb;
        self.regs.write(*dst, result);
        Row { pc, op: op.tag(), in0: va, in1: vb, in2: 0, out: result, flags: 0, advice: 0 }
      }

      // ── Xor32 ─────────────────────────────────────────────────────────────
      MicroOp::Xor32 { dst, a, b } => {
        let va = self.regs.read(*a);
        let vb = self.regs.read(*b);
        let result = va ^ vb;
        self.regs.write(*dst, result);
        Row { pc, op: op.tag(), in0: va, in1: vb, in2: 0, out: result, flags: 0, advice: 0 }
      }

      // ── Not32 ─────────────────────────────────────────────────────────────
      MicroOp::Not32 { dst, src } => {
        let vs = self.regs.read(*src);
        let result = !vs;
        self.regs.write(*dst, result);
        Row { pc, op: op.tag(), in0: vs, in1: 0, in2: 0, out: result, flags: 0, advice: 0 }
      }

      // ── Rot32 ─────────────────────────────────────────────────────────────
      MicroOp::Rot32 { dst, src, shift } => {
        let vs = self.regs.read(*src);
        let result = vs.rotate_right(*shift as u32);
        self.regs.write(*dst, result);
        Row { pc, op: op.tag(), in0: vs, in1: 0, in2: 0, out: result, flags: 0, advice: 0 }
      }

      // ── Shr32 ─────────────────────────────────────────────────────────────
      MicroOp::Shr32 { dst, src, cin, shift } => {
        let vs = self.regs.read(*src);
        let vc = self.regs.read(*cin);
        let result = if *shift == 0 {
          vs
        } else if *shift >= 32 {
          vc
        } else {
          (vs >> shift) | (vc << (32 - *shift as u32))
        };
        self.regs.write(*dst, result);
        Row { pc, op: op.tag(), in0: vs, in1: vc, in2: 0, out: result, flags: 0, advice: 0 }
      }

      // ── Shl32 ─────────────────────────────────────────────────────────────
      MicroOp::Shl32 { dst, src, cin, shift } => {
        let vs = self.regs.read(*src);
        let vc = self.regs.read(*cin);
        let result = if *shift == 0 {
          vs
        } else if *shift >= 32 {
          vc
        } else {
          (vs << shift) | (vc >> (32 - *shift as u32))
        };
        self.regs.write(*dst, result);
        Row { pc, op: op.tag(), in0: vs, in1: vc, in2: 0, out: result, flags: 0, advice: 0 }
      }

      // ── Chi32 ─────────────────────────────────────────────────────────────
      MicroOp::Chi32 { dst, a, b, c } => {
        let va = self.regs.read(*a);
        let vb = self.regs.read(*b);
        let vc = self.regs.read(*c);
        let result = va ^ ((!vb) & vc);
        self.regs.write(*dst, result);
        Row { pc, op: op.tag(), in0: va, in1: vb, in2: vc, out: result, flags: 0, advice: 0 }
      }

      // ── Const ─────────────────────────────────────────────────────────────
      MicroOp::Const { dst, val } => {
        self.regs.write(*dst, *val);
        Row { pc, op: op.tag(), in0: 0, in1: 0, in2: 0, out: *val, flags: 0, advice: 0 }
      }

      // ── Mov ───────────────────────────────────────────────────────────────
      MicroOp::Mov { dst, src } => {
        let vs = self.regs.read(*src);
        self.regs.write(*dst, vs);
        Row { pc, op: op.tag(), in0: vs, in1: 0, in2: 0, out: vs, flags: 0, advice: 0 }
      }

      // ── AdviceLoad ──────────────────────────────────────────────────────────
      MicroOp::AdviceLoad { dst } => {
        let val = self.advice.next().ok_or(VmError::AdviceExhausted(pc))?;
        self.regs.write(*dst, val);
        Row { pc, op: op.tag(), in0: 0, in1: 0, in2: 0, out: val, flags: 0, advice: val }
      }

      // ── CheckDiv ──────────────────────────────────────────────────────────
      MicroOp::CheckDiv { quot, rem, dividend, divisor } => {
        let p = self.regs.read(*dividend);
        let d = self.regs.read(*divisor);
        let q = self.regs.read(*quot);
        let r = self.regs.read(*rem);
        if d == 0 {
          return Err(VmError::DivisionByZero(pc));
        }
        if d.wrapping_mul(q).wrapping_add(r) != p {
          return Err(VmError::AdviceCheckFailed(pc, "dividend ≠ divisor×quot + rem"));
        }
        if r >= d {
          return Err(VmError::AdviceCheckFailed(pc, "remainder ≥ divisor"));
        }
        Row { pc, op: op.tag(), in0: p, in1: d, in2: q, out: r, flags: 0, advice: 0 }
      }

      // ── CheckMul ──────────────────────────────────────────────────────────
      MicroOp::CheckMul { q_lo, q_hi, a, b } => {
        let va = self.regs.read(*a);
        let vb = self.regs.read(*b);
        let ql = self.regs.read(*q_lo);
        let qh = self.regs.read(*q_hi);
        let product = (va as u64) * (vb as u64);
        let expected_lo = product as u32;
        let expected_hi = (product >> 32) as u32;
        if ql != expected_lo || qh != expected_hi {
          return Err(VmError::AdviceCheckFailed(pc, "a*b ≠ q_hi:q_lo"));
        }
        Row { pc, op: op.tag(), in0: va, in1: vb, in2: ql, out: qh, flags: 0, advice: 0 }
      }

      // ── CheckInv ──────────────────────────────────────────────────────────
      MicroOp::CheckInv { a, a_inv } => {
        let va = self.regs.read(*a);
        let vi = self.regs.read(*a_inv);
        if va.wrapping_mul(vi) != 1 {
          return Err(VmError::AdviceCheckFailed(pc, "a * a_inv ≠ 1 (mod 2^32)"));
        }
        Row { pc, op: op.tag(), in0: va, in1: vi, in2: 0, out: 1, flags: 0, advice: 0 }
      }

      // ── RangeCheck ────────────────────────────────────────────────────────
      MicroOp::RangeCheck { r, bits } => {
        let vr = self.regs.read(*r);
        let b = *bits;
        if b < 32 && vr >= (1u32 << b) {
          return Err(VmError::AdviceCheckFailed(pc, "value out of range"));
        }
        // b >= 32 always passes for u32
        Row { pc, op: op.tag(), in0: vr, in1: b as u32, in2: 0, out: 0, flags: 0, advice: 0 }
      }

      // ── Load ──────────────────────────────────────────────────────────────
      MicroOp::Load { dst, addr } => {
        let a = *addr as usize;
        if a >= self.memory.len() {
          return Err(VmError::MemoryOutOfBounds(pc, *addr));
        }
        let val = self.memory[a];
        self.regs.write(*dst, val);
        Row { pc, op: op.tag(), in0: *addr, in1: 0, in2: 0, out: val, flags: 0, advice: 0 }
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
        Row { pc, op: op.tag(), in0: *addr, in1: val, in2: 0, out: 0, flags: 0, advice: 0 }
      }

      // ── KeccakLeaf ────────────────────────────────────────────────────────
      MicroOp::KeccakLeaf { dst_commit, input } => {
        let vi = self.regs.read(*input);
        // At VM level, the commitment is a placeholder handle; the real
        // Keccak sub-proof is handled outside the main circuit.
        let commit_handle = vi.wrapping_mul(0x9E37_79B9); // non-cryptographic tag
        self.regs.write(*dst_commit, commit_handle);
        Row { pc, op: op.tag(), in0: vi, in1: 0, in2: 0, out: commit_handle, flags: 0, advice: 0 }
      }

      // ── Compose ────────────────────────────────────────────────────────────
      MicroOp::Compose { dst, left, right } => {
        let vl = self.regs.read(*left);
        let vr = self.regs.read(*right);
        let handle = vl ^ vr;
        self.regs.write(*dst, handle);
        Row { pc, op: op.tag(), in0: vl, in1: vr, in2: 0, out: handle, flags: 0, advice: 0 }
      }

      // ── TypeCheck ───────────────────────────────────────────────────────────
      MicroOp::TypeCheck { opcode, pre, post } => {
        let vp = self.regs.read(*pre);
        let vq = self.regs.read(*post);
        Row { pc, op: op.tag(), in0: vp, in1: vq, in2: *opcode as u32, out: 0, flags: 0, advice: 0 }
      }

      // ── MLoad ─────────────────────────────────────────────────────────────
      MicroOp::MLoad { dst, offset_reg } => {
        let byte_off = self.regs.read(*offset_reg) as usize;
        let word_base = byte_off / 4;
        let end = word_base + 8;
        if end > self.memory.len() {
          return Err(VmError::MemoryOutOfBounds(pc, byte_off as u32));
        }
        for i in 0..8u8 {
          // EVM is big-endian: memory word at lowest address → highest limb.
          let val = self.memory[word_base + i as usize];
          self.regs.write(*dst + (7 - i), val);
        }
        Row { pc, op: op.tag(), in0: byte_off as u32, in1: 0, in2: 0, out: self.regs.read(*dst), flags: 0, advice: 0 }
      }

      // ── MStore ────────────────────────────────────────────────────────────
      MicroOp::MStore { offset_reg, src } => {
        let byte_off = self.regs.read(*offset_reg) as usize;
        let word_base = byte_off / 4;
        let end = word_base + 8;
        // Auto-extend memory.
        if end > self.memory.len() {
          self.memory.resize(end, 0);
        }
        for i in 0..8u8 {
          let val = self.regs.read(*src + (7 - i));
          self.memory[word_base + i as usize] = val;
        }
        Row { pc, op: op.tag(), in0: byte_off as u32, in1: self.regs.read(*src), in2: 0, out: 0, flags: 0, advice: 0 }
      }

      // ── MStore8 ───────────────────────────────────────────────────────────
      MicroOp::MStore8 { offset_reg, src } => {
        let byte_off = self.regs.read(*offset_reg) as usize;
        let word_idx = byte_off / 4;
        let byte_pos = byte_off % 4;
        if word_idx >= self.memory.len() {
          self.memory.resize(word_idx + 1, 0);
        }
        let byte_val = (self.regs.read(*src) & 0xFF) as u8;
        // Big-endian byte ordering within each u32 word.
        let shift = (3 - byte_pos) * 8;
        let mask = !(0xFFu32 << shift);
        self.memory[word_idx] = (self.memory[word_idx] & mask) | ((byte_val as u32) << shift);
        Row { pc, op: op.tag(), in0: byte_off as u32, in1: byte_val as u32, in2: 0, out: 0, flags: 0, advice: 0 }
      }

      // ── SLoad ─────────────────────────────────────────────────────────────
      MicroOp::SLoad { dst, key_reg } => {
        let mut key = [0u32; 8];
        for i in 0..8u8 {
          key[i as usize] = self.regs.read(*key_reg + i);
        }
        let val = self.storage.get(&key).copied().unwrap_or([0u32; 8]);
        for i in 0..8u8 {
          self.regs.write(*dst + i, val[i as usize]);
        }
        Row { pc, op: op.tag(), in0: key[0], in1: 0, in2: 0, out: val[0], flags: 0, advice: 0 }
      }

      // ── SStore ────────────────────────────────────────────────────────────
      MicroOp::SStore { key_reg, val_reg } => {
        let mut key = [0u32; 8];
        let mut val = [0u32; 8];
        for i in 0..8u8 {
          key[i as usize] = self.regs.read(*key_reg + i);
          val[i as usize] = self.regs.read(*val_reg + i);
        }
        self.storage.insert(key, val);
        Row { pc, op: op.tag(), in0: key[0], in1: val[0], in2: 0, out: 0, flags: 0, advice: 0 }
      }

      // ── TLoad ─────────────────────────────────────────────────────────────
      MicroOp::TLoad { dst, key_reg } => {
        let mut key = [0u32; 8];
        for i in 0..8u8 {
          key[i as usize] = self.regs.read(*key_reg + i);
        }
        let val = self.transient_storage.get(&key).copied().unwrap_or([0u32; 8]);
        for i in 0..8u8 {
          self.regs.write(*dst + i, val[i as usize]);
        }
        Row { pc, op: op.tag(), in0: key[0], in1: 0, in2: 0, out: val[0], flags: 0, advice: 0 }
      }

      // ── TStore ────────────────────────────────────────────────────────────
      MicroOp::TStore { key_reg, val_reg } => {
        let mut key = [0u32; 8];
        let mut val = [0u32; 8];
        for i in 0..8u8 {
          key[i as usize] = self.regs.read(*key_reg + i);
          val[i as usize] = self.regs.read(*val_reg + i);
        }
        self.transient_storage.insert(key, val);
        Row { pc, op: op.tag(), in0: key[0], in1: val[0], in2: 0, out: 0, flags: 0, advice: 0 }
      }

      // ── Done ──────────────────────────────────────────────────────────────
      MicroOp::Done => {
        self.halted = true;
        Row { pc, op: op.tag(), in0: 0, in1: 0, in2: 0, out: 0, flags: 0, advice: 0 }
      }
    };

    self.trace.push(row);
    Ok(())
  }
}
