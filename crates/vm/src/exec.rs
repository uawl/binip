//! VM interpreter: executes a [`MicroOp`] program and produces a trace.
//!
//! Registers and trace fields are all native `u32`.  A 256-bit EVM value
//! is represented as 8 consecutive registers (little-endian limb order).

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
}

// ─────────────────────────────────────────────────────────────────────────────────

/// Register-based Meta-VM.
pub struct Vm {
  pub regs:   RegisterFile,
  pub advice: AdviceTape,
  pub trace:  Vec<Row>,
  pub pc:     u32,
}

impl Vm {
  pub fn new(advice: AdviceTape) -> Self {
    Self { regs: RegisterFile::new(), advice, trace: Vec::new(), pc: 0 }
  }

  pub fn run(&mut self, program: &[MicroOp]) -> Result<(), VmError> {
    for op in program {
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
    };

    self.trace.push(row);
    Ok(())
  }
}
