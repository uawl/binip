//! Fixed-width execution trace row — one entry per [`crate::isa::MicroOp`].
//!
//! Every field is a native `u32` matching the register width.  The circuit
//! layer reads `Vec<Row>` and emits one constraint set per row; unused slots
//! are zero.

/// One row of the execution trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Row {
  /// Program counter at the start of this step.
  pub pc: u32,
  /// [`crate::isa::MicroOp::tag`] discriminant.
  pub op: u32,
  /// First source operand (snapshotted *before* execution).
  pub in0: u32,
  /// Second source operand.
  pub in1: u32,
  /// Third source operand (Chi32 only; zero otherwise).
  pub in2: u32,
  /// Value written to the destination register.
  pub out: u32,
  /// Carry/flag bits produced by this step (bit 0 = carry-out).
  pub flags: u32,
  /// Advice limb consumed by this step (0 if none).
  pub advice: u32,
}
