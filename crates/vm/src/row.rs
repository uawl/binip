//! Fixed-width execution trace row — one entry per [`crate::isa::MicroOp`].
//!
//! Every field is a native `u128` matching the register width.  The circuit
//! layer reads `Vec<Row>` and emits one constraint set per row; unused slots
//! are zero.

/// One row of the execution trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Row {
  /// Program counter at the start of this step.
  pub pc: u128,
  /// [`crate::isa::MicroOp::tag`] discriminant.
  pub op: u128,
  /// First source operand (snapshotted *before* execution).
  pub in0: u128,
  /// Second source operand.
  pub in1: u128,
  /// Third source operand (Chi128 only; zero otherwise).
  pub in2: u128,
  /// Value written to the destination register.
  pub out: u128,
  /// Carry/flag bits produced by this step (bit 0 = carry-out).
  pub flags: u128,
  /// Advice limb consumed by this step (0 if none).
  pub advice: u128,
}
