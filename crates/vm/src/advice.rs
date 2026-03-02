//! Advice witness tape — pre-computed limb values consumed by
//! [`crate::isa::MicroOp::AdviceLoad`].
//!
//! ## Design (Advice Witness pattern)
//! Expensive computations (e.g., 256-bit division) are *non-deterministically*
//! provided by the prover as sequences of **u128 limbs** rather than
//! recomputed inside the circuit.  The prover fills this tape before calling
//! [`crate::exec::Vm::run`].  Each consumed limb is embedded in
//! [`crate::row::Row::advice`] and later verified by the circuit via a paired
//! `Check*` constraint row.

/// Sequential tape of pre-computed u128 advice limbs.
#[derive(Debug, Clone, Default)]
pub struct AdviceTape {
  values: Vec<u128>,
  cursor: usize,
}

impl AdviceTape {
  pub fn new(values: impl IntoIterator<Item = u128>) -> Self {
    Self {
      values: values.into_iter().collect(),
      cursor: 0,
    }
  }

  /// Number of limbs not yet consumed.
  pub fn remaining(&self) -> usize {
    self.values.len().saturating_sub(self.cursor)
  }

  /// Current read position.
  pub fn cursor(&self) -> usize {
    self.cursor
  }
}

impl Iterator for AdviceTape {
  type Item = u128;

  fn next(&mut self) -> Option<u128> {
    let v = self.values.get(self.cursor).copied();
    if v.is_some() {
      self.cursor += 1;
    }
    v
  }
}
