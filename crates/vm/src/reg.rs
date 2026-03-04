//! Register file: 16 × u128 general-purpose registers.
//!
//! Each register holds exactly one 128-bit limb.  A 256-bit EVM word
//! occupies **2 consecutive registers** (r_base, r_base+1, little-endian
//! limb order).  This matches the circuit's native column width and
//! eliminates the U256 wrapper entirely from the trace.
//!
//! Carry/borrow values from `Add128` are stored as 0 or 1 (zero-extended)
//! in ordinary registers — there is no separate flag register file.

use crate::isa::Reg;

/// Number of general-purpose data registers.
/// 16 slots = 5 EVM-word stack slots (×2 limbs each) + 6 scratch.
pub const NUM_REGS: usize = 16;

/// The register file state at a single execution step.
#[derive(Debug, Clone)]
pub struct RegisterFile {
  regs: [u128; NUM_REGS],
}

impl Default for RegisterFile {
  fn default() -> Self {
    Self {
      regs: [0u128; NUM_REGS],
    }
  }
}

impl RegisterFile {
  pub fn new() -> Self {
    Self::default()
  }

  #[inline]
  pub fn read(&self, r: Reg) -> u128 {
    self.regs[r as usize]
  }

  #[inline]
  pub fn write(&mut self, r: Reg, val: u128) {
    self.regs[r as usize] = val;
  }
}
