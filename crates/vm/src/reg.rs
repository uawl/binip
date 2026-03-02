//! Register file: 64 × u32 data registers + 8 boolean flag registers.
//!
//! Each register holds exactly one 32-bit limb.  A 256-bit EVM word
//! occupies **8 consecutive registers** (r_base .. r_base+7, little-endian
//! limb order).  This matches the circuit's native column width and
//! eliminates the U256 wrapper entirely from the trace.

use crate::isa::{FlagReg, Reg};

/// Number of general-purpose data registers.
/// 64 slots = 8 EVM words of headroom per execution step.
pub const NUM_REGS: usize = 64;
/// Number of carry/boolean flag registers.
pub const NUM_FLAGS: usize = 8;

/// The register file state at a single execution step.
#[derive(Debug, Clone)]
pub struct RegisterFile {
  regs:  [u32; NUM_REGS],
  flags: [bool; NUM_FLAGS],
}

impl Default for RegisterFile {
  fn default() -> Self {
    Self { regs: [0u32; NUM_REGS], flags: [false; NUM_FLAGS] }
  }
}

impl RegisterFile {
  pub fn new() -> Self {
    Self::default()
  }

  #[inline]
  pub fn read(&self, r: Reg) -> u32 {
    self.regs[r as usize]
  }

  #[inline]
  pub fn write(&mut self, r: Reg, val: u32) {
    self.regs[r as usize] = val;
  }

  #[inline]
  pub fn flag(&self, f: FlagReg) -> bool {
    self.flags[f as usize]
  }

  #[inline]
  pub fn set_flag(&mut self, f: FlagReg, val: bool) {
    self.flags[f as usize] = val;
  }
}
