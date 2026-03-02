pub mod advice;
pub mod exec;
pub mod isa;
pub mod reg;
pub mod row;

pub use advice::AdviceTape;
pub use exec::{Vm, VmError};
pub use isa::{FlagReg, MicroOp, Reg};
pub use reg::RegisterFile;
pub use row::Row;

#[cfg(test)]
mod tests {
  use super::*;

  // ── Add32 ───────────────────────────────────────────────────────────────────

  #[test]
  fn add32_no_carry() {
    let mut vm = Vm::new(AdviceTape::default());
    vm.regs.write(0, 3u32);
    vm.regs.write(1, 5u32);
    vm.run(&[MicroOp::Add32 { dst: 2, a: 0, b: 1, cin: 0, cout: 0 }]).unwrap();
    assert_eq!(vm.regs.read(2), 8u32);
    assert!(!vm.regs.flag(0));
    assert_eq!(vm.trace.len(), 1);
  }

  #[test]
  fn add32_carry_out() {
    let mut vm = Vm::new(AdviceTape::default());
    vm.regs.write(0, 0xFFFF_FFFFu32);
    vm.regs.write(1, 1u32);
    vm.run(&[MicroOp::Add32 { dst: 2, a: 0, b: 1, cin: 0, cout: 1 }]).unwrap();
    assert_eq!(vm.regs.read(2), 0u32);
    assert!(vm.regs.flag(1));
  }

  #[test]
  fn add32_carry_chain() {
    let mut vm = Vm::new(AdviceTape::default());
    vm.regs.write(0, 0xFFFF_FFFFu32); // lo a
    vm.regs.write(1, 1u32);           // lo b
    vm.regs.write(2, 0u32);           // hi a
    vm.regs.write(3, 0u32);           // hi b
    vm.run(&[
      MicroOp::Add32 { dst: 4, a: 0, b: 1, cin: 0, cout: 1 },
      MicroOp::Add32 { dst: 5, a: 2, b: 3, cin: 1, cout: 2 },
    ]).unwrap();
    assert_eq!(vm.regs.read(4), 0u32);
    assert_eq!(vm.regs.read(5), 1u32);
    assert!(!vm.regs.flag(2));
  }

  // ── Mul32 ───────────────────────────────────────────────────────────────────

  #[test]
  fn mul32_split() {
    let mut vm = Vm::new(AdviceTape::default());
    vm.regs.write(0, 0xFFFFu32);
    vm.regs.write(1, 0xFFFFu32);
    vm.run(&[MicroOp::Mul32 { dst_lo: 2, dst_hi: 3, a: 0, b: 1 }]).unwrap();
    let product: u64 = 0xFFFF * 0xFFFF;
    assert_eq!(vm.regs.read(2), (product & 0xFFFF_FFFF) as u32);
    assert_eq!(vm.regs.read(3), (product >> 32) as u32);
  }

  // ── Bitwise ────────────────────────────────────────────────────────────────

  #[test]
  fn and_xor_not() {
    let mut vm = Vm::new(AdviceTape::default());
    vm.regs.write(0, 0b1100u32);
    vm.regs.write(1, 0b1010u32);
    vm.run(&[
      MicroOp::And32 { dst: 2, a: 0, b: 1 },
      MicroOp::Xor32 { dst: 3, a: 0, b: 1 },
      MicroOp::Not32 { dst: 4, src: 0 },
    ]).unwrap();
    assert_eq!(vm.regs.read(2), 0b1000u32);
    assert_eq!(vm.regs.read(3), 0b0110u32);
    assert_eq!(vm.regs.read(4), !0b1100u32);
  }

  #[test]
  fn rot32_correctness() {
    let mut vm = Vm::new(AdviceTape::default());
    vm.regs.write(0, 0x8000_0001u32);
    vm.run(&[MicroOp::Rot32 { dst: 1, src: 0, shift: 1 }]).unwrap();
    assert_eq!(vm.regs.read(1), 0xC000_0000u32);
  }

  // ── Shl32 ────────────────────────────────────────────────────────────────

  #[test]
  fn shl32_no_carry_in() {
    let mut vm = Vm::new(AdviceTape::default());
    vm.regs.write(0, 0x0Fu32);
    vm.regs.write(1, 0u32);
    vm.run(&[MicroOp::Shl32 { dst: 2, src: 0, cin: 1, shift: 4 }]).unwrap();
    assert_eq!(vm.regs.read(2), 0xF0u32);
  }

  #[test]
  fn shl32_carry_chain() {
    let mut vm = Vm::new(AdviceTape::default());
    vm.regs.write(0, 0x8000_0000u32); // lo
    vm.regs.write(1, 0u32);           // hi
    vm.regs.write(2, 0u32);           // cin for lo
    vm.run(&[
      MicroOp::Shl32 { dst: 3, src: 0, cin: 2, shift: 1 },
      MicroOp::Shl32 { dst: 4, src: 1, cin: 0, shift: 1 },
    ]).unwrap();
    assert_eq!(vm.regs.read(3), 0u32);
    assert_eq!(vm.regs.read(4), 1u32);
  }

  // ── Chi32 (Keccak χ) ─────────────────────────────────────────────────────────

  #[test]
  fn chi32_keccak() {
    let mut vm = Vm::new(AdviceTape::default());
    vm.regs.write(0, 0b1010u32);
    vm.regs.write(1, 0b1100u32);
    vm.regs.write(2, 0b1111u32);
    vm.run(&[MicroOp::Chi32 { dst: 3, a: 0, b: 1, c: 2 }]).unwrap();
    assert_eq!(vm.regs.read(3), 0b1001u32);
  }

  // ── Const / Mov ─────────────────────────────────────────────────────────────

  #[test]
  fn const_and_mov() {
    let mut vm = Vm::new(AdviceTape::default());
    vm.run(&[
      MicroOp::Const { dst: 0, val: 0xDEAD_BEEFu32 },
      MicroOp::Mov   { dst: 1, src: 0 },
    ]).unwrap();
    assert_eq!(vm.regs.read(0), 0xDEAD_BEEFu32);
    assert_eq!(vm.regs.read(1), 0xDEAD_BEEFu32);
    assert_eq!(vm.trace.len(), 2);
  }

  // ── AdviceLoad + CheckDiv ──────────────────────────────────────────────────

  #[test]
  fn advice_check_div_ok() {
    // 17 / 5 = 3 remainder 2
    let mut vm = Vm::new(AdviceTape::new([3u32, 2u32]));
    vm.regs.write(0, 17u32);
    vm.regs.write(1, 5u32);
    vm.run(&[
      MicroOp::AdviceLoad { dst: 2 },
      MicroOp::AdviceLoad { dst: 3 },
      MicroOp::CheckDiv { quot: 2, rem: 3, dividend: 0, divisor: 1 },
    ]).unwrap();
    assert_eq!(vm.regs.read(2), 3u32);
    assert_eq!(vm.regs.read(3), 2u32);
    assert_eq!(vm.trace.len(), 3);
    assert_eq!(vm.trace[0].advice, 3u32);
    assert_eq!(vm.trace[1].advice, 2u32);
  }

  #[test]
  fn advice_check_div_wrong_quot_fails() {
    let mut vm = Vm::new(AdviceTape::new([4u32, 2u32])); // quot=4 is wrong
    vm.regs.write(0, 17u32);
    vm.regs.write(1, 5u32);
    let res = vm.run(&[
      MicroOp::AdviceLoad { dst: 2 },
      MicroOp::AdviceLoad { dst: 3 },
      MicroOp::CheckDiv { quot: 2, rem: 3, dividend: 0, divisor: 1 },
    ]);
    assert!(matches!(res, Err(VmError::AdviceCheckFailed(_, _))));
  }

  #[test]
  fn advice_check_div_by_zero_fails() {
    let mut vm = Vm::new(AdviceTape::new([1u32, 0u32]));
    vm.regs.write(0, 10u32);
    vm.regs.write(1, 0u32); // divisor = 0
    let res = vm.run(&[
      MicroOp::AdviceLoad { dst: 2 },
      MicroOp::AdviceLoad { dst: 3 },
      MicroOp::CheckDiv { quot: 2, rem: 3, dividend: 0, divisor: 1 },
    ]);
    assert!(matches!(res, Err(VmError::DivisionByZero(_))));
  }

  #[test]
  fn advice_tape_exhausted_fails() {
    let mut vm = Vm::new(AdviceTape::default());
    let res = vm.run(&[MicroOp::AdviceLoad { dst: 0 }]);
    assert!(matches!(res, Err(VmError::AdviceExhausted(_))));
  }
}
