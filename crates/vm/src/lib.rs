pub mod advice;
pub mod compile;
pub mod exec;
pub mod isa;
pub mod reg;
pub mod row;
pub mod schedule;

pub use advice::AdviceTape;
pub use compile::{compile, compile_push, Compiled};
pub use exec::{Vm, VmError};
pub use isa::{FlagReg, MicroOp, Reg};
pub use reg::RegisterFile;
pub use row::Row;
pub use schedule::{Schedule, ScheduleError, Workgroup, schedule};

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

  // ── CheckMul ─────────────────────────────────────────────────────────────

  #[test]
  fn check_mul_ok() {
    // 7 * 9 = 63.  q_lo=63, q_hi=0.
    let mut vm = Vm::new(AdviceTape::new([63u32, 0u32]));
    vm.regs.write(0, 7u32);
    vm.regs.write(1, 9u32);
    vm.run(&[
      MicroOp::AdviceLoad { dst: 2 },
      MicroOp::AdviceLoad { dst: 3 },
      MicroOp::CheckMul { q_lo: 2, q_hi: 3, a: 0, b: 1 },
    ]).unwrap();
    assert_eq!(vm.trace.len(), 3);
  }

  #[test]
  fn check_mul_large() {
    // 0xFFFF_FFFF * 0xFFFF_FFFF = 0xFFFF_FFFE_0000_0001
    let a = 0xFFFF_FFFFu32;
    let product = (a as u64) * (a as u64);
    let lo = product as u32;
    let hi = (product >> 32) as u32;
    let mut vm = Vm::new(AdviceTape::new([lo, hi]));
    vm.regs.write(0, a);
    vm.regs.write(1, a);
    vm.run(&[
      MicroOp::AdviceLoad { dst: 2 },
      MicroOp::AdviceLoad { dst: 3 },
      MicroOp::CheckMul { q_lo: 2, q_hi: 3, a: 0, b: 1 },
    ]).unwrap();
  }

  #[test]
  fn check_mul_wrong_fails() {
    let mut vm = Vm::new(AdviceTape::new([99u32, 0u32]));
    vm.regs.write(0, 7u32);
    vm.regs.write(1, 9u32);
    let res = vm.run(&[
      MicroOp::AdviceLoad { dst: 2 },
      MicroOp::AdviceLoad { dst: 3 },
      MicroOp::CheckMul { q_lo: 2, q_hi: 3, a: 0, b: 1 },
    ]);
    assert!(matches!(res, Err(VmError::AdviceCheckFailed(_, _))));
  }

  // ── CheckInv ─────────────────────────────────────────────────────────────

  #[test]
  fn check_inv_ok() {
    // 3 * 0xAAAAAAAB == 1 (mod 2^32)
    let a = 3u32;
    let a_inv = 0xAAAA_AAABu32;
    assert_eq!(a.wrapping_mul(a_inv), 1);
    let mut vm = Vm::new(AdviceTape::new([a_inv]));
    vm.regs.write(0, a);
    vm.run(&[
      MicroOp::AdviceLoad { dst: 1 },
      MicroOp::CheckInv { a: 0, a_inv: 1 },
    ]).unwrap();
  }

  #[test]
  fn check_inv_wrong_fails() {
    let mut vm = Vm::new(AdviceTape::new([42u32]));
    vm.regs.write(0, 3u32);
    let res = vm.run(&[
      MicroOp::AdviceLoad { dst: 1 },
      MicroOp::CheckInv { a: 0, a_inv: 1 },
    ]);
    assert!(matches!(res, Err(VmError::AdviceCheckFailed(_, _))));
  }

  // ── RangeCheck ───────────────────────────────────────────────────────────

  #[test]
  fn range_check_ok() {
    let mut vm = Vm::new(AdviceTape::default());
    vm.regs.write(0, 255u32);
    vm.run(&[MicroOp::RangeCheck { r: 0, bits: 8 }]).unwrap();
  }

  #[test]
  fn range_check_exact_boundary() {
    // 2^8 - 1 = 255 fits in 8 bits
    let mut vm = Vm::new(AdviceTape::default());
    vm.regs.write(0, 255u32);
    vm.run(&[MicroOp::RangeCheck { r: 0, bits: 8 }]).unwrap();

    // 256 does NOT fit in 8 bits
    let mut vm2 = Vm::new(AdviceTape::default());
    vm2.regs.write(0, 256u32);
    let res = vm2.run(&[MicroOp::RangeCheck { r: 0, bits: 8 }]);
    assert!(matches!(res, Err(VmError::AdviceCheckFailed(_, _))));
  }

  #[test]
  fn range_check_32_bits_always_ok() {
    let mut vm = Vm::new(AdviceTape::default());
    vm.regs.write(0, u32::MAX);
    vm.run(&[MicroOp::RangeCheck { r: 0, bits: 32 }]).unwrap();
  }

  // ── Load / Store ─────────────────────────────────────────────────────────

  #[test]
  fn store_then_load() {
    let mut vm = Vm::with_memory(AdviceTape::default(), 16);
    vm.regs.write(0, 0xCAFE_BABEu32);
    vm.run(&[
      MicroOp::Store { addr: 5, src: 0 },
      MicroOp::Load  { dst: 1, addr: 5 },
    ]).unwrap();
    assert_eq!(vm.regs.read(1), 0xCAFE_BABEu32);
  }

  #[test]
  fn store_auto_extends_memory() {
    let mut vm = Vm::new(AdviceTape::default()); // empty memory
    vm.regs.write(0, 42u32);
    vm.run(&[MicroOp::Store { addr: 100, src: 0 }]).unwrap();
    assert_eq!(vm.memory.len(), 101);
    assert_eq!(vm.memory[100], 42u32);
  }

  #[test]
  fn load_out_of_bounds_fails() {
    let mut vm = Vm::new(AdviceTape::default()); // empty memory
    let res = vm.run(&[MicroOp::Load { dst: 0, addr: 0 }]);
    assert!(matches!(res, Err(VmError::MemoryOutOfBounds(_, _))));
  }

  // ── KeccakLeaf ───────────────────────────────────────────────────────────

  #[test]
  fn keccak_leaf_produces_handle() {
    let mut vm = Vm::new(AdviceTape::default());
    vm.regs.write(0, 12345u32);
    vm.run(&[MicroOp::KeccakLeaf { dst_commit: 1, input: 0 }]).unwrap();
    assert_ne!(vm.regs.read(1), 0);
  }

  // ── Done ─────────────────────────────────────────────────────────────────

  #[test]
  fn done_halts_execution() {
    let mut vm = Vm::new(AdviceTape::default());
    vm.run(&[
      MicroOp::Const { dst: 0, val: 1 },
      MicroOp::Done,
      MicroOp::Const { dst: 1, val: 2 }, // should NOT execute
    ]).unwrap();
    assert_eq!(vm.regs.read(0), 1u32);
    assert_eq!(vm.regs.read(1), 0u32); // still zero — skipped
    assert!(vm.halted);
    // trace has 2 rows: Const + Done (the third op was skipped)
    assert_eq!(vm.trace.len(), 2);
  }

  // ── MStore + MLoad ─────────────────────────────────────────────────────

  #[test]
  fn mstore_then_mload_roundtrip() {
    let mut vm = Vm::new(AdviceTape::default());
    // offset = 0 (byte offset), value = U256 in slot1
    vm.regs.write(0, 0u32);  // offset in limb(0,0)
    // Write a recognizable U256 into slot1 (regs 8..15)
    for i in 0..8u8 {
      vm.regs.write(8 + i, (i as u32 + 1) * 0x11);
    }
    vm.run(&[
      MicroOp::MStore { offset_reg: 0, src: 8 },
      MicroOp::MLoad  { dst: 16, offset_reg: 0 },
    ]).unwrap();
    // Loaded value in regs 16..23 should match slot1 (regs 8..15)
    for i in 0..8u8 {
      assert_eq!(vm.regs.read(16 + i), vm.regs.read(8 + i));
    }
  }

  #[test]
  fn mstore_auto_extends() {
    let mut vm = Vm::new(AdviceTape::default());
    vm.regs.write(0, 0u32); // offset = 0
    vm.regs.write(8, 42u32);
    vm.run(&[MicroOp::MStore { offset_reg: 0, src: 8 }]).unwrap();
    assert_eq!(vm.memory.len(), 8); // 8 u32 words for one U256
  }

  #[test]
  fn mload_out_of_bounds() {
    let mut vm = Vm::new(AdviceTape::default());
    vm.regs.write(0, 0u32);
    let res = vm.run(&[MicroOp::MLoad { dst: 8, offset_reg: 0 }]);
    assert!(matches!(res, Err(VmError::MemoryOutOfBounds(_, _))));
  }

  // ── MStore8 ──────────────────────────────────────────────────────────────

  #[test]
  fn mstore8_writes_single_byte() {
    let mut vm = Vm::with_memory(AdviceTape::default(), 4);
    vm.regs.write(0, 0u32);    // offset = byte 0
    vm.regs.write(1, 0xABu32); // value — only low byte (0xAB)
    vm.run(&[MicroOp::MStore8 { offset_reg: 0, src: 1 }]).unwrap();
    // Byte 0 is in the highest byte of word 0 (big-endian within word).
    assert_eq!(vm.memory[0] >> 24, 0xAB);
  }

  // ── SStore + SLoad ─────────────────────────────────────────────────────

  #[test]
  fn sstore_then_sload_roundtrip() {
    let mut vm = Vm::new(AdviceTape::default());
    // key in regs 0..7 = [1, 0, 0, 0, 0, 0, 0, 0]
    vm.regs.write(0, 1u32);
    // value in regs 8..15 = [42, 0, 0, 0, 0, 0, 0, 0]
    vm.regs.write(8, 42u32);
    vm.run(&[
      MicroOp::SStore { key_reg: 0, val_reg: 8 },
      MicroOp::SLoad  { dst: 16, key_reg: 0 },
    ]).unwrap();
    assert_eq!(vm.regs.read(16), 42u32);
    for i in 1..8u8 {
      assert_eq!(vm.regs.read(16 + i), 0u32);
    }
    assert_eq!(vm.storage.len(), 1);
  }

  #[test]
  fn sload_missing_key_returns_zero() {
    let mut vm = Vm::new(AdviceTape::default());
    vm.regs.write(0, 99u32); // key that doesn't exist
    vm.run(&[MicroOp::SLoad { dst: 8, key_reg: 0 }]).unwrap();
    for i in 0..8u8 {
      assert_eq!(vm.regs.read(8 + i), 0u32);
    }
  }

  #[test]
  fn sstore_overwrites() {
    let mut vm = Vm::new(AdviceTape::default());
    vm.regs.write(0, 1u32); // key
    vm.regs.write(8, 10u32); // first value
    vm.run(&[MicroOp::SStore { key_reg: 0, val_reg: 8 }]).unwrap();
    vm.regs.write(8, 20u32); // second value
    vm.pc = 0;
    vm.run(&[MicroOp::SStore { key_reg: 0, val_reg: 8 }]).unwrap();
    vm.pc = 0;
    vm.run(&[MicroOp::SLoad { dst: 16, key_reg: 0 }]).unwrap();
    assert_eq!(vm.regs.read(16), 20u32); // latest value
    assert_eq!(vm.storage.len(), 1);
  }

  // ── Jumpdest table ───────────────────────────────────────────────────────

  #[test]
  fn vm_with_jumpdests() {
    use std::collections::BTreeSet;
    let jt: BTreeSet<u32> = [10, 20, 30].into_iter().collect();
    let vm = Vm::with_jumpdests(AdviceTape::default(), jt.clone());
    assert_eq!(vm.jumpdest_table, jt);
  }
}
