pub mod advice;
pub mod compile;
pub mod exec;
pub mod isa;
pub mod reg;
pub mod row;
pub mod schedule;

pub use advice::AdviceTape;
pub use compile::{Compiled, compile, compile_push};
pub use exec::{Vm, VmError};
pub use isa::{FlagReg, MicroOp, Reg};
pub use reg::RegisterFile;
pub use row::Row;
pub use schedule::{Schedule, ScheduleError, Workgroup, schedule};

#[cfg(test)]
mod tests {
  use super::*;

  // ── Add128 ──────────────────────────────────────────────────────────────────

  #[test]
  fn add128_no_carry() {
    let mut vm = Vm::new(AdviceTape::default());
    vm.regs.write(0, 3u128);
    vm.regs.write(1, 5u128);
    vm.run(&[MicroOp::Add128 {
      dst: 2,
      a: 0,
      b: 1,
      cin: 0,
      cout: 0,
    }])
    .unwrap();
    assert_eq!(vm.regs.read(2), 8u128);
    assert!(!vm.regs.flag(0));
    assert_eq!(vm.trace.len(), 1);
  }

  #[test]
  fn add128_carry_out() {
    let mut vm = Vm::new(AdviceTape::default());
    vm.regs.write(0, u128::MAX);
    vm.regs.write(1, 1u128);
    vm.run(&[MicroOp::Add128 {
      dst: 2,
      a: 0,
      b: 1,
      cin: 0,
      cout: 1,
    }])
    .unwrap();
    assert_eq!(vm.regs.read(2), 0u128);
    assert!(vm.regs.flag(1));
  }

  #[test]
  fn add128_carry_chain() {
    let mut vm = Vm::new(AdviceTape::default());
    vm.regs.write(0, u128::MAX); // lo a
    vm.regs.write(1, 1u128); // lo b
    vm.regs.write(2, 0u128); // hi a
    vm.regs.write(3, 0u128); // hi b
    vm.run(&[
      MicroOp::Add128 {
        dst: 4,
        a: 0,
        b: 1,
        cin: 0,
        cout: 1,
      },
      MicroOp::Add128 {
        dst: 5,
        a: 2,
        b: 3,
        cin: 1,
        cout: 2,
      },
    ])
    .unwrap();
    assert_eq!(vm.regs.read(4), 0u128);
    assert_eq!(vm.regs.read(5), 1u128);
    assert!(!vm.regs.flag(2));
  }

  // ── Mul128 ──────────────────────────────────────────────────────────────────

  #[test]
  fn mul128_split() {
    let mut vm = Vm::new(AdviceTape::default());
    vm.regs.write(0, 0xFFFF_FFFFu128);
    vm.regs.write(1, 0xFFFF_FFFFu128);
    vm.run(&[MicroOp::Mul128 {
      dst_lo: 2,
      dst_hi: 3,
      a: 0,
      b: 1,
    }])
    .unwrap();
    let product: u128 = 0xFFFF_FFFF * 0xFFFF_FFFF;
    assert_eq!(vm.regs.read(2), product);
    assert_eq!(vm.regs.read(3), 0u128);
  }

  // ── Bitwise ────────────────────────────────────────────────────────────────

  #[test]
  fn and_xor_not() {
    let mut vm = Vm::new(AdviceTape::default());
    vm.regs.write(0, 0b1100u128);
    vm.regs.write(1, 0b1010u128);
    vm.run(&[
      MicroOp::And128 { dst: 2, a: 0, b: 1 },
      MicroOp::Xor128 { dst: 3, a: 0, b: 1 },
      MicroOp::Not128 { dst: 4, src: 0 },
    ])
    .unwrap();
    assert_eq!(vm.regs.read(2), 0b1000u128);
    assert_eq!(vm.regs.read(3), 0b0110u128);
    assert_eq!(vm.regs.read(4), !0b1100u128);
  }

  #[test]
  fn rot128_correctness() {
    let mut vm = Vm::new(AdviceTape::default());
    vm.regs.write(0, (1u128 << 127) | 1);
    vm.run(&[MicroOp::Rot128 {
      dst: 1,
      src: 0,
      shift: 1,
    }])
    .unwrap();
    assert_eq!(vm.regs.read(1), (1u128 << 127) | (1u128 << 126));
  }

  // ── Shl128 ────────────────────────────────────────────────────────────────

  #[test]
  fn shl128_no_carry_in() {
    let mut vm = Vm::new(AdviceTape::default());
    vm.regs.write(0, 0x0Fu128);
    vm.regs.write(1, 0u128);
    vm.run(&[MicroOp::Shl128 {
      dst: 2,
      src: 0,
      cin: 1,
      shift: 4,
    }])
    .unwrap();
    assert_eq!(vm.regs.read(2), 0xF0u128);
  }

  #[test]
  fn shl128_carry_chain() {
    let mut vm = Vm::new(AdviceTape::default());
    vm.regs.write(0, 1u128 << 127); // lo
    vm.regs.write(1, 0u128); // hi
    vm.regs.write(2, 0u128); // cin for lo
    vm.run(&[
      MicroOp::Shl128 {
        dst: 3,
        src: 0,
        cin: 2,
        shift: 1,
      },
      MicroOp::Shl128 {
        dst: 4,
        src: 1,
        cin: 0,
        shift: 1,
      },
    ])
    .unwrap();
    assert_eq!(vm.regs.read(3), 0u128);
    assert_eq!(vm.regs.read(4), 1u128);
  }

  // ── Chi128 (Keccak χ) ────────────────────────────────────────────────────

  #[test]
  fn chi128_keccak() {
    let mut vm = Vm::new(AdviceTape::default());
    vm.regs.write(0, 0b1010u128);
    vm.regs.write(1, 0b1100u128);
    vm.regs.write(2, 0b1111u128);
    vm.run(&[MicroOp::Chi128 {
      dst: 3,
      a: 0,
      b: 1,
      c: 2,
    }])
    .unwrap();
    assert_eq!(vm.regs.read(3), 0b1001u128);
  }

  // ── Const / Mov ────────────────────────────────────────────────────────────

  #[test]
  fn const_and_mov() {
    let mut vm = Vm::new(AdviceTape::default());
    vm.run(&[
      MicroOp::Const {
        dst: 0,
        val: 0xDEAD_BEEFu128,
      },
      MicroOp::Mov { dst: 1, src: 0 },
    ])
    .unwrap();
    assert_eq!(vm.regs.read(0), 0xDEAD_BEEFu128);
    assert_eq!(vm.regs.read(1), 0xDEAD_BEEFu128);
    assert_eq!(vm.trace.len(), 2);
  }

  // ── AdviceLoad + CheckDiv ─────────────────────────────────────────────────

  #[test]
  fn advice_check_div_ok() {
    // 17 / 5 = 3 remainder 2
    let mut vm = Vm::new(AdviceTape::new([3u128, 2u128]));
    vm.regs.write(0, 17u128);
    vm.regs.write(1, 5u128);
    vm.run(&[
      MicroOp::AdviceLoad { dst: 2 },
      MicroOp::AdviceLoad { dst: 3 },
      MicroOp::CheckDiv {
        quot: 2,
        rem: 3,
        dividend: 0,
        divisor: 1,
      },
    ])
    .unwrap();
    assert_eq!(vm.regs.read(2), 3u128);
    assert_eq!(vm.regs.read(3), 2u128);
    assert_eq!(vm.trace.len(), 3);
    assert_eq!(vm.trace[0].advice, 3u128);
    assert_eq!(vm.trace[1].advice, 2u128);
  }

  #[test]
  fn advice_check_div_wrong_quot_fails() {
    let mut vm = Vm::new(AdviceTape::new([4u128, 2u128]));
    vm.regs.write(0, 17u128);
    vm.regs.write(1, 5u128);
    let res = vm.run(&[
      MicroOp::AdviceLoad { dst: 2 },
      MicroOp::AdviceLoad { dst: 3 },
      MicroOp::CheckDiv {
        quot: 2,
        rem: 3,
        dividend: 0,
        divisor: 1,
      },
    ]);
    assert!(matches!(res, Err(VmError::AdviceCheckFailed(_, _))));
  }

  #[test]
  fn advice_check_div_by_zero_fails() {
    let mut vm = Vm::new(AdviceTape::new([1u128, 0u128]));
    vm.regs.write(0, 10u128);
    vm.regs.write(1, 0u128);
    let res = vm.run(&[
      MicroOp::AdviceLoad { dst: 2 },
      MicroOp::AdviceLoad { dst: 3 },
      MicroOp::CheckDiv {
        quot: 2,
        rem: 3,
        dividend: 0,
        divisor: 1,
      },
    ]);
    assert!(matches!(res, Err(VmError::DivisionByZero(_))));
  }

  #[test]
  fn advice_tape_exhausted_fails() {
    let mut vm = Vm::new(AdviceTape::default());
    let res = vm.run(&[MicroOp::AdviceLoad { dst: 0 }]);
    assert!(matches!(res, Err(VmError::AdviceExhausted(_))));
  }

  // ── CheckMul ──────────────────────────────────────────────────────────────

  #[test]
  fn check_mul_ok() {
    // 7 * 9 = 63.  q_lo=63, q_hi=0.
    let mut vm = Vm::new(AdviceTape::new([63u128, 0u128]));
    vm.regs.write(0, 7u128);
    vm.regs.write(1, 9u128);
    vm.run(&[
      MicroOp::AdviceLoad { dst: 2 },
      MicroOp::AdviceLoad { dst: 3 },
      MicroOp::CheckMul {
        q_lo: 2,
        q_hi: 3,
        a: 0,
        b: 1,
      },
    ])
    .unwrap();
    assert_eq!(vm.trace.len(), 3);
  }

  #[test]
  fn check_mul_large() {
    // Test with large u128 values
    let a = u128::MAX;
    // a * 1 = a, with hi = 0
    let mut vm = Vm::new(AdviceTape::new([a, 0u128]));
    vm.regs.write(0, a);
    vm.regs.write(1, 1u128);
    vm.run(&[
      MicroOp::AdviceLoad { dst: 2 },
      MicroOp::AdviceLoad { dst: 3 },
      MicroOp::CheckMul {
        q_lo: 2,
        q_hi: 3,
        a: 0,
        b: 1,
      },
    ])
    .unwrap();
  }

  #[test]
  fn check_mul_wrong_fails() {
    let mut vm = Vm::new(AdviceTape::new([99u128, 0u128]));
    vm.regs.write(0, 7u128);
    vm.regs.write(1, 9u128);
    let res = vm.run(&[
      MicroOp::AdviceLoad { dst: 2 },
      MicroOp::AdviceLoad { dst: 3 },
      MicroOp::CheckMul {
        q_lo: 2,
        q_hi: 3,
        a: 0,
        b: 1,
      },
    ]);
    assert!(matches!(res, Err(VmError::AdviceCheckFailed(_, _))));
  }

  // ── CheckInv ──────────────────────────────────────────────────────────────

  #[test]
  fn check_inv_ok() {
    // Find inverse of 3 mod 2^128.
    // 3 * inv = 1 mod 2^128
    // inv = modinv(3, 2^128)
    // 3 * 0xAAAA_AAAA_AAAA_AAAA_AAAA_AAAA_AAAA_AAAB = 1 (mod 2^128)
    let a = 3u128;
    let a_inv = 0xAAAA_AAAA_AAAA_AAAA_AAAA_AAAA_AAAA_AAABu128;
    assert_eq!(a.wrapping_mul(a_inv), 1);
    let mut vm = Vm::new(AdviceTape::new([a_inv]));
    vm.regs.write(0, a);
    vm.run(&[
      MicroOp::AdviceLoad { dst: 1 },
      MicroOp::CheckInv { a: 0, a_inv: 1 },
    ])
    .unwrap();
  }

  #[test]
  fn check_inv_wrong_fails() {
    let mut vm = Vm::new(AdviceTape::new([42u128]));
    vm.regs.write(0, 3u128);
    let res = vm.run(&[
      MicroOp::AdviceLoad { dst: 1 },
      MicroOp::CheckInv { a: 0, a_inv: 1 },
    ]);
    assert!(matches!(res, Err(VmError::AdviceCheckFailed(_, _))));
  }

  // ── RangeCheck ────────────────────────────────────────────────────────────

  #[test]
  fn range_check_ok() {
    let mut vm = Vm::new(AdviceTape::default());
    vm.regs.write(0, 255u128);
    vm.run(&[MicroOp::RangeCheck { r: 0, bits: 8 }]).unwrap();
  }

  #[test]
  fn range_check_exact_boundary() {
    let mut vm = Vm::new(AdviceTape::default());
    vm.regs.write(0, 255u128);
    vm.run(&[MicroOp::RangeCheck { r: 0, bits: 8 }]).unwrap();

    let mut vm2 = Vm::new(AdviceTape::default());
    vm2.regs.write(0, 256u128);
    let res = vm2.run(&[MicroOp::RangeCheck { r: 0, bits: 8 }]);
    assert!(matches!(res, Err(VmError::AdviceCheckFailed(_, _))));
  }

  #[test]
  fn range_check_128_bits_always_ok() {
    let mut vm = Vm::new(AdviceTape::default());
    vm.regs.write(0, u128::MAX);
    vm.run(&[MicroOp::RangeCheck { r: 0, bits: 128 }]).unwrap();
  }

  // ── Load / Store ──────────────────────────────────────────────────────────

  #[test]
  fn store_then_load() {
    let mut vm = Vm::with_memory(AdviceTape::default(), 16);
    vm.regs.write(0, 0xCAFE_BABEu128);
    vm.run(&[
      MicroOp::Store { addr: 5, src: 0 },
      MicroOp::Load { dst: 1, addr: 5 },
    ])
    .unwrap();
    assert_eq!(vm.regs.read(1), 0xCAFE_BABEu128);
  }

  #[test]
  fn store_auto_extends_memory() {
    let mut vm = Vm::new(AdviceTape::default());
    vm.regs.write(0, 42u128);
    vm.run(&[MicroOp::Store { addr: 100, src: 0 }]).unwrap();
    assert_eq!(vm.memory.len(), 101);
    assert_eq!(vm.memory[100], 42u128);
  }

  #[test]
  fn load_out_of_bounds_fails() {
    let mut vm = Vm::new(AdviceTape::default());
    let res = vm.run(&[MicroOp::Load { dst: 0, addr: 0 }]);
    assert!(matches!(res, Err(VmError::MemoryOutOfBounds(_, _))));
  }

  // ── KeccakLeaf ────────────────────────────────────────────────────────────

  #[test]
  fn keccak_leaf_produces_handle() {
    let mut vm = Vm::new(AdviceTape::default());
    vm.regs.write(0, 12345u128);
    vm.run(&[MicroOp::KeccakLeaf {
      dst_commit: 1,
      input: 0,
    }])
    .unwrap();
    assert_ne!(vm.regs.read(1), 0);
  }

  // ── Done ──────────────────────────────────────────────────────────────────

  #[test]
  fn done_halts_execution() {
    let mut vm = Vm::new(AdviceTape::default());
    vm.run(&[
      MicroOp::Const { dst: 0, val: 1 },
      MicroOp::Done,
      MicroOp::Const { dst: 1, val: 2 },
    ])
    .unwrap();
    assert_eq!(vm.regs.read(0), 1u128);
    assert_eq!(vm.regs.read(1), 0u128);
    assert!(vm.halted);
    assert_eq!(vm.trace.len(), 2);
  }

  // ── MStore + MLoad ────────────────────────────────────────────────────────

  #[test]
  fn mstore_then_mload_roundtrip() {
    let mut vm = Vm::new(AdviceTape::default());
    vm.regs.write(0, 0u128); // offset
    // Write a U256 into slot1 (regs 2..3)
    vm.regs.write(2, 0x11u128);
    vm.regs.write(3, 0x22u128);
    vm.run(&[
      MicroOp::MStore {
        offset_reg: 0,
        src: 2,
      },
      MicroOp::MLoad {
        dst: 4,
        offset_reg: 0,
      },
    ])
    .unwrap();
    assert_eq!(vm.regs.read(4), vm.regs.read(2));
    assert_eq!(vm.regs.read(5), vm.regs.read(3));
  }

  #[test]
  fn mstore_auto_extends() {
    let mut vm = Vm::new(AdviceTape::default());
    vm.regs.write(0, 0u128); // offset = 0
    vm.regs.write(2, 42u128);
    vm.run(&[MicroOp::MStore {
      offset_reg: 0,
      src: 2,
    }])
    .unwrap();
    assert_eq!(vm.memory.len(), 2); // 2 u128 words for one U256
  }

  #[test]
  fn mload_out_of_bounds() {
    let mut vm = Vm::new(AdviceTape::default());
    vm.regs.write(0, 0u128);
    let res = vm.run(&[MicroOp::MLoad {
      dst: 2,
      offset_reg: 0,
    }]);
    assert!(matches!(res, Err(VmError::MemoryOutOfBounds(_, _))));
  }

  // ── MStore8 ───────────────────────────────────────────────────────────────

  #[test]
  fn mstore8_writes_single_byte() {
    let mut vm = Vm::with_memory(AdviceTape::default(), 4);
    vm.regs.write(0, 0u128); // offset = byte 0
    vm.regs.write(1, 0xABu128);
    vm.run(&[MicroOp::MStore8 {
      offset_reg: 0,
      src: 1,
    }])
    .unwrap();
    // Byte 0 is in the highest byte of word 0 (big-endian within word).
    assert_eq!(vm.memory[0] >> 120, 0xAB);
  }

  // ── SStore + SLoad ────────────────────────────────────────────────────────

  #[test]
  fn sstore_then_sload_roundtrip() {
    let mut vm = Vm::new(AdviceTape::default());
    // key in regs 0..1 = [1, 0]
    vm.regs.write(0, 1u128);
    // value in regs 2..3 = [42, 0]
    vm.regs.write(2, 42u128);
    vm.run(&[
      MicroOp::SStore {
        key_reg: 0,
        val_reg: 2,
      },
      MicroOp::SLoad {
        dst: 4,
        key_reg: 0,
      },
    ])
    .unwrap();
    assert_eq!(vm.regs.read(4), 42u128);
    assert_eq!(vm.regs.read(5), 0u128);
    assert_eq!(vm.storage.len(), 1);
  }

  #[test]
  fn sload_missing_key_returns_zero() {
    let mut vm = Vm::new(AdviceTape::default());
    vm.regs.write(0, 99u128);
    vm.run(&[MicroOp::SLoad { dst: 2, key_reg: 0 }]).unwrap();
    assert_eq!(vm.regs.read(2), 0u128);
    assert_eq!(vm.regs.read(3), 0u128);
  }

  #[test]
  fn sstore_overwrites() {
    let mut vm = Vm::new(AdviceTape::default());
    vm.regs.write(0, 1u128); // key
    vm.regs.write(2, 10u128); // first value
    vm.run(&[MicroOp::SStore {
      key_reg: 0,
      val_reg: 2,
    }])
    .unwrap();
    vm.regs.write(2, 20u128); // second value
    vm.pc = 0;
    vm.run(&[MicroOp::SStore {
      key_reg: 0,
      val_reg: 2,
    }])
    .unwrap();
    vm.pc = 0;
    vm.run(&[MicroOp::SLoad {
      dst: 4,
      key_reg: 0,
    }])
    .unwrap();
    assert_eq!(vm.regs.read(4), 20u128);
    assert_eq!(vm.storage.len(), 1);
  }

  // ── Jumpdest table ────────────────────────────────────────────────────────

  #[test]
  fn vm_with_jumpdests() {
    use std::collections::BTreeSet;
    let jt: BTreeSet<u32> = [10, 20, 30].into_iter().collect();
    let vm = Vm::with_jumpdests(AdviceTape::default(), jt.clone());
    assert_eq!(vm.jumpdest_table, jt);
  }
}
