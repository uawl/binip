//! Row → GF(2^128) column encoding.
//!
//! Each [`vm::Row`] has 8 `u32` fields.  We embed each field into
//! [`GF2_128`] as the low 32 bits (the canonical unsigned embedding).
//! This gives 8 "columns" per row that can be assembled into MLE
//! evaluation tables.

use field::GF2_128;
use vm::Row;

/// Number of field-element columns per row.
pub const NUM_COLS: usize = 8;

/// Column indices.
pub const COL_PC: usize = 0;
pub const COL_OP: usize = 1;
pub const COL_IN0: usize = 2;
pub const COL_IN1: usize = 3;
pub const COL_IN2: usize = 4;
pub const COL_OUT: usize = 5;
pub const COL_FLAGS: usize = 6;
pub const COL_ADVICE: usize = 7;

/// Encode a single [`Row`] into `NUM_COLS` field elements.
///
/// The order matches the `COL_*` constants above.
#[inline]
pub fn encode_row(row: &Row) -> [GF2_128; NUM_COLS] {
  [
    GF2_128::from(row.pc as u64),
    GF2_128::from(row.op as u64),
    GF2_128::from(row.in0 as u64),
    GF2_128::from(row.in1 as u64),
    GF2_128::from(row.in2 as u64),
    GF2_128::from(row.out as u64),
    GF2_128::from(row.flags as u64),
    GF2_128::from(row.advice as u64),
  ]
}

#[cfg(test)]
mod tests {
  use super::*;
  use field::FieldElem;

  #[test]
  fn encode_row_zero() {
    let row = Row { pc: 0, op: 0, in0: 0, in1: 0, in2: 0, out: 0, flags: 0, advice: 0 };
    let cols = encode_row(&row);
    for c in &cols {
      assert!(c.is_zero());
    }
  }

  #[test]
  fn encode_row_preserves_fields() {
    let row = Row { pc: 1, op: 2, in0: 3, in1: 4, in2: 5, out: 6, flags: 7, advice: 8 };
    let cols = encode_row(&row);
    assert_eq!(cols[COL_PC], GF2_128::from(1u64));
    assert_eq!(cols[COL_OP], GF2_128::from(2u64));
    assert_eq!(cols[COL_IN0], GF2_128::from(3u64));
    assert_eq!(cols[COL_IN1], GF2_128::from(4u64));
    assert_eq!(cols[COL_IN2], GF2_128::from(5u64));
    assert_eq!(cols[COL_OUT], GF2_128::from(6u64));
    assert_eq!(cols[COL_FLAGS], GF2_128::from(7u64));
    assert_eq!(cols[COL_ADVICE], GF2_128::from(8u64));
  }

  #[test]
  fn encode_row_max_u32() {
    let row = Row {
      pc: u32::MAX,
      op: u32::MAX,
      in0: u32::MAX,
      in1: u32::MAX,
      in2: u32::MAX,
      out: u32::MAX,
      flags: u32::MAX,
      advice: u32::MAX,
    };
    let expected = GF2_128::from(u32::MAX as u64);
    let cols = encode_row(&row);
    for c in &cols {
      assert_eq!(*c, expected);
    }
  }
}
