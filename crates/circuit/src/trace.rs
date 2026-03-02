//! Execution trace → column-major table → MLE polynomials.
//!
//! [`TraceTable`] stores the encoded columns and converts them to
//! [`poly::MlePoly`] instances (one per column, padded to a power of two)
//! suitable for the sumcheck prover.

use field::{FieldElem, GF2_128};
use poly::MlePoly;
use vm::Row;

use crate::constraint::{self, ConstraintError, NUM_TAGS};
use crate::encoder::{self, NUM_COLS};

/// Column-major representation of an execution trace.
///
/// `columns[c][i]` is the value of column `c` in row `i`.
#[derive(Debug, Clone)]
pub struct TraceTable {
  /// Column-major data.  `columns.len() == NUM_COLS`.
  pub columns: Vec<Vec<GF2_128>>,
  /// Number of rows (before padding).
  pub n_rows: usize,
}

impl TraceTable {
  /// Build a [`TraceTable`] from a slice of [`Row`]s.
  pub fn from_rows(rows: &[Row]) -> Self {
    let n = rows.len();
    let mut columns = vec![Vec::with_capacity(n); NUM_COLS];
    for row in rows {
      let encoded = encoder::encode_row(row);
      for (c, val) in encoded.iter().enumerate() {
        columns[c].push(*val);
      }
    }
    TraceTable { columns, n_rows: n }
  }

  /// Convert each column into an [`MlePoly`] by padding the evaluation
  /// table to the next power of two with zeros.
  pub fn to_mle_polys(&self) -> Vec<MlePoly> {
    let n_vars = n_vars_for(self.n_rows);
    let padded_len = 1usize << n_vars;

    self.columns.iter().map(|col| {
      let mut evals = col.clone();
      evals.resize(padded_len, GF2_128::zero());
      MlePoly::new(evals)
    }).collect()
  }

  /// Evaluate the batched constraint polynomial on every row and return
  /// the result as an MLE polynomial.
  ///
  /// Each evaluation is `sel(op_tag, β) · constraint(op_tag, cols)`.
  /// The sumcheck protocol then proves that this MLE sums to zero.
  pub fn constraint_mle(&self, beta: GF2_128) -> Result<MlePoly, ConstraintError> {
    let n_vars = n_vars_for(self.n_rows);
    let padded_len = 1usize << n_vars;
    let mut evals = Vec::with_capacity(padded_len);

    for i in 0..self.n_rows {
      let cols = std::array::from_fn::<GF2_128, 8, _>(|c| self.columns[c][i]);
      // Recover tag from the op column.
      // We stored `op` as GF2_128::from(tag as u64), so for small tags
      // the value is just the tag integer embedded in the field.
      let tag = row_tag(&cols);
      let val = if tag < NUM_TAGS {
        constraint::batched_constraint_with_tag(tag, &cols, beta)?
      } else {
        return Err(ConstraintError::UnknownTag(tag));
      };
      evals.push(val);
    }

    // Padding rows (no-op) contribute zero.
    evals.resize(padded_len, GF2_128::zero());
    Ok(MlePoly::new(evals))
  }
}

/// Extract the opcode tag (u32) from the encoded op column element.
///
/// Since tags are small integers (0..27), the GF2_128 encoding stores them
/// in the low bits.  We reconstruct the tag by testing which small integer
/// matches.
fn row_tag(cols: &[GF2_128; 8]) -> u32 {
  let op = cols[1];
  for t in 0..NUM_TAGS {
    if op == GF2_128::from(t as u64) {
      return t;
    }
  }
  u32::MAX
}

/// Minimum number of variables so that `2^n >= len`.
fn n_vars_for(len: usize) -> usize {
  if len <= 1 {
    return 0;
  }
  (usize::BITS - (len - 1).leading_zeros()) as usize
}

#[cfg(test)]
mod tests {
  use super::*;

  fn make_xor_row(a: u32, b: u32) -> Row {
    Row {
      pc: 0,
      op: 3, // Xor32
      in0: a,
      in1: b,
      in2: 0,
      out: a ^ b,
      flags: 0,
      advice: 0,
    }
  }

  #[test]
  fn trace_table_from_rows() {
    let rows = vec![make_xor_row(5, 3), make_xor_row(10, 7)];
    let table = TraceTable::from_rows(&rows);
    assert_eq!(table.n_rows, 2);
    assert_eq!(table.columns.len(), NUM_COLS);
    // Check op column
    assert_eq!(table.columns[1][0], GF2_128::from(3u64));
    assert_eq!(table.columns[1][1], GF2_128::from(3u64));
  }

  #[test]
  fn to_mle_polys_pads_to_power_of_two() {
    let rows = vec![make_xor_row(1, 2), make_xor_row(3, 4), make_xor_row(5, 6)];
    let table = TraceTable::from_rows(&rows);
    let mles = table.to_mle_polys();
    assert_eq!(mles.len(), NUM_COLS);
    // 3 rows → padded to 4 = 2^2
    for mle in &mles {
      assert_eq!(mle.n_vars, 2);
    }
  }

  #[test]
  fn constraint_mle_zero_for_valid_trace() {
    // Xor32 constraint: out + in0 + in1 = 0  (char 2)
    // With correct trace rows, the constraint MLE should sum to zero.
    let rows = vec![
      make_xor_row(5, 3),
      make_xor_row(10, 7),
      make_xor_row(0xFF, 0x0F),
      make_xor_row(0, 0),
    ];
    let table = TraceTable::from_rows(&rows);
    let beta = GF2_128::from(100u64);
    let cmle = table.constraint_mle(beta).unwrap();
    assert!(cmle.sum().is_zero(), "valid trace should have zero constraint sum");
  }

  #[test]
  fn constraint_mle_nonzero_for_invalid_trace() {
    // Deliberately bad row: out ≠ in0 XOR in1
    let bad_row = Row {
      pc: 0,
      op: 3, // Xor32
      in0: 5,
      in1: 3,
      in2: 0,
      out: 999, // wrong!
      flags: 0,
      advice: 0,
    };
    let rows = vec![bad_row];
    let table = TraceTable::from_rows(&rows);
    let beta = GF2_128::from(42u64);
    let cmle = table.constraint_mle(beta).unwrap();
    assert!(!cmle.sum().is_zero(), "invalid trace should have nonzero constraint sum");
  }

  #[test]
  fn n_vars_for_various() {
    assert_eq!(n_vars_for(0), 0);
    assert_eq!(n_vars_for(1), 0);
    assert_eq!(n_vars_for(2), 1);
    assert_eq!(n_vars_for(3), 2);
    assert_eq!(n_vars_for(4), 2);
    assert_eq!(n_vars_for(5), 3);
    assert_eq!(n_vars_for(8), 3);
    assert_eq!(n_vars_for(9), 4);
  }

  #[test]
  fn row_tag_extraction() {
    for t in 0..NUM_TAGS {
      let mut cols = [GF2_128::zero(); 8];
      cols[1] = GF2_128::from(t as u64);
      assert_eq!(row_tag(&cols), t);
    }
  }

  #[test]
  fn mixed_ops_valid_trace() {
    // Mix of Xor32 (tag 3) and Mov (tag 10)
    let rows = vec![
      make_xor_row(5, 3),
      Row { pc: 1, op: 10, in0: 42, in1: 0, in2: 0, out: 42, flags: 0, advice: 0 }, // Mov
    ];
    let table = TraceTable::from_rows(&rows);
    let beta = GF2_128::from(77u64);
    let cmle = table.constraint_mle(beta).unwrap();
    assert!(cmle.sum().is_zero(), "mixed valid trace should have zero constraint sum");
  }

  #[test]
  fn check_inv_in_trace() {
    let a = GF2_128::from(7u64);
    let a_inv = a.inv();
    // We need to encode a_inv as u32... but GF2_128 is a 128-bit field element.
    // For this test, use small values where inverse is representable.
    // In GF(2^128), GF2_128::from(7).inv() is a full 128-bit element — not u32.
    // So CheckInv in the trace uses the full field element in the constraint,
    // but the Row columns are u32.  In practice, CheckInv operates on 32-bit
    // limbs and the "multiply + check == 1" is across the full 256-bit value
    // assembled from multiple rows.
    //
    // For a unit test, we verify the constraint function directly.
    let mut cols = [GF2_128::zero(); 8];
    cols[2] = a;
    cols[3] = a_inv;
    assert_eq!(constraint::eval_constraint(14, &cols).unwrap(), GF2_128::zero());
  }

  #[test]
  fn done_rows_contribute_zero() {
    let rows = vec![
      make_xor_row(1, 2),
      Row { pc: 1, op: 21, in0: 0, in1: 0, in2: 0, out: 0, flags: 0, advice: 0 }, // Done
    ];
    let table = TraceTable::from_rows(&rows);
    let beta = GF2_128::from(55u64);
    let cmle = table.constraint_mle(beta).unwrap();
    assert!(cmle.sum().is_zero());
  }
}
