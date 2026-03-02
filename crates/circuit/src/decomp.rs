//! Byte-level decomposition of trace operands for STARK ↔ LUT binding.
//!
//! # Problem
//!
//! Algebraic constraints for Add/Mul/And/Chi are `zero()` — soundness is
//! delegated to LUT lookups.  But if the prover commits *different* values
//! to the STARK trace than what goes into the LUT witness, both pass
//! independently with no binding.
//!
//! # Solution: Decomposition + Reconstruction
//!
//! 1. **Decompose** each operand into 16 bytes (auxiliary committed columns).
//! 2. **Reconstruct** algebraically: `v = Σ_{k=0}^{15} b[k] · x^{8k}`.
//!    In GF(2^128) this is exact when each `b[k] ∈ [0,256)`, because byte
//!    windows `[8k, 8k+8)` don't overlap and addition is XOR.
//! 3. **Range LUT** proves each committed byte column ∈ `[0, 256)`.
//! 4. **Operation LUT** proves byte-level operation correctness from the
//!    committed byte columns.
//!
//! Since reconstruction, range, and operation checks all reference the
//! *same* committed polynomials, the prover cannot use different values.

use field::{FieldElem, GF2_128};
use vm::Row;

// ── DecompRow ────────────────────────────────────────────────────────────────

/// Byte decomposition of the five operand columns.
///
/// For tags that don't require a particular bank, the bytes are zero and
/// no reconstruction constraint fires.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecompRow {
  pub in0_bytes: [u8; 16],
  pub in1_bytes: [u8; 16],
  pub in2_bytes: [u8; 16],
  pub out_bytes: [u8; 16],
  pub flags_bytes: [u8; 16],
}

impl DecompRow {
  /// Deterministic byte decomposition of a trace row.
  pub fn compute(row: &Row) -> Self {
    DecompRow {
      in0_bytes: row.in0.to_le_bytes(),
      in1_bytes: row.in1.to_le_bytes(),
      in2_bytes: row.in2.to_le_bytes(),
      out_bytes: row.out.to_le_bytes(),
      flags_bytes: row.flags.to_le_bytes(),
    }
  }

  pub fn zero() -> Self {
    DecompRow {
      in0_bytes: [0; 16],
      in1_bytes: [0; 16],
      in2_bytes: [0; 16],
      out_bytes: [0; 16],
      flags_bytes: [0; 16],
    }
  }
}

// ── Decomposition mask ───────────────────────────────────────────────────────

/// Which operand banks need decomposition for a given tag.
#[derive(Debug, Clone, Copy)]
pub struct DecompMask {
  pub in0: bool,
  pub in1: bool,
  pub in2: bool,
  pub out: bool,
  pub flags: bool,
}

/// Returns which operand banks need byte decomposition for a given tag.
pub fn decomp_mask(tag: u32) -> DecompMask {
  match tag {
    // Add128: in0, in1, out
    0 => DecompMask { in0: true, in1: true, in2: false, out: true, flags: false },
    // Mul128: in0, in1, out, flags  (out=low 128, flags=high 128)
    1 => DecompMask { in0: true, in1: true, in2: false, out: true, flags: true },
    // And128: in0, in1, out
    2 => DecompMask { in0: true, in1: true, in2: false, out: true, flags: false },
    // Chi128: in0, in1, in2, out
    8 => DecompMask { in0: true, in1: true, in2: true, out: true, flags: false },
    _ => DecompMask { in0: false, in1: false, in2: false, out: false, flags: false },
  }
}

/// Returns true if the given tag requires any byte decomposition.
pub fn needs_decomp(tag: u32) -> bool {
  matches!(tag, 0 | 1 | 2 | 8)
}

// ── Reconstruction ───────────────────────────────────────────────────────────

/// Reconstruct a GF(2^128) element from 16 bytes.
///
/// `reconstruct(b) = Σ_{k=0}^{15} b[k] · x^{8k}`
///
/// Since each byte occupies a disjoint 8-bit window and addition is XOR,
/// this equals the standard little-endian embedding `GF2_128::new(lo, hi)`.
#[inline]
pub fn reconstruct(bytes: &[u8; 16]) -> GF2_128 {
  let lo = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
  let hi = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
  GF2_128::new(lo, hi)
}

/// Reconstruction residual for one operand: `operand + reconstruct(bytes)`.
///
/// Returns zero iff the committed field element matches its byte columns.
#[inline]
fn recon_residual(operand: GF2_128, bytes: &[u8; 16]) -> GF2_128 {
  operand + reconstruct(bytes)
}

/// Evaluate the batched reconstruction constraint for a single row.
///
/// Combines up to 5 operand reconstruction residuals using powers of `gamma`:
///
///   `Σ_{i=0}^{4} γ^i · [mask_i] · (operand_i + reconstruct(bytes_i))`
///
/// Returns zero when all binding constraints hold.
pub fn eval_reconstruction(
  tag: u32,
  main_cols: &[GF2_128; 8],
  decomp: &DecompRow,
  gamma: GF2_128,
) -> GF2_128 {
  let mask = decomp_mask(tag);

  // (needed, column index, byte bank)
  let banks: [(bool, usize, &[u8; 16]); 5] = [
    (mask.in0, 2, &decomp.in0_bytes),
    (mask.in1, 3, &decomp.in1_bytes),
    (mask.in2, 4, &decomp.in2_bytes),
    (mask.out, 5, &decomp.out_bytes),
    (mask.flags, 6, &decomp.flags_bytes),
  ];

  let mut result = GF2_128::zero();
  let mut g = GF2_128::one(); // γ^0

  for &(needed, col_idx, bytes) in &banks {
    if needed {
      result = result + g * recon_residual(main_cols[col_idx], bytes);
    }
    g = g * gamma;
  }
  result
}

// ── Auxiliary column encoding ────────────────────────────────────────────────

/// Number of auxiliary columns: 5 banks × 16 bytes.
pub const NUM_DECOMP_COLS: usize = 80;

/// Encode a [`DecompRow`] into `NUM_DECOMP_COLS` field elements (one per byte).
pub fn encode_decomp(decomp: &DecompRow) -> [GF2_128; NUM_DECOMP_COLS] {
  let mut cols = [GF2_128::zero(); NUM_DECOMP_COLS];
  for k in 0..16 {
    cols[k] = GF2_128::from(decomp.in0_bytes[k] as u64);
    cols[16 + k] = GF2_128::from(decomp.in1_bytes[k] as u64);
    cols[32 + k] = GF2_128::from(decomp.in2_bytes[k] as u64);
    cols[48 + k] = GF2_128::from(decomp.out_bytes[k] as u64);
    cols[64 + k] = GF2_128::from(decomp.flags_bytes[k] as u64);
  }
  cols
}

// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
  use super::*;
  use crate::encoder;

  fn make_row(op: u32, in0: u128, in1: u128, in2: u128, out: u128, flags: u128) -> Row {
    Row { pc: 0, op: op as u128, in0, in1, in2, out, flags, advice: 0 }
  }

  // ── reconstruct ───────────────────────────────────────────────────

  #[test]
  fn reconstruct_roundtrip_small() {
    let v: u128 = 0xDEAD_BEEF;
    let bytes = v.to_le_bytes();
    let got = reconstruct(&bytes);
    let expect = GF2_128::new(v as u64, 0);
    assert_eq!(got, expect);
  }

  #[test]
  fn reconstruct_roundtrip_full() {
    let v: u128 = 0xDEAD_BEEF_CAFE_BABE_1234_5678_9ABC_DEF0;
    let bytes = v.to_le_bytes();
    let got = reconstruct(&bytes);
    let expect = GF2_128::new(v as u64, (v >> 64) as u64);
    assert_eq!(got, expect);
  }

  // ── eval_reconstruction ───────────────────────────────────────────

  #[test]
  fn correct_decomp_gives_zero() {
    // Add128: in0=100, in1=200, out=300
    let row = make_row(0, 100, 200, 0, 300, 0);
    let decomp = DecompRow::compute(&row);
    let main = encoder::encode_row(&row);
    let gamma = GF2_128::from(42);
    assert!(eval_reconstruction(0, &main, &decomp, gamma).is_zero());
  }

  #[test]
  fn correct_decomp_all_tags() {
    let gamma = GF2_128::from(7777);
    let cases: Vec<Row> = vec![
      make_row(0, 100, 200, 0, 300, 0),                   // Add128
      make_row(1, 6, 7, 0, 42, 0),                        // Mul128
      make_row(2, 0xFF, 0x0F, 0, 0x0F, 0),                // And128
      make_row(8, 0xAA, 0xBB, 0xCC, 0xAA ^ ((!0xBB) & 0xCC), 0), // Chi128
      make_row(3, 5, 3, 0, 6, 0),                         // Xor128 (no decomp)
    ];
    for row in &cases {
      let d = DecompRow::compute(row);
      let m = encoder::encode_row(row);
      let tag = row.op as u32;
      assert!(
        eval_reconstruction(tag, &m, &d, gamma).is_zero(),
        "tag {} should have zero reconstruction", tag
      );
    }
  }

  #[test]
  fn tampered_in0_detected() {
    let row = make_row(0, 100, 200, 0, 300, 0); // Add128
    let main = encoder::encode_row(&row);
    let gamma = GF2_128::from(42);

    let mut tampered = DecompRow::compute(&row);
    tampered.in0_bytes = 999u128.to_le_bytes(); // in0 committed as 100, but bytes say 999
    assert!(!eval_reconstruction(0, &main, &tampered, gamma).is_zero());
  }

  #[test]
  fn tampered_out_detected() {
    let row = make_row(2, 0xFF, 0x0F, 0, 0x0F, 0); // And128
    let main = encoder::encode_row(&row);
    let gamma = GF2_128::from(99);

    let mut tampered = DecompRow::compute(&row);
    tampered.out_bytes = 0xFFu128.to_le_bytes(); // out committed as 0x0F, but bytes say 0xFF
    assert!(!eval_reconstruction(2, &main, &tampered, gamma).is_zero());
  }

  #[test]
  fn non_lut_tag_always_zero() {
    // Xor128 (tag 3): no decomposition needed, reconstruction is zero
    // even with arbitrary decomp values.
    let row = make_row(3, 5, 3, 0, 6, 0);
    let main = encoder::encode_row(&row);
    let gamma = GF2_128::from(42);

    let bogus = DecompRow {
      in0_bytes: 111u128.to_le_bytes(),
      in1_bytes: 222u128.to_le_bytes(),
      in2_bytes: 333u128.to_le_bytes(),
      out_bytes: 444u128.to_le_bytes(),
      flags_bytes: 555u128.to_le_bytes(),
    };
    // Should still be zero — tag 3 doesn't check any bank.
    assert!(eval_reconstruction(3, &main, &bogus, gamma).is_zero());
  }

  // ── decomp_mask ───────────────────────────────────────────────────

  #[test]
  fn mask_add128() {
    let m = decomp_mask(0);
    assert!(m.in0 && m.in1 && !m.in2 && m.out && !m.flags);
  }

  #[test]
  fn mask_mul128() {
    let m = decomp_mask(1);
    assert!(m.in0 && m.in1 && !m.in2 && m.out && m.flags);
  }

  #[test]
  fn mask_chi128() {
    let m = decomp_mask(8);
    assert!(m.in0 && m.in1 && m.in2 && m.out && !m.flags);
  }

  // ── encode_decomp ─────────────────────────────────────────────────

  #[test]
  fn encode_decomp_size() {
    let d = DecompRow::zero();
    let cols = encode_decomp(&d);
    assert_eq!(cols.len(), NUM_DECOMP_COLS);
    for c in &cols {
      assert!(c.is_zero());
    }
  }

  #[test]
  fn encode_decomp_layout() {
    let row = make_row(0, 0x0102, 0x0304, 0, 0x0506, 0);
    let d = DecompRow::compute(&row);
    let cols = encode_decomp(&d);
    // in0 bank: bytes of 0x0102 = [0x02, 0x01, 0, 0, ...]
    assert_eq!(cols[0], GF2_128::from(0x02));
    assert_eq!(cols[1], GF2_128::from(0x01));
    // in1 bank: bytes of 0x0304 = [0x04, 0x03, 0, 0, ...]
    assert_eq!(cols[16], GF2_128::from(0x04));
    assert_eq!(cols[17], GF2_128::from(0x03));
    // out bank: bytes of 0x0506 = [0x06, 0x05, 0, 0, ...]
    assert_eq!(cols[48], GF2_128::from(0x06));
    assert_eq!(cols[49], GF2_128::from(0x05));
  }
}
