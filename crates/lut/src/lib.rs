//! 8-bit arithmetic lookup tables for LogUp over GF(2^128).
//!
//! Each table encodes all valid `(a, b) → result` triples for an 8-bit
//! operation as GF(2^128) field elements.  The prover packs actual execution
//! values the same way to form a *witness*, then uses LogUp to prove
//! `witness ⊆ table`.
//!
//! # Encoding layout (little-endian in `GF2_128.lo`)
//!
//! | Operation | Bits 7:0 | 15:8 | 23:16  | 31:24  | Size  |
//! |-----------|----------|------|--------|--------|-------|
//! | And / Xor | a        | b    | c      | —      | 2^16  |
//! | Add       | a        | b    | sum    | carry  | 2^16  |
//! | Mul       | a        | b    | lo     | hi     | 2^16  |
//! | Range     | v        | —    | —      | —      | 2^8   |

use field::GF2_128;
use logup::LookupTable;

// ── Operation kind ───────────────────────────────────────────────────────────

/// Supported 8-bit operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Op8 {
  /// Bitwise AND: `c = a & b`.
  And,
  /// Bitwise XOR: `c = a ^ b`.
  Xor,
  /// Integer addition mod 256 with carry: `(sum, carry) = a + b`.
  Add,
  /// Integer multiplication: `(lo, hi) = a * b` (16-bit result split).
  Mul,
  /// Range check: value ∈ \[0, 256).
  Range,
}

// ── Encoding helpers ─────────────────────────────────────────────────────────

/// Pack a binary-op triple `(a, b, c)` into a field element.
#[inline]
pub fn encode_binary(a: u8, b: u8, c: u8) -> GF2_128 {
  GF2_128::from(a as u64 | (b as u64) << 8 | (c as u64) << 16)
}

/// Unpack a binary-op field element back into `(a, b, c)`.
#[inline]
pub fn decode_binary(e: GF2_128) -> (u8, u8, u8) {
  let v = e.lo;
  (
    (v & 0xFF) as u8,
    ((v >> 8) & 0xFF) as u8,
    ((v >> 16) & 0xFF) as u8,
  )
}

/// Pack an addition result `(a, b, sum, carry)`.
#[inline]
pub fn encode_add(a: u8, b: u8, sum: u8, carry: u8) -> GF2_128 {
  debug_assert!(carry <= 1);
  GF2_128::from(a as u64 | (b as u64) << 8 | (sum as u64) << 16 | (carry as u64) << 24)
}

/// Unpack an addition field element into `(a, b, sum, carry)`.
#[inline]
pub fn decode_add(e: GF2_128) -> (u8, u8, u8, u8) {
  let v = e.lo;
  (
    (v & 0xFF) as u8,
    ((v >> 8) & 0xFF) as u8,
    ((v >> 16) & 0xFF) as u8,
    ((v >> 24) & 0xFF) as u8,
  )
}

/// Pack a multiplication result `(a, b, lo, hi)`.
#[inline]
pub fn encode_mul(a: u8, b: u8, lo: u8, hi: u8) -> GF2_128 {
  GF2_128::from(a as u64 | (b as u64) << 8 | (lo as u64) << 16 | (hi as u64) << 24)
}

/// Unpack a multiplication field element into `(a, b, lo, hi)`.
#[inline]
pub fn decode_mul(e: GF2_128) -> (u8, u8, u8, u8) {
  let v = e.lo;
  (
    (v & 0xFF) as u8,
    ((v >> 8) & 0xFF) as u8,
    ((v >> 16) & 0xFF) as u8,
    ((v >> 24) & 0xFF) as u8,
  )
}

/// Pack a range-check value.
#[inline]
pub fn encode_range(v: u8) -> GF2_128 {
  GF2_128::from(v as u64)
}

/// Unpack a range-check field element.
#[inline]
pub fn decode_range(e: GF2_128) -> u8 {
  (e.lo & 0xFF) as u8
}

// ── Table generation ─────────────────────────────────────────────────────────

use std::sync::OnceLock;

static AND_TABLE: OnceLock<LookupTable> = OnceLock::new();
static XOR_TABLE: OnceLock<LookupTable> = OnceLock::new();
static ADD_TABLE: OnceLock<LookupTable> = OnceLock::new();
static MUL_TABLE: OnceLock<LookupTable> = OnceLock::new();
static RANGE_TABLE: OnceLock<LookupTable> = OnceLock::new();

fn build_table_inner(op: Op8) -> LookupTable {
  let entries: Vec<GF2_128> = match op {
    Op8::And => (0u32..=255)
      .flat_map(|a| {
        (0u32..=255).map(move |b| {
          let c = (a & b) as u8;
          encode_binary(a as u8, b as u8, c)
        })
      })
      .collect(),

    Op8::Xor => (0u32..=255)
      .flat_map(|a| {
        (0u32..=255).map(move |b| {
          let c = (a ^ b) as u8;
          encode_binary(a as u8, b as u8, c)
        })
      })
      .collect(),

    Op8::Add => (0u32..=255)
      .flat_map(|a| {
        (0u32..=255).map(move |b| {
          let full = a + b;
          let sum = (full & 0xFF) as u8;
          let carry = (full >> 8) as u8;
          encode_add(a as u8, b as u8, sum, carry)
        })
      })
      .collect(),

    Op8::Mul => (0u32..=255)
      .flat_map(|a| {
        (0u32..=255).map(move |b| {
          let full = a * b;
          let lo = (full & 0xFF) as u8;
          let hi = ((full >> 8) & 0xFF) as u8;
          encode_mul(a as u8, b as u8, lo, hi)
        })
      })
      .collect(),

    Op8::Range => (0u32..=255).map(|v| encode_range(v as u8)).collect(),
  };

  LookupTable::new(entries)
}

/// Build (or return cached) the full lookup table for an 8-bit operation.
///
/// Binary ops (And, Xor, Add, Mul) enumerate all 256 × 256 = 2^16 pairs.
/// Range enumerates all 256 = 2^8 values.
/// Tables are computed once and cached for the lifetime of the process.
pub fn build_table(op: Op8) -> &'static LookupTable {
  match op {
    Op8::And => AND_TABLE.get_or_init(|| build_table_inner(Op8::And)),
    Op8::Xor => XOR_TABLE.get_or_init(|| build_table_inner(Op8::Xor)),
    Op8::Add => ADD_TABLE.get_or_init(|| build_table_inner(Op8::Add)),
    Op8::Mul => MUL_TABLE.get_or_init(|| build_table_inner(Op8::Mul)),
    Op8::Range => RANGE_TABLE.get_or_init(|| build_table_inner(Op8::Range)),
  }
}

/// Encode a witness vector for a binary op (And / Xor) from `(a, b)` pairs.
pub fn witness_binary(op: Op8, pairs: &[(u8, u8)]) -> Vec<GF2_128> {
  pairs
    .iter()
    .map(|&(a, b)| {
      let c = match op {
        Op8::And => a & b,
        Op8::Xor => a ^ b,
        _ => panic!("witness_binary only supports And/Xor"),
      };
      encode_binary(a, b, c)
    })
    .collect()
}

/// Encode a witness vector for Add from `(a, b)` pairs.
pub fn witness_add(pairs: &[(u8, u8)]) -> Vec<GF2_128> {
  pairs
    .iter()
    .map(|&(a, b)| {
      let full = a as u16 + b as u16;
      encode_add(a, b, (full & 0xFF) as u8, (full >> 8) as u8)
    })
    .collect()
}

/// Encode a witness vector for Mul from `(a, b)` pairs.
pub fn witness_mul(pairs: &[(u8, u8)]) -> Vec<GF2_128> {
  pairs
    .iter()
    .map(|&(a, b)| {
      let full = a as u16 * b as u16;
      encode_mul(a, b, (full & 0xFF) as u8, ((full >> 8) & 0xFF) as u8)
    })
    .collect()
}

/// Encode a witness vector for Range from byte values.
pub fn witness_range(values: &[u8]) -> Vec<GF2_128> {
  values.iter().map(|&v| encode_range(v)).collect()
}

// ── Prove / Verify convenience ───────────────────────────────────────────────

/// Prove that every element in `witness` is a valid 8-bit operation result.
///
/// Returns `(proof, table)` so the verifier can access the table entries.
pub fn prove(
  op: Op8,
  witness: &mut Vec<GF2_128>,
  transcript: &mut transcript::Blake3Transcript,
) -> (logup::LogUpProof, LookupTable) {
  let table = build_table(op);

  // Pad witness to next power of two with a valid table entry.
  let n = witness.len().next_power_of_two();
  let pad = table.entries[0];
  witness.reserve(n.saturating_sub(witness.len()));
  witness.resize(n, pad);

  let proof = logup::prove(witness, table, transcript);
  (proof, table.clone())
}

/// Prove an 8-bit LUT relation with a succinct witness commitment.
///
/// Returns `(proof, table, witness_digest)`.
pub fn prove_committed(
  op: Op8,
  witness: &mut Vec<GF2_128>,
  transcript: &mut transcript::Blake3Transcript,
) -> (logup::LogUpProof, LookupTable, [u8; 32]) {
  let table = build_table(op);

  let n = witness.len().next_power_of_two();
  let pad = table.entries[0];
  witness.reserve(n.saturating_sub(witness.len()));
  witness.resize(n, pad);

  let (proof, digest) = logup::prove_committed(witness, &table, transcript);
  (proof, table.clone(), digest)
}

/// Verify an 8-bit LUT proof.
pub fn verify(
  proof: &logup::LogUpProof,
  witness: &[GF2_128],
  table: &LookupTable,
  transcript: &mut transcript::Blake3Transcript,
) -> Option<logup::LogUpClaims> {
  logup::verify(proof, witness, table, transcript)
}

/// Verify an 8-bit LUT proof using a succinct witness commitment.
pub fn verify_committed(
  proof: &logup::LogUpProof,
  n_witness: usize,
  table: &LookupTable,
  transcript: &mut transcript::Blake3Transcript,
  witness_digest: &[u8; 32],
) -> Option<logup::LogUpClaims> {
  logup::verify_committed(proof, n_witness, table, transcript, witness_digest)
}

// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
  use super::*;
  use transcript::Blake3Transcript;

  // ── Encoding roundtrip ──────────────────────────────────────────────

  #[test]
  fn binary_encode_decode() {
    for a in [0u8, 1, 127, 255] {
      for b in [0u8, 1, 127, 255] {
        let c = a & b;
        let e = encode_binary(a, b, c);
        assert_eq!(decode_binary(e), (a, b, c));
      }
    }
  }

  #[test]
  fn add_encode_decode() {
    for &(a, b) in &[(0u8, 0u8), (1, 1), (200, 100), (255, 255)] {
      let full = a as u16 + b as u16;
      let sum = (full & 0xFF) as u8;
      let carry = (full >> 8) as u8;
      let e = encode_add(a, b, sum, carry);
      assert_eq!(decode_add(e), (a, b, sum, carry));
    }
  }

  #[test]
  fn mul_encode_decode() {
    for &(a, b) in &[(0u8, 0u8), (1, 1), (16, 16), (255, 255)] {
      let full = a as u16 * b as u16;
      let lo = (full & 0xFF) as u8;
      let hi = ((full >> 8) & 0xFF) as u8;
      let e = encode_mul(a, b, lo, hi);
      assert_eq!(decode_mul(e), (a, b, lo, hi));
    }
  }

  #[test]
  fn range_encode_decode() {
    for v in [0u8, 1, 127, 255] {
      assert_eq!(decode_range(encode_range(v)), v);
    }
  }

  // ── Table sizes ──────────────────────────────────────────────────────

  #[test]
  fn table_sizes() {
    assert_eq!(build_table(Op8::And).entries.len(), 1 << 16);
    assert_eq!(build_table(Op8::Xor).entries.len(), 1 << 16);
    assert_eq!(build_table(Op8::Add).entries.len(), 1 << 16);
    assert_eq!(build_table(Op8::Mul).entries.len(), 1 << 16);
    assert_eq!(build_table(Op8::Range).entries.len(), 1 << 8);
  }

  #[test]
  fn table_entries_unique_and() {
    let t = build_table(Op8::And);
    let mut sorted: Vec<_> = t.entries.iter().map(|e| e.lo).collect();
    sorted.sort();
    sorted.dedup();
    // AND has duplicates: e.g. 0&0 = 0&1 = … all give c=0 with different a,b.
    // But the full triple (a,b,c) is unique because (a,b) is unique.
    assert_eq!(sorted.len(), 1 << 16);
  }

  // ── LogUp integration ────────────────────────────────────────────────

  fn prove_and_verify_op(op: Op8, pairs: &[(u8, u8)]) -> bool {
    let mut witness = match op {
      Op8::And | Op8::Xor => witness_binary(op, pairs),
      Op8::Add => witness_add(pairs),
      Op8::Mul => witness_mul(pairs),
      Op8::Range => unreachable!(),
    };

    let mut tp = Blake3Transcript::new();
    let (proof, table) = prove(op, &mut witness, &mut tp);

    let mut tv = Blake3Transcript::new();
    verify(&proof, &witness, &table, &mut tv).is_some()
  }

  fn prove_and_verify_range(values: &[u8]) -> bool {
    let mut witness = witness_range(values);

    let mut tp = Blake3Transcript::new();
    let (proof, table) = prove(Op8::Range, &mut witness, &mut tp);

    let mut tv = Blake3Transcript::new();
    verify(&proof, &witness, &table, &mut tv).is_some()
  }

  #[test]
  fn and_simple() {
    assert!(prove_and_verify_op(
      Op8::And,
      &[(0xFF, 0x0F), (0xAB, 0xCD), (0, 0), (255, 255)]
    ));
  }

  #[test]
  fn xor_simple() {
    assert!(prove_and_verify_op(
      Op8::Xor,
      &[(0xFF, 0x0F), (0xAB, 0xCD), (0, 0), (255, 255)]
    ));
  }

  #[test]
  fn add_simple() {
    assert!(prove_and_verify_op(
      Op8::Add,
      &[(100, 50), (200, 100), (0, 0), (255, 1)]
    ));
  }

  #[test]
  fn mul_simple() {
    assert!(prove_and_verify_op(
      Op8::Mul,
      &[(10, 20), (255, 255), (0, 1), (16, 16)]
    ));
  }

  #[test]
  fn range_simple() {
    assert!(prove_and_verify_range(&[0, 1, 127, 255]));
  }

  #[test]
  fn add_repeated_inputs() {
    // Same operation inputs repeated — exercises γ-weighting
    assert!(prove_and_verify_op(
      Op8::Add,
      &[(42, 42), (42, 42), (42, 42), (42, 42)]
    ));
  }

  #[test]
  fn range_all_bytes() {
    // All 256 values
    let all: Vec<u8> = (0..=255).collect();
    assert!(prove_and_verify_range(&all));
  }
}
