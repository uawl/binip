//! LUT-backed lookup arguments for trace soundness.
//!
//! # Architecture
//!
//! 1. [`collect_witnesses`] walks the execution trace and emits byte-level
//!    lookup entries for each operation that needs enforcement beyond the
//!    existing algebraic constraint.
//!
//! 2. [`prove_lookups`] runs LogUp for each non-empty witness table.
//!
//! 3. [`verify_lookups`] pads witnesses identically, then verifies all
//!    LogUp proofs.  [`verify_deterministic_checks`] handles conditions
//!    that don't need lookups (boolean range, upper-byte zeros).
//!
//! ## STARK ↔ LUT binding
//!
//! To prevent a prover from committing different values to the STARK trace
//! vs. the LUT witness, each operand of a LUT-backed opcode is:
//!
//! 1. **Decomposed** into 16 committed byte columns ([`crate::decomp`]).
//! 2. **Reconstructed** algebraically: `v = Σ byte[k] · x^{8k}`.
//! 3. **Range-checked** via the Range LUT (each byte ∈ [0, 256)).
//! 4. **Operation-checked** via the relevant op LUT.
//!
//! ## Covered operations
//!
//! | Tag | Op         | LUT tables | Purpose                                 |
//! |-----|------------|------------|-----------------------------------------|
//! | 0   | Add128     | Add        | Byte carry-chain for integer addition   |
//! | 1   | Mul128     | Mul        | Byte×byte schoolbook products           |
//! | 2   | And128     | And        | Byte-level bitwise AND                  |
//! | 8   | Chi128     | And        | AND component of Keccak χ               |
//! | 15  | RangeCheck | Range      | Byte decomposition for `v < 2^bits`     |

use field::{FieldElem, GF2_128};
use logup::LookupTable;
use rayon::prelude::*;
use transcript::{Blake3Transcript, Transcript};
use vm::Row;

use crate::decomp;

pub use logup::{LogUpClaims, LogUpProof};

/// Lookup witnesses partitioned by LUT table type.
#[derive(Debug, Clone)]
pub struct LookupWitness {
  /// Range table: `lut::encode_range(byte)`.
  pub range: Vec<GF2_128>,
  /// And table: `lut::encode_binary(a, b, a&b)`.
  pub and_op: Vec<GF2_128>,
  /// Add table: `lut::encode_add(a, b, sum, carry)`.
  pub add_op: Vec<GF2_128>,
  /// Mul table: `lut::encode_mul(a, b, lo, hi)`.
  pub mul_op: Vec<GF2_128>,
}

/// LogUp proofs, one per active LUT table.
#[derive(Debug, Clone)]
pub struct LookupProofs {
  pub range: Option<(LogUpProof, LookupTable)>,
  pub and_op: Option<(LogUpProof, LookupTable)>,
  pub add_op: Option<(LogUpProof, LookupTable)>,
  pub mul_op: Option<(LogUpProof, LookupTable)>,
}

#[inline]
fn bytes_of(v: u128) -> [u8; 16] {
  v.to_le_bytes()
}

// ── Witness collection ───────────────────────────────────────────────────────

/// Collect byte-level lookup witnesses from every trace row.
pub fn collect_witnesses(rows: &[Row]) -> LookupWitness {
  let mut w = LookupWitness {
    range: Vec::new(),
    and_op: Vec::new(),
    add_op: Vec::new(),
    mul_op: Vec::new(),
  };
  for row in rows {
    emit_decomp_ranges(&mut w, row);
    match row.op as u32 {
      0 => emit_add128(&mut w, row),
      1 => emit_mul128(&mut w, row),
      2 => emit_and128(&mut w, row),
      8 => emit_chi128(&mut w, row),
      15 => emit_range_check(&mut w, row),
      _ => {}
    }
  }
  w
}

/// Parallel variant of [`collect_witnesses`] — per-row witnesses computed
/// via rayon, then concatenated.
pub fn collect_witnesses_par(rows: &[Row]) -> LookupWitness {
  let per_row: Vec<LookupWitness> = rows
    .par_iter()
    .map(|row| {
      let mut w = LookupWitness {
        range: Vec::new(),
        and_op: Vec::new(),
        add_op: Vec::new(),
        mul_op: Vec::new(),
      };
      emit_decomp_ranges(&mut w, row);
      match row.op as u32 {
        0 => emit_add128(&mut w, row),
        1 => emit_mul128(&mut w, row),
        2 => emit_and128(&mut w, row),
        8 => emit_chi128(&mut w, row),
        15 => emit_range_check(&mut w, row),
        _ => {}
      }
      w
    })
    .collect();

  // Merge: pre-compute total sizes, then concatenate.
  let total_range: usize = per_row.iter().map(|w| w.range.len()).sum();
  let total_and: usize = per_row.iter().map(|w| w.and_op.len()).sum();
  let total_add: usize = per_row.iter().map(|w| w.add_op.len()).sum();
  let total_mul: usize = per_row.iter().map(|w| w.mul_op.len()).sum();

  let mut merged = LookupWitness {
    range: Vec::with_capacity(total_range),
    and_op: Vec::with_capacity(total_and),
    add_op: Vec::with_capacity(total_add),
    mul_op: Vec::with_capacity(total_mul),
  };
  for w in per_row {
    merged.range.extend(w.range);
    merged.and_op.extend(w.and_op);
    merged.add_op.extend(w.add_op);
    merged.mul_op.extend(w.mul_op);
  }
  merged
}

/// Emit Range LUT entries for all byte-decomposed operands (STARK ↔ LUT
/// binding).  Each committed byte column must be proven ∈ [0, 256).
fn emit_decomp_ranges(w: &mut LookupWitness, row: &Row) {
  let tag = row.op as u32;
  let mask = decomp::decomp_mask(tag);
  if mask.in0 {
    for b in row.in0.to_le_bytes() { w.range.push(lut::encode_range(b)); }
  }
  if mask.in1 {
    for b in row.in1.to_le_bytes() { w.range.push(lut::encode_range(b)); }
  }
  if mask.in2 {
    for b in row.in2.to_le_bytes() { w.range.push(lut::encode_range(b)); }
  }
  if mask.out {
    for b in row.out.to_le_bytes() { w.range.push(lut::encode_range(b)); }
  }
  if mask.flags {
    for b in row.flags.to_le_bytes() { w.range.push(lut::encode_range(b)); }
  }
}

/// RangeCheck (tag 15): decompose the lower bytes into Range LUT entries.
///
/// `bits == 1` is handled algebraically in [`verify_deterministic_checks`].
/// `bits >= 128` or `bits == 0` need no proof (always valid for u128).
fn emit_range_check(w: &mut LookupWitness, row: &Row) {
  let val = row.in0;
  let bits = row.in1 as u32;
  if bits <= 1 || bits >= 128 {
    return;
  }
  let bs = bytes_of(val);
  let n_bytes = ((bits + 7) / 8) as usize; // ceil(bits / 8)
  for i in 0..n_bytes.min(16) {
    w.range.push(lut::encode_range(bs[i]));
  }
}

/// Mul128 (tag 1): schoolbook byte×byte products via Mul LUT.
///
/// Emits 256 entries: one `encode_mul(a_i, b_j, lo, hi)` for each (i,j) pair.
/// Column-sum accumulation is verified deterministically in
/// [`verify_deterministic_checks`].
fn emit_mul128(w: &mut LookupWitness, row: &Row) {
  let a = bytes_of(row.in0);
  let b = bytes_of(row.in1);
  for i in 0..16 {
    for j in 0..16 {
      let prod = a[i] as u16 * b[j] as u16;
      let lo = (prod & 0xFF) as u8;
      let hi = ((prod >> 8) & 0xFF) as u8;
      w.mul_op.push(lut::encode_mul(a[i], b[j], lo, hi));
    }
  }
}

/// And128 (tag 2): for each byte position emit an And LUT entry.
fn emit_and128(w: &mut LookupWitness, row: &Row) {
  let a = bytes_of(row.in0);
  let b = bytes_of(row.in1);
  let c = bytes_of(row.out);
  for k in 0..16 {
    w.and_op.push(lut::encode_binary(a[k], b[k], c[k]));
  }
}

/// Chi128 (tag 8): `out = in0 ^ ((~in1) & in2)`.
///
/// The AND component `(~in1_byte) & in2_byte` is proven via the And LUT.
/// XOR with `in0` is already algebraically sound.
fn emit_chi128(w: &mut LookupWitness, row: &Row) {
  let in0 = bytes_of(row.in0);
  let in1 = bytes_of(row.in1);
  let in2 = bytes_of(row.in2);
  let out = bytes_of(row.out);
  for k in 0..16 {
    let not_b = !in1[k];
    let and_result = out[k] ^ in0[k]; // = (~in1) & in2
    w.and_op.push(lut::encode_binary(not_b, in2[k], and_result));
  }
}

/// Add128 (tag 0): byte-level carry chain via two Add LUT lookups per byte.
///
/// Carry-in is recovered from the trace values: `result ∈ {a+b, a+b+1}`.
fn emit_add128(w: &mut LookupWitness, row: &Row) {
  let a = row.in0;
  let b = row.in1;
  let result = row.out;

  let cin = recover_cin(a, b, result);

  let a_bs = bytes_of(a);
  let b_bs = bytes_of(b);
  let r_bs = bytes_of(result);

  let mut carry = cin;
  for k in 0..16 {
    // Step 1: a[k] + b[k] → (partial, c1)
    let sum1 = a_bs[k] as u16 + b_bs[k] as u16;
    let p = (sum1 & 0xFF) as u8;
    let c1 = (sum1 >> 8) as u8;
    w.add_op.push(lut::encode_add(a_bs[k], b_bs[k], p, c1));

    // Step 2: partial + carry_in → (result_byte, c2)
    let sum2 = p as u16 + carry as u16;
    let s = (sum2 & 0xFF) as u8;
    let c2 = (sum2 >> 8) as u8;
    w.add_op.push(lut::encode_add(p, carry, s, c2));

    debug_assert_eq!(s, r_bs[k]);
    carry = c1 ^ c2; // at most 1, proven by non-overlap
  }
}

fn recover_cin(a: u128, b: u128, result: u128) -> u8 {
  if result == a.wrapping_add(b) {
    0
  } else if result == a.wrapping_add(b).wrapping_add(1) {
    1
  } else {
    panic!("Add128: result is neither a+b nor a+b+1");
  }
}

// ── Deterministic checks ─────────────────────────────────────────────────────

/// Verify conditions that don't need lookups:
///
/// - **RangeCheck bits=1**: boolean constraint `v*(v+1) = 0` in GF(2^128).
/// - **RangeCheck bits>1**: upper bytes of value must be zero.
pub fn verify_deterministic_checks(rows: &[Row]) -> bool {
  for row in rows {
    match row.op as u32 {
      // Mul128: verify full 256-bit product matches out||flags.
      1 => {
        let (expected_lo, expected_hi) = widening_mul(row.in0, row.in1);
        if row.out != expected_lo || row.flags != expected_hi {
          return false;
        }
      }
      // RangeCheck: boolean constraint or upper-byte zeros.
      15 => {
        let val = row.in0;
        let bits = row.in1 as u32;
        if bits == 1 {
          let v = GF2_128::new(val as u64, (val >> 64) as u64);
          if !(v * (v + GF2_128::one())).is_zero() {
            return false;
          }
        } else if bits > 1 && bits < 128 {
          if val >= (1u128 << bits) {
            return false;
          }
        }
      }
      _ => {}
    }
  }
  true
}

/// 128×128 → 256-bit schoolbook multiplication (u64-limb decomposition).
fn widening_mul(a: u128, b: u128) -> (u128, u128) {
  let al = a as u64 as u128;
  let ah = (a >> 64) as u64 as u128;
  let bl = b as u64 as u128;
  let bh = (b >> 64) as u64 as u128;

  let p0 = al * bl;
  let p1 = al * bh;
  let p2 = ah * bl;
  let p3 = ah * bh;

  let (mid, mid_overflow) = p1.overflowing_add(p2);
  let mid_lo = (mid as u64) as u128;
  let mid_hi = mid >> 64;

  let (lo, lo_carry) = p0.overflowing_add(mid_lo << 64);
  let mut hi = p3.wrapping_add(mid_hi).wrapping_add(lo_carry as u128);
  if mid_overflow {
    hi = hi.wrapping_add(1u128 << 64);
  }
  (lo, hi)
}

// ── Prove / Verify ───────────────────────────────────────────────────────────

/// Prove all lookup arguments.
///
/// Witnesses in `w` are padded to powers of two after this call.
pub fn prove_lookups(
  w: &mut LookupWitness,
  transcript: &mut Blake3Transcript,
) -> LookupProofs {
  transcript.absorb_bytes(b"circuit:lookups");

  let range = if !w.range.is_empty() {
    transcript.absorb_bytes(b"lut:range");
    let (proof, table) = lut::prove(lut::Op8::Range, &mut w.range, transcript);
    Some((proof, table))
  } else {
    None
  };

  let and_op = if !w.and_op.is_empty() {
    transcript.absorb_bytes(b"lut:and");
    let (proof, table) = lut::prove(lut::Op8::And, &mut w.and_op, transcript);
    Some((proof, table))
  } else {
    None
  };

  let add_op = if !w.add_op.is_empty() {
    transcript.absorb_bytes(b"lut:add");
    let (proof, table) = lut::prove(lut::Op8::Add, &mut w.add_op, transcript);
    Some((proof, table))
  } else {
    None
  };

  let mul_op = if !w.mul_op.is_empty() {
    transcript.absorb_bytes(b"lut:mul");
    let (proof, table) = lut::prove(lut::Op8::Mul, &mut w.mul_op, transcript);
    Some((proof, table))
  } else {
    None
  };

  LookupProofs { range, and_op, add_op, mul_op }
}

/// Parallel variant of [`prove_lookups`] — each table gets a forked
/// transcript so the four LogUp proofs can run concurrently via rayon.
pub fn prove_lookups_par(
  w: &mut LookupWitness,
  transcript: &mut Blake3Transcript,
) -> LookupProofs {
  transcript.absorb_bytes(b"circuit:lookups");

  // Fork transcripts — independent and deterministic per table.
  let mut t_range = transcript.fork("lut:range", 0);
  let mut t_and   = transcript.fork("lut:and",   1);
  let mut t_add   = transcript.fork("lut:add",   2);
  let mut t_mul   = transcript.fork("lut:mul",   3);

  let ((range, and_op), (add_op, mul_op)) = rayon::join(
    || {
      rayon::join(
        || {
          if w.range.is_empty() { None }
          else {
            let (p, t) = lut::prove(lut::Op8::Range, &mut w.range, &mut t_range);
            Some((p, t))
          }
        },
        || {
          if w.and_op.is_empty() { None }
          else {
            let (p, t) = lut::prove(lut::Op8::And, &mut w.and_op, &mut t_and);
            Some((p, t))
          }
        },
      )
    },
    || {
      rayon::join(
        || {
          if w.add_op.is_empty() { None }
          else {
            let (p, t) = lut::prove(lut::Op8::Add, &mut w.add_op, &mut t_add);
            Some((p, t))
          }
        },
        || {
          if w.mul_op.is_empty() { None }
          else {
            let (p, t) = lut::prove(lut::Op8::Mul, &mut w.mul_op, &mut t_mul);
            Some((p, t))
          }
        },
      )
    },
  );

  LookupProofs { range, and_op, add_op, mul_op }
}

/// Verify all lookup proofs.
///
/// Witnesses in `w` are padded to match the prover's padding.
pub fn verify_lookups(
  proofs: &LookupProofs,
  w: &mut LookupWitness,
  transcript: &mut Blake3Transcript,
) -> bool {
  transcript.absorb_bytes(b"circuit:lookups");

  if let Some((proof, table)) = &proofs.range {
    transcript.absorb_bytes(b"lut:range");
    let n = w.range.len().next_power_of_two();
    w.range.resize(n, table.entries[0]);
    if lut::verify(proof, &w.range, table, transcript).is_none() {
      return false;
    }
  } else if !w.range.is_empty() {
    return false;
  }

  if let Some((proof, table)) = &proofs.and_op {
    transcript.absorb_bytes(b"lut:and");
    let n = w.and_op.len().next_power_of_two();
    w.and_op.resize(n, table.entries[0]);
    if lut::verify(proof, &w.and_op, table, transcript).is_none() {
      return false;
    }
  } else if !w.and_op.is_empty() {
    return false;
  }

  if let Some((proof, table)) = &proofs.add_op {
    transcript.absorb_bytes(b"lut:add");
    let n = w.add_op.len().next_power_of_two();
    w.add_op.resize(n, table.entries[0]);
    if lut::verify(proof, &w.add_op, table, transcript).is_none() {
      return false;
    }
  } else if !w.add_op.is_empty() {
    return false;
  }

  if let Some((proof, table)) = &proofs.mul_op {
    transcript.absorb_bytes(b"lut:mul");
    let n = w.mul_op.len().next_power_of_two();
    w.mul_op.resize(n, table.entries[0]);
    if lut::verify(proof, &w.mul_op, table, transcript).is_none() {
      return false;
    }
  } else if !w.mul_op.is_empty() {
    return false;
  }

  true
}

/// Parallel variant of [`verify_lookups`] — uses forked transcripts
/// matching [`prove_lookups_par`].
pub fn verify_lookups_par(
  proofs: &LookupProofs,
  w: &mut LookupWitness,
  transcript: &mut Blake3Transcript,
) -> bool {
  transcript.absorb_bytes(b"circuit:lookups");

  let mut t_range = transcript.fork("lut:range", 0);
  let mut t_and   = transcript.fork("lut:and",   1);
  let mut t_add   = transcript.fork("lut:add",   2);
  let mut t_mul   = transcript.fork("lut:mul",   3);

  // Pad witnesses (must happen before verify reads them).
  if let Some((_, table)) = &proofs.range {
    let n = w.range.len().next_power_of_two();
    w.range.resize(n, table.entries[0]);
  }
  if let Some((_, table)) = &proofs.and_op {
    let n = w.and_op.len().next_power_of_two();
    w.and_op.resize(n, table.entries[0]);
  }
  if let Some((_, table)) = &proofs.add_op {
    let n = w.add_op.len().next_power_of_two();
    w.add_op.resize(n, table.entries[0]);
  }
  if let Some((_, table)) = &proofs.mul_op {
    let n = w.mul_op.len().next_power_of_two();
    w.mul_op.resize(n, table.entries[0]);
  }

  // Check presence consistency first (proof ↔ witness non-emptiness).
  if proofs.range.is_none() && !w.range.is_empty() { return false; }
  if proofs.and_op.is_none() && !w.and_op.is_empty() { return false; }
  if proofs.add_op.is_none() && !w.add_op.is_empty() { return false; }
  if proofs.mul_op.is_none() && !w.mul_op.is_empty() { return false; }

  let ((r1, r2), (r3, r4)) = rayon::join(
    || rayon::join(
      || proofs.range.as_ref().map_or(true, |(proof, table)|
        lut::verify(proof, &w.range, table, &mut t_range).is_some()),
      || proofs.and_op.as_ref().map_or(true, |(proof, table)|
        lut::verify(proof, &w.and_op, table, &mut t_and).is_some()),
    ),
    || rayon::join(
      || proofs.add_op.as_ref().map_or(true, |(proof, table)|
        lut::verify(proof, &w.add_op, table, &mut t_add).is_some()),
      || proofs.mul_op.as_ref().map_or(true, |(proof, table)|
        lut::verify(proof, &w.mul_op, table, &mut t_mul).is_some()),
    ),
  );

  r1 && r2 && r3 && r4
}

// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
  use super::*;
  use transcript::Blake3Transcript;

  fn make_row(op: u32, in0: u128, in1: u128, in2: u128, out: u128, flags: u128) -> Row {
    Row { pc: 0, op: op as u128, in0, in1, in2, out, flags, advice: 0 }
  }

  /// Helper: collect → prove → verify.
  fn run_lookup(rows: &[Row]) -> bool {
    let mut w_prover = collect_witnesses(rows);
    let mut tp = Blake3Transcript::new();
    let proofs = prove_lookups(&mut w_prover, &mut tp);

    let mut w_verifier = collect_witnesses(rows);
    let mut tv = Blake3Transcript::new();
    let ok = verify_lookups(&proofs, &mut w_verifier, &mut tv);
    ok && verify_deterministic_checks(rows)
  }

  // ── RangeCheck ──────────────────────────────────────────────────────

  #[test]
  fn range_check_8bit() {
    let rows = vec![
      make_row(15, 0, 8, 0, 0, 0),
      make_row(15, 42, 8, 0, 42, 0),
      make_row(15, 255, 8, 0, 255, 0),
      make_row(15, 0, 8, 0, 0, 0), // pad to pow2
    ];
    assert!(run_lookup(&rows));
  }

  #[test]
  fn range_check_1bit_boolean() {
    let rows = vec![
      make_row(15, 0, 1, 0, 0, 0),
      make_row(15, 1, 1, 0, 1, 0),
    ];
    // bits=1 → no LUT (algebraic), but deterministic check should pass.
    assert!(verify_deterministic_checks(&rows));
  }

  #[test]
  fn range_check_1bit_fails_for_2() {
    let rows = vec![make_row(15, 2, 1, 0, 2, 0)];
    assert!(!verify_deterministic_checks(&rows));
  }

  #[test]
  fn range_check_upper_bytes_fail() {
    // value = 256 with bits=8 → should fail deterministic check
    let rows = vec![make_row(15, 256, 8, 0, 256, 0)];
    assert!(!verify_deterministic_checks(&rows));
  }

  // ── And128 ──────────────────────────────────────────────────────────

  #[test]
  fn and128_correct() {
    let a: u128 = 0xFF00_FF00_FF00_FF00_FF00_FF00_FF00_FF00;
    let b: u128 = 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
    let rows = vec![make_row(2, a, b, 0, a & b, 0)];
    assert!(run_lookup(&rows));
  }

  #[test]
  fn and128_all_ones() {
    let a = u128::MAX;
    let b = 0x1234_5678_9ABC_DEF0_1234_5678_9ABC_DEF0u128;
    let rows = vec![make_row(2, a, b, 0, a & b, 0)];
    assert!(run_lookup(&rows));
  }

  // ── Add128 ──────────────────────────────────────────────────────────

  #[test]
  fn add128_no_carry() {
    let a: u128 = 100;
    let b: u128 = 200;
    let result = a.wrapping_add(b);
    let rows = vec![make_row(0, a, b, 0, result, 0)];
    assert!(run_lookup(&rows));
  }

  #[test]
  fn add128_with_carry() {
    let a = u128::MAX;
    let b: u128 = 1;
    let result = a.wrapping_add(b); // wraps to 0
    let rows = vec![make_row(0, a, b, 0, result, 1)];
    assert!(run_lookup(&rows));
  }

  #[test]
  fn add128_byte_boundary_carries() {
    // 0xFF + 0x01 = 0x100 → carry at byte boundary
    let a: u128 = 0xFF;
    let b: u128 = 0x01;
    let rows = vec![make_row(0, a, b, 0, a + b, 0)];
    assert!(run_lookup(&rows));
  }

  #[test]
  fn add128_large_values() {
    let a: u128 = 0xDEAD_BEEF_CAFE_BABE_1234_5678_9ABC_DEF0;
    let b: u128 = 0x1111_1111_1111_1111_1111_1111_1111_1111;
    let result = a.wrapping_add(b);
    let carry = if a.checked_add(b).is_none() { 1 } else { 0 };
    let rows = vec![make_row(0, a, b, 0, result, carry)];
    assert!(run_lookup(&rows));
  }

  // ── Chi128 ──────────────────────────────────────────────────────────

  #[test]
  fn chi128_correct() {
    let a: u128 = 0xAAAA_BBBB_CCCC_DDDD_EEEE_FFFF_0000_1111;
    let b: u128 = 0x1234_5678_9ABC_DEF0_1234_5678_9ABC_DEF0;
    let c: u128 = 0xFEDC_BA98_7654_3210_FEDC_BA98_7654_3210;
    let result = a ^ ((!b) & c);
    let rows = vec![make_row(8, a, b, c, result, 0)];
    assert!(run_lookup(&rows));
  }

  // ── Mul128 ──────────────────────────────────────────────────────────

  /// Helper: compute 128×128→256-bit product for test rows.
  fn mul_row(a: u128, b: u128) -> Row {
    let (lo, hi) = widening_mul(a, b);
    make_row(1, a, b, 0, lo, hi)
  }

  #[test]
  fn mul128_small() {
    let rows = vec![mul_row(7, 11)];
    assert!(run_lookup(&rows));
  }

  #[test]
  fn mul128_large() {
    let a: u128 = 0xDEAD_BEEF_CAFE_BABE_1234_5678_9ABC_DEF0;
    let b: u128 = 0x1111_1111_1111_1111_1111_1111_1111_1111;
    let rows = vec![mul_row(a, b)];
    assert!(run_lookup(&rows));
  }

  #[test]
  fn mul128_max() {
    let rows = vec![mul_row(u128::MAX, u128::MAX)];
    assert!(run_lookup(&rows));
  }

  #[test]
  fn mul128_by_zero() {
    let rows = vec![mul_row(0xABCD, 0)];
    assert!(run_lookup(&rows));
  }

  #[test]
  fn mul128_deterministic_wrong_result() {
    // Correct product lo, but tamper flags (hi).
    let a: u128 = 100;
    let b: u128 = 200;
    let (lo, _hi) = widening_mul(a, b);
    let rows = vec![make_row(1, a, b, 0, lo, 999)];
    assert!(!verify_deterministic_checks(&rows));
  }

  // ── Mixed ops ──────────────────────────────────────────────────────

  #[test]
  fn mixed_operations() {
    let rows = vec![
      // And128
      make_row(2, 0xFF, 0x0F, 0, 0x0F, 0),
      // Add128
      make_row(0, 100, 200, 0, 300, 0),
      // Mul128
      mul_row(6, 7),
      // RangeCheck bits=8
      make_row(15, 42, 8, 0, 42, 0),
      // Xor128 (tag 3 — no lookup needed, just padding)
      make_row(3, 5, 3, 0, 6, 0),
    ];
    assert!(run_lookup(&rows));
  }

  // ── Soundness: tampered traces ─────────────────────────────────────

  #[test]
  #[should_panic(expected = "not found in table")]
  fn and128_wrong_result_fails() {
    let a: u128 = 0xFF;
    let b: u128 = 0x0F;
    // Correct AND = 0x0F, but we put 0xFF (wrong).
    let rows = vec![make_row(2, a, b, 0, 0xFF, 0)];
    // This should fail at prove time because (0xFF, 0x0F, 0xFF)
    // is not a valid AND triple and won't be in the table.
    run_lookup(&rows);
  }

  #[test]
  #[should_panic(expected = "neither a+b nor a+b+1")]
  fn add128_wrong_result_panics() {
    let rows = vec![make_row(0, 100, 200, 0, 999, 0)];
    run_lookup(&rows);
  }

  /// Parallel collect → prove → verify matches sequential.
  fn run_lookup_par(rows: &[Row]) -> bool {
    let mut w_prover = collect_witnesses_par(rows);
    let mut tp = Blake3Transcript::new();
    let proofs = prove_lookups_par(&mut w_prover, &mut tp);

    let mut w_verifier = collect_witnesses_par(rows);
    let mut tv = Blake3Transcript::new();
    verify_lookups_par(&proofs, &mut w_verifier, &mut tv)
      && verify_deterministic_checks(rows)
  }

  #[test]
  fn par_mixed_operations() {
    let rows = vec![
      make_row(2, 0xFF, 0x0F, 0, 0x0F, 0),
      make_row(0, 100, 200, 0, 300, 0),
      mul_row(6, 7),
      make_row(15, 42, 8, 0, 42, 0),
      make_row(3, 5, 3, 0, 6, 0),
    ];
    assert!(run_lookup_par(&rows));
  }

  #[test]
  fn par_witnesses_match_sequential() {
    let rows = vec![
      make_row(2, 0xFF00, 0x0F0F, 0, 0xFF00 & 0x0F0F, 0),
      make_row(0, 100, 200, 0, 300, 0),
      mul_row(3, 5),
    ];
    let seq = collect_witnesses(&rows);
    let par = collect_witnesses_par(&rows);
    assert_eq!(seq.range, par.range);
    assert_eq!(seq.and_op, par.and_op);
    assert_eq!(seq.add_op, par.add_op);
    assert_eq!(seq.mul_op, par.mul_op);
  }
}
