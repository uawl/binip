//! Per-opcode algebraic constraints over GF(2^128).
//!
//! Every valid trace row must satisfy the constraint corresponding to its
//! opcode tag.  Constraints are expressed as polynomial identities over
//! the 8 column field elements; on a correct row the constraint evaluates
//! to [`GF2_128::zero()`].
//!
//! # Constraint formulation
//!
//! In GF(2) arithmetic:
//! - Addition is XOR (free).
//! - Multiplication is AND (expensive).
//! - `a + a = 0` for all `a`.
//!
//! Most constraints check simple algebraic relationships between the row
//! columns (`in0`, `in1`, `in2`, `out`, `flags`, `advice`).  The "heavy"
//! constraints (CheckDiv, CheckMul, CheckInv) verify advice-tape values.

use field::{FieldElem, GF2_128};

/// Constraint evaluation error.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ConstraintError {
  /// Unknown opcode tag.
  #[error("unknown opcode tag {0}")]
  UnknownTag(u32),
}

/// Total number of opcode tags this module handles (0..=32).
pub const NUM_TAGS: u32 = 33;

/// Evaluate the constraint polynomial for opcode `tag` on a row whose
/// columns are given as 8 [`GF2_128`] values (same order as [`encoder::COL_*`]).
///
/// Returns [`GF2_128::zero()`] when the row satisfies the constraint.
///
/// # Column mapping
///
/// | index | field  |
/// |-------|--------|
/// | 0     | pc     |
/// | 1     | op     |
/// | 2     | in0    |
/// | 3     | in1    |
/// | 4     | in2    |
/// | 5     | out    |
/// | 6     | flags  |
/// | 7     | advice |
pub fn eval_constraint(tag: u32, cols: &[GF2_128; 8]) -> Result<GF2_128, ConstraintError> {
  let _pc = cols[0];
  let _op = cols[1];
  let in0 = cols[2];
  let in1 = cols[3];
  let in2 = cols[4];
  let out = cols[5];
  let flags = cols[6];
  let _advice = cols[7];

  match tag {
    // ── ALU (tag 0): Add128 ─────────────────────────────────────────────
    // Integer addition cannot be checked algebraically in GF(2^128):
    // field addition is XOR, not integer add.  Soundness is enforced by
    // the byte-level Add LUT in `lookup::emit_add128`.
    0 => Ok(GF2_128::zero()),

    // ── ALU (tag 1): Mul128 ─────────────────────────────────────────────
    // GF(2^128) field multiplication ≠ integer multiplication.
    // Soundness is enforced by the byte-level Mul LUT for each partial
    // product plus a deterministic column-sum check in
    // `lookup::verify_deterministic_checks`.
    1 => Ok(GF2_128::zero()),

    // ── Bitwise (tag 2): And128 ─────────────────────────────────────────
    // GF(2^128) multiplication ≠ bitwise AND (AND ≅ mult only in GF(2)).
    // Soundness is enforced by the byte-level And LUT in
    // `lookup::emit_and128`.
    2 => Ok(GF2_128::zero()),

    // ── Bitwise (tag 3): Xor128 ─────────────────────────────────────────
    // out = in0 XOR in1 → in field: out = in0 + in1
    3 => Ok(out + in0 + in1),

    // ── Bitwise (tag 4): Not128 ─────────────────────────────────────────
    // out = NOT in0.  In GF(2) arithmetic over 128-bit words:
    // out = in0 ⊕ mask128.  We check: out + in0 + mask128 = 0.
    4 => {
      let mask128 = GF2_128::new(u64::MAX, u64::MAX);
      Ok(out + in0 + mask128)
    }

    // ── Bitwise (tag 5): Rot128 ─────────────────────────────────────────
    // Wire permutation — in1 encodes the shift amount.
    // For the MLE-level constraint we use: out + in0 + flags = 0
    // (flags encodes the correction bits inserted/removed by rotation).
    5 => Ok(out + in0 + flags),

    // ── Shift (tag 6): Shr128, (tag 7): Shl128 ──────────────────────────
    // Similar to Rot128: wire substitution + cin provides shifted-in bits.
    // Constraint: out + in0 + in1 + flags = 0   (in1 = cin contribution)
    6 | 7 => Ok(out + in0 + in1 + flags),

    // ── Keccak (tag 8): Chi128 ──────────────────────────────────────────
    // χ(a,b,c) = a ⊕ ((¬b) ∧ c).  The AND component cannot be
    // checked algebraically (field mult ≠ bitwise AND).  Soundness
    // is enforced by the byte-level And LUT in `lookup::emit_chi128`.
    8 => Ok(GF2_128::zero()),

    // ── Data movement (tag 9): Const ────────────────────────────────────
    // out = advice (the immediate value is stored in the advice column).
    9 => Ok(out + _advice),

    // ── Data movement (tag 10): Mov ─────────────────────────────────────
    // out = in0
    10 => Ok(out + in0),

    // ── Advice (tag 11): AdviceLoad ─────────────────────────────────────
    // out = advice (loaded from tape; circuit only checks column consistency).
    11 => Ok(out + _advice),

    // ── Check (tag 12): CheckDiv ────────────────────────────────────────
    // dividend == divisor * quot + rem
    // Columns: in0 = quot, in1 = rem, in2 = dividend, advice = divisor
    // Integer multiplication cannot be checked algebraically in GF(2^128).
    // Soundness is enforced by:
    //   1. Byte-level Mul LUT for divisor×quot products
    //   2. Mul accumulation Add LUT for column-sum verification
    //   3. Add LUT carry chain for product+rem = dividend
    //   4. Reconstruction binds all operands to committed trace
    // Note: range check (rem < divisor) is a separate RangeCheck op.
    12 => Ok(GF2_128::zero()),

    // ── Check (tag 13): CheckMul ────────────────────────────────────────
    // a * b == (q_hi << 128) | q_lo
    // Columns: in0 = q_lo, in1 = q_hi, in2 = a, advice = b
    // Integer multiplication cannot be checked algebraically in GF(2^128).
    // Soundness is enforced by Mul LUT byte products + accumulation
    // Add LUT + reconstruction binding.
    13 => Ok(GF2_128::zero()),

    // ── Check (tag 14): CheckInv ────────────────────────────────────────
    // a * a_inv ≡ 1 mod 2^128 → widening_mul(a, a_inv).lo = 1
    // Columns: in0 = a, in1 = a_inv
    // Integer multiplication cannot be checked algebraically in GF(2^128).
    // Soundness is enforced by Mul LUT byte products + accumulation
    // Add LUT proving the low 128 bits of a×a_inv equal 1.
    14 => Ok(GF2_128::zero()),

    // ── Check (tag 15): RangeCheck ──────────────────────────────────────
    // Verify in0 < 2^bits.  At the field level the prover provides a
    // witness decomposition; here we just enforce:
    //   out = in0  (the range-checked value passes through unchanged)
    15 => Ok(out + in0),

    // ── Memory (tag 16): Load ───────────────────────────────────────────
    // out = memory[addr].  Like advice, the value is witness-provided.
    // Constraint: out + advice = 0  (advice holds the loaded value).
    16 => Ok(out + _advice),

    // ── Memory (tag 17): Store ──────────────────────────────────────────
    // No output; constraint is identity (verified by memory consistency
    // argument at a higher level).  Return zero unconditionally.
    17 => Ok(GF2_128::zero()),

    // ── Keccak leaf (tag 18): KeccakLeaf ────────────────────────────────
    // Sub-proof commitment absorption — no arithmetic constraint locally.
    18 => Ok(GF2_128::zero()),

    // ── Structure (tag 19): Compose ─────────────────────────────────────
    // Structural annotation only — verified at the proof-tree level.
    19 => Ok(GF2_128::zero()),

    // ── Structure (tag 20): TypeCheck ───────────────────────────────────
    // Structural annotation — verified by evm-types type checker.
    20 => Ok(GF2_128::zero()),

    // ── Control (tag 21): Done ──────────────────────────────────────────
    // No state change; all columns must be zero except pc/op.
    21 => Ok(out + in0 + in1 + in2 + flags + _advice),

    // ── EVM Memory (tag 22): MLoad ──────────────────────────────────────
    // Witness-provided load; consistency checked at memory argument level.
    22 => Ok(out + _advice),

    // ── EVM Memory (tag 23): MStore ─────────────────────────────────────
    // Write-only, no local output constraint.
    23 => Ok(GF2_128::zero()),

    // ── EVM Memory (tag 24): MStore8 ────────────────────────────────────
    // Write a single byte.  No local output constraint.
    24 => Ok(GF2_128::zero()),

    // ── EVM Storage (tag 25): SLoad ─────────────────────────────────────
    // Witness-provided; checked by storage consistency argument.
    25 => Ok(out + _advice),

    // ── EVM Storage (tag 26): SStore ────────────────────────────────────
    // Write-only storage, no local constraint.
    26 => Ok(GF2_128::zero()),

    // ── EVM Transient Storage (tag 27): TLoad ────────────────────────────
    // Witness-provided; checked by storage consistency argument.
    27 => Ok(out + _advice),

    // ── EVM Transient Storage (tag 28): TStore ───────────────────────────
    // Write-only transient storage, no local constraint.
    28 => Ok(GF2_128::zero()),
    // ── Wide advice (tag 29): Advice2 ─────────────────────────────
    // 2 values loaded from advice tape. No algebraic constraint;
    // values are verified by downstream Check* operations.
    29 => Ok(GF2_128::zero()),

    // ── Wide advice (tag 30): Advice4 ─────────────────────────────
    // 4 values loaded from advice tape. No algebraic constraint;
    // values are verified by downstream Check* operations.
    30 => Ok(GF2_128::zero()),

    // ── Zero-check (tag 31): CheckZeroInv ────────────────────────
    // acc * inv + result + 1 = 0  in GF(2^128).
    // Pure field-algebraic constraint (no LUT needed).
    31 => {
      let one = GF2_128::from(1u64);
      Ok(in0 * in1 + out + one)
    }

    // ── Zero-check (tag 32): CheckZeroMul ────────────────────────
    // acc * result = 0  in GF(2^128).
    // GF(2^128) is a field: no zero divisors, so this forces
    // acc = 0 or result = 0.
    32 => Ok(in0 * in1),

    _ => Err(ConstraintError::UnknownTag(tag)),
  }
}

/// Build the *selector polynomial* value for a given tag and random
/// challenge `β`.
///
/// The selector polynomial at a row with opcode `t` is:
///
/// ```text
/// sel(t, β) = ∏_{j ≠ t} (β − j)
/// ```
///
/// This is nonzero when `β ≠ t` and zero when `β = j` for some other tag `j`.
/// Multiplied with the constraint, it "selects" only the correct constraint
/// per-row in the batched sumcheck.
pub fn selector(tag: u32, beta: GF2_128) -> GF2_128 {
  let mut acc = GF2_128::one();
  for j in 0..NUM_TAGS {
    if j != tag {
      acc = acc * (beta + GF2_128::from(j as u64));
    }
  }
  acc
}

/// Evaluate the *batched constraint* on a single encoded row.
///
/// ```text
/// C(row, β) = sel(op, β) · constraint(op, cols)
/// ```
///
/// Evaluate the batched constraint using a known tag (avoids selector iteration).
///
/// This is more efficient when the tag is known (e.g. during trace encoding).
/// Returns `sel(tag, beta) * constraint(tag, cols)`.
pub fn batched_constraint_with_tag(
  tag: u32,
  cols: &[GF2_128; 8],
  beta: GF2_128,
) -> Result<GF2_128, ConstraintError> {
  let c = eval_constraint(tag, cols)?;
  Ok(selector(tag, beta) * c)
}

/// Precompute selector values for all tags at a fixed `beta`.
///
/// `result[t] = sel(t, beta)` for `t` in `0..NUM_TAGS`.
/// Avoids 28 GF multiplications per row when the selector table is reused.
pub fn precompute_selectors(beta: GF2_128) -> [GF2_128; NUM_TAGS as usize] {
  std::array::from_fn(|t| selector(t as u32, beta))
}

/// Like [`batched_constraint_with_tag`] but uses a precomputed selector value.
#[inline]
pub fn batched_constraint_with_precomputed_sel(
  tag: u32,
  cols: &[GF2_128; 8],
  sel: GF2_128,
) -> Result<GF2_128, ConstraintError> {
  let c = eval_constraint(tag, cols)?;
  Ok(sel * c)
}

#[cfg(test)]
mod tests {
  use super::*;

  fn f(v: u64) -> GF2_128 {
    GF2_128::from(v)
  }

  fn zero_cols() -> [GF2_128; 8] {
    [GF2_128::zero(); 8]
  }

  // ── Add128 ───────────────────────────────────────────────────────────────────

  #[test]
  fn add128_deferred_to_lookup() {
    // Add128 constraint is zero — soundness comes from Add LUT.
    let cols = zero_cols();
    assert_eq!(eval_constraint(0, &cols).unwrap(), GF2_128::zero());
    // Any column values → still zero (no algebraic check).
    let mut cols2 = zero_cols();
    cols2[2] = f(100);
    cols2[3] = f(200);
    cols2[5] = f(999);
    assert_eq!(eval_constraint(0, &cols2).unwrap(), GF2_128::zero());
  }

  // ── And128 ───────────────────────────────────────────────────────────────────

  #[test]
  fn and128_deferred_to_lookup() {
    // And128 constraint is zero — soundness comes from And LUT.
    let cols = zero_cols();
    assert_eq!(eval_constraint(2, &cols).unwrap(), GF2_128::zero());
  }

  // ── Mul128 ───────────────────────────────────────────────────────────────────

  #[test]
  fn mul128_deferred_to_lookup() {
    // Mul128 constraint is zero — soundness comes from Mul LUT + deterministic check.
    let cols = zero_cols();
    assert_eq!(eval_constraint(1, &cols).unwrap(), GF2_128::zero());
    let mut cols2 = zero_cols();
    cols2[2] = f(42);
    cols2[3] = f(99);
    cols2[5] = f(12345);
    assert_eq!(eval_constraint(1, &cols2).unwrap(), GF2_128::zero());
  }

  // ── Xor128 ───────────────────────────────────────────────────────────────────

  #[test]
  fn xor128_valid() {
    // out = in0 + in1
    let mut cols = zero_cols();
    cols[2] = f(0xA); // in0
    cols[3] = f(0x5); // in1
    cols[5] = f(0xA) + f(0x5); // out
    assert_eq!(eval_constraint(3, &cols).unwrap(), GF2_128::zero());
  }

  // ── Not128 ───────────────────────────────────────────────────────────────────

  #[test]
  fn not128_valid() {
    // out = in0 + mask128
    let mut cols = zero_cols();
    cols[2] = f(0); // in0
    cols[5] = GF2_128::new(u64::MAX, u64::MAX); // out = NOT 0
    assert_eq!(eval_constraint(4, &cols).unwrap(), GF2_128::zero());
  }

  // ── Mov ───────────────────────────────────────────────────────────────

  #[test]
  fn mov_valid() {
    let mut cols = zero_cols();
    cols[2] = f(42); // in0
    cols[5] = f(42); // out = in0
    assert_eq!(eval_constraint(10, &cols).unwrap(), GF2_128::zero());
  }

  // ── Done ──────────────────────────────────────────────────────────────

  #[test]
  fn done_valid() {
    // All data columns zero
    let cols = zero_cols();
    assert_eq!(eval_constraint(21, &cols).unwrap(), GF2_128::zero());
  }

  // ── CheckInv, CheckMul, CheckDiv ────────────────────────────────────────
  // Tags 12, 13, 14 now return zero() — soundness is delegated to LogUp.
  // The algebraic constraint no longer rejects invalid values; this is
  // tested at the lookup level instead.

  #[test]
  fn check_inv_returns_zero() {
    // Tag 14 always returns zero (delegated to LogUp)
    let mut cols = zero_cols();
    cols[2] = f(3);
    cols[3] = f(5);
    assert_eq!(eval_constraint(14, &cols).unwrap(), GF2_128::zero());
  }

  #[test]
  fn check_div_returns_zero() {
    // Tag 12 always returns zero (delegated to LogUp)
    let cols = zero_cols();
    assert_eq!(eval_constraint(12, &cols).unwrap(), GF2_128::zero());
  }

  #[test]
  fn check_mul_returns_zero() {
    // Tag 13 always returns zero (delegated to LogUp)
    let cols = zero_cols();
    assert_eq!(eval_constraint(13, &cols).unwrap(), GF2_128::zero());
  }

  // ── Unknown tag ───────────────────────────────────────────────────────

  #[test]
  fn unknown_tag_error() {
    let cols = zero_cols();
    assert!(eval_constraint(99, &cols).is_err());
  }

  // ── Selector ──────────────────────────────────────────────────────────

  #[test]
  fn selector_nonzero_for_matching_tag() {
    // When β equals the tag value, the selector for OTHER tags should be
    // nonzero.  The selector for the matching tag ∏_{j≠t}(β−j) is nonzero
    // when β = t because none of the (β−j) factors are zero (j ≠ t).
    let beta = f(5);
    assert!(!selector(5, beta).is_zero());
  }

  // ── L-1 coverage gap: AdviceLoad (tag 11) accepts arbitrary values ──
  //
  // Comparison opcodes (LT, GT, EQ, ISZERO) use AdviceLoad to inject
  // the boolean result.  The circuit constraint for tag 11 is:
  //   `out + advice = 0`  (i.e., out == advice)
  // Combined with a 1-bit RangeCheck (tag 15), this only ensures the
  // result is 0 or 1 — NOT that it is the correct comparison result.
  //
  // These tests document that a malicious prover can inject wrong
  // comparison results that still satisfy all circuit constraints.
  // Correctness currently depends on `consistency_check`.

  #[test]
  fn advice_load_accepts_any_value() {
    // Tag 11: out + advice = 0.  Any (out, advice) pair with out == advice passes.
    let mut cols_true = zero_cols();
    cols_true[5] = f(1); // out = 1
    cols_true[7] = f(1); // advice = 1
    assert_eq!(eval_constraint(11, &cols_true).unwrap(), GF2_128::zero());

    let mut cols_false = zero_cols();
    cols_false[5] = f(0); // out = 0
    cols_false[7] = f(0); // advice = 0
    assert_eq!(eval_constraint(11, &cols_false).unwrap(), GF2_128::zero());

    // Both pass — circuit cannot distinguish correct from incorrect advice.
  }

  #[test]
  fn range_check_1bit_accepts_both_values() {
    // Tag 15: out + in0 = 0.  For 1-bit range check, both 0 and 1 pass.
    let mut cols_zero = zero_cols();
    cols_zero[2] = f(0); // in0 = 0
    cols_zero[5] = f(0); // out = 0
    assert_eq!(eval_constraint(15, &cols_zero).unwrap(), GF2_128::zero());

    let mut cols_one = zero_cols();
    cols_one[2] = f(1); // in0 = 1
    cols_one[5] = f(1); // out = 1
    assert_eq!(eval_constraint(15, &cols_one).unwrap(), GF2_128::zero());
  }

  #[test]
  fn comparison_gap_wrong_iszero_passes_constraints() {
    // ISZERO(5) should return 0, but a malicious prover claims 1.
    // AdviceLoad row: out=1, advice=1 → tag 11 passes.
    // RangeCheck row: in0=1, out=1 → tag 15 passes.
    // Mov row: in0=1, out=1 → tag 10 passes.
    // The circuit accepts the wrong result.
    for wrong_result in [0u64, 1u64] {
      let v = f(wrong_result);
      // AdviceLoad
      let mut adv = zero_cols();
      adv[5] = v;
      adv[7] = v;
      assert_eq!(eval_constraint(11, &adv).unwrap(), GF2_128::zero());
      // RangeCheck (1-bit)
      let mut rc = zero_cols();
      rc[2] = v;
      rc[5] = v;
      assert_eq!(eval_constraint(15, &rc).unwrap(), GF2_128::zero());
      // Mov
      let mut mv = zero_cols();
      mv[2] = v;
      mv[5] = v;
      assert_eq!(eval_constraint(10, &mv).unwrap(), GF2_128::zero());
    }
    // Both wrong_result=0 and wrong_result=1 pass all constraints.
    // Only one is correct for a given input, but the circuit cannot tell.
  }
}
