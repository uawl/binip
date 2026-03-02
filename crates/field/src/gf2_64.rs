//! GF(2^64) — degree-2 tower extension over GF(2^32).
//!
//! ## Construction
//! `GF(2^64) = GF(2^32)[t] / (t^2 + t + β)` where `β = GF2_32(2)`.
//!
//! ## Representation
//! `GF2_64(pub u64)` encodes `lo + hi·t` as `lo | (hi << 32)` where
//! `lo, hi ∈ GF(2^32)` are stored as their raw `u32` bit patterns.
//!
//! ## Multiplication
//! Schoolbook over GF(2^32):
//!   `(a0+a1·t)(b0+b1·t) = (a0·b0 + a1·b1·β) + (a0·b1 + a1·b0 + a1·b1)·t`

use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::{gf2_32::GF2_32, traits::FieldElem};

/// β constant in the tower polynomial `t^2 + t + β` over GF(2^32).
const BETA32: GF2_32 = GF2_32(2);

/// An element of GF(2^64) = GF(2^32)[t]/(t^2+t+β).
/// Stored as `u64 = (lo as u32) | ((hi as u32) << 32)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GF2_64(pub u64);

#[inline] fn lo32(x: GF2_64) -> GF2_32 { GF2_32(x.0 as u32) }
#[inline] fn hi32(x: GF2_64) -> GF2_32 { GF2_32((x.0 >> 32) as u32) }
#[inline] fn from_lohi(lo: GF2_32, hi: GF2_32) -> GF2_64 {
  GF2_64((lo.0 as u64) | ((hi.0 as u64) << 32))
}

// GF(2) addition = XOR.
#[allow(clippy::suspicious_arithmetic_impl, clippy::suspicious_op_assign_impl)]
impl Add for GF2_64 {
  type Output = Self;
  #[inline]
  fn add(self, rhs: Self) -> Self { Self(self.0 ^ rhs.0) }
}
#[allow(clippy::suspicious_arithmetic_impl, clippy::suspicious_op_assign_impl)]
impl Sub for GF2_64 {
  type Output = Self;
  #[inline]
  fn sub(self, rhs: Self) -> Self { Self(self.0 ^ rhs.0) } // same as add in GF(2)
}
impl Neg for GF2_64 {
  type Output = Self;
  #[inline]
  fn neg(self) -> Self { self } // characteristic 2: -a = a
}
#[allow(clippy::suspicious_op_assign_impl)]
impl AddAssign for GF2_64 {
  #[inline]
  fn add_assign(&mut self, rhs: Self) { self.0 ^= rhs.0; }
}
#[allow(clippy::suspicious_op_assign_impl)]
impl SubAssign for GF2_64 {
  #[inline]
  fn sub_assign(&mut self, rhs: Self) { self.0 ^= rhs.0; }
}

// ─── tower multiplication over GF(2^32) ──────────────────────────────────────
// (a0+a1·t)(b0+b1·t) = (a0·b0 + a1·b1·β) + (a0·b1 + a1·b0 + a1·b1)·t

impl Mul for GF2_64 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: Self) -> Self {
    let (a0, a1) = (lo32(self), hi32(self));
    let (b0, b1) = (lo32(rhs), hi32(rhs));
    let a0b0 = a0 * b0;
    let a1b1 = a1 * b1;
    from_lohi(a0b0 + a1b1 * BETA32, a0 * b1 + a1 * b0 + a1b1)
  }
}
impl MulAssign for GF2_64 {
  #[inline]
  fn mul_assign(&mut self, rhs: Self) { *self = *self * rhs; }
}

// ─── From<u64> ────────────────────────────────────────────────────────────────

impl From<u64> for GF2_64 {
  #[inline]
  fn from(v: u64) -> Self { Self(v) }
}

// ─── FieldElem ────────────────────────────────────────────────────────────────

impl FieldElem for GF2_64 {
  #[inline] fn zero() -> Self { Self(0) }
  #[inline] fn one() -> Self { Self(1) }
  #[inline] fn is_zero(self) -> bool { self.0 == 0 }

  /// Extended Euclidean algorithm over GF(2^64).
  fn inv(self) -> Self {
    assert!(!self.is_zero(), "GF2_64::inv called on zero");
    // Fermat: a^(2^64 - 2) = a^(-1).
    // 2^64 - 2 = 2*(2^63 - 1).  Use square-and-multiply.
    self.pow(u64::MAX - 1)
  }
}

// ─── random sampling (tests / fuzzing) ───────────────────────────────────────

impl GF2_64 {
  /// Sample a uniformly random non-zero element.
  pub fn random_nonzero<R: rand::Rng>(rng: &mut R) -> Self {
    loop {
      let v = Self(rng.random::<u64>());
      if !v.is_zero() { return v; }
    }
  }

  /// Sample a uniformly random element (may be zero).
  pub fn random<R: rand::Rng>(rng: &mut R) -> Self {
    Self(rng.random::<u64>())
  }
}

// ─── unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn add_is_xor() {
    let a = GF2_64(0xDEAD_BEEF_CAFE_BABEu64);
    let b = GF2_64(0x1234_5678_9ABC_DEF0u64);
    assert_eq!(a + b, GF2_64(a.0 ^ b.0));
  }

  #[test]
  fn neg_is_identity() {
    let a = GF2_64(42);
    assert_eq!(-a, a);
    assert_eq!(a - a, GF2_64::zero());
  }

  #[test]
  fn mul_by_zero_is_zero() {
    let a = GF2_64(0xFFFF_FFFF_FFFF_FFFFu64);
    assert_eq!(a * GF2_64::zero(), GF2_64::zero());
  }

  #[test]
  fn mul_by_one_is_identity() {
    let a = GF2_64(0xDEAD_BEEF_0000_0001u64);
    assert_eq!(a * GF2_64::one(), a);
  }

  #[test]
  fn mul_commutativity() {
    let a = GF2_64(0x1111_2222_3333_4444u64);
    let b = GF2_64(0xAAAA_BBBB_CCCC_DDDDu64);
    assert_eq!(a * b, b * a);
  }

  #[test]
  fn mul_associativity() {
    let a = GF2_64(0x0102_0304_0506_0708u64);
    let b = GF2_64(0x0807_0605_0403_0201u64);
    let c = GF2_64(0xFEDC_BA98_7654_3210u64);
    assert_eq!((a * b) * c, a * (b * c));
  }

  #[test]
  fn mul_distributivity() {
    let a = GF2_64(0xABCD_EF01_2345_6789u64);
    let b = GF2_64(0x1111_1111_1111_1111u64);
    let c = GF2_64(0x2222_2222_2222_2222u64);
    assert_eq!(a * (b + c), a * b + a * c);
  }

  #[test]
  fn inv_roundtrip() {
    let a = GF2_64(0xDEAD_BEEF_CAFE_0001u64);
    assert_eq!(a * a.inv(), GF2_64::one());
  }

  #[test]
  fn pow_small() {
    let a = GF2_64(2); // x
    // x^2 = x*x
    let x2 = a * a;
    assert_eq!(a.pow(2), x2);
    assert_eq!(a.pow(0), GF2_64::one());
    assert_eq!(a.pow(1), a);
  }

  #[test]
  fn from_u64_roundtrip() {
    let v: u64 = 0x1234_5678_9ABC_DEF0;
    assert_eq!(GF2_64::from(v).0, v);
  }
}
