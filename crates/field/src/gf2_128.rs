//! GF(2^128) — degree-2 extension tower over GF(2^64).
//!
//! ## Construction
//! `GF(2^128) = GF(2^64)[t] / (t^2 + t + β)`
//!
//! where `β = GF2_64(2)` (`x` in the base field).  This polynomial is
//! irreducible over GF(2^64) because `t^2 + t + β` has no root in GF(2^64):
//! evaluating at 0 gives `β ≠ 0`, at 1 gives `1 + 1 + β = β ≠ 0`.
//!
//! ## Representation
//! `GF2_128 { lo, hi }` represents `lo + hi * t`.
//!
//! ## Operations
//! - **Add**: `(a0 + a1*t) + (b0 + b1*t) = (a0+b0) + (a1+b1)*t`
//! - **Mul**: schoolbook then reduce with `t^2 = t + β`
//!   ```text
//!   (a0 + a1 t)(b0 + b1 t) = a0 b0 + (a0 b1 + a1 b0) t + a1 b1 t^2
//!                           = (a0 b0 + a1 b1 β) + (a0 b1 + a1 b0 + a1 b1) t
//!   ```

use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::{gf2_64::GF2_64, traits::FieldElem};

/// β = x ∈ GF(2^64), the constant in the extension polynomial `t^2 + t + β`.
const BETA: GF2_64 = GF2_64(2);

/// An element of GF(2^128) = GF(2^64)[t] / (t^2 + t + β).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GF2_128 {
  pub lo: GF2_64, // coefficient of 1
  pub hi: GF2_64, // coefficient of t
}

impl GF2_128 {
  #[inline]
  pub const fn new(lo: GF2_64, hi: GF2_64) -> Self { Self { lo, hi } }

  /// Embed a base-field element as `lo + 0*t`.
  #[inline]
  pub const fn from_base(e: GF2_64) -> Self { Self { lo: e, hi: GF2_64(0) } }
}

// ─── Add / Sub (characteristic 2: same) ──────────────────────────────────────

impl Add for GF2_128 {
  type Output = Self;
  #[inline]
  fn add(self, rhs: Self) -> Self {
    Self { lo: self.lo + rhs.lo, hi: self.hi + rhs.hi }
  }
}
#[allow(clippy::suspicious_arithmetic_impl, clippy::suspicious_op_assign_impl)]
impl Sub for GF2_128 {
  type Output = Self;
  #[inline]
  fn sub(self, rhs: Self) -> Self { self + rhs } // char 2: sub = add
}
impl Neg for GF2_128 {
  type Output = Self;
  #[inline]
  fn neg(self) -> Self { self }
}
impl AddAssign for GF2_128 {
  #[inline]
  fn add_assign(&mut self, rhs: Self) { self.lo += rhs.lo; self.hi += rhs.hi; }
}
#[allow(clippy::suspicious_op_assign_impl)]
impl SubAssign for GF2_128 {
  #[inline]
  fn sub_assign(&mut self, rhs: Self) { *self += rhs; }
}

// ─── Mul ──────────────────────────────────────────────────────────────────────

impl Mul for GF2_128 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: Self) -> Self {
    // Schoolbook (Karatsuba would save one GF2_64 mul):
    //   c0 = a0*b0 + a1*b1*β
    //   c1 = a0*b1 + a1*b0 + a1*b1
    let a0b0 = self.lo * rhs.lo;
    let a1b1 = self.hi * rhs.hi;
    let a0b1 = self.lo * rhs.hi;
    let a1b0 = self.hi * rhs.lo;
    Self {
      lo: a0b0 + a1b1 * BETA,
      hi: a0b1 + a1b0 + a1b1,
    }
  }
}
impl MulAssign for GF2_128 {
  #[inline]
  fn mul_assign(&mut self, rhs: Self) { *self = *self * rhs; }
}

// ─── From<u64> ────────────────────────────────────────────────────────────────

impl From<u64> for GF2_128 {
  #[inline]
  fn from(v: u64) -> Self { Self::from_base(GF2_64(v)) }
}

// ─── FieldElem ────────────────────────────────────────────────────────────────

impl FieldElem for GF2_128 {
  #[inline] fn zero() -> Self { Self { lo: GF2_64(0), hi: GF2_64(0) } }
  #[inline] fn one()  -> Self { Self { lo: GF2_64(1), hi: GF2_64(0) } }
  #[inline] fn is_zero(self) -> bool { self.lo.is_zero() && self.hi.is_zero() }

  /// Inverse via norm: a⁻¹ = conj(a) · N(a)⁻¹.
  fn inv(self) -> Self {
    assert!(!self.is_zero(), "GF2_128::inv called on zero");
    let n_inv = (self.lo * self.lo + self.lo * self.hi + self.hi * self.hi * BETA).inv();
    let c = Self { lo: self.lo + self.hi, hi: self.hi };
    Self { lo: c.lo * n_inv, hi: c.hi * n_inv }
  }
}

impl GF2_128 {
  /// Frobenius conjugate: `conj(a0 + a1·t) = (a0 + a1) + a1·t`.
  #[inline]
  pub fn conjugate(self) -> Self {
    Self { lo: self.lo + self.hi, hi: self.hi }
  }

  /// Field norm N: GF(2^128) → GF(2^64): `N(a) = a0² + a0·a1 + a1²·β`.
  #[inline]
  pub fn norm(self) -> GF2_64 {
    self.lo * self.lo + self.lo * self.hi + self.hi * self.hi * BETA
  }

  /// Sample a uniformly random element.
  pub fn random<R: rand::Rng>(rng: &mut R) -> Self {
    Self { lo: GF2_64::random(rng), hi: GF2_64::random(rng) }
  }

  /// Sample a uniformly random non-zero element.
  pub fn random_nonzero<R: rand::Rng>(rng: &mut R) -> Self {
    loop {
      let v = Self::random(rng);
      if !v.is_zero() { return v; }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::traits::FieldElem;

  fn a() -> GF2_128 { GF2_128::new(GF2_64(0xDEAD_BEEF_0000_0001), GF2_64(0x1234_5678_9ABC_DEF0)) }
  fn b() -> GF2_128 { GF2_128::new(GF2_64(0xAAAA_BBBB_CCCC_DDDD), GF2_64(0xFEDC_BA98_7654_3210)) }
  fn c() -> GF2_128 { GF2_128::new(GF2_64(0x0102_0304_0506_0708), GF2_64(0x0807_0605_0403_0201)) }

  #[test]
  fn add_sub_roundtrip() {
    assert_eq!((a() + b()) - b(), a());
  }

  #[test]
  fn neg_is_self() {
    assert_eq!(-a(), a());
  }

  #[test]
  fn mul_zero() { assert_eq!(a() * GF2_128::zero(), GF2_128::zero()); }

  #[test]
  fn mul_one() { assert_eq!(a() * GF2_128::one(), a()); }

  #[test]
  fn mul_commutative() { assert_eq!(a() * b(), b() * a()); }

  #[test]
  fn mul_associative() { assert_eq!((a() * b()) * c(), a() * (b() * c())); }

  #[test]
  fn mul_distributive() { assert_eq!(a() * (b() + c()), a() * b() + a() * c()); }

  #[test]
  fn norm_in_base_field_for_one() {
    // N(1) = 1^2 + 1*0 + 0^2*β = 1
    assert_eq!(GF2_128::one().norm(), GF2_64::one());
  }

  #[test]
  fn conjugate_of_base_elem_is_self() {
    let e = GF2_128::from_base(GF2_64(42));
    assert_eq!(e.conjugate(), e);
  }

  #[test]
  fn inv_roundtrip() {
    let x = a();
    assert_eq!(x * x.inv(), GF2_128::one());
  }

  #[test]
  fn norm_is_multiplicative() {
    // N(a*b) == N(a)*N(b)
    assert_eq!((a() * b()).norm(), a().norm() * b().norm());
  }

  #[test]
  fn from_u64_is_base_embed() {
    let v = 0xCAFE_BABEu64;
    assert_eq!(GF2_128::from(v), GF2_128::from_base(GF2_64(v)));
  }

  #[test]
  fn pow_matches_repeated_mul() {
    let x = a();
    let x3 = x * x * x;
    assert_eq!(x.pow(3), x3);
  }

  #[test]
  fn random_inv_roundtrip() {
    let mut rng = rand::rng();
    for _ in 0..20 {
      let x = GF2_128::random_nonzero(&mut rng);
      assert_eq!(x * x.inv(), GF2_128::one());
    }
  }

}
