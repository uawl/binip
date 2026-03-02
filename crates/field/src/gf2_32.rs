//! GF(2^32) — binary extension field.
//!
//! ## Irreducible polynomial
//! `p(x) = x^32 + x^7 + x^3 + x^2 + 1`
//!
//! ## Multiplication
//! Portable 32×32 carry-less multiply then reduction modulo p(x).
//! `x^32 ≡ x^7 + x^3 + x^2 + 1`, so:
//! `x^(32+k) ≡ x^(k+7) + x^(k+3) + x^(k+2) + x^k`

use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::traits::FieldElem;

/// An element of GF(2^32).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GF2_32(pub u32);

// GF(2) addition = XOR.
#[allow(clippy::suspicious_arithmetic_impl, clippy::suspicious_op_assign_impl)]
impl Add for GF2_32 {
  type Output = Self;
  #[inline]
  fn add(self, rhs: Self) -> Self { Self(self.0 ^ rhs.0) }
}
#[allow(clippy::suspicious_arithmetic_impl, clippy::suspicious_op_assign_impl)]
impl Sub for GF2_32 {
  type Output = Self;
  #[inline]
  fn sub(self, rhs: Self) -> Self { Self(self.0 ^ rhs.0) }
}
impl Neg for GF2_32 {
  type Output = Self;
  #[inline]
  fn neg(self) -> Self { self }
}
#[allow(clippy::suspicious_op_assign_impl)]
impl AddAssign for GF2_32 {
  #[inline]
  fn add_assign(&mut self, rhs: Self) { self.0 ^= rhs.0; }
}
#[allow(clippy::suspicious_op_assign_impl)]
impl SubAssign for GF2_32 {
  #[inline]
  fn sub_assign(&mut self, rhs: Self) { self.0 ^= rhs.0; }
}

// ─── carry-less multiplication ────────────────────────────────────────────────

/// Portable 32×32 → 64-bit carry-less multiply.
#[inline]
fn clmul32(a: u32, b: u32) -> u64 {
  let mut lo: u32 = 0;
  let mut hi: u32 = 0;
  for i in 0..32u32 {
    if (a >> i) & 1 == 1 {
      lo ^= b << i;
      if i > 0 { hi ^= b >> (32 - i); }
    }
  }
  (lo as u64) | ((hi as u64) << 32)
}

/// Reduce a 64-bit carry-less product modulo `p(x) = x^32+x^7+x^3+x^2+1`.
/// `x^(32+k) ≡ x^(k+7)+x^(k+3)+x^(k+2)+x^k`
#[inline]
fn reduce32(product: u64) -> u32 {
  let lo = product as u32;
  let hi = (product >> 32) as u32;
  // Each bit k of hi represents x^(32+k); reduce to lo bits.
  let tail = hi ^ (hi << 7) ^ (hi << 3) ^ (hi << 2);
  // Overflow from the shifts past bit 31 (hi<<7 overflows at bit 25, etc.)
  let overflow = (hi >> 25) ^ (hi >> 29) ^ (hi >> 30);
  // overflow has at most bits 0..6, so one more reduction step suffices.
  let extra = overflow ^ (overflow << 7) ^ (overflow << 3) ^ (overflow << 2);
  lo ^ tail ^ extra
}

impl Mul for GF2_32 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: Self) -> Self {
    Self(reduce32(clmul32(self.0, rhs.0)))
  }
}
impl MulAssign for GF2_32 {
  #[inline]
  fn mul_assign(&mut self, rhs: Self) { *self = *self * rhs; }
}

// ─── From<u64> ────────────────────────────────────────────────────────────────

impl From<u64> for GF2_32 {
  #[inline]
  fn from(v: u64) -> Self { Self(v as u32) }
}

// ─── FieldElem ────────────────────────────────────────────────────────────────

impl FieldElem for GF2_32 {
  #[inline] fn zero() -> Self { Self(0) }
  #[inline] fn one()  -> Self { Self(1) }
  #[inline] fn is_zero(self) -> bool { self.0 == 0 }

  /// Fermat: a^(2^32 − 2) = a^(−1).
  fn inv(self) -> Self {
    assert!(!self.is_zero(), "GF2_32::inv called on zero");
    self.pow((1u64 << 32) - 2)
  }
}

// ─── random sampling ─────────────────────────────────────────────────────────

impl GF2_32 {
  /// Sample a uniformly random non-zero element.
  pub fn random_nonzero<R: rand::Rng>(rng: &mut R) -> Self {
    loop {
      let v = Self(rng.random::<u32>());
      if !v.is_zero() { return v; }
    }
  }

  /// Sample a uniformly random element (may be zero).
  pub fn random<R: rand::Rng>(rng: &mut R) -> Self {
    Self(rng.random::<u32>())
  }
}

// ─── unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn add_is_xor() {
    let a = GF2_32(0xDEAD_BEEF);
    let b = GF2_32(0x1234_5678);
    assert_eq!(a + b, GF2_32(a.0 ^ b.0));
  }

  #[test]
  fn neg_is_identity() {
    assert_eq!(-GF2_32(42), GF2_32(42));
    assert_eq!(GF2_32(7) - GF2_32(7), GF2_32::zero());
  }

  #[test]
  fn mul_by_zero() {
    assert_eq!(GF2_32(0xFFFF_FFFF) * GF2_32::zero(), GF2_32::zero());
  }

  #[test]
  fn mul_by_one() {
    let a = GF2_32(0xABCD_1234);
    assert_eq!(a * GF2_32::one(), a);
  }

  #[test]
  fn mul_commutative() {
    let a = GF2_32(0x1111_2222);
    let b = GF2_32(0xAAAA_BBBB);
    assert_eq!(a * b, b * a);
  }

  #[test]
  fn mul_associative() {
    let a = GF2_32(0x0102_0304);
    let b = GF2_32(0x0807_0605);
    let c = GF2_32(0xFEDC_BA98);
    assert_eq!((a * b) * c, a * (b * c));
  }

  #[test]
  fn mul_distributive() {
    let a = GF2_32(0xABCD_EF01);
    let b = GF2_32(0x1111_1111);
    let c = GF2_32(0x2222_2222);
    assert_eq!(a * (b + c), a * b + a * c);
  }

  #[test]
  fn inv_roundtrip() {
    let a = GF2_32(0xDEAD_0001);
    assert_eq!(a * a.inv(), GF2_32::one());
  }

  #[test]
  fn pow_matches_repeated_mul() {
    let a = GF2_32(0xABCD_0001);
    assert_eq!(a.pow(4), a * a * a * a);
  }
}
