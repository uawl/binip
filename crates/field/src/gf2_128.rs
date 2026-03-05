//! GF(2^128) — flat binary extension field via CLMUL.
//!
//! ## Construction
//! `GF(2^128) = GF(2)[x] / (x^128 + x^7 + x^2 + x + 1)`
//!
//! This is the standard GCM/GHASH irreducible polynomial.
//!
//! ## Representation
//! `GF2_128 { lo: u64, hi: u64 }` stores bits 0–63 in `lo` and 64–127 in `hi`.
//!
//! ## Multiplication
//! Karatsuba via 3× PCLMULQDQ/PMULL (carry-less 64×64→128 bit multiply)
//! followed by an inexpensive reduction.

use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::traits::FieldElem;

/// An element of GF(2^128) = GF(2)[x] / (x^128 + x^7 + x^2 + x + 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash, bincode::Encode, bincode::Decode)]
#[repr(C)]
pub struct GF2_128 {
  pub lo: u64, // bits 0..63
  pub hi: u64, // bits 64..127
}

impl GF2_128 {
  #[inline]
  pub const fn new(lo: u64, hi: u64) -> Self {
    Self { lo, hi }
  }
}

// ─── Add / Sub (characteristic 2: same) ──────────────────────────────────────

impl Add for GF2_128 {
  type Output = Self;
  #[inline]
  fn add(self, rhs: Self) -> Self {
    Self {
      lo: self.lo ^ rhs.lo,
      hi: self.hi ^ rhs.hi,
    }
  }
}
#[allow(clippy::suspicious_arithmetic_impl, clippy::suspicious_op_assign_impl)]
impl Sub for GF2_128 {
  type Output = Self;
  #[inline]
  fn sub(self, rhs: Self) -> Self {
    self + rhs
  } // char 2: sub = add
}
impl Neg for GF2_128 {
  type Output = Self;
  #[inline]
  fn neg(self) -> Self {
    self
  }
}
impl AddAssign for GF2_128 {
  #[inline]
  fn add_assign(&mut self, rhs: Self) {
    self.lo ^= rhs.lo;
    self.hi ^= rhs.hi;
  }
}
#[allow(clippy::suspicious_op_assign_impl)]
impl SubAssign for GF2_128 {
  #[inline]
  fn sub_assign(&mut self, rhs: Self) {
    *self += rhs;
  }
}

// ─── Mul (Karatsuba + reduction) ─────────────────────────────────────────────

impl Mul for GF2_128 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: Self) -> Self {
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    {
      // Fused Karatsuba: pack once → 3× PCLMULQDQ → recombine → reduce.
      unsafe {
        use std::arch::x86_64::*;
        let va = _mm_set_epi64x(self.hi as i64, self.lo as i64);
        let vb = _mm_set_epi64x(rhs.hi as i64, rhs.lo as i64);

        let m0 = _mm_clmulepi64_si128(va, vb, 0x00); // lo×lo
        let m1 = _mm_clmulepi64_si128(va, vb, 0x11); // hi×hi
        let a_xor = _mm_xor_si128(va, _mm_bsrli_si128(va, 8));
        let b_xor = _mm_xor_si128(vb, _mm_bsrli_si128(vb, 8));
        let m2 = _mm_clmulepi64_si128(a_xor, b_xor, 0x00);

        let mid = _mm_xor_si128(_mm_xor_si128(m2, m0), m1);
        let low = _mm_xor_si128(m0, _mm_bslli_si128(mid, 8));
        let high = _mm_xor_si128(m1, _mm_bsrli_si128(mid, 8));

        let d0 = _mm_cvtsi128_si64(low) as u64;
        let d1 = _mm_cvtsi128_si64(_mm_bsrli_si128(low, 8)) as u64;
        let d2 = _mm_cvtsi128_si64(high) as u64;
        let d3 = _mm_cvtsi128_si64(_mm_bsrli_si128(high, 8)) as u64;

        reduce256(d0, d1, d2, d3)
      }
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
      // Fused Karatsuba: 3× PMULL → recombine → scalar reduce.
      unsafe {
        use std::arch::aarch64::*;

        // vmull_p64 does poly-mul 64×64 → 128.
        let m0: u128 = std::mem::transmute(vmull_p64(self.lo, rhs.lo));
        let m1: u128 = std::mem::transmute(vmull_p64(self.hi, rhs.hi));
        let m2: u128 = std::mem::transmute(vmull_p64(self.lo ^ self.hi, rhs.lo ^ rhs.hi));

        let mid = m2 ^ m0 ^ m1;
        let d0 = m0 as u64;
        let d1 = (m0 >> 64) as u64 ^ mid as u64;
        let d2 = (mid >> 64) as u64 ^ m1 as u64;
        let d3 = (m1 >> 64) as u64;

        reduce256(d0, d1, d2, d3)
      }
    }
    #[cfg(not(any(
      all(target_arch = "x86_64", target_feature = "pclmulqdq"),
      all(target_arch = "aarch64", target_feature = "aes"),
    )))]
    {
      let m0 = clmul64(self.lo, rhs.lo);
      let m1 = clmul64(self.hi, rhs.hi);
      let m2 = clmul64(self.lo ^ self.hi, rhs.lo ^ rhs.hi);
      let mid = m2 ^ m0 ^ m1;
      let d0 = m0 as u64;
      let d1 = (m0 >> 64) as u64 ^ mid as u64;
      let d2 = (mid >> 64) as u64 ^ m1 as u64;
      let d3 = (m1 >> 64) as u64;
      reduce256(d0, d1, d2, d3)
    }
  }
}
impl MulAssign for GF2_128 {
  #[inline]
  fn mul_assign(&mut self, rhs: Self) {
    *self = *self * rhs;
  }
}

// ─── 256-bit → 128-bit reduction mod x^128 + x^7 + x^2 + x + 1 ─────────────
//
// x^128 ≡ x^7 + x^2 + x + 1.  For a bit at position 128+k:
//   x^(128+k) ≡ x^(k+7) + x^(k+2) + x^(k+1) + x^k
//
// We reduce [d2, d3] (bits 128..255) into [d0, d1].

#[inline]
fn reduce256(d0: u64, d1: u64, d2: u64, d3: u64) -> GF2_128 {
  // Reduce d3 (bits 192..255) → bits 64..191
  //   d3 shifted by 7, 2, 1, 0 produces contributions to d1 and d2-positions.
  let t3_lo = d3 ^ (d3 << 7) ^ (d3 << 2) ^ (d3 << 1);
  let t3_hi = (d3 >> 57) ^ (d3 >> 62) ^ (d3 >> 63);

  // After folding d3: t3_lo → d1 (main contribution), t3_hi → d2 (overflow).
  let e1 = d1 ^ t3_lo;
  let e2 = d2 ^ t3_hi;

  // Now reduce e2 (bits 128..191) → bits 0..127
  let t2_lo = e2 ^ (e2 << 7) ^ (e2 << 2) ^ (e2 << 1);
  let t2_hi = (e2 >> 57) ^ (e2 >> 62) ^ (e2 >> 63);

  GF2_128 {
    lo: d0 ^ t2_lo,
    hi: e1 ^ t2_hi,
  }
}

// ─── 64×64 → 128-bit carry-less multiply ────────────────────────────────────

#[inline(always)]
#[allow(unused)]
fn clmul64(a: u64, b: u64) -> u128 {
  #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
  {
    return unsafe { clmul64_pclmul(a, b) };
  }

  #[cfg(all(target_arch = "x86_64", not(target_feature = "pclmulqdq")))]
  {
    if std::arch::is_x86_feature_detected!("pclmulqdq") {
      return unsafe { clmul64_pclmul(a, b) };
    }
  }

  #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
  {
    return unsafe { clmul64_pmull(a, b) };
  }

  #[cfg(all(target_arch = "aarch64", not(target_feature = "aes")))]
  {
    if std::arch::is_aarch64_feature_detected!("aes") {
      return unsafe { clmul64_pmull(a, b) };
    }
  }

  #[allow(unreachable_code)]
  clmul64_portable(a, b)
}

// ── x86_64 PCLMULQDQ ────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "pclmulqdq")]
#[allow(unused)]
unsafe fn clmul64_pclmul(a: u64, b: u64) -> u128 {
  use std::arch::x86_64::*;
  let va = _mm_set_epi64x(0, a as i64);
  let vb = _mm_set_epi64x(0, b as i64);
  let r = _mm_clmulepi64_si128(va, vb, 0x00);
  let lo = _mm_cvtsi128_si64(r) as u64;
  let hi_vec = _mm_srli_si128(r, 8);
  let hi = _mm_cvtsi128_si64(hi_vec) as u64;
  (lo as u128) | ((hi as u128) << 64)
}

// ── aarch64 PMULL ────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
unsafe fn clmul64_pmull(a: u64, b: u64) -> u128 {
  use std::arch::aarch64::*;
  unsafe { std::mem::transmute(vmull_p64(a, b)) }
}

// ── portable fallback ────────────────────────────────────────────────────────

#[allow(unused)]
fn clmul64_portable(a: u64, b: u64) -> u128 {
  let mut result: u128 = 0;
  let mut b_acc = b as u128;
  let mut a_rem = a;
  while a_rem != 0 {
    if a_rem & 1 != 0 {
      result ^= b_acc;
    }
    b_acc <<= 1;
    a_rem >>= 1;
  }
  result
}

// ─── From<u64> ────────────────────────────────────────────────────────────────

impl From<u64> for GF2_128 {
  #[inline]
  fn from(v: u64) -> Self {
    Self { lo: v, hi: 0 }
  }
}

// ─── FieldElem ────────────────────────────────────────────────────────────────

impl FieldElem for GF2_128 {
  #[inline]
  fn zero() -> Self {
    Self { lo: 0, hi: 0 }
  }
  #[inline]
  fn one() -> Self {
    Self { lo: 1, hi: 0 }
  }
  #[inline]
  fn is_zero(self) -> bool {
    self.lo == 0 && self.hi == 0
  }

  /// Inverse via Fermat little theorem: a⁻¹ = a^(2^128 − 2).
  fn inv(self) -> Self {
    assert!(!self.is_zero(), "GF2_128::inv called on zero");
    // Compute a^(2^128 - 2) via addition chain on squarings.
    // 2^128 - 2 = 2 * (2^127 - 1).
    // Build a^(2^k - 1) iteratively.
    let mut r = self; // a^(2^1 - 1) = a
    let mut b = self;
    for _ in 1..127 {
      b = b * b; // b = a^(2^k)
      r = r * b; // r = a^(2^(k+1) - 1)
    }
    // r = a^(2^127 - 1), square once: a^(2^128 - 2)
    r * r
  }
}

impl GF2_128 {
  /// Sample a uniformly random element.
  pub fn random<R: rand::Rng>(rng: &mut R) -> Self {
    Self {
      lo: rng.random::<u64>(),
      hi: rng.random::<u64>(),
    }
  }

  /// Sample a uniformly random non-zero element.
  pub fn random_nonzero<R: rand::Rng>(rng: &mut R) -> Self {
    loop {
      let v = Self::random(rng);
      if !v.is_zero() {
        return v;
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::traits::FieldElem;

  fn a() -> GF2_128 {
    GF2_128::new(0xDEAD_BEEF_0000_0001, 0x1234_5678_9ABC_DEF0)
  }
  fn b() -> GF2_128 {
    GF2_128::new(0xAAAA_BBBB_CCCC_DDDD, 0xFEDC_BA98_7654_3210)
  }
  fn c() -> GF2_128 {
    GF2_128::new(0x0102_0304_0506_0708, 0x0807_0605_0403_0201)
  }

  #[test]
  fn add_sub_roundtrip() {
    assert_eq!((a() + b()) - b(), a());
  }

  #[test]
  fn neg_is_self() {
    assert_eq!(-a(), a());
  }

  #[test]
  fn mul_zero() {
    assert_eq!(a() * GF2_128::zero(), GF2_128::zero());
  }

  #[test]
  fn mul_one() {
    assert_eq!(a() * GF2_128::one(), a());
  }

  #[test]
  fn mul_commutative() {
    assert_eq!(a() * b(), b() * a());
  }

  #[test]
  fn mul_associative() {
    assert_eq!((a() * b()) * c(), a() * (b() * c()));
  }

  #[test]
  fn mul_distributive() {
    assert_eq!(a() * (b() + c()), a() * b() + a() * c());
  }

  #[test]
  fn inv_roundtrip() {
    let x = a();
    assert_eq!(x * x.inv(), GF2_128::one());
  }

  #[test]
  fn from_u64_embed() {
    let v = 0xCAFE_BABEu64;
    assert_eq!(GF2_128::from(v), GF2_128::new(v, 0));
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
