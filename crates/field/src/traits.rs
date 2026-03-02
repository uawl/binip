//! Core algebraic traits for finite fields.
//!
//! Every field element must implement `FieldElem`.  The associated type
//! `F` is the field itself, so that `pow`, `inv`, etc. return the same type
//! without boxing.

use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Marker + arithmetic trait for a finite-field element.
///
/// All operations are **pure** (take values or references, never mutate in
/// place via `self`).  The `AddAssign` / `MulAssign` blanket constraints
/// let callers accumulate into a slot efficiently.
pub trait FieldElem:
  Sized
  + Copy
  + Clone
  + PartialEq
  + Eq
  + std::fmt::Debug
  + Add<Output = Self>
  + Sub<Output = Self>
  + Neg<Output = Self>
  + Mul<Output = Self>
  + AddAssign
  + SubAssign
  + MulAssign
  + From<u64>
{
  /// Additive identity.
  fn zero() -> Self;

  /// Multiplicative identity.
  fn one() -> Self;

  /// Return `true` iff `self == zero()`.
  fn is_zero(self) -> bool;

  /// Multiplicative inverse.  Panics if `self.is_zero()`.
  fn inv(self) -> Self;

  /// Repeated squaring.
  fn pow(self, mut exp: u64) -> Self {
    let mut base = self;
    let mut result = Self::one();
    while exp > 0 {
      if exp & 1 == 1 {
        result *= base;
      }
      base *= base;
      exp >>= 1;
    }
    result
  }

  /// Division.  Panics if `rhs.is_zero()`.
  fn div(self, rhs: Self) -> Self {
    self * rhs.inv()
  }
}
