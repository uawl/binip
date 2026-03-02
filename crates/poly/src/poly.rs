//! Multilinear Extension (MLE) polynomial over GF(2^128).
//!
//! An MLE with `n_vars` variables `x_0, …, x_{n-1}` is uniquely determined by
//! its 2^n evaluations on the Boolean hypercube {0,1}^n stored in `evals`.
//! Index encoding: `i = x_{n-1} << (n-1) | … | x_0`.
//!
//! ## Core operation: fold
//! Fixing the last variable `x_{n-1} = r` halves the evaluation table:
//! ```text
//! evals'[i] = evals[i] * (1 + r) + evals[i + half] * r   (GF(2^128) arith)
//! ```
//! (`1 + r` = `1 XOR r` in GF(2), but `GF2_128` uses `+` / `*` generically.)
//! Repeated folding at r_0, …, r_{n-1} yields a single scalar = MLE(r_0,…,r_{n-1}).

use field::{FieldElem, GF2_128};
use rand::Rng;
use rayon::prelude::*;

/// Multilinear Extension polynomial — evaluation table over {0,1}^n.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlePoly {
  /// Evaluation values; length must be a power of two.
  pub evals: Vec<GF2_128>,
  /// Number of variables: `evals.len() == 1 << n_vars`.
  pub n_vars: u32,
}

impl MlePoly {
  /// Construct from a flat evaluation table.
  ///
  /// Panics if `evals.len()` is not a power of two.
  pub fn new(evals: Vec<GF2_128>) -> Self {
    let len = evals.len();
    assert!(
      len.is_power_of_two(),
      "MlePoly: evals.len() must be a power of two, got {len}"
    );
    let n_vars = len.trailing_zeros();
    Self { evals, n_vars }
  }

  /// Construct the zero polynomial with `n_vars` variables.
  pub fn zero(n_vars: u32) -> Self {
    Self {
      evals: vec![GF2_128::zero(); 1 << n_vars],
      n_vars,
    }
  }

  /// Evaluate the MLE at a full point `r ∈ GF(2^128)^n` using CPU fold.
  ///
  /// Panics if `r.len() != n_vars`.
  pub fn evaluate(&self, r: &[GF2_128]) -> GF2_128 {
    assert_eq!(
      r.len() as u32,
      self.n_vars,
      "MlePoly::evaluate: wrong number of vars"
    );
    let mut table = self.evals.clone();
    let mut half = table.len() >> 1;
    for &ri in r {
      let one_minus_r = GF2_128::one() + ri; // 1 XOR r in GF(2^128)
      for lo in 0..half {
        let hi = lo + half;
        table[lo] = table[lo] * one_minus_r + table[hi] * ri;
      }
      table.truncate(half);
      half >>= 1;
    }
    table[0]
  }

  /// Fold the last variable: fix `x_{n-1} = r`, returning an (n-1)-var MLE.
  pub fn fold_last(&self, r: GF2_128) -> Self {
    assert!(self.n_vars > 0, "cannot fold a constant polynomial");
    let half = self.evals.len() >> 1;
    let one_minus_r = GF2_128::one() + r;
    let new_evals: Vec<GF2_128> = (0..half)
      .map(|i| self.evals[i] * one_minus_r + self.evals[i + half] * r)
      .collect();
    Self {
      evals: new_evals,
      n_vars: self.n_vars - 1,
    }
  }

  /// Sum all evaluations: Σ evals[i]. Equals MLE summed over {0,1}^n.
  pub fn sum(&self) -> GF2_128 {
    self.evals.iter().fold(GF2_128::zero(), |acc, &e| acc + e)
  }

  /// Construct the equality polynomial `eq(r, ·)` over {0,1}^n.
  ///
  /// `eq(r, x) = Π_i (r_i * x_i + (1-r_i)*(1-x_i))`
  ///
  /// Evaluated at all 2^n Boolean points, stored in the natural index order.
  pub fn eq_poly(r: &[GF2_128]) -> Self {
    let n = r.len();
    let size = 1usize << n;
    let mut evals = vec![GF2_128::one(); size];
    // Build via tensor product: for each variable i, split and scale.
    let mut stride = size >> 1;
    for &ri in r {
      let one_minus_ri = GF2_128::one() + ri;
      let mut i = 0;
      while i < size {
        for j in i..i + stride {
          let lo = evals[j];
          evals[j] = lo * one_minus_ri; // x_i = 0
          evals[j + stride] = lo * ri; // x_i = 1
        }
        i += stride * 2;
      }
      stride >>= 1;
    }
    Self {
      evals,
      n_vars: n as u32,
    }
  }

  /// Pointwise add two MLEs (same n_vars).
  pub fn add_poly(&self, other: &Self) -> Self {
    assert_eq!(self.n_vars, other.n_vars);
    let evals = self
      .evals
      .iter()
      .zip(&other.evals)
      .map(|(&a, &b)| a + b)
      .collect();
    Self {
      evals,
      n_vars: self.n_vars,
    }
  }

  /// Pointwise multiply two MLEs (product-of-MLEs, NOT standard composition).
  pub fn mul_poly(&self, other: &Self) -> Self {
    assert_eq!(self.n_vars, other.n_vars);
    let evals = self
      .evals
      .iter()
      .zip(&other.evals)
      .map(|(&a, &b)| a * b)
      .collect();
    Self {
      evals,
      n_vars: self.n_vars,
    }
  }

  /// Generate a random multilinear polynomial whose sum over {0,1}^n is zero.
  ///
  /// Construction: sample `2^n - 1` random evaluations, then set the last
  /// evaluation so that the XOR-sum of all evaluations is zero.
  /// This gives a uniformly random MLE conditioned on `Σ evals = 0`.
  pub fn random_zero_sum<R: Rng>(n_vars: u32, rng: &mut R) -> Self {
    let size = 1usize << n_vars;
    let mut evals = Vec::with_capacity(size);
    let mut running_sum = GF2_128::zero();
    for _ in 0..size - 1 {
      let v = GF2_128::random(rng);
      running_sum = running_sum + v;
      evals.push(v);
    }
    // Last element cancels the sum: in char 2, a + a = 0.
    evals.push(running_sum);
    Self { evals, n_vars }
  }

  /// Return a blinded copy: `self + r * B` where `B` is a random zero-sum MLE.
  ///
  /// The blinded polynomial has the same sum over {0,1}^n as the original
  /// (since `Σ B(x) = 0`), but individual evaluations are masked, providing
  /// zero-knowledge in the sumcheck protocol.
  pub fn blind<R: Rng>(&self, rng: &mut R) -> Self {
    let r = GF2_128::random_nonzero(rng);
    let mask = Self::random_zero_sum(self.n_vars, rng);
    let evals = self
      .evals
      .iter()
      .zip(&mask.evals)
      .map(|(&w, &b)| w + r * b)
      .collect();
    Self {
      evals,
      n_vars: self.n_vars,
    }
  }

  /// Parallel variant of [`blind`] — the pointwise masking runs via rayon.
  ///
  /// The random mask generation stays sequential (RNG is inherently serial),
  /// but the expensive `n × 2 field-mul + add` step is parallelised.
  pub fn blind_par<R: Rng>(&self, rng: &mut R) -> Self {
    let r = GF2_128::random_nonzero(rng);
    let mask = Self::random_zero_sum(self.n_vars, rng);
    let evals: Vec<GF2_128> = self
      .evals
      .par_iter()
      .zip(mask.evals.par_iter())
      .map(|(&w, &b)| w + r * b)
      .collect();
    Self {
      evals,
      n_vars: self.n_vars,
    }
  }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use super::*;

  fn g(n: u64) -> GF2_128 {
    GF2_128::from(n)
  }

  // Evaluate at each Boolean corner must return the stored eval.
  // For a 2-var poly, evaluate(&[r0, r1]) fixes x_1=r0 then x_0=r1.
  // So evaluate(&[0,0])=evals[0], (&[0,1])=evals[1], (&[1,0])=evals[2], (&[1,1])=evals[3].
  #[test]
  fn evaluate_at_boolean_hypercube() {
    let e: Vec<GF2_128> = (1u64..=4).map(g).collect();
    let poly = MlePoly::new(e.clone());
    let zero = GF2_128::zero();
    let one = GF2_128::one();
    assert_eq!(poly.evaluate(&[zero, zero]), e[0]);
    assert_eq!(poly.evaluate(&[zero, one]), e[1]);
    assert_eq!(poly.evaluate(&[one, zero]), e[2]);
    assert_eq!(poly.evaluate(&[one, one]), e[3]);
  }

  // fold_last(r) then evaluate(r_rest) == evaluate(&[r, r_rest...]).
  #[test]
  fn fold_consistency() {
    let e: Vec<GF2_128> = (10u64..18).map(g).collect(); // 8 evals, 3-var
    let poly = MlePoly::new(e);
    let r = [g(7), g(13), g(19)]; // 3 random-ish scalars
    let direct = poly.evaluate(&r);
    let folded = poly.fold_last(r[0]);
    let via_fold = folded.evaluate(&r[1..]);
    assert_eq!(direct, via_fold);
  }

  // sum of eq_poly(r).evals == 1.
  #[test]
  fn eq_poly_sum_is_one() {
    let r = [g(3), g(5), g(7)];
    let eq = MlePoly::eq_poly(&r);
    assert_eq!(eq.sum(), GF2_128::one());
  }

  // eq_poly(r).evaluate(r) == 1.
  #[test]
  fn eq_poly_eval_at_r_is_one() {
    let r = [g(11), g(13)];
    let eq = MlePoly::eq_poly(&r);
    assert_eq!(eq.evaluate(&r), GF2_128::one());
  }

  // add_poly and mul_poly are pointwise.
  #[test]
  fn add_mul_poly_pointwise() {
    let a: Vec<GF2_128> = (1u64..=4).map(g).collect();
    let b: Vec<GF2_128> = (5u64..=8).map(g).collect();
    let pa = MlePoly::new(a.clone());
    let pb = MlePoly::new(b.clone());
    let sum = pa.add_poly(&pb);
    let prod = pa.mul_poly(&pb);
    for i in 0..4 {
      assert_eq!(sum.evals[i], a[i] + b[i]);
      assert_eq!(prod.evals[i], a[i] * b[i]);
    }
  }

  // ─── Blinding tests ──────────────────────────────────────────────────

  #[test]
  fn random_zero_sum_has_zero_sum() {
    let mut rng = rand::rng();
    for n_vars in 1..=6 {
      let b = MlePoly::random_zero_sum(n_vars, &mut rng);
      assert_eq!(b.evals.len(), 1 << n_vars);
      assert_eq!(b.n_vars, n_vars);
      assert!(
        b.sum().is_zero(),
        "random_zero_sum n_vars={n_vars} should have sum=0"
      );
    }
  }

  #[test]
  fn random_zero_sum_is_nontrivial() {
    let mut rng = rand::rng();
    let b = MlePoly::random_zero_sum(4, &mut rng);
    // With 16 evaluations over GF(2^128), all being zero is negligible.
    assert!(b.evals.iter().any(|e| !e.is_zero()));
  }

  #[test]
  fn blind_preserves_sum() {
    let mut rng = rand::rng();
    let evals: Vec<GF2_128> = (1u64..=16).map(g).collect();
    let poly = MlePoly::new(evals);
    let original_sum = poly.sum();
    for _ in 0..5 {
      let blinded = poly.blind(&mut rng);
      assert_eq!(blinded.sum(), original_sum, "blinding must preserve sum");
    }
  }

  #[test]
  fn blind_changes_evaluations() {
    let mut rng = rand::rng();
    let evals: Vec<GF2_128> = (1u64..=8).map(g).collect();
    let poly = MlePoly::new(evals);
    let blinded = poly.blind(&mut rng);
    // At least one evaluation should differ (overwhelmingly likely).
    assert_ne!(poly.evals, blinded.evals);
  }

  #[test]
  fn blind_zero_poly_produces_zero_sum() {
    let mut rng = rand::rng();
    let poly = MlePoly::zero(4);
    let blinded = poly.blind(&mut rng);
    assert!(
      blinded.sum().is_zero(),
      "blinded zero-poly should still sum to 0"
    );
    // But evaluations should be nonzero (the blinding mask).
    assert!(blinded.evals.iter().any(|e| !e.is_zero()));
  }

  #[test]
  fn two_blindings_differ() {
    let mut rng = rand::rng();
    let evals: Vec<GF2_128> = (1u64..=4).map(g).collect();
    let poly = MlePoly::new(evals);
    let b1 = poly.blind(&mut rng);
    let b2 = poly.blind(&mut rng);
    assert_ne!(b1.evals, b2.evals, "independent blindings should differ");
  }
}
