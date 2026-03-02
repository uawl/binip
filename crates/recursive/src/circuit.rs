//! Recursive verification circuit builder.
//!
//! At each recursion level, a node aggregates `fan_in` child claims
//! by building a small MLE whose evaluations are the child claimed sums.
//! A sumcheck then proves the aggregate claim.
//!
//! # Aggregation MLE
//!
//! For `fan_in = k` children with claims `c_0, …, c_{k-1}`:
//!
//! 1. Pad to the next power of two `K = 2^m` with zeros.
//! 2. Build MLE `A(x_0, …, x_{m-1})` where `A(i) = c_i` for `i < k`, else 0.
//! 3. The aggregate claim is `Σ A(x) = c_0 + c_1 + … + c_{k-1}`.
//! 4. Sumcheck proves this sum, producing challenges and a final evaluation.
//!
//! The verifier reproduces the same transcript and checks consistency.

use field::{FieldElem, GF2_128};
use poly::MlePoly;

/// Build the aggregation MLE from a set of child claims.
///
/// The claims are padded to the next power of two with zeros,
/// then wrapped in an [`MlePoly`].
pub fn build_aggregation_mle(child_claims: &[GF2_128]) -> MlePoly {
  let k = child_claims.len();
  assert!(k > 0, "must have at least one child claim");

  // Pad to next power of two
  let padded_len = k.next_power_of_two();
  let mut evals = child_claims.to_vec();
  evals.resize(padded_len, GF2_128::zero());
  MlePoly::new(evals)
}

#[cfg(test)]
mod tests {
  use super::*;

  fn g(v: u64) -> GF2_128 {
    GF2_128::from(v)
  }

  #[test]
  fn aggregation_mle_correct_sum() {
    let claims = vec![g(3), g(5), g(7), g(11)];
    let expected_sum = g(3) + g(5) + g(7) + g(11);
    let mle = build_aggregation_mle(&claims);
    assert_eq!(mle.sum(), expected_sum);
    assert_eq!(mle.n_vars, 2); // 4 claims → 2^2
  }

  #[test]
  fn aggregation_mle_pads_to_power_of_two() {
    let claims = vec![g(1), g(2), g(3)]; // 3 → padded to 4
    let mle = build_aggregation_mle(&claims);
    assert_eq!(mle.evals.len(), 4);
    assert_eq!(mle.n_vars, 2);
    // Padded element is zero
    assert!(mle.evals[3].is_zero());
    // Sum should be sum of actual claims only
    assert_eq!(mle.sum(), g(1) + g(2) + g(3));
  }

  #[test]
  fn aggregation_mle_single_claim() {
    let claims = vec![g(42)];
    let mle = build_aggregation_mle(&claims);
    assert_eq!(mle.n_vars, 0);
    assert_eq!(mle.sum(), g(42));
  }

  #[test]
  fn aggregation_mle_power_of_two_claims() {
    let claims: Vec<GF2_128> = (1u64..=8).map(g).collect();
    let mle = build_aggregation_mle(&claims);
    assert_eq!(mle.evals.len(), 8);
    assert_eq!(mle.n_vars, 3);
  }
}
