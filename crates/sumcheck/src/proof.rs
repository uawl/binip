//! Sumcheck proof types.

use field::{FieldElem, GF2_128};

/// Evaluation point α used as the third evaluation point in RoundPoly.
/// α = GF2_128::from(2) = the field element `x` in GF(2^64), embedded in GF(2^128).
/// This is nonzero and ≠ 1, making {0, 1, α} a valid set of 3 distinct points.
pub fn alpha() -> GF2_128 {
    GF2_128::from(2)
}

/// Round polynomial represented as evaluations at three distinct field points: 0, 1, α.
///
/// For degree-1 (MLE sumcheck): g[2] = g(0)*(1+α) + g(1)*α (extrapolated).
/// For degree-2 (GKR): all three values are independently computed by the prover.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoundPoly(pub [GF2_128; 3]);

impl RoundPoly {
    /// Evaluate via degree-1 Lagrange interpolation at points {0, 1}.
    ///
    /// `g(r) = g(0)*(1+r) + g(1)*r`
    ///
    /// Valid when the underlying polynomial is at most degree 1 in this variable,
    /// as is always the case for multilinear (MLE) sumcheck.
    pub fn eval(&self, r: GF2_128) -> GF2_128 {
        let one_plus_r = GF2_128::one() + r; // char 2: 1 - r == 1 + r
        self.0[0] * one_plus_r + self.0[1] * r
    }

    /// Evaluate via full 3-point Lagrange interpolation at {0, 1, α}.
    ///
    /// Use this for degree-2 round polynomials (e.g., GKR-style sumcheck).
    pub fn eval_degree2(&self, r: GF2_128) -> GF2_128 {
        let a = alpha();
        let g0 = self.0[0];
        let g1 = self.0[1];
        let ga = self.0[2];

        // Lagrange basis polynomials at {0, 1, α}:
        //   L₀(r) = (r+1)(r+α) / α
        //   L₁(r) = r(r+α) / (1+α)
        //   L₂(r) = r(r+1) / (α(α+1))
        // (all -1 = +1 in characteristic 2)
        let inv_a = a.inv();
        let inv_1pa = (GF2_128::one() + a).inv();
        let inv_apa1 = (a * (a + GF2_128::one())).inv();

        let l0 = (r + GF2_128::one()) * (r + a) * inv_a;
        let l1 = r * (r + a) * inv_1pa;
        let l2 = r * (r + GF2_128::one()) * inv_apa1;

        g0 * l0 + g1 * l1 + ga * l2
    }
}

/// Complete sumcheck proof for an MLE polynomial.
#[derive(Debug, Clone)]
pub struct SumcheckProof {
    /// Σ_{x ∈ {0,1}^n} f(x) — the value the prover claims.
    pub claimed_sum: GF2_128,
    /// Round polynomials g_0, g_1, …, g_{n-1} sent in order.
    /// Round 0 fixes the highest-index variable x_{n-1}.
    pub round_polys: Vec<RoundPoly>,
    /// f(r_{n-1}, …, r_0) after all rounds folded — the final oracle claim.
    pub final_eval: GF2_128,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(v: u64) -> GF2_128 {
        GF2_128::from(v)
    }

    #[test]
    fn eval_at_zero_returns_g0() {
        let rp = RoundPoly([f(5), f(9), f(13)]);
        assert_eq!(rp.eval(GF2_128::zero()), f(5));
    }

    #[test]
    fn eval_at_one_returns_g1() {
        let rp = RoundPoly([f(5), f(9), f(13)]);
        assert_eq!(rp.eval(GF2_128::one()), f(9));
    }

    #[test]
    fn eval_degree2_at_zero_returns_g0() {
        let rp = RoundPoly([f(3), f(7), f(11)]);
        assert_eq!(rp.eval_degree2(GF2_128::zero()), f(3));
    }

    #[test]
    fn eval_degree2_at_one_returns_g1() {
        let rp = RoundPoly([f(3), f(7), f(11)]);
        assert_eq!(rp.eval_degree2(GF2_128::one()), f(7));
    }

    #[test]
    fn eval_degree2_at_alpha_returns_g2() {
        let a = alpha();
        let rp = RoundPoly([f(3), f(7), f(11)]);
        assert_eq!(rp.eval_degree2(a), f(11));
    }

    /// For a degree-1 round polynomial, degree-1 and degree-2 eval must agree.
    #[test]
    fn degree1_and_degree2_agree_for_linear() {
        // Make g quadratic-free: set g[2] = the degree-1 extrapolation
        let g0 = f(13);
        let g1 = f(7);
        let a = alpha();
        // degree-1 extrapolation at α:
        let g2 = g0 * (GF2_128::one() + a) + g1 * a;
        let rp = RoundPoly([g0, g1, g2]);

        let r = f(42);
        assert_eq!(rp.eval(r), rp.eval_degree2(r));
    }
}
