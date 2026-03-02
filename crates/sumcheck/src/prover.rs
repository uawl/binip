//! Sumcheck prover for multilinear extension (MLE) polynomials.
//!
//! # Protocol (n-variable MLE)
//!
//! 1. Prover absorbs `claimed_sum = Σ f(x)` into transcript.
//! 2. For round i (fixing variable x_{n-1-i}):
//!    a. Computes g_i(t) = Σ_{rest} f(r_{n-1},...,r_{n-i}, t, x_{n-2-i},...,x_0)
//!    b. Sends g_i(0), g_i(1), g_i(α) — all three absorbed into transcript.
//!    c. Receives challenge r_{n-1-i} from transcript.
//!    d. Folds table: `table'[j] = table[j]*(1+r) + table[j+half]*r`
//! 3. `final_eval = table[0]` after all folds = f(r_{n-1},...,r_0).

use field::{FieldElem, GF2_128};
use poly::MlePoly;
use transcript::Transcript;

use crate::proof::{RoundPoly, SumcheckProof, alpha};

/// Prove that `Σ_{x ∈ {0,1}^n} poly(x) = poly.sum()`.
///
/// Consumes challenges from `transcript`. The same transcript must be used
/// by the verifier (initialised in the same state) to reproduce challenges.
pub fn prove<T: Transcript>(poly: &MlePoly, transcript: &mut T) -> SumcheckProof {
    let claimed_sum = poly.sum();
    transcript.absorb_field(claimed_sum);

    let mut table = poly.evals.clone();
    let mut round_polys = Vec::with_capacity(poly.n_vars as usize);
    let a = alpha();

    for _ in 0..poly.n_vars {
        let half = table.len() / 2;

        // g(0) = sum of lower half  (current variable = 0)
        let g0: GF2_128 = table[..half].iter().fold(GF2_128::zero(), |acc, &e| acc + e);
        // g(1) = sum of upper half  (current variable = 1)
        let g1: GF2_128 = table[half..].iter().fold(GF2_128::zero(), |acc, &e| acc + e);
        // g(α) = degree-1 extrapolation at α (redundant for MLE; stored for GKR extension)
        // g(t) = g(0)*(1+t) + g(1)*t  →  g(α) = g(0)*(1+α) + g(1)*α
        let g2: GF2_128 = g0 * (GF2_128::one() + a) + g1 * a;

        let rp = RoundPoly([g0, g1, g2]);
        transcript.absorb_field(rp.0[0]);
        transcript.absorb_field(rp.0[1]);
        transcript.absorb_field(rp.0[2]);
        round_polys.push(rp);

        let r = transcript.squeeze_challenge();

        // Fold: fix current variable to r
        // new[j] = old[j]*(1+r) + old[j+half]*r   (char 2: 1-r = 1+r)
        let one_plus_r = GF2_128::one() + r;
        table = (0..half)
            .map(|j| table[j] * one_plus_r + table[j + half] * r)
            .collect();
    }

    let final_eval = table[0];
    SumcheckProof { claimed_sum, round_polys, final_eval }
}
