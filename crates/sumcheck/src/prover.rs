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
use rayon::prelude::*;
use transcript::Transcript;

use crate::proof::{RoundPoly, SumcheckProof, alpha};

/// Prove that `Σ_{x ∈ {0,1}^n} poly(x) = poly.sum()`.
///
/// Takes ownership of `poly` to fold in-place without cloning.
/// The same transcript must be used by the verifier (initialised in the
/// same state) to reproduce challenges.
pub fn prove<T: Transcript>(poly: MlePoly, transcript: &mut T) -> SumcheckProof {
  let claimed_sum = poly.sum();
  transcript.absorb_field(claimed_sum);

  let mut table = poly.evals;
  let mut round_polys = Vec::with_capacity(poly.n_vars as usize);
  let a = alpha();
  let mut prev_claim = claimed_sum;

  for _ in 0..poly.n_vars {
    let half = table.len() / 2;

    // g(0) = sum of lower half  (current variable = 0)
    let g0: GF2_128 = table[..half]
      .iter()
      .fold(GF2_128::zero(), |acc, &e| acc + e);
    // g(1) = prev_claim + g0  (char 2: g0 + g1 = prev_claim)
    let g1 = prev_claim + g0;
    // g(α) = degree-1 extrapolation at α
    let g2: GF2_128 = g0 * (GF2_128::one() + a) + g1 * a;

    let rp = RoundPoly([g0, g1, g2]);
    transcript.absorb_field(rp.0[0]);
    transcript.absorb_field(rp.0[1]);
    transcript.absorb_field(rp.0[2]);
    round_polys.push(rp);

    let r = transcript.squeeze_challenge();
    prev_claim = rp.eval(r);

    // Fold in-place: fix current variable to r
    // new[j] = old[j]*(1+r) + old[j+half]*r   (char 2: 1-r = 1+r)
    let one_plus_r = GF2_128::one() + r;
    for j in 0..half {
      table[j] = table[j] * one_plus_r + table[j + half] * r;
    }
    table.truncate(half);
  }

  let final_eval = table[0];
  SumcheckProof {
    claimed_sum,
    round_polys,
    final_eval,
  }
}

/// Minimum table half-size for parallel execution.
const PAR_THRESHOLD: usize = 4096;

/// Parallel variant of [`prove`] — uses rayon for large evaluation tables.
///
/// For tables with ≥ 2 × `PAR_THRESHOLD` elements, the g0 summation and
/// fold loop are parallelised via rayon.  Falls back to sequential for
/// smaller tables (where thread overhead dominates).
pub fn prove_par<T: Transcript>(poly: MlePoly, transcript: &mut T) -> SumcheckProof {
  let claimed_sum = poly.sum();
  transcript.absorb_field(claimed_sum);

  let mut table = poly.evals;
  let mut round_polys = Vec::with_capacity(poly.n_vars as usize);
  let a = alpha();
  let mut prev_claim = claimed_sum;

  for _ in 0..poly.n_vars {
    let half = table.len() / 2;

    // g(0) = sum of lower half
    let g0: GF2_128 = if half >= PAR_THRESHOLD {
      table[..half]
        .par_iter()
        .copied()
        .reduce(|| GF2_128::zero(), |a, b| a + b)
    } else {
      table[..half]
        .iter()
        .fold(GF2_128::zero(), |acc, &e| acc + e)
    };
    // g(1) derived from sumcheck invariant
    let g1 = prev_claim + g0;
    let g2: GF2_128 = g0 * (GF2_128::one() + a) + g1 * a;

    let rp = RoundPoly([g0, g1, g2]);
    transcript.absorb_field(rp.0[0]);
    transcript.absorb_field(rp.0[1]);
    transcript.absorb_field(rp.0[2]);
    round_polys.push(rp);

    let r = transcript.squeeze_challenge();
    prev_claim = rp.eval(r);

    // Fold in-place
    let one_plus_r = GF2_128::one() + r;
    if half >= PAR_THRESHOLD {
      let (lo, hi) = table.split_at_mut(half);
      lo.par_iter_mut()
        .zip(hi.par_iter())
        .for_each(|(l, &h)| {
          *l = *l * one_plus_r + h * r;
        });
    } else {
      for j in 0..half {
        table[j] = table[j] * one_plus_r + table[j + half] * r;
      }
    }
    table.truncate(half);
  }

  let final_eval = table[0];
  SumcheckProof {
    claimed_sum,
    round_polys,
    final_eval,
  }
}
