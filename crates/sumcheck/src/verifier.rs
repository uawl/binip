//! Sumcheck verifier.
//!
//! The verifier:
//! 1. Absorbs `claimed_sum` from the proof into the transcript.
//! 2. For each round poly g_i:
//!    a. Checks `g_i(0) + g_i(1) == prev_claim`.
//!    b. Absorbs g_i evaluations into transcript.
//!    c. Squeezes challenge r_i and computes new `prev_claim = g_i(r_i)`.
//! 3. Checks `prev_claim == oracle_eval` (the externally-supplied f(r₀,...) value).
//!
//! The caller provides `oracle_eval` — in production this comes from a PCS opening;
//! in tests it can be computed directly via `MlePoly::evaluate`.
//!
//! Returns the vector of challenges so the caller can open the polynomial
//! commitment at that point.

use field::GF2_128;
use transcript::Transcript;

use crate::proof::SumcheckProof;

/// Verify a sumcheck proof.
///
/// * `proof`       – proof produced by the prover
/// * `oracle_eval` – the claimed evaluation f(challenges) provided by the oracle
/// * `transcript`  – must have the same initial state as the prover's transcript
///
/// Returns `Some(challenges)` on success, `None` on failure.
/// `challenges[0]` is the challenge for the first round (highest variable index).
pub fn verify<T: Transcript>(
    proof: &SumcheckProof,
    oracle_eval: GF2_128,
    transcript: &mut T,
) -> Option<Vec<GF2_128>> {
    transcript.absorb_field(proof.claimed_sum);

    let mut prev_claim = proof.claimed_sum;
    let mut challenges = Vec::with_capacity(proof.round_polys.len());

    for rp in &proof.round_polys {
        // Consistency check: g_i(0) + g_i(1) == prev_claim
        if rp.0[0] + rp.0[1] != prev_claim {
            return None;
        }

        transcript.absorb_field(rp.0[0]);
        transcript.absorb_field(rp.0[1]);
        transcript.absorb_field(rp.0[2]);

        let r = transcript.squeeze_challenge();
        challenges.push(r);

        // New claim: g_i(r)  — degree-1 interpolation for MLE variables
        prev_claim = rp.eval(r);
    }

    // Final oracle check
    if prev_claim != oracle_eval {
        return None;
    }

    Some(challenges)
}
