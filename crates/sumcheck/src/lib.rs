pub mod proof;
pub mod prover;
pub mod verifier;
pub mod gpu_prover;

pub use proof::{RoundPoly, SumcheckProof};
pub use prover::prove;
pub use verifier::verify;
pub use gpu_prover::prove_gpu;

#[cfg(test)]
mod tests {
    use field::{FieldElem, GF2_128};
    use poly::MlePoly;
    use transcript::VisionTranscript;

    use super::*;

    fn g(v: u64) -> GF2_128 {
        GF2_128::from(v)
    }

    fn fresh_transcript() -> VisionTranscript {
        VisionTranscript::new()
    }

    // ─── Helper: prove + verify roundtrip ────────────────────────────────────

    fn roundtrip(poly: &MlePoly) -> bool {
        let mut t_prover = fresh_transcript();
        let proof = prove(poly, &mut t_prover);

        // Derive oracle eval from the MLE at the same challenges the verifier will use.
        // Since both prover and verifier produce the same challenges, we replay
        // the verifier logic manually to get the challenge point.
        let mut t_verifier = fresh_transcript();

        // Compute oracle: evaluate poly at the challenge point.
        // We do a dry-run of the verifier to get challenges, then re-verify.
        let n = poly.n_vars as usize;

        // First pass: collect challenges
        let mut tc_tmp = fresh_transcript();
        if let Some(challenges) = verify(&proof, proof.final_eval, &mut tc_tmp) {
            assert_eq!(challenges.len(), n);

            // Oracle evaluation at derived challenge point
            let oracle = poly.evaluate(&challenges);

            // Real verification
            let result = verify(&proof, oracle, &mut t_verifier);
            result.is_some()
        } else {
            false
        }
    }

    // ─── Tests ───────────────────────────────────────────────────────────────

    #[test]
    fn test_1var_poly() {
        // f(0) = 3, f(1) = 7.  sum = 3 XOR 7 = 4 in GF(2^128)
        let poly = MlePoly::new(vec![g(3), g(7)]);
        assert_eq!(poly.sum(), g(3) + g(7));
        assert!(roundtrip(&poly));
    }

    #[test]
    fn test_2var_poly() {
        let evals = vec![g(1), g(2), g(4), g(8)];
        let poly = MlePoly::new(evals);
        assert!(roundtrip(&poly));
    }

    #[test]
    fn test_4var_poly() {
        let evals: Vec<GF2_128> = (1u64..=16).map(g).collect();
        let poly = MlePoly::new(evals);
        assert!(roundtrip(&poly));
    }

    #[test]
    fn test_8var_poly() {
        let evals: Vec<GF2_128> = (1u64..=256).map(g).collect();
        let poly = MlePoly::new(evals);
        assert!(roundtrip(&poly));
    }

    #[test]
    fn test_tampered_claimed_sum_fails() {
        let poly = MlePoly::new(vec![g(3), g(7), g(2), g(9)]);
        let mut t_prover = fresh_transcript();
        let mut proof = prove(&poly, &mut t_prover);

        // Tamper: flip the claimed sum
        proof.claimed_sum = proof.claimed_sum + g(1);

        let mut t_verifier = fresh_transcript();
        assert!(verify(&proof, proof.final_eval, &mut t_verifier).is_none());
    }

    #[test]
    fn test_tampered_round_poly_fails() {
        let poly = MlePoly::new((1u64..=8).map(g).collect());
        let mut t_prover = fresh_transcript();
        let mut proof = prove(&poly, &mut t_prover);

        // Tamper: modify g(1) of the first round poly
        proof.round_polys[0].0[1] = proof.round_polys[0].0[1] + g(1);

        let mut t_verifier = fresh_transcript();
        // Pass the original final_eval — verification must fail because consistency breaks
        assert!(verify(&proof, proof.final_eval, &mut t_verifier).is_none());
    }

    #[test]
    fn test_zero_poly_sum_is_zero() {
        let poly = MlePoly::zero(4);
        assert_eq!(poly.sum(), GF2_128::zero());
        assert!(roundtrip(&poly));
    }

    #[test]
    fn test_proved_final_eval_matches_direct_evaluate() {
        let evals: Vec<GF2_128> = (10u64..26).map(g).collect();
        let poly = MlePoly::new(evals);

        let mut t_prover = fresh_transcript();
        let proof = prove(&poly, &mut t_prover);

        // Use a fresh transcript to reproduce challenges
        let mut tc = fresh_transcript();
        let challenges = verify(&proof, proof.final_eval, &mut tc).unwrap();

        let direct = poly.evaluate(&challenges);
        assert_eq!(proof.final_eval, direct);
    }

    #[test]
    fn test_correct_oracle_verification_succeeds() {
        let evals: Vec<GF2_128> = (5u64..21).map(g).collect();
        let poly = MlePoly::new(evals);

        let mut t_prover = fresh_transcript();
        let proof = prove(&poly, &mut t_prover);

        // Reproduce challenges
        let mut tc = fresh_transcript();
        let challenges = verify(&proof, proof.final_eval, &mut tc).unwrap();
        let oracle = poly.evaluate(&challenges);

        let mut t_verifier = fresh_transcript();
        assert!(verify(&proof, oracle, &mut t_verifier).is_some());
    }

    #[test]
    fn test_wrong_oracle_fails() {
        let evals: Vec<GF2_128> = (5u64..21).map(g).collect();
        let poly = MlePoly::new(evals.clone());

        let mut t_prover = fresh_transcript();
        let proof = prove(&poly, &mut t_prover);

        let mut t_verifier = fresh_transcript();
        // Pass a wrong oracle value
        let wrong_oracle = g(999999);
        assert!(verify(&proof, wrong_oracle, &mut t_verifier).is_none());
  }
}

#[cfg(test)]
mod gpu_tests {
    use field::{GF2_128};
    use gpu::{GpuContext, PipelineCache};
    use poly::MlePoly;
    use transcript::VisionTranscript;

    use super::*;

    fn g(v: u64) -> GF2_128 { GF2_128::from(v) }

    fn gpu_roundtrip(poly: &MlePoly) -> bool {
        let ctx = GpuContext::new().expect("GPU init");
        let mut cache = PipelineCache::new();

        // Prover transcript
        let mut t_prover = VisionTranscript::new();
        let proof = prove_gpu(poly, &mut t_prover, &ctx, &mut cache);

        // Reproduce challenges with fresh transcript, then verify
        let mut tc = VisionTranscript::new();
        let Some(challenges) = verify(&proof, proof.final_eval, &mut tc) else {
            return false;
        };
        let oracle = poly.evaluate(&challenges);

        let mut t_verifier = VisionTranscript::new();
        verify(&proof, oracle, &mut t_verifier).is_some()
    }

    #[test]
    fn test_gpu_1var() {
        let poly = MlePoly::new(vec![g(3), g(7)]);
        assert!(gpu_roundtrip(&poly));
    }

    #[test]
    fn test_gpu_4var() {
        let evals: Vec<GF2_128> = (1u64..=16).map(g).collect();
        let poly = MlePoly::new(evals);
        assert!(gpu_roundtrip(&poly));
    }

    #[test]
    fn test_gpu_8var() {
        let evals: Vec<GF2_128> = (1u64..=256).map(g).collect();
        let poly = MlePoly::new(evals);
        assert!(gpu_roundtrip(&poly));
    }

    /// GPU 프루버와 CPU 프루버의 round_polys 가 동일한지 교차 검증
    #[test]
    fn test_gpu_cpu_round_polys_match() {
        let evals: Vec<GF2_128> = (10u64..26).map(g).collect();
        let poly = MlePoly::new(evals);

        let mut t_cpu = VisionTranscript::new();
        let cpu_proof = prove(&poly, &mut t_cpu);

        let ctx = GpuContext::new().expect("GPU init");
        let mut cache = PipelineCache::new();
        let mut t_gpu = VisionTranscript::new();
        let gpu_proof = prove_gpu(&poly, &mut t_gpu, &ctx, &mut cache);

        assert_eq!(cpu_proof.claimed_sum, gpu_proof.claimed_sum);
        assert_eq!(cpu_proof.round_polys.len(), gpu_proof.round_polys.len());
        for (cpu_rp, gpu_rp) in cpu_proof.round_polys.iter().zip(&gpu_proof.round_polys) {
            assert_eq!(cpu_rp, gpu_rp, "round poly mismatch");
        }
        assert_eq!(cpu_proof.final_eval, gpu_proof.final_eval);
    }
}
