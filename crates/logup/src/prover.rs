//! LogUp prover — γ-weighted protocol for GF(2^128).
//!
//! # Protocol steps
//!
//! 1. Absorb witness and table values into the transcript.
//! 2. Squeeze challenges `β` and `γ`.
//! 3. Compute γ-weighted fractional MLEs `h_w` and `h_t`.
//! 4. Run sumcheck on `h_w` to prove `Σ h_w(x) = S`.
//! 5. Run sumcheck on `h_t` to prove `Σ h_t(x) = S`.
//! 6. Verifier checks `S_w = S_t`.

use field::{FieldElem, GF2_128};
use poly::MlePoly;
use transcript::{Blake3Transcript, Transcript};

use crate::proof::{LogUpProof, PcsBinding};
use crate::table::{LookupTable, table_fractional_evals, witness_fractional_evals};

/// Prove a single LogUp lookup relation.
///
/// * `witness`   – witness values (length must be a power of two)
/// * `table`     – lookup table
/// * `transcript`– Fiat-Shamir transcript (shared with the verifier)
pub fn prove(
  witness: &[GF2_128],
  table: &LookupTable,
  transcript: &mut Blake3Transcript,
) -> LogUpProof {
  let n_witness = witness.len();
  assert!(
    n_witness.is_power_of_two(),
    "witness length must be a power of two"
  );

  // ── 1. Absorb sizes + raw values ─────────────────────────────────────
  transcript.absorb_bytes(&(n_witness as u64).to_le_bytes());
  transcript.absorb_bytes(&(table.entries.len() as u64).to_le_bytes());

  transcript.absorb_fields(witness);
  // Absorb table content hash instead of all entries (deterministic table).
  transcript.absorb_bytes(&table.content_hash);

  // ── 2. Squeeze β and γ ────────────────────────────────────────────────
  let beta = transcript.squeeze_challenge();
  let gamma = transcript.squeeze_challenge();

  // ── 3. Compute γ-weighted fractional MLEs ─────────────────────────────
  let h_w = MlePoly::new(witness_fractional_evals(witness, beta, gamma));
  let h_t = MlePoly::new(table_fractional_evals(table, witness, beta, gamma));

  let witness_sum = h_w.sum();
  let table_sum = h_t.sum();
  debug_assert_eq!(
    witness_sum, table_sum,
    "LogUp prover: witness sum ≠ table sum"
  );
  let claimed_sum = witness_sum;

  transcript.absorb_field(claimed_sum);

  // ── 4. Witness-side sumcheck ──────────────────────────────────────────
  transcript.absorb_bytes(b"logup:witness_sumcheck");
  let witness_proof = sumcheck::prove_par(h_w, transcript);

  // ── 5. Table-side sumcheck ────────────────────────────────────────────
  transcript.absorb_bytes(b"logup:table_sumcheck");
  let table_proof = sumcheck::prove_par(h_t, transcript);

  LogUpProof {
    beta,
    gamma,
    claimed_sum,
    witness_sumcheck: witness_proof,
    table_sumcheck: table_proof,
    pcs_binding: None,
  }
}

/// Blake3 hash of witness field elements (used as succinct commitment).
pub fn hash_witness(witness: &[GF2_128]) -> [u8; 32] {
  let mut hasher = blake3::Hasher::new();
  for e in witness {
    hasher.update(&e.lo.to_le_bytes());
    hasher.update(&e.hi.to_le_bytes());
  }
  *hasher.finalize().as_bytes()
}

/// Prove a LogUp relation with PCS evaluation binding.
///
/// Same as [`prove`] but:
/// 1. Absorbs a 32-byte witness digest instead of raw witness values.
/// 2. PCS-commits the fractional MLEs h_w and h_t.
///    - If h_w and h_t share the same n_vars, uses a single
///      [`BatchTensorPCS`](pcs::BatchTensorPCS) commit (shared Merkle tree
///      + shared column queries).
///    - Otherwise, commits each individually with `pcs::commit_par`.
/// 3. After both sumchecks, PCS-opens at the challenge points.
///
/// Returns `(proof, witness_digest)`.
pub fn prove_committed(
  witness: &[GF2_128],
  table: &LookupTable,
  transcript: &mut Blake3Transcript,
) -> (LogUpProof, [u8; 32]) {
  let n_witness = witness.len();
  assert!(
    n_witness.is_power_of_two(),
    "witness length must be a power of two"
  );

  let witness_digest = hash_witness(witness);

  // ── 1. Absorb sizes + digest ──────────────────────────────────────────
  transcript.absorb_bytes(&(n_witness as u64).to_le_bytes());
  transcript.absorb_bytes(&(table.entries.len() as u64).to_le_bytes());

  transcript.absorb_bytes(&witness_digest);
  transcript.absorb_bytes(&table.content_hash);

  // ── 2. Squeeze β and γ ────────────────────────────────────────────────
  let beta = transcript.squeeze_challenge();
  let gamma = transcript.squeeze_challenge();

  // ── 3. Compute γ-weighted fractional MLEs ─────────────────────────────
  let h_w_evals = witness_fractional_evals(witness, beta, gamma);
  let h_t_evals = table_fractional_evals(table, witness, beta, gamma);

  let witness_sum: GF2_128 = h_w_evals.iter().fold(GF2_128::zero(), |a, &b| a + b);
  let table_sum: GF2_128 = h_t_evals.iter().fold(GF2_128::zero(), |a, &b| a + b);
  debug_assert_eq!(
    witness_sum, table_sum,
    "LogUp prover: witness sum ≠ table sum"
  );
  let claimed_sum = witness_sum;

  // ── 4. PCS commit ─────────────────────────────────────────────────────
  let h_w_n_vars = h_w_evals.len().trailing_zeros();
  let h_t_n_vars = h_t_evals.len().trailing_zeros();

  if h_w_n_vars == h_t_n_vars {
    // ── Batch path: same n_vars → shared tree + column queries ────────
    prove_committed_batch(
      h_w_evals,
      h_t_evals,
      h_w_n_vars,
      beta,
      gamma,
      claimed_sum,
      witness_digest,
      transcript,
    )
  } else {
    // ── Individual path: different n_vars → separate commits ──────────
    prove_committed_individual(
      h_w_evals,
      h_t_evals,
      h_w_n_vars,
      h_t_n_vars,
      beta,
      gamma,
      claimed_sum,
      witness_digest,
      transcript,
    )
  }
}

/// Batch PCS path: h_w and h_t share the same n_vars.
fn prove_committed_batch(
  h_w_evals: Vec<GF2_128>,
  h_t_evals: Vec<GF2_128>,
  n_vars: u32,
  beta: GF2_128,
  gamma: GF2_128,
  claimed_sum: GF2_128,
  witness_digest: [u8; 32],
  transcript: &mut Blake3Transcript,
) -> (LogUpProof, [u8; 32]) {
  let params = pcs::PcsParams {
    n_vars,
    n_queries: 40,
  };
  let mut batch_pcs = pcs::BatchTensorPCS::new(params);
  let h_w_for_sc = h_w_evals.clone();
  let h_t_for_sc = h_t_evals.clone();
  batch_pcs.add_poly(h_w_evals); // entry 0
  batch_pcs.add_poly(h_t_evals); // entry 1
  let (batch_commit, batch_state) = batch_pcs.commit_par(transcript);

  transcript.absorb_field(claimed_sum);

  // Witness-side sumcheck
  transcript.absorb_bytes(b"logup:witness_sumcheck");
  let pre_w = transcript.clone();
  let witness_proof = sumcheck::prove_par(MlePoly::new(h_w_for_sc), transcript);
  let w_challenges = sumcheck::verify(&witness_proof, witness_proof.final_eval, &mut pre_w.clone())
    .expect("own proof must verify");

  // Table-side sumcheck
  transcript.absorb_bytes(b"logup:table_sumcheck");
  let pre_t = transcript.clone();
  let table_proof = sumcheck::prove_par(MlePoly::new(h_t_for_sc), transcript);
  let t_challenges = sumcheck::verify(&table_proof, table_proof.final_eval, &mut pre_t.clone())
    .expect("own proof must verify");

  // Batch PCS open (shared column queries)
  let queries = vec![
    pcs::BatchOpenQuery {
      entry: 0,
      point: w_challenges,
    },
    pcs::BatchOpenQuery {
      entry: 1,
      point: t_challenges,
    },
  ];
  let (open_claims, batch_open_proof) = pcs::batch_open_par(&batch_state, &queries, transcript);
  debug_assert_eq!(open_claims[0], witness_proof.final_eval);
  debug_assert_eq!(open_claims[1], table_proof.final_eval);

  (
    LogUpProof {
      beta,
      gamma,
      claimed_sum,
      witness_sumcheck: witness_proof,
      table_sumcheck: table_proof,
      pcs_binding: Some(PcsBinding::Batch {
        n_vars,
        commit: batch_commit,
        open_proof: batch_open_proof,
      }),
    },
    witness_digest,
  )
}

/// Individual PCS path: h_w and h_t have different n_vars.
fn prove_committed_individual(
  h_w_evals: Vec<GF2_128>,
  h_t_evals: Vec<GF2_128>,
  h_w_n_vars: u32,
  h_t_n_vars: u32,
  beta: GF2_128,
  gamma: GF2_128,
  claimed_sum: GF2_128,
  witness_digest: [u8; 32],
  transcript: &mut Blake3Transcript,
) -> (LogUpProof, [u8; 32]) {
  let h_w_params = pcs::PcsParams {
    n_vars: h_w_n_vars,
    n_queries: 40,
  };
  let (h_w_commit, h_w_state) = pcs::commit_par(&h_w_evals, &h_w_params, transcript);

  let h_t_params = pcs::PcsParams {
    n_vars: h_t_n_vars,
    n_queries: 40,
  };
  let (h_t_commit, h_t_state) = pcs::commit_par(&h_t_evals, &h_t_params, transcript);

  transcript.absorb_field(claimed_sum);

  // Witness-side sumcheck + PCS open
  transcript.absorb_bytes(b"logup:witness_sumcheck");
  let pre_w = transcript.clone();
  let witness_proof = sumcheck::prove_par(MlePoly::new(h_w_evals), transcript);
  let w_challenges = sumcheck::verify(&witness_proof, witness_proof.final_eval, &mut pre_w.clone())
    .expect("own proof must verify");
  let (h_w_eval, h_w_opening) = pcs::open_par(&h_w_state, &w_challenges, transcript);
  debug_assert_eq!(h_w_eval, witness_proof.final_eval);

  // Table-side sumcheck + PCS open
  transcript.absorb_bytes(b"logup:table_sumcheck");
  let pre_t = transcript.clone();
  let table_proof = sumcheck::prove_par(MlePoly::new(h_t_evals), transcript);
  let t_challenges = sumcheck::verify(&table_proof, table_proof.final_eval, &mut pre_t.clone())
    .expect("own proof must verify");
  let (h_t_eval, h_t_opening) = pcs::open_par(&h_t_state, &t_challenges, transcript);
  debug_assert_eq!(h_t_eval, table_proof.final_eval);

  (
    LogUpProof {
      beta,
      gamma,
      claimed_sum,
      witness_sumcheck: witness_proof,
      table_sumcheck: table_proof,
      pcs_binding: Some(PcsBinding::Individual {
        h_w_commit,
        h_t_commit,
        h_w_opening: (h_w_eval, h_w_opening),
        h_t_opening: (h_t_eval, h_t_opening),
      }),
    },
    witness_digest,
  )
}
