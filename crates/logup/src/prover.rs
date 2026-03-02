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

use field::GF2_128;
use poly::MlePoly;
use transcript::{Blake3Transcript, Transcript};

use crate::proof::LogUpProof;
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

  for &w in witness {
    transcript.absorb_field(w);
  }
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
  let witness_proof = sumcheck::prove(&h_w, transcript);

  // ── 5. Table-side sumcheck ────────────────────────────────────────────
  transcript.absorb_bytes(b"logup:table_sumcheck");
  let table_proof = sumcheck::prove(&h_t, transcript);

  LogUpProof {
    beta,
    gamma,
    claimed_sum,
    witness_sumcheck: witness_proof,
    table_sumcheck: table_proof,
  }
}
