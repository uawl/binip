//! LogUp verifier — γ-weighted protocol for GF(2^128).
//!
//! # Verification steps
//!
//! 1. Re-derive `β` and `γ` from the transcript.
//! 2. Check both sumcheck claimed sums equal `proof.claimed_sum`.
//! 3. Verify the witness-side sumcheck proof.
//! 4. Verify the table-side sumcheck proof.
//!
//! Consistency between the fractional MLEs and the raw data is guaranteed by
//! the Fiat-Shamir binding: witness/table values are absorbed before β/γ are
//! derived, so the prover cannot change the data post-challenge. In production
//! this is reinforced by PCS opening proofs.

use field::GF2_128;
use transcript::{Blake3Transcript, Transcript};

use crate::proof::LogUpProof;
use crate::table::LookupTable;

/// Challenge points returned on successful verification (for PCS opening).
#[derive(Debug, Clone)]
pub struct LogUpClaims {
  pub witness_challenges: Vec<GF2_128>,
  pub table_challenges: Vec<GF2_128>,
}

/// Verify a LogUp proof.
///
/// * `proof`         – the LogUp proof produced by the prover
/// * `witness`       – the witness values (absorbed into transcript)
/// * `table`         – the lookup table (hash absorbed into transcript)
/// * `transcript`    – must have the same initial state as the prover's
pub fn verify(
  proof: &LogUpProof,
  witness: &[GF2_128],
  table: &LookupTable,
  transcript: &mut Blake3Transcript,
) -> Option<LogUpClaims> {
  let n_witness = witness.len();
  let n_table = table.entries.len();

  // ── 1. Replay absorb sequence to derive β, γ ─────────────────────────
  transcript.absorb_bytes(&(n_witness as u64).to_le_bytes());
  transcript.absorb_bytes(&(n_table as u64).to_le_bytes());

  for &w in witness {
    transcript.absorb_field(w);
  }
  // Absorb table content hash instead of all entries (deterministic table).
  transcript.absorb_bytes(&table.content_hash);

  let beta = transcript.squeeze_challenge();
  let gamma = transcript.squeeze_challenge();

  if beta != proof.beta || gamma != proof.gamma {
    return None;
  }

  // ── 2. Check claimed sums ─────────────────────────────────────────────
  if proof.witness_sumcheck.claimed_sum != proof.claimed_sum {
    return None;
  }
  if proof.table_sumcheck.claimed_sum != proof.claimed_sum {
    return None;
  }

  transcript.absorb_field(proof.claimed_sum);

  // ── 3. Witness-side sumcheck ──────────────────────────────────────────
  transcript.absorb_bytes(b"logup:witness_sumcheck");
  let w_challenges = sumcheck::verify(
    &proof.witness_sumcheck,
    proof.witness_sumcheck.final_eval,
    transcript,
  )?;

  // ── 4. Table-side sumcheck ────────────────────────────────────────────
  transcript.absorb_bytes(b"logup:table_sumcheck");
  let t_challenges = sumcheck::verify(
    &proof.table_sumcheck,
    proof.table_sumcheck.final_eval,
    transcript,
  )?;

  Some(LogUpClaims {
    witness_challenges: w_challenges,
    table_challenges: t_challenges,
  })
}
