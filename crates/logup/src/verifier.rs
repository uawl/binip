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

use crate::proof::{LogUpProof, PcsBinding};
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

  transcript.absorb_fields(witness);
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

/// Verify a LogUp proof with PCS evaluation binding.
///
/// Same as [`verify`] but:
/// 1. Absorbs a 32-byte witness digest instead of raw witness values.
/// 2. Dispatches to batch or individual PCS verification depending on
///    the [`PcsBinding`] variant in the proof.
pub fn verify_committed(
  proof: &LogUpProof,
  n_witness: usize,
  table: &LookupTable,
  transcript: &mut Blake3Transcript,
  witness_digest: &[u8; 32],
) -> Option<LogUpClaims> {
  let n_table = table.entries.len();

  // ── 1. Replay absorb sequence to derive β, γ ─────────────────────────
  transcript.absorb_bytes(&(n_witness as u64).to_le_bytes());
  transcript.absorb_bytes(&(n_table as u64).to_le_bytes());

  transcript.absorb_bytes(witness_digest);
  transcript.absorb_bytes(&table.content_hash);

  let beta = transcript.squeeze_challenge();
  let gamma = transcript.squeeze_challenge();

  if beta != proof.beta || gamma != proof.gamma {
    return None;
  }

  let binding = proof.pcs_binding.as_ref()?;

  match binding {
    PcsBinding::Batch { n_vars, commit, .. } => {
      // ── Batch path ──────────────────────────────────────────────────
      transcript.absorb_bytes(&commit.root);

      if proof.witness_sumcheck.claimed_sum != proof.claimed_sum {
        return None;
      }
      if proof.table_sumcheck.claimed_sum != proof.claimed_sum {
        return None;
      }
      transcript.absorb_field(proof.claimed_sum);

      transcript.absorb_bytes(b"logup:witness_sumcheck");
      let w_challenges = sumcheck::verify(
        &proof.witness_sumcheck,
        proof.witness_sumcheck.final_eval,
        transcript,
      )?;

      transcript.absorb_bytes(b"logup:table_sumcheck");
      let t_challenges = sumcheck::verify(
        &proof.table_sumcheck,
        proof.table_sumcheck.final_eval,
        transcript,
      )?;

      let open_proof = match binding {
        PcsBinding::Batch { open_proof, .. } => open_proof,
        _ => unreachable!(),
      };
      let params = pcs::PcsParams {
        n_vars: *n_vars,
        n_queries: 40,
      };
      let queries = vec![
        pcs::BatchOpenQuery {
          entry: 0,
          point: w_challenges.clone(),
        },
        pcs::BatchOpenQuery {
          entry: 1,
          point: t_challenges.clone(),
        },
      ];
      let claims = vec![
        proof.witness_sumcheck.final_eval,
        proof.table_sumcheck.final_eval,
      ];
      if !pcs::batch_verify(commit, &queries, &claims, open_proof, &params, transcript) {
        return None;
      }

      Some(LogUpClaims {
        witness_challenges: w_challenges,
        table_challenges: t_challenges,
      })
    }

    PcsBinding::Individual {
      h_w_commit,
      h_t_commit,
      h_w_opening,
      h_t_opening,
    } => {
      // ── Individual path ─────────────────────────────────────────────
      transcript.absorb_bytes(&h_w_commit.root);
      transcript.absorb_bytes(&h_t_commit.root);

      if proof.witness_sumcheck.claimed_sum != proof.claimed_sum {
        return None;
      }
      if proof.table_sumcheck.claimed_sum != proof.claimed_sum {
        return None;
      }
      transcript.absorb_field(proof.claimed_sum);

      // Witness-side sumcheck
      transcript.absorb_bytes(b"logup:witness_sumcheck");
      let w_challenges = sumcheck::verify(
        &proof.witness_sumcheck,
        proof.witness_sumcheck.final_eval,
        transcript,
      )?;

      // PCS verify h_w
      let (h_w_eval, ref h_w_open) = *h_w_opening;
      if h_w_eval != proof.witness_sumcheck.final_eval {
        return None;
      }
      let h_w_params = pcs::PcsParams {
        n_vars: h_w_commit.n_vars,
        n_queries: 40,
      };
      if !pcs::verify(h_w_commit, &w_challenges, h_w_eval, h_w_open, &h_w_params, transcript) {
        return None;
      }

      // Table-side sumcheck
      transcript.absorb_bytes(b"logup:table_sumcheck");
      let t_challenges = sumcheck::verify(
        &proof.table_sumcheck,
        proof.table_sumcheck.final_eval,
        transcript,
      )?;

      // PCS verify h_t
      let (h_t_eval, ref h_t_open) = *h_t_opening;
      if h_t_eval != proof.table_sumcheck.final_eval {
        return None;
      }
      let h_t_params = pcs::PcsParams {
        n_vars: h_t_commit.n_vars,
        n_queries: 40,
      };
      if !pcs::verify(h_t_commit, &t_challenges, h_t_eval, h_t_open, &h_t_params, transcript) {
        return None;
      }

      Some(LogUpClaims {
        witness_challenges: w_challenges,
        table_challenges: t_challenges,
      })
    }
  }
}
