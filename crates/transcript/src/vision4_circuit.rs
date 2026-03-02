//! In-circuit representation of the Vision-4 permutation.
//!
//! Decomposes the permutation into **linear layers** (B_fwd, B_inv, MDS,
//! round-constant addition — all GF(2)-linear, 0 AND gates) and
//! **inversions** (the only nonlinear step).
//!
//! Each inversion is handled via the advice-and-verify pattern: the prover
//! supplies the inverse value, and the circuit checks `a · a⁻¹ = 1` with
//! a single field multiplication.
//!
//! ```text
//! Inversions:       4 per half-round × 16 half-rounds = 64
//! Total AND cost:   64 field multiplications (CheckInv)
//! Linear steps:     free (XOR / Frobenius / constant-multiply)
//! ```

use super::ghash_field::{batch_inv4, gcm_mul};
use super::vision4::{self, M, NUM_ROUNDS, RATE};

// ── Step type ─────────────────────────────────────────────────────────────────

/// Atomic step in the Vision-4 arithmetic circuit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
  /// XOR round constants\[idx\] into all 4 state elements.  Free.
  AddRoundConst(usize),
  /// Replace each state element with its GF(2^128) inverse.
  /// Produces 4 advice values; verified via `a · a⁻¹ = 1`.
  BatchInv,
  /// Apply the B_inv linearised polynomial to each element.  Free.
  BInv,
  /// Apply the B_fwd linearised polynomial to each element.  Free.
  BFwd,
  /// Circulant MDS mixing of all 4 elements.  Free.
  Mds,
}

// ── Permutation circuit ───────────────────────────────────────────────────────

/// In-circuit Vision-4 permutation (8 full rounds).
///
/// Use [`VisionCircuit::permutation()`] to construct, then
/// [`evaluate()`](VisionCircuit::evaluate) to produce a
/// [`PermutationWitness`] containing the output state and the advice
/// tape (inverse values) required for CheckInv verification.
#[derive(Debug, Clone)]
pub struct VisionCircuit {
  pub steps: Vec<Step>,
}

/// Complete witness for one permutation evaluation.
#[derive(Debug, Clone)]
pub struct PermutationWitness {
  /// State before permutation.
  pub input: [u128; M],
  /// State after permutation.
  pub output: [u128; M],
  /// Advice tape: inverse values, 4 per [`Step::BatchInv`].
  /// Length = `4 × n_batch_inv`.
  pub inverses: Vec<u128>,
  /// Pre-inversion states (one `[u128; 4]` per BatchInv step).
  /// For CheckInv: `pre_inv[k][i] * inverses[4*k+i] = 1`.
  pub pre_inv: Vec<[u128; M]>,
}

impl VisionCircuit {
  /// Build the circuit for the full Vision-4 permutation.
  pub fn permutation() -> Self {
    let mut steps = Vec::with_capacity(1 + NUM_ROUNDS * 8);

    // Initial round-constant addition.
    steps.push(Step::AddRoundConst(0));

    for r in 0..NUM_ROUNDS {
      // First half-round: inv → B_inv → MDS → constants
      steps.push(Step::BatchInv);
      steps.push(Step::BInv);
      steps.push(Step::Mds);
      steps.push(Step::AddRoundConst(1 + 2 * r));

      // Second half-round: inv → B_fwd → MDS → constants
      steps.push(Step::BatchInv);
      steps.push(Step::BFwd);
      steps.push(Step::Mds);
      steps.push(Step::AddRoundConst(1 + 2 * r + 1));
    }

    VisionCircuit { steps }
  }

  /// Number of [`Step::BatchInv`] steps.
  pub fn n_batch_inv(&self) -> usize {
    self.steps.iter().filter(|s| **s == Step::BatchInv).count()
  }

  /// Total advice values (inversions) needed.
  pub fn n_advice(&self) -> usize {
    self.n_batch_inv() * M
  }

  /// Total field multiplications (= AND-gate cost metric).
  /// One multiplication per CheckInv verification.
  pub fn n_multiplications(&self) -> usize {
    self.n_advice()
  }

  /// Evaluate the circuit on a concrete input, producing the full witness.
  pub fn evaluate(&self, input: [u128; M]) -> PermutationWitness {
    let n_inv = self.n_batch_inv();
    let mut state = input;
    let mut inverses = Vec::with_capacity(n_inv * M);
    let mut pre_inv = Vec::with_capacity(n_inv);

    for step in &self.steps {
      match *step {
        Step::AddRoundConst(idx) => {
          let rc = vision4::ROUND_CONSTANTS[idx];
          for i in 0..M {
            state[i] ^= rc[i];
          }
        }
        Step::BatchInv => {
          pre_inv.push(state);
          batch_inv4(&mut state);
          inverses.extend_from_slice(&state);
        }
        Step::BInv => {
          for x in state.iter_mut() {
            *x = vision4::b_inv(*x);
          }
        }
        Step::BFwd => {
          for x in state.iter_mut() {
            *x = vision4::b_fwd(*x);
          }
        }
        Step::Mds => {
          vision4::mds(&mut state);
        }
      }
    }

    PermutationWitness {
      input,
      output: state,
      inverses,
      pre_inv,
    }
  }

  /// Verify a witness: check every `pre * inverse = 1`.
  ///
  /// Returns `true` when all CheckInv pairs are consistent.
  pub fn verify_witness(&self, w: &PermutationWitness) -> bool {
    for (k, pre) in w.pre_inv.iter().enumerate() {
      for i in 0..M {
        let a = pre[i];
        let a_inv = w.inverses[k * M + i];
        if a == 0 {
          if a_inv != 0 {
            return false;
          }
        } else if gcm_mul(a, a_inv) != 1 {
          return false;
        }
      }
    }
    true
  }
}

// ── Sponge circuit ────────────────────────────────────────────────────────────

/// In-circuit Vision-4 sponge (duplex construction).
///
/// Models a fixed-length absorb (field elements, not bytes) followed
/// by a fixed number of squeezes.  Each absorb block overwrites the
/// rate portion of the state, then permutes.  Each squeeze reads one
/// field element from the rate portion (permuting when exhausted).
///
/// # Relation to the byte-level sponge
///
/// The byte-level [`VisionSponge`](super::sponge::VisionSponge) uses
/// Keccak-style padding.  This circuit-level sponge operates on raw
/// field elements and omits byte padding — the caller is responsible
/// for encoding input correctly.
#[derive(Debug, Clone)]
pub struct SpongeCircuit {
  /// Number of field elements to absorb.
  pub n_absorb: usize,
  /// Number of field elements to squeeze.
  pub n_squeeze: usize,
  /// Total permutation calls.
  pub n_perms: usize,
  /// The underlying permutation circuit.
  pub perm: VisionCircuit,
}

/// Witness for a full sponge session.
#[derive(Debug, Clone)]
pub struct SpongeWitness {
  /// Per-permutation witnesses (one per call).
  pub perm_witnesses: Vec<PermutationWitness>,
  /// Squeezed output elements.
  pub squeezed: Vec<u128>,
}

impl SpongeCircuit {
  /// Create a sponge circuit for `n_absorb` absorbed field elements
  /// and `n_squeeze` squeezed elements.
  pub fn new(n_absorb: usize, n_squeeze: usize) -> Self {
    // Absorb: each permutation consumes up to RATE elements.
    let absorb_perms = if n_absorb == 0 {
      0
    } else {
      (n_absorb + RATE - 1) / RATE
    };
    // After the last absorb-permutation, RATE elements are available.
    // Additional permutations are needed if we squeeze more than RATE.
    let squeeze_extra_perms = if n_squeeze <= RATE {
      0
    } else {
      (n_squeeze - RATE + RATE - 1) / RATE
    };
    // At least 1 permutation if we need any squeeze (even if n_absorb=0).
    let n_perms = if absorb_perms == 0 && n_squeeze > 0 {
      1 + squeeze_extra_perms
    } else {
      absorb_perms + squeeze_extra_perms
    };

    SpongeCircuit {
      n_absorb,
      n_squeeze,
      n_perms,
      perm: VisionCircuit::permutation(),
    }
  }

  /// Total advice values across all permutation calls.
  pub fn n_advice(&self) -> usize {
    self.n_perms * self.perm.n_advice()
  }

  /// Total field multiplications across all permutation calls.
  pub fn n_multiplications(&self) -> usize {
    self.n_perms * self.perm.n_multiplications()
  }

  /// Evaluate the sponge circuit on concrete absorb data.
  ///
  /// # Panics
  ///
  /// Panics if `absorb_data.len() != self.n_absorb`.
  pub fn evaluate(&self, absorb_data: &[u128]) -> SpongeWitness {
    assert_eq!(absorb_data.len(), self.n_absorb);

    let mut state = [0u128; M];
    let mut perm_witnesses = Vec::with_capacity(self.n_perms);
    let mut squeezed = Vec::with_capacity(self.n_squeeze);
    let mut squeeze_pos = 0usize; // position within rate for squeezing

    // ── Absorb phase ────────────────────────────────────────────────
    for chunk in absorb_data.chunks(RATE) {
      // Overwrite rate portion.
      for (i, &val) in chunk.iter().enumerate() {
        state[i] = val;
      }
      // Zero remaining rate slots if partial block.
      for i in chunk.len()..RATE {
        state[i] = 0;
      }
      let w = self.perm.evaluate(state);
      state = w.output;
      perm_witnesses.push(w);
    }

    // If no absorb but we need squeezes, do one permutation on the zero state.
    if self.n_absorb == 0 && self.n_squeeze > 0 {
      let w = self.perm.evaluate(state);
      state = w.output;
      perm_witnesses.push(w);
    }

    // ── Squeeze phase ───────────────────────────────────────────────
    for _ in 0..self.n_squeeze {
      if squeeze_pos >= RATE {
        // Exhausted current rate; permute for more output.
        let w = self.perm.evaluate(state);
        state = w.output;
        perm_witnesses.push(w);
        squeeze_pos = 0;
      }
      squeezed.push(state[squeeze_pos]);
      squeeze_pos += 1;
    }

    SpongeWitness {
      perm_witnesses,
      squeezed,
    }
  }

  /// Verify all permutation witnesses in the sponge.
  pub fn verify_witness(&self, w: &SpongeWitness) -> bool {
    w.perm_witnesses
      .iter()
      .all(|pw| self.perm.verify_witness(pw))
  }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use super::*;
  use crate::vision4::permutation;

  #[test]
  fn permutation_circuit_structure() {
    let c = VisionCircuit::permutation();
    // 1 initial AddRoundConst + 8 rounds × (4+4) = 65 steps
    assert_eq!(c.steps.len(), 1 + NUM_ROUNDS * 8);
    // 2 BatchInv per round × 8 rounds = 16
    assert_eq!(c.n_batch_inv(), 2 * NUM_ROUNDS);
    // 4 inversions per BatchInv → 64 total
    assert_eq!(c.n_advice(), 64);
    assert_eq!(c.n_multiplications(), 64);
  }

  #[test]
  fn evaluate_matches_permutation_zero() {
    let c = VisionCircuit::permutation();
    let w = c.evaluate([0u128; M]);
    // Must match the reference permutation.
    let mut expected = [0u128; M];
    permutation(&mut expected);
    assert_eq!(w.output, expected);
  }

  #[test]
  fn evaluate_matches_permutation_deadbeef() {
    let c = VisionCircuit::permutation();
    let input = [0xdeadbeef, 0, 0xdeadbeef, 0];
    let w = c.evaluate(input);
    let mut expected = input;
    permutation(&mut expected);
    assert_eq!(w.output, expected);
  }

  #[test]
  fn evaluate_matches_permutation_random_like() {
    let c = VisionCircuit::permutation();
    let input = [
      0xabcdef0123456789abcdef0123456789u128,
      0x1111111122222222333333334444444u128,
      0xfedcba9876543210fedcba9876543210u128,
      0x99887766554433221100aabbccddeeffu128,
    ];
    let w = c.evaluate(input);
    let mut expected = input;
    permutation(&mut expected);
    assert_eq!(w.output, expected);
  }

  #[test]
  fn witness_verify_passes() {
    let c = VisionCircuit::permutation();
    let w = c.evaluate([0u128; M]);
    assert!(c.verify_witness(&w));
  }

  #[test]
  fn witness_verify_passes_nonzero() {
    let c = VisionCircuit::permutation();
    let w = c.evaluate([42, 0x1337, 0xdead, 0xbeef]);
    assert!(c.verify_witness(&w));
  }

  #[test]
  fn witness_verify_fails_on_tampered_inverse() {
    let c = VisionCircuit::permutation();
    let mut w = c.evaluate([1, 2, 3, 4]);
    // Corrupt one inverse value.
    w.inverses[0] ^= 1;
    assert!(!c.verify_witness(&w));
  }

  #[test]
  fn witness_inverse_count() {
    let c = VisionCircuit::permutation();
    let w = c.evaluate([0u128; M]);
    assert_eq!(w.inverses.len(), 64);
    assert_eq!(w.pre_inv.len(), 16);
  }

  // ── Sponge circuit tests ────────────────────────────────────────────────

  #[test]
  fn sponge_circuit_no_absorb_one_squeeze() {
    let sc = SpongeCircuit::new(0, 1);
    assert_eq!(sc.n_perms, 1);
    let w = sc.evaluate(&[]);
    assert_eq!(w.squeezed.len(), 1);
    assert!(sc.verify_witness(&w));
  }

  #[test]
  fn sponge_circuit_one_block_absorb() {
    // Absorb 2 elements (= 1 rate block), squeeze 1.
    let sc = SpongeCircuit::new(2, 1);
    assert_eq!(sc.n_perms, 1);
    let w = sc.evaluate(&[0xAAu128, 0xBBu128]);
    assert_eq!(w.squeezed.len(), 1);
    assert_eq!(w.perm_witnesses.len(), 1);
    assert!(sc.verify_witness(&w));

    // Verify output matches manual computation.
    let mut state = [0xAAu128, 0xBBu128, 0, 0];
    permutation(&mut state);
    assert_eq!(w.squeezed[0], state[0]);
  }

  #[test]
  fn sponge_circuit_two_block_absorb() {
    // Absorb 3 elements → 2 blocks (partial second block).
    let sc = SpongeCircuit::new(3, 2);
    assert_eq!(sc.n_perms, 2); // 2 absorb blocks
    let w = sc.evaluate(&[1, 2, 3]);
    assert_eq!(w.squeezed.len(), 2);
    assert!(sc.verify_witness(&w));
  }

  #[test]
  fn sponge_circuit_squeeze_more_than_rate() {
    // Absorb 2 elements, squeeze 4 (> RATE=2) → needs extra permutation.
    let sc = SpongeCircuit::new(2, 4);
    assert_eq!(sc.n_perms, 2); // 1 absorb + 1 extra squeeze
    let w = sc.evaluate(&[0x11, 0x22]);
    assert_eq!(w.squeezed.len(), 4);
    assert_eq!(w.perm_witnesses.len(), 2);
    assert!(sc.verify_witness(&w));

    // First 2 squeezes come from the absorb permutation,
    // next 2 from the extra permutation.
    let mut state = [0x11u128, 0x22u128, 0, 0];
    permutation(&mut state);
    assert_eq!(w.squeezed[0], state[0]);
    assert_eq!(w.squeezed[1], state[1]);
    permutation(&mut state);
    assert_eq!(w.squeezed[2], state[0]);
    assert_eq!(w.squeezed[3], state[1]);
  }

  #[test]
  fn sponge_circuit_advice_count() {
    let sc = SpongeCircuit::new(4, 2);
    // 4 absorb elements → 2 blocks → 2 perms, each with 64 advice
    assert_eq!(sc.n_perms, 2);
    assert_eq!(sc.n_advice(), 128);
    assert_eq!(sc.n_multiplications(), 128);
  }

  #[test]
  fn sponge_deterministic() {
    let sc = SpongeCircuit::new(2, 2);
    let w1 = sc.evaluate(&[42, 99]);
    let w2 = sc.evaluate(&[42, 99]);
    assert_eq!(w1.squeezed, w2.squeezed);
  }

  #[test]
  fn sponge_different_input_different_output() {
    let sc = SpongeCircuit::new(2, 1);
    let w1 = sc.evaluate(&[1, 2]);
    let w2 = sc.evaluate(&[3, 4]);
    assert_ne!(w1.squeezed, w2.squeezed);
  }
}
