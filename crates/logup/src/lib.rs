//! LogUp — γ-Weighted Logarithmic Derivative Lookup Argument over GF(2^128).
//!
//! # Protocol
//!
//! The γ-weighted LogUp protocol proves that every element in a **witness**
//! vector `w` appears in a **table** vector `t`, using the identity:
//!
//! ```text
//!   Σ_{j=0}^{N-1}  γ^j / (β + w_j)  =  Σ_{i=0}^{T-1}  M_i / (β + t_i)
//! ```
//!
//! where `M_i = Σ_{j: w_j=t_i} γ^j` is the γ-weighted multiplicity, and
//! `β, γ ∈ GF(2^128)` are verifier-chosen random challenges.
//!
//! ## Why γ-weighting?
//!
//! In characteristic 2, `a + a = 0` for any field element. The standard LogUp
//! sum `Σ 1/(β−w_j)` suffers from cancellation when the same witness value
//! appears an even number of times, making the protocol unsound. The γ-weight
//! `γ^j` makes every term unique (since `γ^j ≠ γ^k` for `j ≠ k`), preventing
//! any cancellation regardless of multiplicities.
//!
//! ## Reduction to sumcheck
//!
//! Both sides are encoded as MLE polynomials and verified via two sumcheck
//! instances. The verifier checks that both sums are equal.

pub mod proof;
pub mod prover;
pub mod table;
pub mod verifier;

pub use proof::{LogUpProof, PcsBinding};
pub use prover::{hash_witness, prove, prove_committed};
pub use table::LookupTable;
pub use verifier::{LogUpClaims, verify, verify_committed};

#[cfg(test)]
mod tests {
  use field::{FieldElem, GF2_128};
  use transcript::Blake3Transcript;

  use super::*;

  fn g(v: u64) -> GF2_128 {
    GF2_128::from(v)
  }

  /// Build a table, prove, and verify.
  fn prove_and_verify(table_entries: Vec<GF2_128>, witness: Vec<GF2_128>) -> bool {
    let mut witness = witness;
    let n = witness.len().next_power_of_two();
    while witness.len() < n {
      witness.push(table_entries[0]);
    }

    let table = LookupTable::new(table_entries.clone());

    let mut t_prover = Blake3Transcript::new();
    let proof = prove(&witness, &table, &mut t_prover);

    let mut t_verifier = Blake3Transcript::new();
    let result = verify(&proof, &witness, &table, &mut t_verifier);
    result.is_some()
  }

  // ── Basic tests ─────────────────────────────────────────────────────

  #[test]
  fn simple_identity_lookup() {
    let table = vec![g(1), g(2), g(3), g(4)];
    let witness = vec![g(1), g(2), g(3), g(4)];
    assert!(prove_and_verify(table, witness));
  }

  #[test]
  fn repeated_lookups() {
    let table = vec![g(10), g(20)];
    let witness = vec![g(10), g(10), g(20), g(20)];
    assert!(prove_and_verify(table, witness));
  }

  #[test]
  fn single_entry_table() {
    let table = vec![g(42)];
    let witness = vec![g(42), g(42), g(42), g(42)];
    assert!(prove_and_verify(table, witness));
  }

  #[test]
  fn large_table_small_witness() {
    let table: Vec<GF2_128> = (1u64..=16).map(g).collect();
    let witness = vec![g(3), g(7), g(15), g(1)];
    assert!(prove_and_verify(table, witness));
  }

  #[test]
  fn nonzero_field_elements() {
    let t1 = GF2_128::new(0xDEAD_BEEF_CAFE_BABE, 0x1234_5678_9ABC_DEF0);
    let t2 = GF2_128::new(0xAAAA_BBBB_CCCC_DDDD, 0xFEDC_BA98_7654_3210);
    let t3 = GF2_128::new(0x0102_0304_0506_0708, 0x0807_0605_0403_0201);
    let t4 = GF2_128::new(0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF);
    let table = vec![t1, t2, t3, t4];
    let witness = vec![t2, t1, t4, t3];
    assert!(prove_and_verify(table, witness));
  }

  #[test]
  fn high_multiplicity_char2_safe() {
    // Multiplicity > 2: exercises γ-weighting correctness
    let table = vec![g(7), g(8)];
    let witness = vec![g(7), g(7), g(7), g(8)];
    assert!(prove_and_verify(table, witness));
  }

  // ── Soundness tests ─────────────────────────────────────────────────

  #[test]
  fn tampered_claimed_sum_fails() {
    let table_entries = vec![g(1), g(2), g(3), g(4)];
    let witness = vec![g(1), g(2), g(3), g(4)];
    let table = LookupTable::new(table_entries.clone());

    let mut t_prover = Blake3Transcript::new();
    let mut proof = prove(&witness, &table, &mut t_prover);

    proof.claimed_sum = proof.claimed_sum + g(1);

    let mut t_verifier = Blake3Transcript::new();
    assert!(verify(&proof, &witness, &table, &mut t_verifier).is_none());
  }

  #[test]
  #[should_panic(expected = "not found in table")]
  fn witness_not_in_table_panics() {
    let table_entries = vec![g(1), g(2)];
    let witness = vec![g(1), g(999), g(1), g(2)]; // 999 not in table
    let table = LookupTable::new(table_entries);
    let mut t = Blake3Transcript::new();
    prove(&witness, &table, &mut t);
  }

  // ── Table unit tests ─────────────────────────────────────────────────

  #[test]
  fn fractional_sums_match() {
    let table_entries = vec![g(5), g(10), g(15), g(20)];
    let witness = vec![g(5), g(10), g(15), g(20)];
    let table = LookupTable::new(table_entries);

    let beta = GF2_128::new(0x123456789ABCDEF0, 0xFEDCBA9876543210);
    let gamma = GF2_128::new(0xABCDABCDABCDABCD, 0x1111222233334444);
    let h_w = crate::table::witness_fractional_evals(&witness, beta, gamma);
    let h_t = crate::table::table_fractional_evals(&table, &witness, beta, gamma);

    let sum_w: GF2_128 = h_w.iter().fold(GF2_128::zero(), |a, &b| a + b);
    let sum_t: GF2_128 = h_t.iter().fold(GF2_128::zero(), |a, &b| a + b);
    assert_eq!(sum_w, sum_t);
  }

  #[test]
  fn fractional_sums_match_with_repeats() {
    let table_entries = vec![g(5), g(10)];
    let witness = vec![g(5), g(5), g(10), g(10)];
    let table = LookupTable::new(table_entries);

    let beta = GF2_128::new(0x123456789ABCDEF0, 0xFEDCBA9876543210);
    let gamma = GF2_128::new(0xABCDABCDABCDABCD, 0x1111222233334444);
    let h_w = crate::table::witness_fractional_evals(&witness, beta, gamma);
    let h_t = crate::table::table_fractional_evals(&table, &witness, beta, gamma);

    let sum_w: GF2_128 = h_w.iter().fold(GF2_128::zero(), |a, &b| a + b);
    let sum_t: GF2_128 = h_t.iter().fold(GF2_128::zero(), |a, &b| a + b);
    assert_eq!(sum_w, sum_t);
  }
}
