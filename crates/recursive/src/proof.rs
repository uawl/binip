//! Proof types for recursive aggregation.

use field::GF2_128;
use sumcheck::SumcheckProof;

/// Proof produced by one recursive aggregation node.
///
/// Each node takes `fan_in` child claims and produces a sumcheck proof
/// that the sum of those claims is correct.
#[derive(Debug, Clone)]
pub struct LevelProof {
  /// Recursion level (0 = first aggregation above shards).
  pub level: u32,
  /// Index of this node within its level.
  pub node_idx: u32,
  /// The `fan_in` child claimed sums that this node aggregates.
  pub child_claims: Vec<GF2_128>,
  /// Sumcheck proof over the MLE built from child claims.
  pub sumcheck: SumcheckProof,
}

/// Complete recursive proof from shards to root.
#[derive(Debug, Clone)]
pub struct RecursiveProof {
  /// Level proofs, outer index = level (0 .. depth-1).
  /// `levels[l]` contains the proofs at recursion level `l`.
  pub levels: Vec<Vec<LevelProof>>,
  /// The single root claim after all recursion levels.
  pub root_claim: GF2_128,
}
