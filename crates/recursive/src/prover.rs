//! Recursive prover — aggregates shard/level proofs bottom-up.
//!
//! # Recursion loop
//!
//! Starting from `n_shards` shard claimed sums:
//!
//! 1. Group consecutive `fan_in` claims into aggregation nodes.
//! 2. Each node builds an aggregation MLE from its children's claims.
//! 3. Run sumcheck on each node's MLE with a forked transcript.
//! 4. Collect the node's aggregate claim (= sum of children).
//! 5. Feed the aggregate claims into the next level.
//! 6. Repeat until a single root claim remains.

use std::sync::Mutex;

use field::{FieldElem, GF2_128};
use poly::MlePoly;
use rayon::prelude::*;
use shard::{RecursiveConfig, ShardProofBatch};
use transcript::{Blake3Transcript, Transcript};

use crate::circuit::build_aggregation_mle;
use crate::proof::{LevelProof, RecursiveProof};

/// Run the full recursive aggregation from shard proofs to root.
///
/// The `root_transcript` must be in the same state as the one used for
/// shard proving (after absorbing the commitment).
///
/// Returns a [`RecursiveProof`] containing all level proofs and the root claim.
pub fn prove_recursive(
  shard_batch: &ShardProofBatch,
  config: &RecursiveConfig,
  root_transcript: &Blake3Transcript,
) -> RecursiveProof {
  let depth = config.depth();

  // Collect initial claims from shards.
  let mut current_claims: Vec<GF2_128> = shard_batch
    .shard_proofs
    .iter()
    .map(|sp| sp.sumcheck.claimed_sum)
    .collect();

  let mut levels = Vec::with_capacity(depth as usize);

  for level in 0..depth {
    let (level_proofs, next_claims) =
      prove_level(level, &current_claims, config.fan_in, root_transcript);
    levels.push(level_proofs);
    current_claims = next_claims;
  }

  // After all levels, we should have exactly 1 claim.
  assert_eq!(
    current_claims.len(),
    1,
    "recursive aggregation should produce exactly 1 root claim, got {}",
    current_claims.len()
  );

  RecursiveProof {
    levels,
    root_claim: current_claims[0],
  }
}

/// Prove one recursion level: group `claims` into `fan_in`-sized chunks,
/// run sumcheck on each, and return level proofs + the aggregate claims
/// for the next level.
fn prove_level(
  level: u32,
  claims: &[GF2_128],
  fan_in: u32,
  root_transcript: &Blake3Transcript,
) -> (Vec<LevelProof>, Vec<GF2_128>) {
  let fan = fan_in as usize;
  let n_nodes = (claims.len() + fan - 1) / fan;

  let mut level_proofs = Vec::with_capacity(n_nodes);
  let mut next_claims = Vec::with_capacity(n_nodes);

  for node_idx in 0..n_nodes {
    let start = node_idx * fan;
    let end = (start + fan).min(claims.len());
    let child_claims = &claims[start..end];

    // Build aggregation MLE from child claims.
    let agg_mle = build_aggregation_mle(child_claims);
    let aggregate_sum = agg_mle.sum();

    // Fork transcript for this node with domain separation.
    let mut t = root_transcript.fork("recursive", level * 0x1_0000 + node_idx as u32);
    // Absorb the level and node index for extra binding.
    t.absorb_bytes(&level.to_le_bytes());
    t.absorb_bytes(&(node_idx as u32).to_le_bytes());

    let sumcheck = sumcheck::prove(&agg_mle, &mut t);

    level_proofs.push(LevelProof {
      level,
      node_idx: node_idx as u32,
      child_claims: child_claims.to_vec(),
      sumcheck,
    });

    next_claims.push(aggregate_sum);
  }

  (level_proofs, next_claims)
}

/// DAG-parallel variant of [`prove_recursive`].
///
/// Unlike the sequential version (or a level-barrier parallel version),
/// this schedules each node as soon as **its own children** complete.
/// A node at level L+1 can start while unrelated subtrees at level L are
/// still running.
///
/// Implementation: A recursive function [`prove_dag_node`] is called from
/// the root.  At each internal node, `rayon::par_iter` fans out to child
/// nodes, which recursively fan out further.  Rayon's work-stealing
/// scheduler automatically balances the load.
pub fn prove_recursive_par(
  shard_batch: &ShardProofBatch,
  config: &RecursiveConfig,
  root_transcript: &Blake3Transcript,
) -> RecursiveProof {
  let depth = config.depth();
  let fan = config.fan_in as usize;

  if depth == 0 {
    return RecursiveProof {
      levels: vec![],
      root_claim: shard_batch.shard_proofs[0].sumcheck.claimed_sum,
    };
  }

  let shard_claims: Vec<GF2_128> = shard_batch
    .shard_proofs
    .iter()
    .map(|sp| sp.sumcheck.claimed_sum)
    .collect();

  // Pre-compute number of nodes at each level for bounds checking.
  let mut nodes_per_level = Vec::with_capacity(depth as usize);
  let mut count = shard_claims.len();
  for _ in 0..depth {
    count = (count + fan - 1) / fan;
    nodes_per_level.push(count);
  }

  // Thread-safe collectors for level proofs.
  let collectors: Vec<Mutex<Vec<LevelProof>>> =
    (0..depth).map(|_| Mutex::new(Vec::new())).collect();

  // Kick off from the root — it recursively spawns the whole tree.
  let root_claim = prove_dag_node(
    depth - 1,
    0,
    &shard_claims,
    fan,
    &nodes_per_level,
    root_transcript,
    &collectors,
  );

  // Extract and sort proofs by node_idx within each level.
  let levels: Vec<Vec<LevelProof>> = collectors
    .into_iter()
    .map(|m| {
      let mut proofs = m.into_inner().unwrap();
      proofs.sort_by_key(|p| p.node_idx);
      proofs
    })
    .collect();

  RecursiveProof {
    levels,
    root_claim,
  }
}

/// Recursively prove a single DAG node.
///
/// - **Level 0** nodes take shard claims directly.
/// - **Level > 0** nodes spawn children via `par_iter`, wait for them,
///   then prove this node.  This means a parent starts the instant all
///   its children finish — no level-wide barrier.
fn prove_dag_node(
  level: u32,
  node_idx: usize,
  shard_claims: &[GF2_128],
  fan: usize,
  nodes_per_level: &[usize],
  root_transcript: &Blake3Transcript,
  collectors: &[Mutex<Vec<LevelProof>>],
) -> GF2_128 {
  let child_claims: Vec<GF2_128> = if level == 0 {
    // Leaf: slice shard claims.
    let start = node_idx * fan;
    let end = (start + fan).min(shard_claims.len());
    shard_claims[start..end].to_vec()
  } else {
    // Internal: prove children in parallel, collect their aggregate sums.
    let child_level = (level - 1) as usize;
    let n_children_total = nodes_per_level[child_level];
    let start = node_idx * fan;
    let end = (start + fan).min(n_children_total);
    (start..end)
      .into_par_iter()
      .map(|child_idx| {
        prove_dag_node(
          level - 1,
          child_idx,
          shard_claims,
          fan,
          nodes_per_level,
          root_transcript,
          collectors,
        )
      })
      .collect()
  };

  // Prove this node.
  let agg_mle = build_aggregation_mle(&child_claims);
  let aggregate_sum = agg_mle.sum();

  let mut t = root_transcript.fork("recursive", level * 0x1_0000 + node_idx as u32);
  t.absorb_bytes(&level.to_le_bytes());
  t.absorb_bytes(&(node_idx as u32).to_le_bytes());

  let sumcheck = sumcheck::prove(&agg_mle, &mut t);

  collectors[level as usize].lock().unwrap().push(LevelProof {
    level,
    node_idx: node_idx as u32,
    child_claims,
    sumcheck,
  });

  aggregate_sum
}

#[cfg(test)]
mod tests {
  use super::*;
  use field::GF2_128;
  use poly::MlePoly;
  use shard::{RecursiveConfig, prove_all};

  fn g(v: u64) -> GF2_128 {
    GF2_128::from(v)
  }

  fn make_shard_batch(
    total_vars: u32,
    shard_vars: u32,
    fan_in: u32,
  ) -> (MlePoly, RecursiveConfig, Blake3Transcript, ShardProofBatch) {
    let n = 1usize << total_vars;
    let evals: Vec<GF2_128> = (1..=(n as u64)).map(g).collect();
    let poly = MlePoly::new(evals);
    let cfg = RecursiveConfig {
      total_vars,
      shard_vars,
      fan_in,
    };
    let root_t = Blake3Transcript::new();
    let batch = prove_all(&poly, &cfg, &root_t);
    (poly, cfg, root_t, batch)
  }

  #[test]
  fn recursive_proof_basic() {
    // 4-var MLE, 4 shards (2-var each), fan_in=2 → depth 2
    let (poly, cfg, root_t, batch) = make_shard_batch(4, 2, 2);
    assert_eq!(cfg.n_shards(), 4);
    assert_eq!(cfg.depth(), 2);

    let rproof = prove_recursive(&batch, &cfg, &root_t);
    assert_eq!(rproof.levels.len(), 2);
    // Level 0: 4 shards / 2 = 2 nodes
    assert_eq!(rproof.levels[0].len(), 2);
    // Level 1: 2 nodes / 2 = 1 node
    assert_eq!(rproof.levels[1].len(), 1);
    // Root claim should equal total MLE sum
    assert_eq!(rproof.root_claim, poly.sum());
  }

  #[test]
  fn recursive_proof_fan4() {
    // 4-var MLE, 4 shards, fan_in=4 → depth 1
    let (poly, cfg, root_t, batch) = make_shard_batch(4, 2, 4);
    assert_eq!(cfg.depth(), 1);

    let rproof = prove_recursive(&batch, &cfg, &root_t);
    assert_eq!(rproof.levels.len(), 1);
    assert_eq!(rproof.levels[0].len(), 1); // 4/4 = 1 node
    assert_eq!(rproof.root_claim, poly.sum());
  }

  #[test]
  fn recursive_proof_single_shard() {
    // total_vars == shard_vars → 1 shard, depth 0 → no recursion levels
    let (poly, cfg, root_t, batch) = make_shard_batch(3, 3, 2);
    assert_eq!(cfg.depth(), 0);

    let rproof = prove_recursive(&batch, &cfg, &root_t);
    assert_eq!(rproof.levels.len(), 0);
    assert_eq!(rproof.root_claim, poly.sum());
  }

  #[test]
  fn recursive_proof_larger() {
    // 6-var MLE, 4 shards (4-var each), fan_in=2 → depth 2
    let (poly, cfg, root_t, batch) = make_shard_batch(6, 4, 2);
    assert_eq!(cfg.n_shards(), 4);
    assert_eq!(cfg.depth(), 2);

    let rproof = prove_recursive(&batch, &cfg, &root_t);
    assert_eq!(rproof.levels.len(), 2);
    assert_eq!(rproof.root_claim, poly.sum());
  }

  #[test]
  fn level_proof_child_claims_correct() {
    let (_, cfg, root_t, batch) = make_shard_batch(4, 2, 2);
    let rproof = prove_recursive(&batch, &cfg, &root_t);

    // Level 0, node 0 should have shard 0 and shard 1 claims
    let node0 = &rproof.levels[0][0];
    assert_eq!(node0.child_claims.len(), 2);
    assert_eq!(
      node0.child_claims[0],
      batch.shard_proofs[0].sumcheck.claimed_sum
    );
    assert_eq!(
      node0.child_claims[1],
      batch.shard_proofs[1].sumcheck.claimed_sum
    );
  }

  #[test]
  fn level_proof_sumcheck_has_correct_rounds() {
    let (_, cfg, root_t, batch) = make_shard_batch(4, 2, 2);
    let rproof = prove_recursive(&batch, &cfg, &root_t);

    // fan_in=2 → aggregation MLE has 2 evals → 1 var → 1 round
    for lp in &rproof.levels[0] {
      assert_eq!(lp.sumcheck.round_polys.len(), 1);
    }
  }
}
