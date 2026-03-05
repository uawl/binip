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

use std::sync::atomic::{AtomicUsize, Ordering};

use field::{FieldElem, GF2_128};
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
    // Absorb the level, node index, and fan_in for extra binding.
    t.absorb_bytes(&level.to_le_bytes());
    t.absorb_bytes(&(node_idx as u32).to_le_bytes());
    t.absorb_bytes(&fan_in.to_le_bytes());

    let sumcheck = sumcheck::prove(agg_mle, &mut t);

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
/// Uses per-node countdown latches so that a parent node starts the
/// instant **its own `fan_in` children** finish — no level-wide barrier.
///
/// Each node writes its result (claim + proof) into a pre-allocated slot
/// indexed by `(level, node_idx)`, eliminating Mutex contention entirely.
/// When the last child of a group decrements its parent's `AtomicUsize`
/// latch to 0, it spawns the parent task on rayon's scoped thread pool.
pub fn prove_recursive_par(
  shard_batch: &ShardProofBatch,
  config: &RecursiveConfig,
  root_transcript: &Blake3Transcript,
) -> RecursiveProof {
  use std::cell::UnsafeCell;

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

  // Pre-compute node counts per level.
  let mut nodes_per_level = Vec::with_capacity(depth as usize);
  let mut count = shard_claims.len();
  for _ in 0..depth {
    count = (count + fan - 1) / fan;
    nodes_per_level.push(count);
  }

  // ── Pre-allocate slots for results ───────────────────────────────────
  // Each slot holds Option<(LevelProof, GF2_128)>.
  // Written exactly once by the owning node, read after the scope ends.
  struct Slot(UnsafeCell<Option<(LevelProof, GF2_128)>>);
  // SAFETY: each slot is written by exactly one rayon task and never read
  // concurrently — the parent only reads after its latch fires, which
  // happens-after the child's write.
  unsafe impl Sync for Slot {}

  let level_slots: Vec<Vec<Slot>> = nodes_per_level
    .iter()
    .map(|&n| (0..n).map(|_| Slot(UnsafeCell::new(None))).collect())
    .collect();

  // ── Per-node latches ─────────────────────────────────────────────────
  // latch[level][node_idx] counts remaining children.
  // Level 0 nodes have no latch (triggered directly from shard claims).
  let latches: Vec<Vec<AtomicUsize>> = nodes_per_level
    .iter()
    .enumerate()
    .map(|(level, &n)| {
      (0..n)
        .map(|node_idx| {
          let n_children = if level == 0 {
            // Children are shard claims — all immediately available.
            0
          } else {
            let child_level_nodes = nodes_per_level[level - 1];
            let start = node_idx * fan;
            (start + fan).min(child_level_nodes) - start
          };
          AtomicUsize::new(n_children)
        })
        .collect()
    })
    .collect();

  // ── Node proving + latch trigger ─────────────────────────────────────
  fn prove_node<'scope>(
    scope: &rayon::Scope<'scope>,
    level: u32,
    node_idx: usize,
    depth: u32,
    fan: usize,
    shard_claims: &'scope [GF2_128],
    nodes_per_level: &'scope [usize],
    level_slots: &'scope [Vec<Slot>],
    latches: &'scope [Vec<AtomicUsize>],
    root_transcript: &'scope Blake3Transcript,
  ) {
    let lv = level as usize;

    // Gather child claims.
    let child_claims: Vec<GF2_128> = if level == 0 {
      let start = node_idx * fan;
      let end = (start + fan).min(shard_claims.len());
      shard_claims[start..end].to_vec()
    } else {
      let child_lv = lv - 1;
      let child_level_nodes = nodes_per_level[child_lv];
      let start = node_idx * fan;
      let end = (start + fan).min(child_level_nodes);
      (start..end)
        .map(|ci| {
          // SAFETY: child slot was written before this node's latch fired.
          let slot = unsafe { &*level_slots[child_lv][ci].0.get() };
          slot.as_ref().unwrap().1
        })
        .collect()
    };

    // Prove.
    let agg_mle = build_aggregation_mle(&child_claims);
    let aggregate_sum = agg_mle.sum();

    let mut t = root_transcript.fork("recursive", level * 0x1_0000 + node_idx as u32);
    t.absorb_bytes(&level.to_le_bytes());
    t.absorb_bytes(&(node_idx as u32).to_le_bytes());
    t.absorb_bytes(&(fan as u32).to_le_bytes());

    let sumcheck = sumcheck::prove(agg_mle, &mut t);

    let proof = LevelProof {
      level,
      node_idx: node_idx as u32,
      child_claims,
      sumcheck,
    };

    // Write result to slot.
    // SAFETY: this slot is written exactly once.
    unsafe {
      *level_slots[lv][node_idx].0.get() = Some((proof, aggregate_sum));
    }

    // Notify parent.
    if (level + 1) < depth {
      let parent_lv = lv + 1;
      let parent_idx = node_idx / fan;
      let prev = latches[parent_lv][parent_idx].fetch_sub(1, Ordering::AcqRel);
      if prev == 1 {
        // Last child — spawn parent on the scoped thread pool.
        scope.spawn(move |s| {
          prove_node(
            s,
            level + 1,
            parent_idx,
            depth,
            fan,
            shard_claims,
            nodes_per_level,
            level_slots,
            latches,
            root_transcript,
          );
        });
      }
    }
  }

  // ── Kick off level-0 nodes ───────────────────────────────────────────
  // Level-0 nodes have all inputs ready (shard claims), so fire them all.
  let sc_ref: &[GF2_128] = &shard_claims;
  let npl_ref: &[usize] = &nodes_per_level;
  let ls_ref: &[Vec<Slot>] = &level_slots;
  let la_ref: &[Vec<AtomicUsize>] = &latches;

  rayon::scope(|s| {
    let n_level0 = npl_ref[0];
    for node_idx in 0..n_level0 {
      s.spawn(move |inner_s| {
        prove_node(
          inner_s,
          0,
          node_idx,
          depth,
          fan,
          sc_ref,
          npl_ref,
          ls_ref,
          la_ref,
          root_transcript,
        );
      });
    }
  });

  // ── Collect results ──────────────────────────────────────────────────
  let mut levels: Vec<Vec<LevelProof>> = Vec::with_capacity(depth as usize);
  for lv_slots in &level_slots {
    let lv_proofs: Vec<LevelProof> = lv_slots
      .iter()
      .map(|slot| {
        let inner = unsafe { &mut *slot.0.get() };
        inner.take().expect("all nodes must complete").0
      })
      .collect();
    levels.push(lv_proofs);
  }

  let root_claim = {
    let last_level = levels.last().unwrap();
    assert_eq!(last_level.len(), 1);
    last_level[0]
      .child_claims
      .iter()
      .fold(GF2_128::zero(), |a, &b| a + b)
  };

  RecursiveProof { levels, root_claim }
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
