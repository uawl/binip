//! GPU workgroup scheduling from Proof Tree structure.
//!
//! Walks the [`ProofNode`] tree and partitions the flat execution trace
//! into workgroups aligned with composition boundaries (Seq / Branch
//! nodes).  Each workgroup maps to an independent GPU dispatch and
//! is an ideal shard boundary for the recursive proving pipeline.
//!
//! # Algorithm
//!
//! 1. **Annotate** — DFS-walk the tree, computing the cumulative row
//!    range `[start, start+size)` for every node using the per-leaf
//!    row counts provided by the compiler.
//! 2. **Atomise** — recursively collect *atoms*: subtrees whose total
//!    row count is ≤ `target_size`.  Atoms are the smallest units that
//!    respect composition boundaries.
//! 3. **Merge** — greedily combine adjacent atoms into workgroups of
//!    approximately `target_size` rows.  A workgroup is closed once its
//!    accumulated size reaches the target; the last workgroup may be
//!    merged into the previous one if it would be unreasonably small.

use evm_types::ProofNode;

// ── Public types ────────────────────────────────────────────────────────────

/// A workgroup spans a contiguous range of trace rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Workgroup {
  /// Zero-based workgroup index.
  pub id: u32,
  /// First row in this workgroup (inclusive).
  pub row_start: u32,
  /// One-past-the-last row (exclusive).
  pub row_end: u32,
}

impl Workgroup {
  /// Number of rows in this workgroup.
  pub fn len(&self) -> u32 {
    self.row_end - self.row_start
  }

  pub fn is_empty(&self) -> bool {
    self.row_end == self.row_start
  }
}

/// Workgroup schedule for GPU dispatch.
///
/// Invariants:
/// - Workgroups are ordered by ascending `row_start`.
/// - They are non-overlapping and contiguous, covering `[0, n_rows)`.
#[derive(Debug, Clone)]
pub struct Schedule {
  /// Ordered, non-overlapping workgroups covering `[0, n_rows)`.
  pub workgroups: Vec<Workgroup>,
  /// Total number of trace rows (before padding).
  pub n_rows: u32,
}

impl Schedule {
  /// Look up the workgroup that contains `row`.
  ///
  /// Returns `None` if `row >= n_rows`.
  pub fn workgroup_of(&self, row: u32) -> Option<u32> {
    // Binary search by row_start.
    let idx = self
      .workgroups
      .partition_point(|wg| wg.row_start <= row)
      .checked_sub(1)?;
    let wg = &self.workgroups[idx];
    if row < wg.row_end { Some(wg.id) } else { None }
  }
}

/// Errors during schedule construction.
#[derive(Debug, thiserror::Error)]
pub enum ScheduleError {
  #[error(
    "leaf count mismatch: tree has {tree} leaves but {given} row counts provided"
  )]
  LeafCountMismatch { tree: usize, given: usize },

  #[error("target workgroup size must be > 0")]
  ZeroTarget,
}

// ── Internal tree annotation ────────────────────────────────────────────────

/// A tree node annotated with its row span.
struct NodeSpan {
  start: u32,
  size: u32,
  children: Vec<NodeSpan>,
}

/// DFS-walk the Proof Tree, assigning each node its `[start, start+size)`.
fn annotate(
  tree: &ProofNode,
  row_counts: &[usize],
  leaf_idx: &mut usize,
  offset: &mut u32,
) -> NodeSpan {
  match tree {
    ProofNode::Leaf { .. } => {
      let start = *offset;
      let size = row_counts[*leaf_idx] as u32;
      *leaf_idx += 1;
      *offset += size;
      NodeSpan { start, size, children: vec![] }
    }
    ProofNode::Seq { left, right } => {
      let start = *offset;
      let l = annotate(left, row_counts, leaf_idx, offset);
      let r = annotate(right, row_counts, leaf_idx, offset);
      NodeSpan { start, size: l.size + r.size, children: vec![l, r] }
    }
    ProofNode::Branch { cond, taken, not_taken } => {
      let start = *offset;
      let c = annotate(cond, row_counts, leaf_idx, offset);
      let t = annotate(taken, row_counts, leaf_idx, offset);
      let n = annotate(not_taken, row_counts, leaf_idx, offset);
      NodeSpan {
        start,
        size: c.size + t.size + n.size,
        children: vec![c, t, n],
      }
    }
  }
}

/// Recursively collect indivisible *atoms*: subtrees with ≤ `target` rows,
/// or leaves (no children to split further).
///
/// Each atom is a `(start, size)` pair.
fn collect_atoms(node: &NodeSpan, target: u32, out: &mut Vec<(u32, u32)>) {
  if node.size == 0 {
    return;
  }
  if node.size <= target || node.children.is_empty() {
    out.push((node.start, node.size));
  } else {
    for child in &node.children {
      collect_atoms(child, target, out);
    }
  }
}

// ── Public API ──────────────────────────────────────────────────────────────

/// Build a workgroup schedule from the Proof Tree.
///
/// Each `Leaf` in the tree produced some number of micro-op rows during
/// VM execution.  `row_counts[i]` gives the row count for the *i*-th
/// leaf in DFS order.
///
/// `target_size` is the desired number of rows per workgroup (typically
/// `2^shard_vars`, e.g. 256).
///
/// # Algorithm
///
/// 1. Annotate the tree with row ranges.
/// 2. Collect *atoms*: composition-boundary-aligned subtrees of
///    ≤ `target_size` rows.
/// 3. Greedily merge adjacent atoms into workgroups.  A workgroup is
///    closed once adding the next atom would push it past `target_size`.
///    If the final workgroup is smaller than `target_size / 4`, it is
///    absorbed into the preceding one.
pub fn schedule(
  tree: &ProofNode,
  row_counts: &[usize],
  target_size: u32,
) -> Result<Schedule, ScheduleError> {
  let n_leaves = tree.leaf_count();
  if row_counts.len() != n_leaves {
    return Err(ScheduleError::LeafCountMismatch {
      tree: n_leaves,
      given: row_counts.len(),
    });
  }
  if target_size == 0 {
    return Err(ScheduleError::ZeroTarget);
  }

  // Phase 1: annotate
  let mut leaf_idx = 0usize;
  let mut offset = 0u32;
  let root = annotate(tree, row_counts, &mut leaf_idx, &mut offset);
  let n_rows = offset;

  if n_rows == 0 {
    return Ok(Schedule { workgroups: vec![], n_rows: 0 });
  }

  // Phase 2: atomise
  let mut atoms = Vec::new();
  collect_atoms(&root, target_size, &mut atoms);

  if atoms.is_empty() {
    return Ok(Schedule { workgroups: vec![], n_rows: 0 });
  }

  // Phase 3: greedy merge
  let mut workgroups: Vec<Workgroup> = Vec::new();
  let mut wg_start = atoms[0].0;
  let mut wg_end = atoms[0].0 + atoms[0].1;

  for &(start, size) in &atoms[1..] {
    let current_len = wg_end - wg_start;
    // If adding this atom would exceed target, close the workgroup.
    if current_len >= target_size {
      workgroups.push(Workgroup {
        id: workgroups.len() as u32,
        row_start: wg_start,
        row_end: wg_end,
      });
      wg_start = start;
      wg_end = start + size;
    } else {
      wg_end = start + size;
    }
  }

  // Close the last workgroup.
  // If it is unreasonably small, merge it into the previous one.
  let last_len = wg_end - wg_start;
  if !workgroups.is_empty() && last_len < target_size / 4 {
    // Absorb into previous workgroup.
    workgroups.last_mut().unwrap().row_end = wg_end;
  } else {
    workgroups.push(Workgroup {
      id: workgroups.len() as u32,
      row_start: wg_start,
      row_end: wg_end,
    });
  }

  Ok(Schedule { workgroups, n_rows })
}

#[cfg(test)]
mod tests {
  use super::*;
  use evm_types::proof_tree::LeafProof;
  use evm_types::state::EvmState;

  // ── Helpers ─────────────────────────────────────────────────────────────

  fn leaf(opcode: u8) -> ProofNode {
    ProofNode::Leaf {
      opcode,
      pre_state: EvmState::new(0, 0),
      post_state: EvmState::new(0, 0),
      leaf_proof: LeafProof::placeholder(),
    }
  }

  fn seq(left: ProofNode, right: ProofNode) -> ProofNode {
    ProofNode::Seq { left: Box::new(left), right: Box::new(right) }
  }

  fn branch(cond: ProofNode, taken: ProofNode, not_taken: ProofNode) -> ProofNode {
    ProofNode::Branch {
      cond: Box::new(cond),
      taken: Box::new(taken),
      not_taken: Box::new(not_taken),
    }
  }

  // ── Tests ───────────────────────────────────────────────────────────────

  #[test]
  fn single_leaf() {
    let tree = leaf(0x01);
    let sched = schedule(&tree, &[10], 256).unwrap();
    assert_eq!(sched.n_rows, 10);
    assert_eq!(sched.workgroups.len(), 1);
    assert_eq!(sched.workgroups[0].row_start, 0);
    assert_eq!(sched.workgroups[0].row_end, 10);
  }

  #[test]
  fn two_leaves_fit_one_workgroup() {
    let tree = seq(leaf(0x01), leaf(0x02));
    // 5 + 5 = 10, target=256 → one workgroup
    let sched = schedule(&tree, &[5, 5], 256).unwrap();
    assert_eq!(sched.workgroups.len(), 1);
    assert_eq!(sched.workgroups[0].len(), 10);
  }

  #[test]
  fn two_leaves_split_at_seq() {
    let tree = seq(leaf(0x01), leaf(0x02));
    // 100 + 100 = 200, target=100 → splits at Seq boundary
    let sched = schedule(&tree, &[100, 100], 100).unwrap();
    assert_eq!(sched.workgroups.len(), 2);
    assert_eq!(sched.workgroups[0].row_start, 0);
    assert_eq!(sched.workgroups[0].row_end, 100);
    assert_eq!(sched.workgroups[1].row_start, 100);
    assert_eq!(sched.workgroups[1].row_end, 200);
  }

  #[test]
  fn four_leaves_balanced() {
    // Seq(Seq(L0, L1), Seq(L2, L3))
    let tree = seq(seq(leaf(0x01), leaf(0x02)), seq(leaf(0x03), leaf(0x04)));
    // Each leaf 50 rows, target=100 → 2 workgroups at top Seq boundary
    let sched = schedule(&tree, &[50, 50, 50, 50], 100).unwrap();
    assert_eq!(sched.workgroups.len(), 2);
    assert_eq!(sched.workgroups[0].len(), 100);
    assert_eq!(sched.workgroups[1].len(), 100);
  }

  #[test]
  fn branch_splits_into_three() {
    let tree = branch(leaf(0x01), leaf(0x02), leaf(0x03));
    // 200 + 200 + 200 = 600, target=200
    let sched = schedule(&tree, &[200, 200, 200], 200).unwrap();
    assert_eq!(sched.workgroups.len(), 3);
    for (i, wg) in sched.workgroups.iter().enumerate() {
      assert_eq!(wg.id, i as u32);
      assert_eq!(wg.len(), 200);
    }
  }

  #[test]
  fn small_tail_absorbed() {
    // Seq(big, tiny) where tiny < target/4
    let tree = seq(leaf(0x01), leaf(0x02));
    // 300 + 10 = 310, target=256 → tiny tail absorbed into first
    let sched = schedule(&tree, &[300, 10], 256).unwrap();
    assert_eq!(sched.workgroups.len(), 1);
    assert_eq!(sched.workgroups[0].len(), 310);
  }

  #[test]
  fn small_tail_not_absorbed_when_large_enough() {
    let tree = seq(leaf(0x01), leaf(0x02));
    // 300 + 200 = 500, target=256 → 200 ≥ 64 (256/4), so two workgroups
    let sched = schedule(&tree, &[300, 200], 256).unwrap();
    assert_eq!(sched.workgroups.len(), 2);
  }

  #[test]
  fn many_small_leaves_merge() {
    // 20 leaves of 10 rows each → 200 total, target=100
    // Should get 2 workgroups of ~100 rows each.
    let leaves: Vec<ProofNode> = (0..20).map(|i| leaf(i as u8)).collect();
    let tree = leaves
      .into_iter()
      .reduce(|a, b| seq(a, b))
      .unwrap();
    let row_counts = vec![10usize; 20];
    let sched = schedule(&tree, &row_counts, 100).unwrap();
    assert_eq!(sched.n_rows, 200);
    assert_eq!(sched.workgroups.len(), 2);
    assert_eq!(sched.workgroups[0].len(), 100);
    assert_eq!(sched.workgroups[1].len(), 100);
  }

  #[test]
  fn workgroup_of_lookup() {
    let tree = seq(leaf(0x01), leaf(0x02));
    let sched = schedule(&tree, &[100, 100], 100).unwrap();
    assert_eq!(sched.workgroup_of(0), Some(0));
    assert_eq!(sched.workgroup_of(99), Some(0));
    assert_eq!(sched.workgroup_of(100), Some(1));
    assert_eq!(sched.workgroup_of(199), Some(1));
    assert_eq!(sched.workgroup_of(200), None);
  }

  #[test]
  fn leaf_count_mismatch_err() {
    let tree = seq(leaf(0x01), leaf(0x02));
    let err = schedule(&tree, &[10], 100);
    assert!(err.is_err());
  }

  #[test]
  fn zero_target_err() {
    let tree = leaf(0x01);
    let err = schedule(&tree, &[10], 0);
    assert!(err.is_err());
  }

  #[test]
  fn empty_leaf_rows() {
    // A leaf with 0 rows (e.g. a no-op that produces no trace rows).
    let tree = seq(leaf(0x01), leaf(0x02));
    let sched = schedule(&tree, &[0, 10], 256).unwrap();
    assert_eq!(sched.n_rows, 10);
    assert_eq!(sched.workgroups.len(), 1);
    assert_eq!(sched.workgroups[0].len(), 10);
  }

  #[test]
  fn deeply_nested_tree() {
    // Right-skewed: Seq(Seq(Seq(L0, L1), L2), L3)
    let tree = seq(seq(seq(leaf(0x01), leaf(0x02)), leaf(0x03)), leaf(0x04));
    // 50 each, target=100 → 2 workgroups
    let sched = schedule(&tree, &[50, 50, 50, 50], 100).unwrap();
    assert_eq!(sched.workgroups.len(), 2);
    assert_eq!(sched.workgroups[0].row_start, 0);
    assert_eq!(sched.workgroups[0].row_end, 100);
    assert_eq!(sched.workgroups[1].row_start, 100);
    assert_eq!(sched.workgroups[1].row_end, 200);
  }

  #[test]
  fn realistic_mixed_tree() {
    // Simulate a small EVM execution:
    //   Seq(
    //     Seq(PUSH, PUSH),        -- 2 leaves, 16+16 rows
    //     Branch(
    //       JUMPI,                 -- 1 leaf, 1 row
    //       Seq(ADD, MUL),         -- 2 leaves, 8+8 rows
    //       Seq(SUB, DIV),         -- 2 leaves, 8+24 rows
    //     )
    //   )
    let tree = seq(
      seq(leaf(0x60), leaf(0x60)),
      branch(
        leaf(0x57),
        seq(leaf(0x01), leaf(0x02)),
        seq(leaf(0x03), leaf(0x04)),
      ),
    );
    let row_counts = [16, 16, 1, 8, 8, 8, 24];
    let sched = schedule(&tree, &row_counts, 32).unwrap();
    assert_eq!(sched.n_rows, 81);

    // All rows covered
    let total: u32 = sched.workgroups.iter().map(|wg| wg.len()).sum();
    assert_eq!(total, 81);

    // Workgroups are contiguous
    for pair in sched.workgroups.windows(2) {
      assert_eq!(pair[0].row_end, pair[1].row_start);
    }

    // Each workgroup ID is sequential
    for (i, wg) in sched.workgroups.iter().enumerate() {
      assert_eq!(wg.id, i as u32);
    }
  }
}
