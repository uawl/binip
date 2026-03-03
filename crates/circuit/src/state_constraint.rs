//! State-transition constraints proved inside the ZKP.
//!
//! At every [`ProofNode::Seq`] boundary in the proof tree, the left sub-tree's
//! post-state must equal the right sub-tree's pre-state.  This module encodes
//! those equalities as rows of a **boundary trace** and evaluates polynomial
//! constraints over GF(2^128) that a separate sumcheck instance proves.
//!
//! # What is proved in ZK
//!
//! For each Seq boundary the following are checked:
//!
//! | Field | Constraint (char 2) |
//! |-------|---------------------|
//! | PC | `left_pc + right_pc = 0` |
//! | stack depth | `left_depth + right_depth = 0` |
//! | gas (low 32) | `left_gas_lo + right_gas_lo = 0` |
//! | gas (high 32) | `left_gas_hi + right_gas_hi = 0` |
//! | memory size | `left_mem + right_mem = 0` |
//! | storage count | `left_stor + right_stor = 0` |
//! | jumpdest hash | `left_jd_hash + right_jd_hash = 0` |
//!
//! Each Seq boundary yields 2 rows (8 columns each):
//!
//! - **Row type 0**: `[left_pc, right_pc, left_depth, right_depth, left_gas_lo, right_gas_lo, left_gas_hi, right_gas_hi]`
//! - **Row type 1**: `[left_mem, right_mem, left_stor, right_stor, left_jd_hash, right_jd_hash, 0, 0]`
//!
//! The constraint is the same for both rows:
//!
//! $$C(\text{row}, \beta) = \sum_{k=0}^{3} \beta^k \cdot (\text{col}[2k] + \text{col}[2k+1])$$
//!
//! which equals zero iff all 4 pairs in the row match (by Schwartz-Zippel
//! with the Fiat-Shamir challenge $\beta$).

use evm_types::proof_tree::ProofNode;
use evm_types::state::EvmState;
use field::{FieldElem, GF2_128};
use poly::MlePoly;

/// Number of columns in a boundary row (matches main trace width).
pub const BOUNDARY_COLS: usize = 8;

/// A single row in the boundary trace.
///
/// Columns are organised as 4 `(left, right)` pairs. The constraint
/// forces each pair to be equal.
#[derive(Debug, Clone, Copy)]
pub struct BoundaryRow {
  pub cols: [GF2_128; BOUNDARY_COLS],
}

/// Column-major boundary trace table.
#[derive(Debug, Clone)]
pub struct BoundaryTraceTable {
  pub columns: Vec<Vec<GF2_128>>,
  pub n_rows: usize,
}

impl BoundaryTraceTable {
  /// Build from a slice of [`BoundaryRow`]s.
  pub fn from_rows(rows: &[BoundaryRow]) -> Self {
    let n = rows.len();
    let mut columns = vec![Vec::with_capacity(n); BOUNDARY_COLS];
    for row in rows {
      for (c, val) in row.cols.iter().enumerate() {
        columns[c].push(*val);
      }
    }
    BoundaryTraceTable { columns, n_rows: n }
  }

  /// Evaluate the boundary constraint MLE.
  ///
  /// Each row's constraint is:
  /// $$C = \sum_{k=0}^{3} \beta^k \cdot (\text{col}[2k] + \text{col}[2k+1])$$
  ///
  /// Returns zero-sum iff all boundary equalities hold.
  pub fn constraint_mle(&self, beta: GF2_128) -> MlePoly {
    let n_vars = n_vars_for(self.n_rows.max(1));
    let padded_len = 1usize << n_vars;
    let mut evals = Vec::with_capacity(padded_len);

    let mut beta_powers = [GF2_128::one(); 4];
    for k in 1..4 {
      beta_powers[k] = beta_powers[k - 1] * beta;
    }

    for i in 0..self.n_rows {
      let mut val = GF2_128::zero();
      for k in 0..4 {
        let left = self.columns[2 * k][i];
        let right = self.columns[2 * k + 1][i];
        val = val + beta_powers[k] * (left + right);
      }
      evals.push(val);
    }

    evals.resize(padded_len, GF2_128::zero());
    MlePoly::new(evals)
  }
}

/// Minimum number of variables so that `2^n >= len`.
fn n_vars_for(len: usize) -> usize {
  if len <= 1 {
    return 0;
  }
  (usize::BITS - (len - 1).leading_zeros()) as usize
}

// ─────────────────────────────────────────────────────────────────────────────
// Boundary extraction from the proof tree
// ─────────────────────────────────────────────────────────────────────────────

/// Extract all Seq boundary rows from a [`ProofNode`] tree.
///
/// Each `Seq { left, right }` produces 2 rows: one for PC/depth/gas,
/// one for memory/storage/jumpdest-hash.
pub fn extract_boundaries(tree: &ProofNode) -> Vec<BoundaryRow> {
  let mut rows = Vec::new();
  walk_boundaries(tree, &mut rows);
  rows
}

fn walk_boundaries(node: &ProofNode, rows: &mut Vec<BoundaryRow>) {
  match node {
    ProofNode::Leaf { .. } => {}
    ProofNode::Seq { left, right } => {
      // Recurse into children first.
      walk_boundaries(left, rows);
      walk_boundaries(right, rows);

      // Extract boundary between left's post-state and right's pre-state.
      let left_post = rightmost_post(left);
      let right_pre = leftmost_pre(right);
      emit_boundary_rows(left_post, right_pre, rows);
    }
    ProofNode::Branch {
      cond,
      taken,
      not_taken,
    } => {
      walk_boundaries(cond, rows);
      walk_boundaries(taken, rows);
      walk_boundaries(not_taken, rows);
    }
  }
}

fn emit_boundary_rows(left: &EvmState, right: &EvmState, rows: &mut Vec<BoundaryRow>) {
  let f = |v: u32| GF2_128::from(v as u64);

  let left_gas_lo = left.gas as u32;
  let left_gas_hi = (left.gas >> 32) as u32;
  let right_gas_lo = right.gas as u32;
  let right_gas_hi = (right.gas >> 32) as u32;

  // Row 0: PC, stack depth, gas (lo + hi)
  rows.push(BoundaryRow {
    cols: [
      f(left.pc),
      f(right.pc),
      f(left.stack.len() as u32),
      f(right.stack.len() as u32),
      f(left_gas_lo),
      f(right_gas_lo),
      f(left_gas_hi),
      f(right_gas_hi),
    ],
  });

  // Row 1: memory size, storage count, jumpdest table hash (128-bit)
  let left_jd_hash = jumpdest_hash(&left.jumpdest_table);
  let right_jd_hash = jumpdest_hash(&right.jumpdest_table);

  rows.push(BoundaryRow {
    cols: [
      f(left.memory.len() as u32),
      f(right.memory.len() as u32),
      f(left.storage.len() as u32),
      f(right.storage.len() as u32),
      left_jd_hash,
      right_jd_hash,
      GF2_128::zero(),
      GF2_128::zero(),
    ],
  });
}

/// Hash a jumpdest table into a GF(2^128) element for concise comparison.
///
/// Two tables are equal iff their hashes match.  Using the full 128-bit
/// Blake3 output provides ~2^64 collision resistance (birthday bound),
/// eliminating the old 32-bit truncation weakness (H-2).
fn jumpdest_hash(table: &std::collections::BTreeSet<u32>) -> GF2_128 {
  let mut hasher = blake3::Hasher::new();
  for &offset in table {
    hasher.update(&offset.to_le_bytes());
  }
  let hash = hasher.finalize();
  let bytes = hash.as_bytes();
  let lo = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
  let hi = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
  GF2_128::new(lo, hi)
}

/// Leftmost pre-state in a subtree.
fn leftmost_pre(node: &ProofNode) -> &EvmState {
  match node {
    ProofNode::Leaf { pre_state, .. } => pre_state,
    ProofNode::Seq { left, .. } => leftmost_pre(left),
    ProofNode::Branch { cond, .. } => leftmost_pre(cond),
  }
}

/// Rightmost post-state in a subtree.
fn rightmost_post(node: &ProofNode) -> &EvmState {
  match node {
    ProofNode::Leaf { post_state, .. } => post_state,
    ProofNode::Seq { right, .. } => rightmost_post(right),
    ProofNode::Branch { taken, .. } => rightmost_post(taken),
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use super::*;
  use evm_types::proof_tree::LeafProof;
  use evm_types::state::EvmState;
  use revm::primitives::U256;

  fn st_gas(stack: Vec<U256>, pc: u32, gas: u64) -> EvmState {
    let mut s = EvmState::with_stack(stack, pc);
    s.gas = gas;
    s
  }

  fn leaf(opcode: u8, pre: EvmState, post: EvmState) -> ProofNode {
    ProofNode::Leaf {
      opcode,
      pre_state: pre,
      post_state: post,
      leaf_proof: LeafProof::placeholder(),
    }
  }

  #[test]
  fn single_leaf_no_boundaries() {
    let node = leaf(
      0x01,
      st_gas(vec![U256::from(1u64), U256::from(2u64)], 0, 1000),
      st_gas(vec![U256::from(3u64)], 1, 997),
    );
    let rows = extract_boundaries(&node);
    assert!(rows.is_empty());
  }

  #[test]
  fn seq_produces_two_boundary_rows() {
    let left = leaf(
      0x01,
      st_gas(vec![U256::from(1u64), U256::from(2u64)], 0, 1000),
      st_gas(vec![U256::from(3u64)], 1, 997),
    );
    let right = leaf(
      0x50, // POP
      st_gas(vec![U256::from(3u64)], 1, 997),
      st_gas(vec![], 2, 995),
    );
    let seq = ProofNode::Seq {
      left: Box::new(left),
      right: Box::new(right),
    };
    let rows = extract_boundaries(&seq);
    assert_eq!(rows.len(), 2); // 2 rows per Seq boundary
  }

  #[test]
  fn consistent_boundary_has_zero_constraint() {
    // left.post == right.pre → all pairs match → constraint = 0
    let post_state = st_gas(vec![U256::from(3u64)], 1, 997);
    let pre_state = post_state.clone();

    let left = leaf(
      0x01,
      st_gas(vec![U256::from(1u64), U256::from(2u64)], 0, 1000),
      post_state,
    );
    let right = leaf(0x50, pre_state, st_gas(vec![], 2, 995));
    let seq = ProofNode::Seq {
      left: Box::new(left),
      right: Box::new(right),
    };

    let rows = extract_boundaries(&seq);
    let table = BoundaryTraceTable::from_rows(&rows);
    let beta = GF2_128::from(0x1234_5678u64);
    let mle = table.constraint_mle(beta);

    assert!(
      mle.sum().is_zero(),
      "consistent boundary should sum to zero"
    );
  }

  #[test]
  fn inconsistent_pc_has_nonzero_constraint() {
    // left.post.pc = 1, right.pre.pc = 5 → mismatch
    let left = leaf(
      0x01,
      st_gas(vec![U256::from(1u64), U256::from(2u64)], 0, 1000),
      st_gas(vec![U256::from(3u64)], 1, 997),
    );
    let right = leaf(
      0x50,
      st_gas(vec![U256::from(3u64)], 5, 997), // PC mismatch
      st_gas(vec![], 6, 995),
    );
    let seq = ProofNode::Seq {
      left: Box::new(left),
      right: Box::new(right),
    };

    let rows = extract_boundaries(&seq);
    let table = BoundaryTraceTable::from_rows(&rows);
    let beta = GF2_128::from(0x1234_5678u64);
    let mle = table.constraint_mle(beta);

    assert!(
      !mle.sum().is_zero(),
      "PC mismatch should produce non-zero constraint sum"
    );
  }

  #[test]
  fn inconsistent_gas_has_nonzero_constraint() {
    let left = leaf(
      0x01,
      st_gas(vec![U256::from(1u64), U256::from(2u64)], 0, 1000),
      st_gas(vec![U256::from(3u64)], 1, 997),
    );
    let right = leaf(
      0x50,
      st_gas(vec![U256::from(3u64)], 1, 990), // gas mismatch
      st_gas(vec![], 2, 988),
    );
    let seq = ProofNode::Seq {
      left: Box::new(left),
      right: Box::new(right),
    };

    let rows = extract_boundaries(&seq);
    let table = BoundaryTraceTable::from_rows(&rows);
    let beta = GF2_128::from(0xABCDu64);
    let mle = table.constraint_mle(beta);

    assert!(!mle.sum().is_zero());
  }

  #[test]
  fn nested_seq_multiple_boundaries() {
    // Seq(Seq(A, B), C) has 2 Seq nodes → 2 boundaries → 4 rows
    let a = leaf(
      0x01,
      st_gas(vec![U256::from(1u64), U256::from(2u64)], 0, 1000),
      st_gas(vec![U256::from(3u64)], 1, 997),
    );
    let b = leaf(
      0x50, // POP: consumes 1 item
      st_gas(vec![U256::from(3u64)], 1, 997),
      st_gas(vec![], 2, 995),
    );
    let c = leaf(
      0x00, // STOP: no stack change
      st_gas(vec![], 2, 995),
      st_gas(vec![], 2, 995),
    );

    let inner = ProofNode::Seq {
      left: Box::new(a),
      right: Box::new(b),
    };
    let outer = ProofNode::Seq {
      left: Box::new(inner),
      right: Box::new(c),
    };

    let rows = extract_boundaries(&outer);
    assert_eq!(rows.len(), 4); // 2 Seq boundaries × 2 rows each

    let table = BoundaryTraceTable::from_rows(&rows);
    let beta = GF2_128::from(999u64);
    let mle = table.constraint_mle(beta);
    assert!(mle.sum().is_zero(), "all boundaries consistent");
  }
}
