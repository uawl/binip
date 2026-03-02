pub mod consistency;
pub mod opcode;
pub mod proof_tree;
pub mod state;
pub mod type_check;

pub use consistency::{ConsistencyError, consistency_check, consistency_check_all};
pub use opcode::{StackEffect, stack_effect};
pub use proof_tree::{LeafProof, ProofNode, TypeCert};
pub use state::{EvmState, EvmStateType};
pub use type_check::{TypeError, build_cert, type_check};

#[cfg(test)]
mod tests {
  use revm::primitives::U256;

  use crate::{
    opcode,
    proof_tree::{LeafProof, ProofNode},
    state::EvmState,
    type_check::{self, TypeError},
  };

  fn st(stack: Vec<U256>, pc: u32) -> EvmState {
    EvmState::with_stack(stack, pc)
  }

  /// State with explicit memory bytes.
  fn st_mem(stack: Vec<U256>, pc: u32, mem_bytes: usize) -> EvmState {
    let mut s = EvmState::with_stack(stack, pc);
    s.memory = vec![0u8; mem_bytes];
    s
  }

  /// State with explicit storage entries.
  fn st_stor(stack: Vec<U256>, pc: u32, keys: &[u64]) -> EvmState {
    let mut s = EvmState::with_stack(stack, pc);
    for &k in keys {
      s.storage.insert(U256::from(k), U256::ZERO);
    }
    s
  }

  /// State with jumpdest table.
  fn st_jmp(stack: Vec<U256>, pc: u32, jumpdests: &[u32]) -> EvmState {
    EvmState::with_jumpdests(stack, pc, jumpdests.iter().copied().collect())
  }

  fn leaf(opcode: u8, pre: EvmState, post: EvmState) -> ProofNode {
    ProofNode::Leaf {
      opcode,
      pre_state: pre,
      post_state: post,
      leaf_proof: LeafProof::placeholder(),
    }
  }

  // ── Leaf: ADD ─────────────────────────────────────────────────────────────

  #[test]
  fn leaf_add_ok() {
    let node = leaf(
      opcode::ADD,
      st(vec![U256::from(3u64), U256::from(5u64)], 0),
      st(vec![U256::from(8u64)], 1),
    );
    let (pre, post) = type_check::type_check(&node).unwrap();
    assert_eq!(pre.stack_depth, 2);
    assert_eq!(post.stack_depth, 1);
  }

  // ── Leaf: PUSH1 ───────────────────────────────────────────────────────────

  #[test]
  fn leaf_push1_ok() {
    let node = leaf(
      0x60, // PUSH1
      st(vec![], 0),
      st(vec![U256::from(42u64)], 2),
    );
    let (pre, post) = type_check::type_check(&node).unwrap();
    assert_eq!(pre.stack_depth, 0);
    assert_eq!(post.stack_depth, 1);
  }

  // ── Leaf: DUP1 ────────────────────────────────────────────────────────────

  #[test]
  fn leaf_dup1_ok() {
    let v = U256::from(99u64);
    let node = leaf(
      0x80, // DUP1
      st(vec![v], 0),
      st(vec![v, v], 1),
    );
    let (pre, post) = type_check::type_check(&node).unwrap();
    assert_eq!(pre.stack_depth, 1);
    assert_eq!(post.stack_depth, 2);
  }

  // DUP1 on empty stack → underflow
  #[test]
  fn leaf_dup1_underflow() {
    let node = leaf(0x80, st(vec![], 0), st(vec![U256::ZERO], 1));
    let err = type_check::type_check(&node).unwrap_err();
    assert!(matches!(err, TypeError::StackUnderflow { .. }));
  }

  // ── Leaf: SWAP1 ───────────────────────────────────────────────────────────

  #[test]
  fn leaf_swap1_ok() {
    let a = U256::from(1u64);
    let b = U256::from(2u64);
    let node = leaf(
      0x90, // SWAP1
      st(vec![b, a], 0),
      st(vec![a, b], 1),
    );
    let (pre, post) = type_check::type_check(&node).unwrap();
    assert_eq!(pre.stack_depth, 2);
    assert_eq!(post.stack_depth, 2);
  }

  // ── Leaf: stack underflow for ADD ─────────────────────────────────────────

  #[test]
  fn leaf_add_underflow() {
    let node = leaf(
      opcode::ADD,
      st(vec![U256::from(1u64)], 0), // only 1 item on stack, ADD needs 2
      st(vec![], 1),
    );
    let err = type_check::type_check(&node).unwrap_err();
    assert!(matches!(
      err,
      TypeError::StackUnderflow { opcode: 0x01, .. }
    ));
  }

  // ── Leaf: post-state depth mismatch ───────────────────────────────────────

  #[test]
  fn leaf_add_wrong_post_depth() {
    let node = leaf(
      opcode::ADD,
      st(vec![U256::from(1u64), U256::from(2u64)], 0),
      st(vec![U256::from(3u64), U256::from(0u64)], 1), // wrong: should be depth 1
    );
    let err = type_check::type_check(&node).unwrap_err();
    assert!(matches!(
      err,
      TypeError::LeafDepthMismatch {
        expected: 1,
        actual: 2,
        ..
      }
    ));
  }

  // ── Leaf: undefined opcode ────────────────────────────────────────────────

  #[test]
  fn leaf_unknown_opcode() {
    let node = leaf(0x0c, st(vec![], 0), st(vec![], 1));
    let err = type_check::type_check(&node).unwrap_err();
    assert!(matches!(err, TypeError::UnknownOpcode { opcode: 0x0c }));
  }

  // ── Seq: ADD then MUL ─────────────────────────────────────────────────────

  #[test]
  fn seq_add_mul_ok() {
    // [a, b, c] → ADD → [a+b, c] → MUL → [(a+b)*c]
    let a = U256::from(2u64);
    let b = U256::from(3u64);
    let c = U256::from(4u64);
    let ab = U256::from(5u64);
    let abc = U256::from(20u64);

    let mid = st(vec![ab, c], 1);
    let seq = ProofNode::Seq {
      left: Box::new(leaf(opcode::ADD, st(vec![a, b, c], 0), mid.clone())),
      right: Box::new(leaf(opcode::MUL, mid, st(vec![abc], 2))),
    };
    let (pre, post) = type_check::type_check(&seq).unwrap();
    assert_eq!(pre.stack_depth, 3);
    assert_eq!(post.stack_depth, 1);
  }

  // ── Seq: depth mismatch at boundary ───────────────────────────────────────

  #[test]
  fn seq_boundary_mismatch() {
    // Left ends at depth 1, right starts at depth 2 → error
    let seq = ProofNode::Seq {
      left: Box::new(leaf(
        opcode::ADD,
        st(vec![U256::from(1u64), U256::from(2u64)], 0),
        st(vec![U256::from(3u64)], 1), // post depth = 1
      )),
      right: Box::new(leaf(
        opcode::ADD,
        // pre depth = 2 ≠ 1 from left post
        st(vec![U256::from(0u64), U256::from(0u64)], 1),
        st(vec![U256::from(0u64)], 2),
      )),
    };
    let err = type_check::type_check(&seq).unwrap_err();
    assert!(matches!(err, TypeError::SeqMismatch { left: 1, right: 2 }));
  }

  // ── Branch ────────────────────────────────────────────────────────────────

  #[test]
  fn branch_ok() {
    // JUMPI pops 2 (dest, cond), pushes 0 → depth 2→0 for cond leaf
    let cond_leaf = leaf(
      opcode::JUMPI,
      st_jmp(vec![U256::from(100u64), U256::from(1u64)], 0, &[100, 2]),
      st_jmp(vec![], 100, &[100, 2]),
    );

    // Both branches: from empty stack, ADD results in depth 0 again
    // (just POP for simplicity: depth 1 → 0)
    let taken = leaf(
      opcode::POP,
      st(vec![U256::from(1u64)], 100),
      st(vec![], 101),
    );
    let not_taken = leaf(opcode::POP, st(vec![U256::from(0u64)], 2), st(vec![], 3));

    let branch = ProofNode::Branch {
      cond: Box::new(cond_leaf),
      taken: Box::new(taken),
      not_taken: Box::new(not_taken),
    };
    let (pre, post) = type_check::type_check(&branch).unwrap();
    assert_eq!(pre.stack_depth, 2);
    assert_eq!(post.stack_depth, 0);
  }

  #[test]
  fn branch_path_type_mismatch() {
    let cond_leaf = leaf(
      opcode::JUMPI,
      st_jmp(vec![U256::from(100u64), U256::from(1u64)], 0, &[100, 2]),
      st_jmp(vec![], 100, &[100, 2]),
    );
    // taken ends at depth 1, not_taken ends at depth 0 → error
    let taken = leaf(
      0x60, // PUSH1 → depth 0+1=1
      st(vec![], 100),
      st(vec![U256::from(1u64)], 102),
    );
    let not_taken = leaf(opcode::JUMPDEST, st(vec![], 2), st(vec![], 3));
    let branch = ProofNode::Branch {
      cond: Box::new(cond_leaf),
      taken: Box::new(taken),
      not_taken: Box::new(not_taken),
    };
    let err = type_check::type_check(&branch).unwrap_err();
    assert!(matches!(
      err,
      TypeError::BranchTypeMismatch {
        taken: 1,
        not_taken: 0
      }
    ));
  }

  // ── TypeCert / build_cert ─────────────────────────────────────────────────

  #[test]
  fn build_cert_ok() {
    let node = leaf(
      opcode::ADD,
      st(vec![U256::from(1u64), U256::from(2u64)], 0),
      st(vec![U256::from(3u64)], 1),
    );
    let cert = type_check::build_cert(&node).unwrap();
    assert_eq!(cert.leaf_count, 1);
    // root_hash must be non-zero (Blake3 of non-empty input)
    assert_ne!(cert.root_hash, [0u8; 32]);
  }

  // ── Memory: can only grow ─────────────────────────────────────────────────

  #[test]
  fn memory_grows_mstore_ok() {
    // MSTORE: pops 2 (offset, value), pushes 0 → depth 2→0
    // Memory grows from 0 to 32 bytes.
    let node = leaf(
      opcode::MSTORE,
      st_mem(vec![U256::from(0u64), U256::from(42u64)], 0, 0),
      st_mem(vec![], 1, 32),
    );
    let (pre, post) = type_check::type_check(&node).unwrap();
    assert_eq!(pre.stack_depth, 2);
    assert_eq!(post.stack_depth, 0);
    assert_eq!(pre.memory_size, 0);
    assert_eq!(post.memory_size, 32);
  }

  #[test]
  fn memory_shrink_error() {
    // Memory goes from 64 bytes to 32 bytes → MemoryShrink error.
    let node = leaf(
      opcode::MSTORE,
      st_mem(vec![U256::from(0u64), U256::from(42u64)], 0, 64),
      st_mem(vec![], 1, 32),
    );
    let err = type_check::type_check(&node).unwrap_err();
    assert!(matches!(
      err,
      TypeError::MemoryShrink {
        pre: 64,
        post: 32,
        ..
      }
    ));
  }

  #[test]
  fn memory_stays_same_ok() {
    // MLOAD: pops 1, pushes 1 → depth stays same, memory stays same.
    let node = leaf(
      opcode::MLOAD,
      st_mem(vec![U256::from(0u64)], 0, 32),
      st_mem(vec![U256::from(99u64)], 1, 32),
    );
    let (pre, post) = type_check::type_check(&node).unwrap();
    assert_eq!(pre.memory_size, 32);
    assert_eq!(post.memory_size, 32);
  }

  // ── Storage: touched count can only grow ──────────────────────────────────

  #[test]
  fn storage_sload_ok() {
    // SLOAD: pops 1 (key), pushes 1 (value) → depth stays same.
    // Storage grows from 0 keys to 1 key.
    let node = leaf(
      opcode::SLOAD,
      st(vec![U256::from(1u64)], 0),
      st_stor(vec![U256::from(99u64)], 1, &[1]),
    );
    let (pre, post) = type_check::type_check(&node).unwrap();
    assert_eq!(pre.stack_depth, 1);
    assert_eq!(post.stack_depth, 1);
    assert_eq!(pre.storage_touched, 0);
    assert_eq!(post.storage_touched, 1);
  }

  #[test]
  fn storage_shrink_error() {
    // Storage goes from 2 keys to 1 key → StorageShrink error.
    let node = leaf(
      opcode::SSTORE,
      st_stor(vec![U256::from(1u64), U256::from(99u64)], 0, &[1, 2]),
      st_stor(vec![], 1, &[1]),
    );
    let err = type_check::type_check(&node).unwrap_err();
    assert!(matches!(
      err,
      TypeError::StorageShrink {
        pre: 2,
        post: 1,
        ..
      }
    ));
  }

  #[test]
  fn storage_sstore_grows_ok() {
    // SSTORE: pops 2 (key, value), pushes 0 → depth decreases.
    // Storage grows from 0 keys to 1 key.
    let node = leaf(
      opcode::SSTORE,
      st(vec![U256::from(1u64), U256::from(42u64)], 0),
      st_stor(vec![], 1, &[1]),
    );
    let (pre, post) = type_check::type_check(&node).unwrap();
    assert_eq!(pre.storage_touched, 0);
    assert_eq!(post.storage_touched, 1);
  }

  // ── Jump destination validation ───────────────────────────────────────────

  #[test]
  fn jump_valid_dest_ok() {
    // JUMP: pops 1 (dest), pushes 0.
    // target pc=10 is in the jumpdest table → ok.
    let node = leaf(
      opcode::JUMP,
      st_jmp(vec![U256::from(10u64)], 0, &[10, 20, 30]),
      st_jmp(vec![], 10, &[10, 20, 30]),
    );
    let (pre, post) = type_check::type_check(&node).unwrap();
    assert_eq!(pre.stack_depth, 1);
    assert_eq!(post.stack_depth, 0);
    assert_eq!(post.pc, 10);
  }

  #[test]
  fn jump_invalid_dest_error() {
    // JUMP to pc=15 which is not in jumpdest table [10, 20, 30].
    let node = leaf(
      opcode::JUMP,
      st_jmp(vec![U256::from(15u64)], 0, &[10, 20, 30]),
      st_jmp(vec![], 15, &[10, 20, 30]),
    );
    let err = type_check::type_check(&node).unwrap_err();
    assert!(matches!(err, TypeError::InvalidJumpDest { target: 15, .. }));
  }

  #[test]
  fn jumpi_valid_dest_ok() {
    // JUMPI: pops 2 (dest, cond), pushes 0. Target 20 is in table.
    let node = leaf(
      opcode::JUMPI,
      st_jmp(vec![U256::from(20u64), U256::from(1u64)], 0, &[10, 20]),
      st_jmp(vec![], 20, &[10, 20]),
    );
    let (pre, post) = type_check::type_check(&node).unwrap();
    assert_eq!(pre.stack_depth, 2);
    assert_eq!(post.stack_depth, 0);
    assert_eq!(post.pc, 20);
  }

  #[test]
  fn jumpi_invalid_dest_error() {
    // JUMPI to pc=99 which is not in jumpdest table [10, 20].
    let node = leaf(
      opcode::JUMPI,
      st_jmp(vec![U256::from(99u64), U256::from(1u64)], 0, &[10, 20]),
      st_jmp(vec![], 99, &[10, 20]),
    );
    let err = type_check::type_check(&node).unwrap_err();
    assert!(matches!(err, TypeError::InvalidJumpDest { target: 99, .. }));
  }

  #[test]
  fn jumpdest_noop_ok() {
    // JUMPDEST: pops 0, pushes 0. No jump target to validate.
    let node = leaf(
      opcode::JUMPDEST,
      st(vec![U256::from(1u64)], 5),
      st(vec![U256::from(1u64)], 6),
    );
    let (pre, post) = type_check::type_check(&node).unwrap();
    assert_eq!(pre.stack_depth, 1);
    assert_eq!(post.stack_depth, 1);
  }

  // ── Seq with memory continuity ────────────────────────────────────────────

  #[test]
  fn seq_memory_continuity_ok() {
    // MSTORE then MLOAD: memory grows from 0→32 in the first leaf,
    // stays 32 in the second leaf. Seq should check full type continuity.
    let mid = st_mem(vec![], 1, 32);
    let seq = ProofNode::Seq {
      left: Box::new(leaf(
        opcode::MSTORE,
        st_mem(vec![U256::from(0u64), U256::from(42u64)], 0, 0),
        mid.clone(),
      )),
      right: Box::new(leaf(
        0x60, // PUSH1: pops 0, pushes 1
        st_mem(vec![], 1, 32),
        st_mem(vec![U256::from(0u64)], 3, 32),
      )),
    };
    let (pre, post) = type_check::type_check(&seq).unwrap();
    assert_eq!(pre.memory_size, 0);
    assert_eq!(post.memory_size, 32);
  }

  // ── build_cert includes memory/pc in shape ────────────────────────────────

  #[test]
  fn cert_differs_with_memory() {
    // Two structurally identical ADD leaves but different memory sizes
    // should produce different certs.
    let node_a = leaf(
      opcode::ADD,
      st_mem(vec![U256::from(1u64), U256::from(2u64)], 0, 0),
      st_mem(vec![U256::from(3u64)], 1, 0),
    );
    let node_b = leaf(
      opcode::ADD,
      st_mem(vec![U256::from(1u64), U256::from(2u64)], 0, 32),
      st_mem(vec![U256::from(3u64)], 1, 32),
    );
    let cert_a = type_check::build_cert(&node_a).unwrap();
    let cert_b = type_check::build_cert(&node_b).unwrap();
    assert_ne!(cert_a.root_hash, cert_b.root_hash);
  }
}
