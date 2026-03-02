pub mod opcode;
pub mod proof_tree;
pub mod state;
pub mod type_check;

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
      st(vec![U256::from(100u64), U256::from(1u64)], 0),
      st(vec![], 1),
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
      st(vec![U256::from(100u64), U256::from(1u64)], 0),
      st(vec![], 1),
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
}
