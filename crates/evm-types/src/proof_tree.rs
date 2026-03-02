use crate::state::EvmState;

/// Opaque ZK arithmetic sub-proof for a single EVM opcode step.
///
/// The type checker treats `proof_bytes` as a black box; only the
/// `commitment` is inspected when building the [`TypeCert`] Merkle tree.
#[derive(Debug, Clone)]
pub struct LeafProof {
  /// Blake3 commitment over the full ZK sub-proof bytes.
  pub commitment: [u8; 32],
  /// Serialised proof messages (backend-specific; empty until ZK layer).
  pub proof_bytes: Vec<u8>,
}

impl LeafProof {
  /// Placeholder used during tree construction before ZK proofs are generated.
  pub fn placeholder() -> Self {
    Self {
      commitment: [0u8; 32],
      proof_bytes: Vec::new(),
    }
  }
}

/// A structural certificate over a [`ProofNode`] tree.
///
/// Produced by [`crate::type_check::build_cert`] after a successful
/// [`crate::type_check::type_check`]. Commits to the *shape* of the
/// derivation tree; arithmetic correctness lives in the [`LeafProof`]s.
#[derive(Debug, Clone)]
pub struct TypeCert {
  /// Blake3 Merkle root over the serialised tree shape.
  pub root_hash: [u8; 32],
  /// Total number of `Leaf` nodes in the tree.
  pub leaf_count: usize,
}

/// Formal derivation tree for EVM execution.
///
/// Each node represents a portion of EVM execution with a well-typed
/// pre- and post-state. The tree mirrors the control-flow structure:
///
/// - Sequential steps compose with [`ProofNode::Seq`].
/// - Conditional branches (JUMPI) become [`ProofNode::Branch`].
/// - Individual opcode steps are [`ProofNode::Leaf`].
///
/// **Type invariants** (enforced by [`crate::type_check::type_check`]):
/// - Leaf: `post_state.stack.len()` equals `pre_state.stack.len() + delta(opcode)`.
/// - Seq: `left` post-depth equals `right` pre-depth.
/// - Branch: `taken` post-depth equals `not_taken` post-depth.
#[derive(Debug, Clone)]
pub enum ProofNode {
  /// A single EVM opcode transition.
  Leaf {
    /// EVM opcode byte (e.g. `0x01` = ADD).
    opcode: u8,
    /// Concrete EVM state *before* executing `opcode`.
    pre_state: EvmState,
    /// Concrete EVM state *after* executing `opcode`.
    post_state: EvmState,
    /// ZK proof of arithmetic correctness for this step.
    leaf_proof: LeafProof,
  },

  /// Sequential composition: `left` executes fully before `right`.
  ///
  /// Invariant: `left.post_state.stack.len() == right.pre_state.stack.len()`.
  Seq {
    left: Box<ProofNode>,
    right: Box<ProofNode>,
  },

  /// Conditional branch modelling a JUMPI followed by two execution paths.
  ///
  /// `cond` is the sub-derivation up to and including the JUMPI step.
  /// `taken` covers the path where the condition != 0 (jump taken).
  /// `not_taken` covers the path where the condition == 0 (fall-through).
  Branch {
    cond: Box<ProofNode>,
    taken: Box<ProofNode>,
    not_taken: Box<ProofNode>,
  },
}

impl ProofNode {
  /// Returns the number of [`Leaf`] nodes in this tree.
  pub fn leaf_count(&self) -> usize {
    match self {
      ProofNode::Leaf { .. } => 1,
      ProofNode::Seq { left, right } => left.leaf_count() + right.leaf_count(),
      ProofNode::Branch {
        cond,
        taken,
        not_taken,
      } => cond.leaf_count() + taken.leaf_count() + not_taken.leaf_count(),
    }
  }

  /// Returns the opcode of this node if it is a `Leaf`.
  pub fn opcode(&self) -> Option<u8> {
    match self {
      ProofNode::Leaf { opcode, .. } => Some(*opcode),
      _ => None,
    }
  }
}
