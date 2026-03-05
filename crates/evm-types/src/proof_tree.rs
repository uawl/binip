use crate::state::EvmState;
use revm::primitives::{Address, B256};

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
#[derive(Debug, Clone, bincode::Encode, bincode::Decode)]
pub struct TypeCert {
  /// Blake3 Merkle root over the serialised tree shape.
  pub root_hash: [u8; 32],
  /// Blake3 hash committing to the actual EVM state values (stack,
  /// gas, memory sizes, storage keys/values) in every `Leaf` node.
  /// This binds the witness-observable state into the Fiat-Shamir
  /// transcript so the verifier can run value-level consistency
  /// checks on an authenticated tree.
  pub state_hash: [u8; 32],
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

  /// A sub-call or contract creation (CALL / DELEGATECALL / STATICCALL / CREATE / CREATE2).
  ///
  /// The parent frame's CALL opcode consumes/produces stack values; the `inner`
  /// tree proves the callee’s execution trace. `pre_state` and `post_state`
  /// are the **parent’s** EVM state before and after the call opcode.
  Call {
    /// CALL / DELEGATECALL / STATICCALL / CREATE / CREATE2 opcode byte.
    opcode: u8,
    /// Address of the contract whose bytecode executes inside the call.
    callee: Address,
    /// Caller address (msg.sender inside the callee).
    caller: Address,
    /// Parent state before the CALL opcode.
    pre_state: EvmState,
    /// Parent state after the CALL opcode returns.
    post_state: EvmState,
    /// Sub-call execution proof tree (callee’s trace).
    inner: Box<ProofNode>,
    /// Whether the sub-call succeeded.
    success: bool,
  },

  /// Transaction boundary within a block.
  ///
  /// Wraps one transaction's full execution proof.  Between consecutive
  /// `TxBoundary` nodes (joined by `Seq`), the verifier checks:
  /// - Persistent storage continuity (state root transition).
  /// - Transient storage is reset to empty (EIP-1153).
  /// - Gas accounting: `gas_used` ≤ `gas_limit`.
  /// - Nonce increment for the sender.
  TxBoundary {
    /// Transaction index within the block (0-based).
    tx_index: u32,
    /// Transaction hash (keccak256 of the RLP-encoded signed tx).
    tx_hash: B256,
    /// Gas limit for this transaction.
    gas_limit: u64,
    /// Gas actually consumed.
    gas_used: u64,
    /// Whether the transaction's top-level call succeeded.
    success: bool,
    /// EVM state before the first opcode of this transaction.
    pre_state: EvmState,
    /// EVM state after the last opcode of this transaction.
    post_state: EvmState,
    /// Proof tree for this transaction's execution.
    inner: Box<ProofNode>,
  },

  /// Block boundary — wraps all transactions in a single Ethereum block.
  ///
  /// The `inner` tree is typically a balanced `Seq` tree of `TxBoundary`
  /// nodes produced by [`build_block_witness`].  Between consecutive
  /// `BlockBoundary` nodes (joined by `Seq`) the verifier checks:
  /// - State root continuity (`state_root_post` of block N == `state_root_pre` of block N+1).
  /// - Block number increments by exactly 1.
  /// - Timestamp is non-decreasing.
  /// - Cumulative `gas_used` ≤ `gas_limit`.
  BlockBoundary {
    /// Ethereum block number.
    block_number: u64,
    /// keccak256(RLP(block_header)).
    block_hash: B256,
    /// Parent block hash (links to previous block proof).
    parent_hash: B256,
    /// Unix timestamp of the block.
    timestamp: u64,
    /// Miner/validator address (coinbase).
    coinbase: Address,
    /// Block-level gas limit.
    gas_limit: u64,
    /// Total gas consumed by all transactions in this block.
    gas_used: u64,
    /// State trie root *before* executing the first transaction.
    state_root_pre: B256,
    /// State trie root *after* executing the last transaction.
    state_root_post: B256,
    /// EVM state before the block's first opcode.
    pre_state: EvmState,
    /// EVM state after the block's last opcode.
    post_state: EvmState,
    /// Proof tree for all transactions in this block.
    inner: Box<ProofNode>,
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
      ProofNode::Call { inner, .. } => 1 + inner.leaf_count(),
      ProofNode::TxBoundary { inner, .. } => inner.leaf_count(),
      ProofNode::BlockBoundary { inner, .. } => inner.leaf_count(),
    }
  }

  /// Returns the opcode of this node if it is a `Leaf` or `Call`.
  pub fn opcode(&self) -> Option<u8> {
    match self {
      ProofNode::Leaf { opcode, .. } | ProofNode::Call { opcode, .. } => Some(*opcode),
      _ => None,
    }
  }
}
