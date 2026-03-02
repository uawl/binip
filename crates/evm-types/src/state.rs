use std::collections::{BTreeSet, HashMap};

use revm::primitives::U256;

/// Concrete EVM execution state at a single step boundary.
///
/// Stored as the prover's witness inside each [`crate::proof_tree::ProofNode::Leaf`].
/// The type checker only inspects `stack.len()` (stack depth); the actual
/// U256 values are consumed by the ZK sub-proof.
///
/// ## Storage model (Advice Witness pattern)
/// Rather than copying the full Merkle-Patricia trie, `storage` and
/// `transient_storage` contain only the **slots accessed during this execution
/// segment**.  The circuit verifies read/write consistency via separate Merkle
/// inclusion proofs; `EvmState` carries the concrete pre/post values the
/// prover claims.
#[derive(Debug, Clone)]
pub struct EvmState {
  /// Operand stack (EVM spec: max 1024 elements).
  pub stack: Vec<U256>,
  /// Linear memory (byte array; length is always a multiple of 32).
  pub memory: Vec<u8>,
  /// Program counter pointing to the current opcode byte.
  pub pc: u32,
  /// Remaining gas budget.
  pub gas: u64,
  /// Persistent storage slots touched by SLOAD / SSTORE in this segment.
  /// Maps `slot → value`.  Cleared across snapshots; survives tx boundaries.
  pub storage: HashMap<U256, U256>,
  /// Transient storage slots touched by TLOAD / TSTORE (EIP-1153).
  /// Same shape as `storage` but reset to zero at the end of each transaction.
  pub transient_storage: HashMap<U256, U256>,
  /// Set of valid JUMPDEST offsets in the deployed bytecode.
  ///
  /// Shared across all states in the same execution context.  The type
  /// checker uses this table to validate JUMP / JUMPI targets.
  pub jumpdest_table: BTreeSet<u32>,
}

impl EvmState {
  pub fn new(pc: u32, gas: u64) -> Self {
    Self {
      stack: Vec::new(),
      memory: Vec::new(),
      pc,
      gas,
      storage: HashMap::new(),
      transient_storage: HashMap::new(),
      jumpdest_table: BTreeSet::new(),
    }
  }

  /// Convenience constructor used in tests.
  pub fn with_stack(stack: Vec<U256>, pc: u32) -> Self {
    Self {
      stack,
      memory: Vec::new(),
      pc,
      gas: 1_000_000,
      storage: HashMap::new(),
      transient_storage: HashMap::new(),
      jumpdest_table: BTreeSet::new(),
    }
  }

  /// Constructor with an explicit jumpdest table.
  pub fn with_jumpdests(stack: Vec<U256>, pc: u32, jumpdests: BTreeSet<u32>) -> Self {
    Self {
      stack,
      memory: Vec::new(),
      pc,
      gas: 1_000_000,
      storage: HashMap::new(),
      transient_storage: HashMap::new(),
      jumpdest_table: jumpdests,
    }
  }
}

/// Type-level abstraction of an [`EvmState`].
///
/// Tracks the structural properties needed to validate derivation trees:
/// - **Stack depth**: the only "type" of EVM stack slots.
/// - **Memory size**: number of accessible 32-byte words (monotonically grows).
/// - **Storage touched**: number of unique storage slots accessed.
/// - **PC**: included so jump-target validation can be checked structurally.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvmStateType {
  /// Current number of items on the operand stack.
  pub stack_depth: usize,
  /// Memory size in bytes (always a multiple of 32).
  pub memory_size: usize,
  /// Number of persistent storage slots touched.
  pub storage_touched: usize,
  /// Current program counter.
  pub pc: u32,
}

impl From<&EvmState> for EvmStateType {
  fn from(s: &EvmState) -> Self {
    Self {
      stack_depth: s.stack.len(),
      memory_size: s.memory.len(),
      storage_touched: s.storage.len(),
      pc: s.pc,
    }
  }
}
