//! Deep consistency checks for EVM proof trees.
//!
//! Where [`type_check`](crate::type_check) validates *structural* properties
//! (stack depth, memory size, storage count), this module validates *value-level*
//! consistency:
//!
//! - **Stack**: overflow, value continuity at `Seq` boundaries, opcode semantics.
//! - **Memory**: 32-byte alignment, content continuity at `Seq` boundaries,
//!   MLOAD / MSTORE correctness.
//! - **Storage**: key preservation, value continuity at `Seq` boundaries,
//!   SLOAD return value correctness.
//! - **Gas**: monotonic decrease within a step, continuity at `Seq` boundaries.
//!
//! # Usage
//! Run [`consistency_check`] *after* [`type_check`](crate::type_check::type_check)
//! passes. The structural invariants are assumed to hold.

use std::collections::HashMap;

use revm::primitives::U256;
use thiserror::Error;

use crate::{opcode, proof_tree::ProofNode, state::EvmState};

/// Maximum EVM operand-stack depth (EIP spec).
const MAX_STACK_DEPTH: usize = 1024;

// ─────────────────────────────────────────────────────────────────────────────
// Error types
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by [`consistency_check`].
#[derive(Debug, Error)]
pub enum ConsistencyError {
  // ── Stack ───────────────────────────────────────────────────────────────
  #[error("stack overflow: depth {depth} exceeds maximum {MAX_STACK_DEPTH}")]
  StackOverflow { depth: usize },

  #[error(
    "stack value mismatch at Seq boundary position {position}: \
     left post = {left_val}, right pre = {right_val}"
  )]
  SeqStackValueMismatch {
    position: usize,
    left_val: U256,
    right_val: U256,
  },

  // ── Memory ──────────────────────────────────────────────────────────────
  #[error("memory size {size} is not a multiple of 32 bytes")]
  MemoryAlignment { size: usize },

  #[error(
    "memory mismatch at Seq boundary: byte offset {offset} \
     left post = 0x{left_byte:02x}, right pre = 0x{right_byte:02x}"
  )]
  SeqMemoryMismatch {
    offset: usize,
    left_byte: u8,
    right_byte: u8,
  },

  // ── Storage ─────────────────────────────────────────────────────────────
  #[error("storage key {key} present in pre-state but lost in post-state")]
  StorageKeyLost { key: U256 },

  #[error("transient storage key {key} present in pre-state but lost in post-state")]
  TransientKeyLost { key: U256 },

  #[error(
    "storage mismatch at Seq boundary: key {key}, \
     left post = {left_val}, right pre = {right_val}"
  )]
  SeqStorageMismatch {
    key: U256,
    left_val: U256,
    right_val: U256,
  },

  #[error(
    "transient storage mismatch at Seq boundary: key {key}, \
     left post = {left_val}, right pre = {right_val}"
  )]
  SeqTransientMismatch {
    key: U256,
    left_val: U256,
    right_val: U256,
  },

  // ── Gas ─────────────────────────────────────────────────────────────────
  #[error(
    "gas increased during opcode 0x{opcode:02x}: \
     pre = {pre}, post = {post}"
  )]
  GasIncrease { opcode: u8, pre: u64, post: u64 },

  #[error(
    "gas cost too low for opcode 0x{opcode:02x} at pc={pc}: \
     consumed {consumed}, minimum static cost {min_cost}"
  )]
  GasCostTooLow {
    opcode: u8,
    pc: u32,
    consumed: u64,
    min_cost: u64,
  },

  #[error(
    "gas cost mismatch for opcode 0x{opcode:02x} at pc={pc}: \
     consumed {consumed}, expected exactly {expected}"
  )]
  GasCostMismatch {
    opcode: u8,
    pc: u32,
    consumed: u64,
    expected: u64,
  },

  #[error("gas mismatch at Seq boundary: left post = {left}, right pre = {right}")]
  SeqGasMismatch { left: u64, right: u64 },

  // ── PC ──────────────────────────────────────────────────────────────────
  #[error("PC mismatch at Seq boundary: left post = {left}, right pre = {right}")]
  SeqPcMismatch { left: u32, right: u32 },

  // ── Jumpdest table ─────────────────────────────────────────────────────
  #[error("jumpdest table differs at Seq boundary")]
  SeqJumpdestMismatch,

  // ── Opcode-specific value checks ───────────────────────────────────────
  #[error(
    "ADD result mismatch at pc={pc}: \
     {a} + {b} should wrap to {expected}, got {actual}"
  )]
  AddMismatch {
    pc: u32,
    a: U256,
    b: U256,
    expected: U256,
    actual: U256,
  },

  #[error(
    "SUB result mismatch at pc={pc}: \
     {a} - {b} should wrap to {expected}, got {actual}"
  )]
  SubMismatch {
    pc: u32,
    a: U256,
    b: U256,
    expected: U256,
    actual: U256,
  },

  #[error(
    "MUL result mismatch at pc={pc}: \
     {a} * {b} should wrap to {expected}, got {actual}"
  )]
  MulMismatch {
    pc: u32,
    a: U256,
    b: U256,
    expected: U256,
    actual: U256,
  },

  #[error(
    "DIV result mismatch at pc={pc}: \
     {a} / {b} should be {expected}, got {actual}"
  )]
  DivMismatch {
    pc: u32,
    a: U256,
    b: U256,
    expected: U256,
    actual: U256,
  },

  #[error(
    "MOD result mismatch at pc={pc}: \
     {a} % {b} should be {expected}, got {actual}"
  )]
  ModMismatch {
    pc: u32,
    a: U256,
    b: U256,
    expected: U256,
    actual: U256,
  },

  #[error("AND result mismatch at pc={pc}: expected {expected}, got {actual}")]
  AndMismatch {
    pc: u32,
    expected: U256,
    actual: U256,
  },

  #[error("OR result mismatch at pc={pc}: expected {expected}, got {actual}")]
  OrMismatch {
    pc: u32,
    expected: U256,
    actual: U256,
  },

  #[error("XOR result mismatch at pc={pc}: expected {expected}, got {actual}")]
  XorMismatch {
    pc: u32,
    expected: U256,
    actual: U256,
  },

  #[error("NOT result mismatch at pc={pc}: expected {expected}, got {actual}")]
  NotMismatch {
    pc: u32,
    expected: U256,
    actual: U256,
  },

  #[error("ISZERO result mismatch at pc={pc}: expected {expected}, got {actual}")]
  IsZeroMismatch {
    pc: u32,
    expected: U256,
    actual: U256,
  },

  #[error("EQ result mismatch at pc={pc}: expected {expected}, got {actual}")]
  EqMismatch {
    pc: u32,
    expected: U256,
    actual: U256,
  },

  #[error("LT result mismatch at pc={pc}: expected {expected}, got {actual}")]
  LtMismatch {
    pc: u32,
    expected: U256,
    actual: U256,
  },

  #[error("GT result mismatch at pc={pc}: expected {expected}, got {actual}")]
  GtMismatch {
    pc: u32,
    expected: U256,
    actual: U256,
  },

  #[error(
    "SLOAD result mismatch at pc={pc}: storage[{key}] should be {expected}, \
     got {actual}"
  )]
  SloadMismatch {
    pc: u32,
    key: U256,
    expected: U256,
    actual: U256,
  },

  #[error(
    "SSTORE value mismatch at pc={pc}: storage[{key}] should be {expected}, \
     got {actual}"
  )]
  SstoreMismatch {
    pc: u32,
    key: U256,
    expected: U256,
    actual: U256,
  },

  #[error(
    "TLOAD result mismatch at pc={pc}: transient[{key}] should be {expected}, \
     got {actual}"
  )]
  TloadMismatch {
    pc: u32,
    key: U256,
    expected: U256,
    actual: U256,
  },

  #[error(
    "TSTORE value mismatch at pc={pc}: transient[{key}] should be {expected}, \
     got {actual}"
  )]
  TstoreMismatch {
    pc: u32,
    key: U256,
    expected: U256,
    actual: U256,
  },

  #[error("MLOAD result mismatch at pc={pc}: expected {expected}, got {actual}")]
  MloadMismatch {
    pc: u32,
    expected: U256,
    actual: U256,
  },

  #[error("DUP{n} result mismatch at pc={pc}: expected {expected}, got {actual}")]
  DupMismatch {
    pc: u32,
    n: u8,
    expected: U256,
    actual: U256,
  },

  #[error(
    "SWAP{n} result mismatch at pc={pc}: \
     position {position} expected {expected}, got {actual}"
  )]
  SwapMismatch {
    pc: u32,
    n: u8,
    position: usize,
    expected: U256,
    actual: U256,
  },

  #[error(
    "stack passthrough mismatch at pc={pc} (opcode 0x{opcode:02x}): \
     position {position} expected {expected}, got {actual}"
  )]
  StackPassthroughMismatch {
    pc: u32,
    opcode: u8,
    position: usize,
    expected: U256,
    actual: U256,
  },
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Run deep value-level consistency checks on a [`ProofNode`] tree.
///
/// This is intended to complement [`type_check`](crate::type_check::type_check)
/// which validates structural (depth/size) properties. `consistency_check`
/// validates that:
///
/// 1. Stack depth never exceeds 1024.
/// 2. Actual stack/memory/storage **values** match at `Seq` boundaries.
/// 3. Gas never increases within a single opcode step.
/// 4. Gas, PC, and jumpdest tables are continuous at `Seq` boundaries.
/// 5. Opcode semantics are correct (ADD, SUB, MUL, etc.).
/// 6. Storage/transient keys are never lost.
/// 7. SLOAD/TLOAD return values match the storage maps.
///
/// Returns `Ok(())` on success or the first error found.
pub fn consistency_check(node: &ProofNode) -> Result<(), ConsistencyError> {
  check_node(node)?;
  Ok(())
}

/// Collect *all* consistency errors (does not stop at the first).
pub fn consistency_check_all(node: &ProofNode) -> Vec<ConsistencyError> {
  let mut errors = Vec::new();
  collect_errors(node, &mut errors);
  errors
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal: recursive tree walk
// ─────────────────────────────────────────────────────────────────────────────

/// Recursively check a node. Returns `(pre_state, post_state)` references
/// for boundary comparisons in parent `Seq`/`Branch` nodes.
fn check_node(node: &ProofNode) -> Result<(&EvmState, &EvmState), ConsistencyError> {
  match node {
    ProofNode::Leaf {
      opcode,
      pre_state,
      post_state,
      ..
    } => {
      check_leaf(*opcode, pre_state, post_state)?;
      Ok((pre_state, post_state))
    }

    ProofNode::Seq { left, right } => {
      let (_pre_l, post_l) = check_node(left)?;
      let (pre_r, post_r) = check_node(right)?;
      check_seq_boundary(post_l, pre_r)?;
      // Return the outermost pre/post for parent boundary checks.
      let pre = node_pre(left);
      Ok((pre, post_r))
    }

    ProofNode::Branch {
      cond,
      taken,
      not_taken,
    } => {
      let (pre_c, _post_c) = check_node(cond)?;
      let _ = check_node(taken)?;
      let _ = check_node(not_taken)?;
      // For branch, pre = cond.pre, post = taken.post (both paths agree by type_check)
      let post = node_post(taken);
      Ok((pre_c, post))
    }
  }
}

/// Collect all errors without early return.
fn collect_errors(node: &ProofNode, errors: &mut Vec<ConsistencyError>) {
  match node {
    ProofNode::Leaf {
      opcode,
      pre_state,
      post_state,
      ..
    } => {
      if let Err(e) = check_leaf(*opcode, pre_state, post_state) {
        errors.push(e);
      }
    }

    ProofNode::Seq { left, right } => {
      collect_errors(left, errors);
      collect_errors(right, errors);
      let post_l = node_post(left);
      let pre_r = node_pre(right);
      if let Err(e) = check_seq_boundary(post_l, pre_r) {
        errors.push(e);
      }
    }

    ProofNode::Branch {
      cond,
      taken,
      not_taken,
    } => {
      collect_errors(cond, errors);
      collect_errors(taken, errors);
      collect_errors(not_taken, errors);
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Node helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Extract the leftmost pre-state from a node.
fn node_pre(node: &ProofNode) -> &EvmState {
  match node {
    ProofNode::Leaf { pre_state, .. } => pre_state,
    ProofNode::Seq { left, .. } => node_pre(left),
    ProofNode::Branch { cond, .. } => node_pre(cond),
  }
}

/// Extract the rightmost post-state from a node.
fn node_post(node: &ProofNode) -> &EvmState {
  match node {
    ProofNode::Leaf { post_state, .. } => post_state,
    ProofNode::Seq { right, .. } => node_post(right),
    // Both branch paths have the same post-type; pick taken.
    ProofNode::Branch { taken, .. } => node_post(taken),
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Leaf checks
// ─────────────────────────────────────────────────────────────────────────────

fn check_leaf(op: u8, pre: &EvmState, post: &EvmState) -> Result<(), ConsistencyError> {
  // 1. Stack overflow
  if post.stack.len() > MAX_STACK_DEPTH {
    return Err(ConsistencyError::StackOverflow {
      depth: post.stack.len(),
    });
  }

  // 2. Memory alignment (must be multiple of 32 bytes)
  if pre.memory.len() % 32 != 0 {
    return Err(ConsistencyError::MemoryAlignment {
      size: pre.memory.len(),
    });
  }
  if post.memory.len() % 32 != 0 {
    return Err(ConsistencyError::MemoryAlignment {
      size: post.memory.len(),
    });
  }

  // 3. Gas: cannot increase within a single opcode
  if post.gas > pre.gas {
    return Err(ConsistencyError::GasIncrease {
      opcode: op,
      pre: pre.gas,
      post: post.gas,
    });
  }

  // 3b. Gas cost must be at least the static minimum for this opcode.
  //     For non-dynamic opcodes, the cost must match exactly.
  if let Some(info) = opcode::gas_cost(op) {
    let consumed = pre.gas - post.gas;
    if consumed < info.static_gas {
      return Err(ConsistencyError::GasCostTooLow {
        opcode: op,
        pc: pre.pc,
        consumed,
        min_cost: info.static_gas,
      });
    }
    if !info.dynamic && consumed != info.static_gas {
      return Err(ConsistencyError::GasCostMismatch {
        opcode: op,
        pc: pre.pc,
        consumed,
        expected: info.static_gas,
      });
    }
  }

  // 4. Storage keys can only be added, never removed
  check_storage_keys_preserved(&pre.storage, &post.storage, false)?;
  check_storage_keys_preserved(&pre.transient_storage, &post.transient_storage, true)?;

  // 5. Opcode-specific value checks
  check_opcode_semantics(op, pre, post)?;

  Ok(())
}

/// Verify that all keys in `pre` are also present in `post`.
fn check_storage_keys_preserved(
  pre: &HashMap<U256, U256>,
  post: &HashMap<U256, U256>,
  transient: bool,
) -> Result<(), ConsistencyError> {
  for key in pre.keys() {
    if !post.contains_key(key) {
      return if transient {
        Err(ConsistencyError::TransientKeyLost { key: *key })
      } else {
        Err(ConsistencyError::StorageKeyLost { key: *key })
      };
    }
  }
  Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Seq boundary checks
// ─────────────────────────────────────────────────────────────────────────────

fn check_seq_boundary(left_post: &EvmState, right_pre: &EvmState) -> Result<(), ConsistencyError> {
  // 1. PC continuity
  if left_post.pc != right_pre.pc {
    return Err(ConsistencyError::SeqPcMismatch {
      left: left_post.pc,
      right: right_pre.pc,
    });
  }

  // 2. Gas continuity
  if left_post.gas != right_pre.gas {
    return Err(ConsistencyError::SeqGasMismatch {
      left: left_post.gas,
      right: right_pre.gas,
    });
  }

  // 3. Stack value continuity (not just depth — actual values)
  let depth = left_post.stack.len().min(right_pre.stack.len());
  for i in 0..depth {
    if left_post.stack[i] != right_pre.stack[i] {
      return Err(ConsistencyError::SeqStackValueMismatch {
        position: i,
        left_val: left_post.stack[i],
        right_val: right_pre.stack[i],
      });
    }
  }

  // 4. Memory content continuity
  let mem_len = left_post.memory.len().min(right_pre.memory.len());
  for i in 0..mem_len {
    if left_post.memory[i] != right_pre.memory[i] {
      return Err(ConsistencyError::SeqMemoryMismatch {
        offset: i,
        left_byte: left_post.memory[i],
        right_byte: right_pre.memory[i],
      });
    }
  }

  // 5. Storage value continuity — all shared keys must agree
  check_storage_boundary(&left_post.storage, &right_pre.storage, false)?;
  check_storage_boundary(
    &left_post.transient_storage,
    &right_pre.transient_storage,
    true,
  )?;

  // 6. Jumpdest table must be identical
  if left_post.jumpdest_table != right_pre.jumpdest_table {
    return Err(ConsistencyError::SeqJumpdestMismatch);
  }

  Ok(())
}

/// Verify storage maps agree on all shared keys at a Seq boundary.
fn check_storage_boundary(
  left: &HashMap<U256, U256>,
  right: &HashMap<U256, U256>,
  transient: bool,
) -> Result<(), ConsistencyError> {
  for (key, left_val) in left {
    if let Some(right_val) = right.get(key) {
      if left_val != right_val {
        return if transient {
          Err(ConsistencyError::SeqTransientMismatch {
            key: *key,
            left_val: *left_val,
            right_val: *right_val,
          })
        } else {
          Err(ConsistencyError::SeqStorageMismatch {
            key: *key,
            left_val: *left_val,
            right_val: *right_val,
          })
        };
      }
    }
  }
  Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Opcode semantics checks
// ─────────────────────────────────────────────────────────────────────────────

/// Check that the opcode computes the correct result from pre→post state.
///
/// Only covers opcodes with deterministic, context-free semantics.
/// Environment-dependent opcodes (ADDRESS, CALLER, etc.) and opcodes
/// verified by the circuit's advice+check pattern (EXP, SAR, etc.) are
/// skipped — their correctness is guaranteed by the ZK sub-proof.
fn check_opcode_semantics(op: u8, pre: &EvmState, post: &EvmState) -> Result<(), ConsistencyError> {
  let pc = pre.pc;

  match op {
    // ── Arithmetic ───────────────────────────────────────────────────────
    opcode::ADD => check_binary(pc, pre, post, |a, b| a.overflowing_add(b).0, "ADD")?,
    opcode::MUL => check_binary(pc, pre, post, |a, b| a.overflowing_mul(b).0, "MUL")?,
    opcode::SUB => check_binary(pc, pre, post, |a, b| a.overflowing_sub(b).0, "SUB")?,
    opcode::DIV => check_binary(
      pc,
      pre,
      post,
      |a, b| {
        if b.is_zero() { U256::ZERO } else { a / b }
      },
      "DIV",
    )?,
    opcode::MOD => check_binary(
      pc,
      pre,
      post,
      |a, b| {
        if b.is_zero() { U256::ZERO } else { a % b }
      },
      "MOD",
    )?,

    // ── Bitwise ──────────────────────────────────────────────────────────
    opcode::AND => check_binary(pc, pre, post, |a, b| a & b, "AND")?,
    opcode::OR => check_binary(pc, pre, post, |a, b| a | b, "OR")?,
    opcode::XOR => check_binary(pc, pre, post, |a, b| a ^ b, "XOR")?,
    opcode::NOT => {
      if let (Some(&a), Some(&result)) = (pre.stack.first(), post.stack.first()) {
        let expected = !a;
        if result != expected {
          return Err(ConsistencyError::NotMismatch {
            pc,
            expected,
            actual: result,
          });
        }
      }
    }

    // ── Comparison ───────────────────────────────────────────────────────
    opcode::LT => check_binary(
      pc,
      pre,
      post,
      |a, b| {
        if a < b { U256::from(1u64) } else { U256::ZERO }
      },
      "LT",
    )?,
    opcode::GT => check_binary(
      pc,
      pre,
      post,
      |a, b| {
        if a > b { U256::from(1u64) } else { U256::ZERO }
      },
      "GT",
    )?,
    opcode::EQ => check_binary(
      pc,
      pre,
      post,
      |a, b| {
        if a == b { U256::from(1u64) } else { U256::ZERO }
      },
      "EQ",
    )?,
    opcode::ISZERO => {
      if let (Some(&a), Some(&result)) = (pre.stack.first(), post.stack.first()) {
        let expected = if a.is_zero() {
          U256::from(1u64)
        } else {
          U256::ZERO
        };
        if result != expected {
          return Err(ConsistencyError::IsZeroMismatch {
            pc,
            expected,
            actual: result,
          });
        }
      }
    }

    // ── DUP1..16 ─────────────────────────────────────────────────────────
    op if (0x80..=0x8f).contains(&op) => {
      let n = (op - 0x80 + 1) as usize; // DUP1 → 1, DUP16 → 16
      // DUP_n: duplicates the n-th stack element to the top.
      // post_stack[0] should equal pre_stack[n-1].
      if pre.stack.len() >= n {
        if let Some(&result) = post.stack.first() {
          let expected = pre.stack[n - 1];
          if result != expected {
            return Err(ConsistencyError::DupMismatch {
              pc,
              n: (op - 0x80 + 1),
              expected,
              actual: result,
            });
          }
        }
      }
      // Remaining elements: post_stack[1..] should equal pre_stack[0..]
      check_stack_passthrough(pc, op, pre, post, 0, 1)?;
    }

    // ── SWAP1..16 ────────────────────────────────────────────────────────
    op if (0x90..=0x9f).contains(&op) => {
      let n = (op - 0x90 + 1) as usize; // SWAP1 → 1, SWAP16 → 16
      // SWAP_n: swaps TOS with the (n+1)-th element.
      // post_stack[0] should equal pre_stack[n]
      // post_stack[n] should equal pre_stack[0]
      // Everything else unchanged.
      if pre.stack.len() > n {
        if post.stack.len() > n {
          // Check swapped positions
          if post.stack[0] != pre.stack[n] {
            return Err(ConsistencyError::SwapMismatch {
              pc,
              n: (op - 0x90 + 1),
              position: 0,
              expected: pre.stack[n],
              actual: post.stack[0],
            });
          }
          if post.stack[n] != pre.stack[0] {
            return Err(ConsistencyError::SwapMismatch {
              pc,
              n: (op - 0x90 + 1),
              position: n,
              expected: pre.stack[0],
              actual: post.stack[n],
            });
          }
          // Check untouched positions (1..n-1 and n+1..)
          for i in 1..n {
            if post.stack[i] != pre.stack[i] {
              return Err(ConsistencyError::SwapMismatch {
                pc,
                n: (op - 0x90 + 1),
                position: i,
                expected: pre.stack[i],
                actual: post.stack[i],
              });
            }
          }
          for i in (n + 1)..post.stack.len().min(pre.stack.len()) {
            if post.stack[i] != pre.stack[i] {
              return Err(ConsistencyError::SwapMismatch {
                pc,
                n: (op - 0x90 + 1),
                position: i,
                expected: pre.stack[i],
                actual: post.stack[i],
              });
            }
          }
        }
      }
    }

    // ── Storage ──────────────────────────────────────────────────────────
    opcode::SLOAD => {
      if let Some(&key) = pre.stack.first() {
        if let Some(&result) = post.stack.first() {
          // Check that result matches the post-state storage value for this key.
          let expected = post.storage.get(&key).copied().unwrap_or(U256::ZERO);
          if result != expected {
            return Err(ConsistencyError::SloadMismatch {
              pc,
              key,
              expected,
              actual: result,
            });
          }
        }
      }
    }

    opcode::SSTORE => {
      if pre.stack.len() >= 2 {
        let key = pre.stack[0];
        let value = pre.stack[1];
        // post-state storage[key] should equal the written value.
        if let Some(&stored) = post.storage.get(&key) {
          if stored != value {
            return Err(ConsistencyError::SstoreMismatch {
              pc,
              key,
              expected: value,
              actual: stored,
            });
          }
        }
      }
    }

    opcode::TLOAD => {
      if let Some(&key) = pre.stack.first() {
        if let Some(&result) = post.stack.first() {
          let expected = post
            .transient_storage
            .get(&key)
            .copied()
            .unwrap_or(U256::ZERO);
          if result != expected {
            return Err(ConsistencyError::TloadMismatch {
              pc,
              key,
              expected,
              actual: result,
            });
          }
        }
      }
    }

    opcode::TSTORE => {
      if pre.stack.len() >= 2 {
        let key = pre.stack[0];
        let value = pre.stack[1];
        if let Some(&stored) = post.transient_storage.get(&key) {
          if stored != value {
            return Err(ConsistencyError::TstoreMismatch {
              pc,
              key,
              expected: value,
              actual: stored,
            });
          }
        }
      }
    }

    // ── Memory ───────────────────────────────────────────────────────────
    opcode::MLOAD => {
      if let Some(&offset_u256) = pre.stack.first() {
        if let Some(&result) = post.stack.first() {
          // Read 32 bytes from memory at offset.
          let offset = offset_u256.saturating_to::<usize>();
          let expected = mload_from_memory(&post.memory, offset);
          if result != expected {
            return Err(ConsistencyError::MloadMismatch {
              pc,
              expected,
              actual: result,
            });
          }
        }
      }
    }

    // ── POP: stack passthrough of remaining elements ─────────────────────
    opcode::POP => {
      // post_stack should equal pre_stack[1..]
      check_stack_passthrough(pc, op, pre, post, 1, 0)?;
    }

    // Other opcodes: environment-dependent or checked by circuit constraints.
    _ => {}
  }

  Ok(())
}

/// Check a binary opcode: `post_stack[0] == f(pre_stack[0], pre_stack[1])`.
fn check_binary(
  pc: u32,
  pre: &EvmState,
  post: &EvmState,
  f: impl Fn(U256, U256) -> U256,
  tag: &str,
) -> Result<(), ConsistencyError> {
  if pre.stack.len() >= 2 {
    if let Some(&result) = post.stack.first() {
      let a = pre.stack[0];
      let b = pre.stack[1];
      let expected = f(a, b);
      if result != expected {
        return Err(match tag {
          "ADD" => ConsistencyError::AddMismatch {
            pc,
            a,
            b,
            expected,
            actual: result,
          },
          "SUB" => ConsistencyError::SubMismatch {
            pc,
            a,
            b,
            expected,
            actual: result,
          },
          "MUL" => ConsistencyError::MulMismatch {
            pc,
            a,
            b,
            expected,
            actual: result,
          },
          "DIV" => ConsistencyError::DivMismatch {
            pc,
            a,
            b,
            expected,
            actual: result,
          },
          "MOD" => ConsistencyError::ModMismatch {
            pc,
            a,
            b,
            expected,
            actual: result,
          },
          "AND" => ConsistencyError::AndMismatch {
            pc,
            expected,
            actual: result,
          },
          "OR" => ConsistencyError::OrMismatch {
            pc,
            expected,
            actual: result,
          },
          "XOR" => ConsistencyError::XorMismatch {
            pc,
            expected,
            actual: result,
          },
          "LT" => ConsistencyError::LtMismatch {
            pc,
            expected,
            actual: result,
          },
          "GT" => ConsistencyError::GtMismatch {
            pc,
            expected,
            actual: result,
          },
          "EQ" => ConsistencyError::EqMismatch {
            pc,
            expected,
            actual: result,
          },
          _ => ConsistencyError::AddMismatch {
            pc,
            a,
            b,
            expected,
            actual: result,
          },
        });
      }
      // Also check passthrough: post_stack[1..] == pre_stack[2..]
      check_stack_passthrough(pc, 0x01, pre, post, 2, 1)?;
    }
  }
  Ok(())
}

/// Verify that stack elements pass through unchanged.
///
/// - `pre_start`: index in `pre.stack` where passthrough begins.
/// - `post_start`: index in `post.stack` where passthrough begins.
///
/// Checks `post.stack[post_start + i] == pre.stack[pre_start + i]`
/// for all valid `i`.
fn check_stack_passthrough(
  pc: u32,
  op: u8,
  pre: &EvmState,
  post: &EvmState,
  pre_start: usize,
  post_start: usize,
) -> Result<(), ConsistencyError> {
  let count =
    (pre.stack.len().saturating_sub(pre_start)).min(post.stack.len().saturating_sub(post_start));
  for i in 0..count {
    let pre_val = pre.stack[pre_start + i];
    let post_val = post.stack[post_start + i];
    if pre_val != post_val {
      return Err(ConsistencyError::StackPassthroughMismatch {
        pc,
        opcode: op,
        position: post_start + i,
        expected: pre_val,
        actual: post_val,
      });
    }
  }
  Ok(())
}

/// Read a big-endian U256 from memory at `offset` (32 bytes).
fn mload_from_memory(memory: &[u8], offset: usize) -> U256 {
  if offset + 32 > memory.len() {
    // Out-of-range reads are zero-padded in EVM.
    let mut buf = [0u8; 32];
    let available = memory.len().saturating_sub(offset);
    if available > 0 {
      buf[..available].copy_from_slice(&memory[offset..offset + available]);
    }
    U256::from_be_bytes(buf)
  } else {
    let mut buf = [0u8; 32];
    buf.copy_from_slice(&memory[offset..offset + 32]);
    U256::from_be_bytes(buf)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use revm::primitives::U256;

  use super::*;
  use crate::proof_tree::{LeafProof, ProofNode};

  fn leaf(opcode: u8, pre: EvmState, post: EvmState) -> ProofNode {
    ProofNode::Leaf {
      opcode,
      pre_state: pre,
      post_state: post,
      leaf_proof: LeafProof::placeholder(),
    }
  }

  const HIGH_GAS: u64 = 1_000_000;

  fn st(stack: Vec<U256>, pc: u32) -> EvmState {
    let mut s = EvmState::with_stack(stack, pc);
    s.gas = HIGH_GAS;
    s
  }

  /// State with exact gas.
  fn st_gas(stack: Vec<U256>, pc: u32, gas: u64) -> EvmState {
    let mut s = EvmState::with_stack(stack, pc);
    s.gas = gas;
    s
  }

  /// Build a leaf node with correct static gas consumption.
  fn leaf_gas(op: u8, pre_stack: Vec<U256>, post_stack: Vec<U256>) -> ProofNode {
    let gas_cost = opcode::gas_cost(op).map(|g| g.static_gas).unwrap_or(0);
    let pre = st_gas(pre_stack, 0, HIGH_GAS);
    let post = st_gas(post_stack, 1, HIGH_GAS - gas_cost);
    leaf(op, pre, post)
  }

  fn st_mem(stack: Vec<U256>, pc: u32, mem: Vec<u8>) -> EvmState {
    let mut s = st(stack, pc);
    s.memory = mem;
    s
  }

  fn st_stor(stack: Vec<U256>, pc: u32, entries: &[(u64, u64)]) -> EvmState {
    let mut s = st(stack, pc);
    for &(k, v) in entries {
      s.storage.insert(U256::from(k), U256::from(v));
    }
    s
  }

  fn st_tstor(stack: Vec<U256>, pc: u32, entries: &[(u64, u64)]) -> EvmState {
    let mut s = st(stack, pc);
    for &(k, v) in entries {
      s.transient_storage.insert(U256::from(k), U256::from(v));
    }
    s
  }

  fn st_gas_tstor(stack: Vec<U256>, pc: u32, gas: u64, entries: &[(u64, u64)]) -> EvmState {
    let mut s = st_gas(stack, pc, gas);
    for &(k, v) in entries {
      s.transient_storage.insert(U256::from(k), U256::from(v));
    }
    s
  }

  // ────────────────── Stack overflow ──────────────────────────────────────

  #[test]
  fn stack_overflow_detected() {
    let big_stack: Vec<U256> = vec![U256::ZERO; 1025];
    let node = leaf(
      opcode::PUSH0,
      st(vec![U256::ZERO; 1024], 0),
      EvmState::with_stack(big_stack, 1),
    );
    let err = consistency_check(&node).unwrap_err();
    assert!(matches!(
      err,
      ConsistencyError::StackOverflow { depth: 1025 }
    ));
  }

  #[test]
  fn stack_at_limit_ok() {
    let node = leaf_gas(
      opcode::PUSH0,
      vec![U256::ZERO; 1023],
      vec![U256::ZERO; 1024],
    );
    assert!(consistency_check(&node).is_ok());
  }

  // ────────────────── Memory alignment ───────────────────────────────────

  #[test]
  fn memory_alignment_error() {
    let node = leaf(
      opcode::ADD,
      st_mem(
        vec![U256::from(1u64), U256::from(2u64)],
        0,
        vec![0u8; 33], // not a multiple of 32
      ),
      st_mem(vec![U256::from(3u64)], 1, vec![0u8; 33]),
    );
    let err = consistency_check(&node).unwrap_err();
    assert!(matches!(
      err,
      ConsistencyError::MemoryAlignment { size: 33 }
    ));
  }

  // ────────────────── Gas ────────────────────────────────────────────────

  #[test]
  fn gas_increase_error() {
    let node = leaf(
      opcode::ADD,
      st_gas(vec![U256::from(1u64), U256::from(2u64)], 0, 100),
      st_gas(vec![U256::from(3u64)], 1, 200), // gas increased!
    );
    let err = consistency_check(&node).unwrap_err();
    assert!(matches!(
      err,
      ConsistencyError::GasIncrease {
        pre: 100,
        post: 200,
        ..
      }
    ));
  }

  #[test]
  fn gas_decrease_ok() {
    let node = leaf(
      opcode::ADD,
      st_gas(vec![U256::from(1u64), U256::from(2u64)], 0, 1000),
      st_gas(vec![U256::from(3u64)], 1, 997),
    );
    assert!(consistency_check(&node).is_ok());
  }

  #[test]
  fn seq_gas_continuity_error() {
    let left = leaf(
      opcode::ADD,
      st_gas(vec![U256::from(1u64), U256::from(2u64)], 0, 1000),
      st_gas(vec![U256::from(3u64)], 1, 997),
    );
    let right = leaf(
      opcode::POP,
      st_gas(vec![U256::from(3u64)], 1, 990), // doesn't match left's post gas
      st_gas(vec![], 2, 988),
    );
    let seq = ProofNode::Seq {
      left: Box::new(left),
      right: Box::new(right),
    };
    let err = consistency_check(&seq).unwrap_err();
    assert!(matches!(
      err,
      ConsistencyError::SeqGasMismatch {
        left: 997,
        right: 990
      }
    ));
  }

  // ────────────────── Seq stack value continuity ─────────────────────────

  #[test]
  fn seq_stack_value_mismatch() {
    let left = leaf(
      opcode::ADD,
      st_gas(vec![U256::from(1u64), U256::from(2u64)], 0, 1000),
      st_gas(vec![U256::from(3u64)], 1, 997),
    );
    let right = leaf(
      opcode::POP,
      st_gas(vec![U256::from(999u64)], 1, 997), // different value!
      st_gas(vec![], 2, 995),
    );
    let seq = ProofNode::Seq {
      left: Box::new(left),
      right: Box::new(right),
    };
    let err = consistency_check(&seq).unwrap_err();
    assert!(matches!(
      err,
      ConsistencyError::SeqStackValueMismatch { position: 0, .. }
    ));
  }

  #[test]
  fn seq_stack_value_match_ok() {
    let left = leaf(
      opcode::ADD,
      st_gas(vec![U256::from(1u64), U256::from(2u64)], 0, 1000),
      st_gas(vec![U256::from(3u64)], 1, 997),
    );
    let right = leaf(
      opcode::POP,
      st_gas(vec![U256::from(3u64)], 1, 997),
      st_gas(vec![], 2, 995),
    );
    let seq = ProofNode::Seq {
      left: Box::new(left),
      right: Box::new(right),
    };
    assert!(consistency_check(&seq).is_ok());
  }

  // ────────────────── Seq memory continuity ──────────────────────────────

  #[test]
  fn seq_memory_mismatch() {
    let mem_a = vec![0u8; 32];
    let mut mem_b = vec![0u8; 32];
    mem_b[0] = 0xff;

    let left = leaf(
      opcode::MSTORE,
      st_gas(vec![U256::from(0u64), U256::from(42u64)], 0, 1000),
      {
        let mut s = st_gas(vec![], 1, 994);
        s.memory = mem_a;
        s
      },
    );
    let right = leaf(
      opcode::PUSH0,
      {
        let mut s = st_gas(vec![], 1, 994);
        s.memory = mem_b;
        s
      },
      {
        let mut s = st_gas(vec![U256::ZERO], 2, 992);
        s.memory = vec![0xffu8; 32];
        s
      },
    );
    let seq = ProofNode::Seq {
      left: Box::new(left),
      right: Box::new(right),
    };
    let err = consistency_check(&seq).unwrap_err();
    assert!(matches!(
      err,
      ConsistencyError::SeqMemoryMismatch { offset: 0, .. }
    ));
  }

  // ────────────────── Storage key preservation ───────────────────────────

  #[test]
  fn storage_key_lost() {
    let node = leaf(
      opcode::SSTORE,
      st_stor(vec![U256::from(1u64), U256::from(42u64)], 0, &[(10, 100)]),
      st_stor(vec![], 1, &[(1, 42)]), // key 10 lost!
    );
    let err = consistency_check(&node).unwrap_err();
    assert!(matches!(err, ConsistencyError::StorageKeyLost { .. }));
  }

  #[test]
  fn transient_key_lost() {
    let node = leaf(
      opcode::TSTORE,
      {
        let mut s = st_gas(vec![U256::from(1u64), U256::from(42u64)], 0, HIGH_GAS);
        s.transient_storage
          .insert(U256::from(10u64), U256::from(100u64));
        s
      },
      st_gas_tstor(vec![], 1, HIGH_GAS - 100, &[(1, 42)]), // key 10 lost!
    );
    let err = consistency_check(&node).unwrap_err();
    assert!(matches!(err, ConsistencyError::TransientKeyLost { .. }));
  }

  // ────────────────── Seq storage continuity ─────────────────────────────

  #[test]
  fn seq_storage_mismatch() {
    let left = leaf(
      opcode::SSTORE,
      st_gas(vec![U256::from(1u64), U256::from(42u64)], 0, 1000),
      {
        let mut s = st_gas(vec![], 1, 995);
        s.storage.insert(U256::from(1u64), U256::from(42u64));
        s
      },
    );
    let right = leaf(
      opcode::PUSH0,
      {
        let mut s = st_gas(vec![], 1, 995);
        // Different value for same key!
        s.storage.insert(U256::from(1u64), U256::from(999u64));
        s
      },
      {
        let mut s = st_gas(vec![U256::ZERO], 2, 993);
        s.storage.insert(U256::from(1u64), U256::from(999u64));
        s
      },
    );
    let seq = ProofNode::Seq {
      left: Box::new(left),
      right: Box::new(right),
    };
    let err = consistency_check(&seq).unwrap_err();
    assert!(matches!(err, ConsistencyError::SeqStorageMismatch { .. }));
  }

  // ────────────────── Opcode: ADD ────────────────────────────────────────

  #[test]
  fn add_correct() {
    let node = leaf_gas(
      opcode::ADD,
      vec![U256::from(3u64), U256::from(5u64)],
      vec![U256::from(8u64)],
    );
    assert!(consistency_check(&node).is_ok());
  }

  #[test]
  fn add_wrong_result() {
    let node = leaf_gas(
      opcode::ADD,
      vec![U256::from(3u64), U256::from(5u64)],
      vec![U256::from(9u64)], // wrong!
    );
    let err = consistency_check(&node).unwrap_err();
    assert!(matches!(err, ConsistencyError::AddMismatch { .. }));
  }

  #[test]
  fn add_overflow_wraps() {
    let node = leaf_gas(
      opcode::ADD,
      vec![U256::MAX, U256::from(1u64)],
      vec![U256::ZERO], // wraps to zero
    );
    assert!(consistency_check(&node).is_ok());
  }

  // ────────────────── Opcode: SUB ────────────────────────────────────────

  #[test]
  fn sub_correct() {
    let node = leaf_gas(
      opcode::SUB,
      vec![U256::from(10u64), U256::from(3u64)],
      vec![U256::from(7u64)],
    );
    assert!(consistency_check(&node).is_ok());
  }

  #[test]
  fn sub_wrong_result() {
    let node = leaf_gas(
      opcode::SUB,
      vec![U256::from(10u64), U256::from(3u64)],
      vec![U256::from(6u64)],
    );
    let err = consistency_check(&node).unwrap_err();
    assert!(matches!(err, ConsistencyError::SubMismatch { .. }));
  }

  // ────────────────── Opcode: MUL ────────────────────────────────────────

  #[test]
  fn mul_correct() {
    let node = leaf_gas(
      opcode::MUL,
      vec![U256::from(6u64), U256::from(7u64)],
      vec![U256::from(42u64)],
    );
    assert!(consistency_check(&node).is_ok());
  }

  // ────────────────── Opcode: DIV ────────────────────────────────────────

  #[test]
  fn div_correct() {
    let node = leaf_gas(
      opcode::DIV,
      vec![U256::from(42u64), U256::from(7u64)],
      vec![U256::from(6u64)],
    );
    assert!(consistency_check(&node).is_ok());
  }

  #[test]
  fn div_by_zero() {
    let node = leaf_gas(
      opcode::DIV,
      vec![U256::from(42u64), U256::ZERO],
      vec![U256::ZERO],
    );
    assert!(consistency_check(&node).is_ok());
  }

  // ────────────────── Opcode: NOT, ISZERO ────────────────────────────────

  #[test]
  fn not_correct() {
    let node = leaf_gas(opcode::NOT, vec![U256::ZERO], vec![U256::MAX]);
    assert!(consistency_check(&node).is_ok());
  }

  #[test]
  fn iszero_correct() {
    let node = leaf_gas(opcode::ISZERO, vec![U256::ZERO], vec![U256::from(1u64)]);
    assert!(consistency_check(&node).is_ok());
  }

  #[test]
  fn iszero_nonzero() {
    let node = leaf_gas(opcode::ISZERO, vec![U256::from(42u64)], vec![U256::ZERO]);
    assert!(consistency_check(&node).is_ok());
  }

  // ────────────────── Opcode: DUP ────────────────────────────────────────

  #[test]
  fn dup1_correct() {
    let v = U256::from(99u64);
    let node = leaf_gas(
      0x80, // DUP1
      vec![v, U256::from(1u64)],
      vec![v, v, U256::from(1u64)],
    );
    assert!(consistency_check(&node).is_ok());
  }

  #[test]
  fn dup1_wrong_value() {
    let node = leaf_gas(
      0x80, // DUP1
      vec![U256::from(99u64)],
      vec![U256::from(100u64), U256::from(99u64)], // wrong dup
    );
    let err = consistency_check(&node).unwrap_err();
    assert!(matches!(err, ConsistencyError::DupMismatch { .. }));
  }

  // ────────────────── Opcode: SWAP ───────────────────────────────────────

  #[test]
  fn swap1_correct() {
    let a = U256::from(1u64);
    let b = U256::from(2u64);
    let c = U256::from(3u64);
    let node = leaf_gas(
      0x90, // SWAP1
      vec![a, b, c],
      vec![b, a, c],
    );
    assert!(consistency_check(&node).is_ok());
  }

  #[test]
  fn swap1_wrong() {
    let a = U256::from(1u64);
    let b = U256::from(2u64);
    let node = leaf_gas(
      0x90, // SWAP1
      vec![a, b],
      vec![a, b], // not swapped!
    );
    let err = consistency_check(&node).unwrap_err();
    assert!(matches!(err, ConsistencyError::SwapMismatch { .. }));
  }

  // ────────────────── Opcode: SLOAD / SSTORE ─────────────────────────────

  #[test]
  fn sload_correct() {
    let node = leaf(
      opcode::SLOAD,
      st(vec![U256::from(1u64)], 0),
      st_stor(vec![U256::from(42u64)], 1, &[(1, 42)]),
    );
    assert!(consistency_check(&node).is_ok());
  }

  #[test]
  fn sload_mismatch() {
    let node = leaf(
      opcode::SLOAD,
      st(vec![U256::from(1u64)], 0),
      st_stor(vec![U256::from(99u64)], 1, &[(1, 42)]), // stack says 99, but storage says 42
    );
    let err = consistency_check(&node).unwrap_err();
    assert!(matches!(err, ConsistencyError::SloadMismatch { .. }));
  }

  #[test]
  fn sstore_correct() {
    let node = leaf(
      opcode::SSTORE,
      st(vec![U256::from(1u64), U256::from(42u64)], 0),
      st_stor(vec![], 1, &[(1, 42)]),
    );
    assert!(consistency_check(&node).is_ok());
  }

  #[test]
  fn sstore_value_mismatch() {
    let node = leaf(
      opcode::SSTORE,
      st(vec![U256::from(1u64), U256::from(42u64)], 0),
      st_stor(vec![], 1, &[(1, 99)]), // stored wrong value
    );
    let err = consistency_check(&node).unwrap_err();
    assert!(matches!(err, ConsistencyError::SstoreMismatch { .. }));
  }

  // ────────────────── Opcode: TLOAD / TSTORE ─────────────────────────────

  #[test]
  fn tload_correct() {
    let node = leaf(
      opcode::TLOAD,
      {
        let mut s = st_gas(vec![U256::from(1u64)], 0, HIGH_GAS);
        s.transient_storage
          .insert(U256::from(1u64), U256::from(42u64));
        s
      },
      st_gas_tstor(vec![U256::from(42u64)], 1, HIGH_GAS - 100, &[(1, 42)]),
    );
    assert!(consistency_check(&node).is_ok());
  }

  #[test]
  fn tstore_correct() {
    let node = leaf(
      opcode::TSTORE,
      st_gas(vec![U256::from(1u64), U256::from(42u64)], 0, HIGH_GAS),
      st_gas_tstor(vec![], 1, HIGH_GAS - 100, &[(1, 42)]),
    );
    assert!(consistency_check(&node).is_ok());
  }

  // ────────────────── Opcode: MLOAD ──────────────────────────────────────

  #[test]
  fn mload_correct() {
    let mut mem = vec![0u8; 64];
    // Write big-endian 42 at offset 0.
    mem[31] = 42;
    let node = leaf(
      opcode::MLOAD,
      {
        let mut s = st_gas(vec![U256::from(0u64)], 0, HIGH_GAS);
        s.memory = mem.clone();
        s
      },
      {
        let mut s = st_gas(vec![U256::from(42u64)], 1, HIGH_GAS - 3);
        s.memory = mem;
        s
      },
    );
    assert!(consistency_check(&node).is_ok());
  }

  #[test]
  fn mload_wrong_value() {
    let mut mem = vec![0u8; 64];
    mem[31] = 42;
    let node = leaf(
      opcode::MLOAD,
      {
        let mut s = st_gas(vec![U256::from(0u64)], 0, HIGH_GAS);
        s.memory = mem.clone();
        s
      },
      {
        let mut s = st_gas(vec![U256::from(99u64)], 1, HIGH_GAS - 3);
        s.memory = mem;
        s
      },
    );
    let err = consistency_check(&node).unwrap_err();
    assert!(matches!(err, ConsistencyError::MloadMismatch { .. }));
  }

  // ────────────────── Opcode: POP ────────────────────────────────────────

  #[test]
  fn pop_passthrough() {
    let a = U256::from(1u64);
    let b = U256::from(2u64);
    let c = U256::from(3u64);
    let node = leaf_gas(opcode::POP, vec![a, b, c], vec![b, c]);
    assert!(consistency_check(&node).is_ok());
  }

  #[test]
  fn pop_corrupted_passthrough() {
    let a = U256::from(1u64);
    let b = U256::from(2u64);
    let c = U256::from(3u64);
    let node = leaf_gas(
      opcode::POP,
      vec![a, b, c],
      vec![b, U256::from(999u64)], // c changed!
    );
    let err = consistency_check(&node).unwrap_err();
    assert!(matches!(
      err,
      ConsistencyError::StackPassthroughMismatch { .. }
    ));
  }

  // ────────────────── Seq PC continuity ──────────────────────────────────

  #[test]
  fn seq_pc_mismatch() {
    let left = leaf(
      opcode::ADD,
      st_gas(vec![U256::from(1u64), U256::from(2u64)], 0, HIGH_GAS),
      st_gas(vec![U256::from(3u64)], 1, HIGH_GAS - 3),
    );
    let right = leaf(
      opcode::POP,
      st_gas(vec![U256::from(3u64)], 5, HIGH_GAS - 3), // PC 5 ≠ left post PC 1
      st_gas(vec![], 6, HIGH_GAS - 5),
    );
    let seq = ProofNode::Seq {
      left: Box::new(left),
      right: Box::new(right),
    };
    let err = consistency_check(&seq).unwrap_err();
    assert!(matches!(
      err,
      ConsistencyError::SeqPcMismatch { left: 1, right: 5 }
    ));
  }

  // ────────────────── Seq jumpdest table mismatch ────────────────────────

  #[test]
  fn seq_jumpdest_mismatch() {
    let mut pre_l = st_gas(vec![U256::from(1u64), U256::from(2u64)], 0, HIGH_GAS);
    pre_l.jumpdest_table = [10, 20].into_iter().collect();
    let mut post_l = st_gas(vec![U256::from(3u64)], 1, HIGH_GAS - 3);
    post_l.jumpdest_table = [10, 20].into_iter().collect();

    let mut pre_r = st_gas(vec![U256::from(3u64)], 1, HIGH_GAS - 3);
    pre_r.jumpdest_table = [10, 30].into_iter().collect(); // differs!
    let post_r = st_gas(vec![], 2, HIGH_GAS - 5);

    let seq = ProofNode::Seq {
      left: Box::new(leaf(opcode::ADD, pre_l, post_l)),
      right: Box::new(leaf(opcode::POP, pre_r, post_r)),
    };
    let err = consistency_check(&seq).unwrap_err();
    assert!(matches!(err, ConsistencyError::SeqJumpdestMismatch));
  }

  // ────────────────── collect_all ────────────────────────────────────────

  #[test]
  fn collect_all_errors() {
    // Seq with both gas mismatch AND stack value mismatch.
    let left = leaf(
      opcode::ADD,
      st_gas(vec![U256::from(1u64), U256::from(2u64)], 0, 1000),
      st_gas(vec![U256::from(3u64)], 1, 997),
    );
    let right = leaf(
      opcode::POP,
      st_gas(vec![U256::from(999u64)], 1, 990),
      st_gas(vec![], 2, 988),
    );
    let seq = ProofNode::Seq {
      left: Box::new(left),
      right: Box::new(right),
    };
    let errs = consistency_check_all(&seq);
    // Should find at least PC-match issue or gas-match issue at the boundary.
    assert!(!errs.is_empty());
  }

  // ────────────────── Gas cost per-opcode ────────────────────────────────

  #[test]
  fn gas_cost_too_low_static() {
    // ADD costs 3.  Give it only 1 gas consumed.
    let node = leaf(
      opcode::ADD,
      st_gas(vec![U256::from(1u64), U256::from(2u64)], 0, 100),
      st_gas(vec![U256::from(3u64)], 1, 99), // consumed=1 < 3
    );
    let err = consistency_check(&node).unwrap_err();
    assert!(matches!(
      err,
      ConsistencyError::GasCostTooLow {
        opcode: 0x01,
        consumed: 1,
        min_cost: 3,
        ..
      }
    ));
  }

  #[test]
  fn gas_cost_mismatch_non_dynamic() {
    // ADD costs exactly 3.  Give it 5 consumed — exact mismatch.
    let node = leaf(
      opcode::ADD,
      st_gas(vec![U256::from(1u64), U256::from(2u64)], 0, 100),
      st_gas(vec![U256::from(3u64)], 1, 95), // consumed=5 ≠ 3
    );
    let err = consistency_check(&node).unwrap_err();
    assert!(matches!(
      err,
      ConsistencyError::GasCostMismatch {
        opcode: 0x01,
        consumed: 5,
        expected: 3,
        ..
      }
    ));
  }

  #[test]
  fn gas_cost_dynamic_over_minimum_ok() {
    // MLOAD static_gas=3, dynamic=true.  consumed=10 >= 3 is fine.
    let node = leaf(
      opcode::MLOAD,
      {
        let mut s = st_gas(vec![U256::from(0u64)], 0, 100);
        s.memory = vec![0u8; 32];
        s
      },
      {
        let mut s = st_gas(vec![U256::ZERO], 1, 90); // consumed=10 > 3
        s.memory = vec![0u8; 32];
        s
      },
    );
    assert!(consistency_check(&node).is_ok());
  }

  #[test]
  fn gas_cost_tstore_exact() {
    // TSTORE costs exactly 100, non-dynamic.
    let node = leaf(
      opcode::TSTORE,
      st_gas(vec![U256::from(1u64), U256::from(42u64)], 0, 200),
      st_gas_tstor(vec![], 1, 100, &[(1, 42)]), // consumed=100 ✓
    );
    assert!(consistency_check(&node).is_ok());
  }

  #[test]
  fn gas_cost_tstore_wrong() {
    // TSTORE costs exactly 100, but give 50.
    let node = leaf(
      opcode::TSTORE,
      st_gas(vec![U256::from(1u64), U256::from(42u64)], 0, 200),
      st_gas_tstor(vec![], 1, 150, &[(1, 42)]), // consumed=50 < 100
    );
    let err = consistency_check(&node).unwrap_err();
    assert!(matches!(
      err,
      ConsistencyError::GasCostTooLow {
        opcode: 0x5d,
        consumed: 50,
        min_cost: 100,
        ..
      }
    ));
  }
}
