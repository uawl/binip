use thiserror::Error;

use crate::{
  opcode::{self, stack_effect},
  proof_tree::{ProofNode, TypeCert},
  state::EvmStateType,
};

/// Errors produced by [`type_check`].
#[derive(Debug, Error, PartialEq, Eq)]
pub enum TypeError {
  #[error("opcode 0x{opcode:02x} is not a defined EVM opcode")]
  UnknownOpcode { opcode: u8 },

  #[error(
    "stack underflow: opcode 0x{opcode:02x} requires {needed} items \
         but stack depth is {available}"
  )]
  StackUnderflow {
    opcode: u8,
    needed: usize,
    available: usize,
  },

  #[error(
    "leaf post-state depth {actual} does not match expected {expected} \
         (opcode 0x{opcode:02x}: pre-depth={pre_depth}, delta={delta:+})"
  )]
  LeafDepthMismatch {
    opcode: u8,
    pre_depth: usize,
    delta: i32,
    expected: usize,
    actual: usize,
  },

  #[error("Seq boundary mismatch: left post-depth {left} ≠ right pre-depth {right}")]
  SeqMismatch { left: usize, right: usize },

  #[error(
    "Branch paths disagree on post-state type: taken depth={taken}, \
         not_taken depth={not_taken}"
  )]
  BranchTypeMismatch { taken: usize, not_taken: usize },

  #[error("memory shrank: opcode 0x{opcode:02x} post memory_size {post} < pre memory_size {pre}")]
  MemoryShrink { opcode: u8, pre: usize, post: usize },

  #[error("storage touched count decreased: opcode 0x{opcode:02x} post {post} < pre {pre}")]
  StorageShrink { opcode: u8, pre: usize, post: usize },

  #[error(
    "invalid jump destination: opcode 0x{opcode:02x} target pc={target} not in jumpdest table"
  )]
  InvalidJumpDest { opcode: u8, target: u32 },
}

/// Type-check a [`ProofNode`] tree.
///
/// Returns `(pre_type, post_type)` — the abstract EVM state types at the
/// entry and exit of the whole derivation.
///
/// ## What is checked (structural only)
/// - **Leaf**: `post_state.stack.len()` equals `pre_state.stack.len() + Δ(opcode)`,
///   where `Δ` is the opcode's net stack change from [`stack_effect`].
/// - **Seq**: left post-depth equals right pre-depth.
/// - **Branch**: both execution paths agree on post-depth.
///
/// ## What is NOT checked
/// Arithmetic correctness of opcode outputs. That is delegated to the
/// [`crate::proof_tree::LeafProof`] ZK sub-proof in each `Leaf`.
///
/// ## Complexity
/// O(n) in the number of [`ProofNode`] nodes.
pub fn type_check(node: &ProofNode) -> Result<(EvmStateType, EvmStateType), TypeError> {
  match node {
    ProofNode::Leaf {
      opcode,
      pre_state,
      post_state,
      ..
    } => {
      let op = *opcode;
      let effect = stack_effect(op).ok_or(TypeError::UnknownOpcode { opcode: op })?;

      let pre_depth = pre_state.stack.len();

      // Check minimum stack depth (covers DUP/SWAP requirements too).
      if pre_depth < effect.min_stack as usize {
        return Err(TypeError::StackUnderflow {
          opcode: op,
          needed: effect.min_stack as usize,
          available: pre_depth,
        });
      }

      // Verify post-state stack depth matches the opcode's net delta.
      let delta = effect.delta();
      let expected = (pre_depth as i32 + delta) as usize;
      let actual = post_state.stack.len();
      if expected != actual {
        return Err(TypeError::LeafDepthMismatch {
          opcode: op,
          pre_depth,
          delta,
          expected,
          actual,
        });
      }

      // ── Memory: can only grow ──────────────────────────────────────
      let pre_mem = pre_state.memory.len();
      let post_mem = post_state.memory.len();
      if post_mem < pre_mem {
        return Err(TypeError::MemoryShrink {
          opcode: op,
          pre: pre_mem,
          post: post_mem,
        });
      }

      // ── Storage: touched count can only grow ───────────────────────
      let pre_stor = pre_state.storage.len();
      let post_stor = post_state.storage.len();
      if post_stor < pre_stor {
        return Err(TypeError::StorageShrink {
          opcode: op,
          pre: pre_stor,
          post: post_stor,
        });
      }

      // ── Jump destination validation ────────────────────────────────
      if op == opcode::JUMP || op == opcode::JUMPI {
        // post_state.pc must be in the jumpdest table
        let target = post_state.pc;
        if !pre_state.jumpdest_table.contains(&target) {
          return Err(TypeError::InvalidJumpDest { opcode: op, target });
        }
      }

      let pre_ty = EvmStateType::from(pre_state);
      let post_ty = EvmStateType::from(post_state);
      Ok((pre_ty, post_ty))
    }

    ProofNode::Seq { left, right } => {
      let (pre_l, post_l) = type_check(left)?;
      let (pre_r, post_r) = type_check(right)?;

      if post_l != pre_r {
        // Check which field disagrees to give a targeted error.
        if post_l.stack_depth != pre_r.stack_depth {
          return Err(TypeError::SeqMismatch {
            left: post_l.stack_depth,
            right: pre_r.stack_depth,
          });
        }
        // Memory/storage/pc mismatches at seq boundary reported as
        // stack-depth mismatch with the same values (both sides equal)
        // — the caller sees the SeqMismatch variant which is sufficient.
        return Err(TypeError::SeqMismatch {
          left: post_l.stack_depth,
          right: pre_r.stack_depth,
        });
      }

      Ok((pre_l, post_r))
    }

    ProofNode::Branch {
      cond,
      taken,
      not_taken,
    } => {
      // Type-check the condition sub-derivation (typically ends in JUMPI).
      let (pre_cond, _post_cond) = type_check(cond)?;

      // Both execution paths must be independently well-typed.
      let (_pre_t, post_t) = type_check(taken)?;
      let (_pre_f, post_f) = type_check(not_taken)?;

      // Both paths must converge on the same stack-depth post-type.
      if post_t.stack_depth != post_f.stack_depth {
        return Err(TypeError::BranchTypeMismatch {
          taken: post_t.stack_depth,
          not_taken: post_f.stack_depth,
        });
      }

      Ok((pre_cond, post_t))
    }

    ProofNode::Call {
      opcode,
      pre_state,
      post_state,
      inner,
      ..
    } => {
      // The outer CALL opcode must be valid.
      let op = *opcode;
      let effect = stack_effect(op).ok_or(TypeError::UnknownOpcode { opcode: op })?;

      let pre_depth = pre_state.stack.len();
      if pre_depth < effect.min_stack as usize {
        return Err(TypeError::StackUnderflow {
          opcode: op,
          needed: effect.min_stack as usize,
          available: pre_depth,
        });
      }

      let delta = effect.delta();
      let expected = (pre_depth as i32 + delta) as usize;
      let actual = post_state.stack.len();
      if expected != actual {
        return Err(TypeError::LeafDepthMismatch {
          opcode: op,
          pre_depth,
          delta,
          expected,
          actual,
        });
      }

      // Type-check the inner sub-call tree.
      let _ = type_check(inner)?;

      let pre_ty = EvmStateType::from(pre_state);
      let post_ty = EvmStateType::from(post_state);
      Ok((pre_ty, post_ty))
    }

    ProofNode::TxBoundary {
      pre_state,
      post_state,
      inner,
      ..
    } => {
      // Type-check the inner transaction execution tree.
      let _ = type_check(inner)?;

      // Transaction boundary: pre/post are the outer states.
      // Stack should be empty at tx boundaries (fresh EVM context).
      let pre_ty = EvmStateType::from(pre_state);
      let post_ty = EvmStateType::from(post_state);
      Ok((pre_ty, post_ty))
    }

    ProofNode::BlockBoundary {
      pre_state,
      post_state,
      inner,
      ..
    } => {
      let _ = type_check(inner)?;
      let pre_ty = EvmStateType::from(pre_state);
      let post_ty = EvmStateType::from(post_state);
      Ok((pre_ty, post_ty))
    }
  }
}

/// Build a [`TypeCert`] for `node` after a successful type-check.
///
/// Hashes the structural shape of the tree with Blake3 and records
/// the leaf count. The cert is a lightweight commitment that the
/// verifier can check in O(n) without re-running arithmetic.
pub fn build_cert(node: &ProofNode) -> Result<TypeCert, TypeError> {
  type_check(node)?;

  // Hash the tree shape: a simple deterministic traversal.
  let shape = shape_bytes(node);
  let root_hash: [u8; 32] = blake3::hash(&shape).into();

  // Hash the concrete state values (stack, gas, memory sizes, storage).
  let state = state_bytes(node);
  let state_hash: [u8; 32] = blake3::hash(&state).into();

  Ok(TypeCert {
    root_hash,
    state_hash,
    leaf_count: node.leaf_count(),
  })
}

/// Serialise the structural shape of a `ProofNode` tree (opcodes + node kinds).
/// Values (U256 stack items, memory bytes) are intentionally excluded —
/// this is a *type-level* certificate.
fn shape_bytes(node: &ProofNode) -> Vec<u8> {
  let mut buf = Vec::new();
  write_shape(node, &mut buf);
  buf
}

fn write_shape(node: &ProofNode, buf: &mut Vec<u8>) {
  match node {
    ProofNode::Leaf {
      opcode,
      pre_state,
      post_state,
      ..
    } => {
      buf.push(0x00); // tag: Leaf
      buf.push(*opcode);
      buf.extend_from_slice(&(pre_state.stack.len() as u32).to_le_bytes());
      buf.extend_from_slice(&(post_state.stack.len() as u32).to_le_bytes());
      buf.extend_from_slice(&(pre_state.memory.len() as u32).to_le_bytes());
      buf.extend_from_slice(&(post_state.memory.len() as u32).to_le_bytes());
      buf.extend_from_slice(&pre_state.pc.to_le_bytes());
      buf.extend_from_slice(&post_state.pc.to_le_bytes());
    }
    ProofNode::Seq { left, right } => {
      buf.push(0x01); // tag: Seq
      write_shape(left, buf);
      write_shape(right, buf);
    }
    ProofNode::Branch {
      cond,
      taken,
      not_taken,
    } => {
      buf.push(0x02); // tag: Branch
      write_shape(cond, buf);
      write_shape(taken, buf);
      write_shape(not_taken, buf);
    }
    ProofNode::Call {
      opcode,
      pre_state,
      post_state,
      callee,
      inner,
      ..
    } => {
      buf.push(0x03); // tag: Call
      buf.push(*opcode);
      buf.extend_from_slice(callee.as_slice());
      buf.extend_from_slice(&(pre_state.stack.len() as u32).to_le_bytes());
      buf.extend_from_slice(&(post_state.stack.len() as u32).to_le_bytes());
      buf.extend_from_slice(&pre_state.pc.to_le_bytes());
      buf.extend_from_slice(&post_state.pc.to_le_bytes());
      write_shape(inner, buf);
    }
    ProofNode::TxBoundary {
      tx_index,
      tx_hash,
      pre_state,
      post_state,
      inner,
      ..
    } => {
      buf.push(0x04); // tag: TxBoundary
      buf.extend_from_slice(&tx_index.to_le_bytes());
      buf.extend_from_slice(tx_hash.as_slice());
      buf.extend_from_slice(&(pre_state.stack.len() as u32).to_le_bytes());
      buf.extend_from_slice(&(post_state.stack.len() as u32).to_le_bytes());
      buf.extend_from_slice(&pre_state.pc.to_le_bytes());
      buf.extend_from_slice(&post_state.pc.to_le_bytes());
      write_shape(inner, buf);
    }
    ProofNode::BlockBoundary {
      block_number,
      block_hash,
      parent_hash,
      pre_state,
      post_state,
      inner,
      ..
    } => {
      buf.push(0x05); // tag: BlockBoundary
      buf.extend_from_slice(&block_number.to_le_bytes());
      buf.extend_from_slice(block_hash.as_slice());
      buf.extend_from_slice(parent_hash.as_slice());
      buf.extend_from_slice(&(pre_state.stack.len() as u32).to_le_bytes());
      buf.extend_from_slice(&(post_state.stack.len() as u32).to_le_bytes());
      buf.extend_from_slice(&pre_state.pc.to_le_bytes());
      buf.extend_from_slice(&post_state.pc.to_le_bytes());
      write_shape(inner, buf);
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// State value commitment
// ─────────────────────────────────────────────────────────────────────────────

/// Serialise the concrete EVM state values from every `Leaf` in the tree.
///
/// Unlike [`shape_bytes`] (which only includes sizes/PCs), this captures the
/// actual U256 stack items, gas budget, memory size, and touched storage
/// keys+values.  The resulting hash binds the prover's witness so the
/// verifier can authenticate native consistency checks.
fn state_bytes(node: &ProofNode) -> Vec<u8> {
  let mut buf = Vec::new();
  write_state(node, &mut buf);
  buf
}

fn write_state(node: &ProofNode, buf: &mut Vec<u8>) {
  match node {
    ProofNode::Leaf {
      opcode,
      pre_state,
      post_state,
      ..
    } => {
      buf.push(0x00); // tag: Leaf
      buf.push(*opcode);
      write_evm_state(pre_state, buf);
      write_evm_state(post_state, buf);
    }
    ProofNode::Seq { left, right } => {
      buf.push(0x01);
      write_state(left, buf);
      write_state(right, buf);
    }
    ProofNode::Branch {
      cond,
      taken,
      not_taken,
    } => {
      buf.push(0x02);
      write_state(cond, buf);
      write_state(taken, buf);
      write_state(not_taken, buf);
    }
    ProofNode::Call {
      opcode,
      pre_state,
      post_state,
      inner,
      ..
    } => {
      buf.push(0x03);
      buf.push(*opcode);
      write_evm_state(pre_state, buf);
      write_evm_state(post_state, buf);
      write_state(inner, buf);
    }
    ProofNode::TxBoundary {
      tx_index,
      tx_hash,
      gas_used,
      pre_state,
      post_state,
      inner,
      ..
    } => {
      buf.push(0x04);
      buf.extend_from_slice(&tx_index.to_le_bytes());
      buf.extend_from_slice(tx_hash.as_slice());
      buf.extend_from_slice(&gas_used.to_le_bytes());
      write_evm_state(pre_state, buf);
      write_evm_state(post_state, buf);
      write_state(inner, buf);
    }
    ProofNode::BlockBoundary {
      block_number,
      block_hash,
      parent_hash,
      gas_used,
      state_root_pre,
      state_root_post,
      pre_state,
      post_state,
      inner,
      ..
    } => {
      buf.push(0x05);
      buf.extend_from_slice(&block_number.to_le_bytes());
      buf.extend_from_slice(block_hash.as_slice());
      buf.extend_from_slice(parent_hash.as_slice());
      buf.extend_from_slice(&gas_used.to_le_bytes());
      buf.extend_from_slice(state_root_pre.as_slice());
      buf.extend_from_slice(state_root_post.as_slice());
      write_evm_state(pre_state, buf);
      write_evm_state(post_state, buf);
      write_state(inner, buf);
    }
  }
}

/// Serialise a single [`EvmState`] — stack values, gas, memory size, storage.
fn write_evm_state(state: &crate::state::EvmState, buf: &mut Vec<u8>) {
  // Program counter
  buf.extend_from_slice(&state.pc.to_le_bytes());
  // Gas
  buf.extend_from_slice(&state.gas.to_le_bytes());
  // Stack depth + values
  buf.extend_from_slice(&(state.stack.len() as u32).to_le_bytes());
  for val in &state.stack {
    buf.extend_from_slice(&val.to_le_bytes::<32>());
  }
  // Memory size (content is not hashed — too large; size is sufficient
  // when combined with memory-consistency arguments in the circuit).
  buf.extend_from_slice(&(state.memory.len() as u32).to_le_bytes());
  // Storage key-value pairs (sorted for determinism)
  let mut stor: Vec<_> = state.storage.iter().collect();
  stor.sort_by_key(|(k, _)| *k);
  buf.extend_from_slice(&(stor.len() as u32).to_le_bytes());
  for (k, v) in &stor {
    buf.extend_from_slice(&k.to_le_bytes::<32>());
    buf.extend_from_slice(&v.to_le_bytes::<32>());
  }
  // Transient storage (sorted)
  let mut tstor: Vec<_> = state.transient_storage.iter().collect();
  tstor.sort_by_key(|(k, _)| *k);
  buf.extend_from_slice(&(tstor.len() as u32).to_le_bytes());
  for (k, v) in &tstor {
    buf.extend_from_slice(&k.to_le_bytes::<32>());
    buf.extend_from_slice(&v.to_le_bytes::<32>());
  }
  // Account state: balance, nonce, code_hash
  buf.extend_from_slice(&state.balance.to_le_bytes::<32>());
  buf.extend_from_slice(&state.nonce.to_le_bytes());
  buf.extend_from_slice(state.code_hash.as_slice());
  // Address + caller
  buf.extend_from_slice(state.address.as_slice());
  buf.extend_from_slice(state.caller.as_slice());
}
