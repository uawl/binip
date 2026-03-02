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

  #[error(
    "memory shrank: opcode 0x{opcode:02x} post memory_size {post} < pre memory_size {pre}"
  )]
  MemoryShrink {
    opcode: u8,
    pre: usize,
    post: usize,
  },

  #[error(
    "storage touched count decreased: opcode 0x{opcode:02x} post {post} < pre {pre}"
  )]
  StorageShrink {
    opcode: u8,
    pre: usize,
    post: usize,
  },

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
          return Err(TypeError::InvalidJumpDest {
            opcode: op,
            target,
          });
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

  Ok(TypeCert {
    root_hash,
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
  }
}
