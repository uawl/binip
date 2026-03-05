//! Phase C: Sparse Merkle Tree (SMT) inclusion proofs for storage state binding.
//!
//! Binds [`RwSummary`] initial/final values to pre/post state roots via
//! Blake3-based 128-bit sparse Merkle trees.
//!
//! # Architecture
//!
//! Storage addresses (`RwSummary.addr`) live in a 128-bit key space.
//! Each leaf is `hash_leaf(addr, value)`.  Empty subtrees use
//! level-dependent default hashes (precomputed).
//!
//! The prover supplies inclusion proofs for every storage slot accessed
//! during execution.  The verifier checks:
//! 1. Each `initial_value` is in the pre-state tree at `addr`.
//! 2. Each `final_value` is in the post-state tree at `addr`.
//! 3. Both roots are absorbed into the Fiat-Shamir transcript to bind
//!    them to the rest of the proof.

use transcript::{Blake3Transcript, Transcript};

use crate::lookup::{RwSummary, RwSummaries};

/// Depth of the sparse Merkle tree (128-bit key space).
pub const SMT_DEPTH: usize = 128;

/// Domain-separated leaf hash: `Blake3("binip:smt:leaf:" ‖ addr ‖ value)`.
pub fn hash_leaf(addr: u128, value: u128) -> [u8; 32] {
  let mut h = blake3::Hasher::new();
  h.update(b"binip:smt:leaf:");
  h.update(&addr.to_le_bytes());
  h.update(&value.to_le_bytes());
  *h.finalize().as_bytes()
}

/// Domain-separated inner hash: `Blake3("binip:smt:inner:" ‖ left ‖ right)`.
fn hash_inner(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
  let mut h = blake3::Hasher::new();
  h.update(b"binip:smt:inner:");
  h.update(left);
  h.update(right);
  *h.finalize().as_bytes()
}

/// Default (empty subtree) hash at each level.
///
/// - `DEFAULT_HASHES[0]` = hash of an empty leaf (addr=0, value=0 conceptually,
///   but we define it as the all-zero hash — the canonical "empty" marker).
/// - `DEFAULT_HASHES[d]` = `hash_inner(DEFAULT_HASHES[d-1], DEFAULT_HASHES[d-1])`.
///
/// Computed lazily via [`default_hash`].
fn default_hash(depth: usize) -> [u8; 32] {
  // Use a simple recursive computation.  In production this would be
  // cached, but 128 Blake3 evaluations is negligible.
  if depth == 0 {
    return [0u8; 32]; // canonical empty leaf
  }
  let child = default_hash(depth - 1);
  hash_inner(&child, &child)
}

// ─── SMT Inclusion Proof ─────────────────────────────────────────────────────

/// Inclusion proof for a single key in a 128-bit sparse Merkle tree.
#[derive(Debug, Clone, bincode::Encode, bincode::Decode)]
pub struct SmtProof {
  /// Sibling hashes from leaf (level 0) up to the root (level SMT_DEPTH).
  /// `siblings[i]` is the sibling at level `i`.
  pub siblings: [[u8; 32]; SMT_DEPTH],
}

impl SmtProof {
  /// Verify that `(addr, value)` is included under `root`.
  ///
  /// The path through the tree is determined by the bits of `addr`:
  /// bit 0 (LSB) chooses left/right at the leaf level, bit 127 at the top.
  pub fn verify(&self, addr: u128, value: u128, root: [u8; 32]) -> bool {
    let mut current = hash_leaf(addr, value);
    for level in 0..SMT_DEPTH {
      let bit = (addr >> level) & 1;
      current = if bit == 0 {
        hash_inner(&current, &self.siblings[level])
      } else {
        hash_inner(&self.siblings[level], &current)
      };
    }
    current == root
  }

  /// Compute the root that would result from inserting `(addr, value)`.
  ///
  /// Same traversal as [`verify`](Self::verify) but returns the computed root.
  pub fn compute_root(&self, addr: u128, value: u128) -> [u8; 32] {
    let mut current = hash_leaf(addr, value);
    for level in 0..SMT_DEPTH {
      let bit = (addr >> level) & 1;
      current = if bit == 0 {
        hash_inner(&current, &self.siblings[level])
      } else {
        hash_inner(&self.siblings[level], &current)
      };
    }
    current
  }
}

// ─── Storage State Proof ─────────────────────────────────────────────────────

/// Phase C proof: binds storage R/W summaries to pre/post state roots.
///
/// For every storage slot accessed during execution (each `RwSummary` in
/// `RwSummaries.storage`), provides SMT inclusion proofs showing:
/// - `initial_value` exists at `addr` in the pre-state tree.
/// - `final_value` exists at `addr` in the post-state tree.
#[derive(Debug, Clone, bincode::Encode, bincode::Decode)]
pub struct StorageProof {
  /// SMT root of storage state before execution.
  pub pre_root: [u8; 32],
  /// SMT root of storage state after execution.
  pub post_root: [u8; 32],
  /// Inclusion proof per storage slot in the pre-state tree.
  /// Order must match `RwSummaries.storage`.
  pub pre_proofs: Vec<SmtProof>,
  /// Inclusion proof per storage slot in the post-state tree.
  /// Order must match `RwSummaries.storage`.
  pub post_proofs: Vec<SmtProof>,
}

/// Error type for Phase C verification.
#[derive(Debug, Clone, thiserror::Error)]
pub enum StorageProofError {
  #[error("proof count mismatch: expected {expected} storage summaries, got {pre_count} pre-proofs and {post_count} post-proofs")]
  CountMismatch {
    expected: usize,
    pre_count: usize,
    post_count: usize,
  },
  #[error("pre-state inclusion failed for slot {addr:#x} (initial_value={value:#x})")]
  PreInclusionFailed { addr: u128, value: u128 },
  #[error("post-state inclusion failed for slot {addr:#x} (final_value={value:#x})")]
  PostInclusionFailed { addr: u128, value: u128 },
}

// ─── Prove / Verify ──────────────────────────────────────────────────────────

/// Build a [`StorageProof`] from an SMT and the verified storage summaries.
///
/// The prover holds the full sparse Merkle tree for pre-state and post-state.
/// This function extracts inclusion proofs for each summary slot.
///
/// `pre_tree` / `post_tree` are functions that, given an address, return the
/// SMT inclusion proof.  This allows the caller to supply any tree backend.
pub fn prove_storage(
  summaries: &[RwSummary],
  pre_root: [u8; 32],
  post_root: [u8; 32],
  pre_tree: impl Fn(u128) -> SmtProof,
  post_tree: impl Fn(u128) -> SmtProof,
  transcript: &mut Blake3Transcript,
) -> StorageProof {
  transcript.absorb_bytes(b"phase_c:storage");
  transcript.absorb_bytes(&pre_root);
  transcript.absorb_bytes(&post_root);
  transcript.absorb_bytes(&(summaries.len() as u64).to_le_bytes());

  let pre_proofs: Vec<SmtProof> = summaries.iter().map(|s| pre_tree(s.addr)).collect();
  let post_proofs: Vec<SmtProof> = summaries.iter().map(|s| post_tree(s.addr)).collect();

  StorageProof { pre_root, post_root, pre_proofs, post_proofs }
}

/// Verify a [`StorageProof`] against the storage summaries from Phase B.
///
/// Checks:
/// 1. Each `initial_value` is in the pre-state SMT at the correct address.
/// 2. Each `final_value` is in the post-state SMT at the correct address.
/// 3. Absorbs pre/post roots into the transcript (matching prover ordering).
///
/// Returns `Ok(())` on success so the caller can propagate errors.
pub fn verify_storage(
  proof: &StorageProof,
  summaries: &[RwSummary],
  transcript: &mut Blake3Transcript,
) -> Result<(), StorageProofError> {
  // Absorb in the same order as the prover.
  transcript.absorb_bytes(b"phase_c:storage");
  transcript.absorb_bytes(&proof.pre_root);
  transcript.absorb_bytes(&proof.post_root);
  transcript.absorb_bytes(&(summaries.len() as u64).to_le_bytes());

  if proof.pre_proofs.len() != summaries.len() || proof.post_proofs.len() != summaries.len() {
    return Err(StorageProofError::CountMismatch {
      expected: summaries.len(),
      pre_count: proof.pre_proofs.len(),
      post_count: proof.post_proofs.len(),
    });
  }

  for (i, summary) in summaries.iter().enumerate() {
    // Pre-state: initial_value at addr
    if !proof.pre_proofs[i].verify(summary.addr, summary.initial_value, proof.pre_root) {
      return Err(StorageProofError::PreInclusionFailed {
        addr: summary.addr,
        value: summary.initial_value,
      });
    }
    // Post-state: final_value at addr
    if !proof.post_proofs[i].verify(summary.addr, summary.final_value, proof.post_root) {
      return Err(StorageProofError::PostInclusionFailed {
        addr: summary.addr,
        value: summary.final_value,
      });
    }
  }

  Ok(())
}

// ─── Sparse Merkle Tree (full in-memory, for prover) ────────────────────────

/// In-memory sparse Merkle tree for the prover.
///
/// Stores only non-empty leaves.  The tree is implicitly 2^128 wide;
/// empty subtrees use precomputed [`default_hash`] values.
#[derive(Debug, Clone)]
pub struct SparseMerkleTree {
  /// Non-empty leaves: `addr → value`.
  leaves: std::collections::BTreeMap<u128, u128>,
}

impl SparseMerkleTree {
  /// Create an empty SMT.
  pub fn new() -> Self {
    Self { leaves: std::collections::BTreeMap::new() }
  }

  /// Insert or update a leaf.
  pub fn insert(&mut self, addr: u128, value: u128) {
    self.leaves.insert(addr, value);
  }

  /// Get the value at an address (0 if absent).
  pub fn get(&self, addr: u128) -> u128 {
    self.leaves.get(&addr).copied().unwrap_or(0)
  }

  /// Compute the SMT root.
  ///
  /// This is O(n log n) where n = number of non-empty leaves.
  /// For small trees (hundreds of slots), this is fast enough.
  pub fn root(&self) -> [u8; 32] {
    self.subtree_hash(0, SMT_DEPTH)
  }

  /// Generate an inclusion proof for `addr`.
  pub fn prove(&self, addr: u128) -> SmtProof {
    let mut siblings = [[0u8; 32]; SMT_DEPTH];
    for level in 0..SMT_DEPTH {
      // Flip bit `level` to get the sibling's subtree prefix.
      let sibling_addr = addr ^ (1u128 << level);
      siblings[level] = self.subtree_hash_at(sibling_addr, level);
    }
    siblings
      .iter()
      .copied()
      .collect::<Vec<_>>()
      .try_into()
      .map(|s| SmtProof { siblings: s })
      .unwrap()
  }

  /// Hash of the subtree rooted at node covering addresses matching
  /// `prefix` in bits `[depth, SMT_DEPTH)`.
  ///
  /// At depth `d`, the subtree covers `2^d` addresses: all addresses whose
  /// bits `[d, 128)` match `prefix_bits`.  i.e., `base..=base+2^d-1` where
  /// `base = prefix_bits & !(2^d - 1)`.
  fn subtree_hash(&self, prefix_bits: u128, depth: usize) -> [u8; 32] {
    if depth == 0 {
      // Leaf level — `prefix_bits` is the full address.
      return match self.leaves.get(&prefix_bits) {
        Some(&value) => hash_leaf(prefix_bits, value),
        None => default_hash(0),
      };
    }

    // Check if any leaf falls in this subtree via range query.
    let (base, end) = if depth >= 128 {
      (0u128, u128::MAX)
    } else {
      let mask = (1u128 << depth) - 1;
      let b = prefix_bits & !mask;
      (b, b | mask)
    };

    let has_any = self.leaves.range(base..=end).next().is_some();
    if !has_any {
      return default_hash(depth);
    }

    let left = self.subtree_hash(prefix_bits & !(1u128 << (depth - 1)), depth - 1);
    let right = self.subtree_hash(prefix_bits | (1u128 << (depth - 1)), depth - 1);
    hash_inner(&left, &right)
  }

  /// Hash of the sibling subtree at `level` for the given `addr`.
  ///
  /// The sibling's subtree covers addresses that differ from `addr` at
  /// bit `level` and share all higher bits.
  fn subtree_hash_at(&self, sibling_addr: u128, level: usize) -> [u8; 32] {
    self.subtree_hash(sibling_addr, level)
  }
}

impl Default for SparseMerkleTree {
  fn default() -> Self {
    Self::new()
  }
}

// ─── Full Phase C entry points ──────────────────────────────────────────────

/// Prove Phase C: generate storage inclusion proofs binding summaries to
/// SMT state roots.
///
/// Takes the pre-state and post-state sparse Merkle trees and the storage
/// summaries output by Phase B.
pub fn prove_phase_c(
  summaries: &RwSummaries,
  pre_tree: &SparseMerkleTree,
  post_tree: &SparseMerkleTree,
  transcript: &mut Blake3Transcript,
) -> Option<StorageProof> {
  if summaries.storage.is_empty() {
    return None;
  }
  let pre_root = pre_tree.root();
  let post_root = post_tree.root();
  Some(prove_storage(
    &summaries.storage,
    pre_root,
    post_root,
    |addr| pre_tree.prove(addr),
    |addr| post_tree.prove(addr),
    transcript,
  ))
}

/// Verify Phase C: check storage inclusion proofs.
///
/// If `storage_proof` is `None`, verifies that there are no storage summaries.
/// Returns `Ok(())` on success.
pub fn verify_phase_c(
  storage_proof: Option<&StorageProof>,
  summaries: &RwSummaries,
  transcript: &mut Blake3Transcript,
) -> Result<(), StorageProofError> {
  match storage_proof {
    Some(proof) => verify_storage(proof, &summaries.storage, transcript),
    None => {
      if summaries.storage.is_empty() {
        Ok(())
      } else {
        Err(StorageProofError::CountMismatch {
          expected: summaries.storage.len(),
          pre_count: 0,
          post_count: 0,
        })
      }
    }
  }
}

// ─── Tests ──────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
  use super::*;
  use transcript::Blake3Transcript;

  #[test]
  fn empty_tree_root_is_default() {
    let tree = SparseMerkleTree::new();
    assert_eq!(tree.root(), default_hash(SMT_DEPTH));
  }

  #[test]
  fn single_leaf_roundtrip() {
    let mut tree = SparseMerkleTree::new();
    tree.insert(42, 999);
    let root = tree.root();
    let proof = tree.prove(42);
    assert!(proof.verify(42, 999, root));
    // Wrong value fails.
    assert!(!proof.verify(42, 1000, root));
    // Wrong addr fails.
    assert!(!proof.verify(43, 999, root));
  }

  #[test]
  fn multiple_leaves() {
    let mut tree = SparseMerkleTree::new();
    tree.insert(1, 100);
    tree.insert(2, 200);
    tree.insert(1000, 300);
    let root = tree.root();

    assert!(tree.prove(1).verify(1, 100, root));
    assert!(tree.prove(2).verify(2, 200, root));
    assert!(tree.prove(1000).verify(1000, 300, root));
    // Absent key with value=0 should NOT verify (it's not a leaf).
    assert!(!tree.prove(3).verify(3, 0, root));
  }

  #[test]
  fn absent_key_default_hash() {
    let tree = SparseMerkleTree::new();
    let root = tree.root();
    // An absent key with value=0 should NOT pass verify because
    // hash_leaf(addr, 0) != default_hash(0) = [0; 32].
    let proof = tree.prove(0);
    assert!(!proof.verify(0, 0, root));
  }

  #[test]
  fn update_changes_root() {
    let mut tree = SparseMerkleTree::new();
    tree.insert(5, 10);
    let root1 = tree.root();
    tree.insert(5, 20);
    let root2 = tree.root();
    assert_ne!(root1, root2);
    // Old proof no longer valid.
    let proof = tree.prove(5);
    assert!(proof.verify(5, 20, root2));
    assert!(!proof.verify(5, 10, root2));
  }

  #[test]
  fn prove_verify_storage_end_to_end() {
    // Pre-state: slot 10 = 0xAA, slot 20 = 0xBB
    let mut pre = SparseMerkleTree::new();
    pre.insert(10, 0xAA);
    pre.insert(20, 0xBB);

    // Post-state: slot 10 = 0xCC (changed), slot 20 = 0xBB (unchanged)
    let mut post = SparseMerkleTree::new();
    post.insert(10, 0xCC);
    post.insert(20, 0xBB);

    let summaries = vec![
      RwSummary { addr: 10, initial_value: 0xAA, final_value: 0xCC },
      RwSummary { addr: 20, initial_value: 0xBB, final_value: 0xBB },
    ];

    let mut tp = Blake3Transcript::new();
    let proof = prove_storage(
      &summaries,
      pre.root(),
      post.root(),
      |addr| pre.prove(addr),
      |addr| post.prove(addr),
      &mut tp,
    );

    let mut tv = Blake3Transcript::new();
    assert!(verify_storage(&proof, &summaries, &mut tv).is_ok());
  }

  #[test]
  fn verify_fails_on_wrong_initial_value() {
    let mut pre = SparseMerkleTree::new();
    pre.insert(10, 0xAA);

    let mut post = SparseMerkleTree::new();
    post.insert(10, 0xCC);

    // Claim initial_value = 0xFF (wrong — tree has 0xAA)
    let summaries = vec![RwSummary { addr: 10, initial_value: 0xFF, final_value: 0xCC }];

    let mut tp = Blake3Transcript::new();
    let proof = prove_storage(
      &summaries,
      pre.root(),
      post.root(),
      |addr| pre.prove(addr),
      |addr| post.prove(addr),
      &mut tp,
    );

    let mut tv = Blake3Transcript::new();
    let result = verify_storage(&proof, &summaries, &mut tv);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), StorageProofError::PreInclusionFailed { .. }));
  }

  #[test]
  fn verify_fails_on_wrong_final_value() {
    let mut pre = SparseMerkleTree::new();
    pre.insert(10, 0xAA);

    let mut post = SparseMerkleTree::new();
    post.insert(10, 0xCC);

    // Claim final_value = 0xDD (wrong — tree has 0xCC)
    let summaries = vec![RwSummary { addr: 10, initial_value: 0xAA, final_value: 0xDD }];

    let mut tp = Blake3Transcript::new();
    let proof = prove_storage(
      &summaries,
      pre.root(),
      post.root(),
      |addr| pre.prove(addr),
      |addr| post.prove(addr),
      &mut tp,
    );

    let mut tv = Blake3Transcript::new();
    let result = verify_storage(&proof, &summaries, &mut tv);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), StorageProofError::PostInclusionFailed { .. }));
  }

  #[test]
  fn phase_c_end_to_end() {
    let mut pre = SparseMerkleTree::new();
    pre.insert(10, 0xAA);
    pre.insert(20, 0xBB);

    let mut post = SparseMerkleTree::new();
    post.insert(10, 0xCC);
    post.insert(20, 0xBB);

    let summaries = RwSummaries {
      mem: vec![],
      emem: vec![],
      storage: vec![
        RwSummary { addr: 10, initial_value: 0xAA, final_value: 0xCC },
        RwSummary { addr: 20, initial_value: 0xBB, final_value: 0xBB },
      ],
      transient: vec![],
    };

    let mut tp = Blake3Transcript::new();
    let proof = prove_phase_c(&summaries, &pre, &post, &mut tp);
    assert!(proof.is_some());

    let mut tv = Blake3Transcript::new();
    assert!(verify_phase_c(proof.as_ref(), &summaries, &mut tv).is_ok());
  }

  #[test]
  fn phase_c_no_storage() {
    let summaries = RwSummaries::default();
    let mut tp = Blake3Transcript::new();
    let proof = prove_phase_c(&summaries, &SparseMerkleTree::new(), &SparseMerkleTree::new(), &mut tp);
    assert!(proof.is_none());

    let mut tv = Blake3Transcript::new();
    assert!(verify_phase_c(None, &summaries, &mut tv).is_ok());
  }

  #[test]
  fn phase_c_missing_proof_fails() {
    let summaries = RwSummaries {
      mem: vec![],
      emem: vec![],
      storage: vec![RwSummary { addr: 1, initial_value: 0, final_value: 0 }],
      transient: vec![],
    };

    let mut tv = Blake3Transcript::new();
    let result = verify_phase_c(None, &summaries, &mut tv);
    assert!(result.is_err());
  }

  #[test]
  fn compute_root_matches_verify() {
    let mut tree = SparseMerkleTree::new();
    tree.insert(7, 77);
    tree.insert(99, 88);
    let root = tree.root();

    let proof = tree.prove(7);
    let computed = proof.compute_root(7, 77);
    assert_eq!(computed, root);
    assert!(proof.verify(7, 77, root));
  }
}
