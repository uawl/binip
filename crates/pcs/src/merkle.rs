//! Blake3-based binary Merkle tree used by the Tensor PCS.
//!
//! Leaves are padded to the next power-of-two with the all-zero hash.
//! The tree is stored in a 1-indexed heap layout:
//!   nodes[1]      = root
//!   nodes[n..2n-1] = leaves (n = next_power_of_two(n_leaves))
//!   parent(i)     = i / 2
//!   children(i)   = 2i, 2i+1

/// A 32-byte Blake3 hash.
pub type Hash = [u8; 32];

// ─── domain-separated hash helpers ───────────────────────────────────────────

fn hash_inner(left: &Hash, right: &Hash) -> Hash {
  let mut h = blake3::Hasher::new();
  h.update(b"binip:pcs:merkle:inner:");
  h.update(left);
  h.update(right);
  *h.finalize().as_bytes()
}

// ─── MerkleTree ───────────────────────────────────────────────────────────────

/// Complete binary Merkle tree over a slice of leaf hashes.
pub struct MerkleTree {
  /// n = next_power_of_two(n_leaves).  Padding leaves are all-zero hashes.
  pub n: usize,
  /// 1-indexed storage.  nodes[0] unused.
  nodes: Vec<Hash>,
}

impl MerkleTree {
  /// Build the tree from the given leaf hashes.  O(n) time.
  pub fn build(leaves: &[Hash]) -> Self {
    let n = leaves.len().next_power_of_two().max(1);
    let mut nodes = vec![[0u8; 32]; 2 * n];
    for (i, h) in leaves.iter().enumerate() {
      nodes[n + i] = *h;
    }
    // inner nodes bottom-up
    for i in (1..n).rev() {
      nodes[i] = hash_inner(&nodes[2 * i], &nodes[2 * i + 1]);
    }
    Self { n, nodes }
  }

  /// The Merkle root.
  pub fn root(&self) -> Hash {
    self.nodes[1]
  }

  /// Generate an inclusion proof for the leaf at `idx` (0-based).
  pub fn prove(&self, idx: usize) -> MerkleProof {
    assert!(idx < self.n);
    let mut siblings = Vec::new();
    let mut pos = self.n + idx;
    while pos > 1 {
      let sib = if pos % 2 == 0 { pos + 1 } else { pos - 1 };
      siblings.push(self.nodes[sib]);
      pos /= 2;
    }
    MerkleProof { idx, siblings }
  }
}

// ─── MerkleProof ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct MerkleProof {
  pub idx: usize,
  /// Sibling hashes from leaf level up to (but not including) the root.
  pub siblings: Vec<Hash>,
}

impl MerkleProof {
  /// Verify this proof.  `n_leaves` must match the committed tree size.
  pub fn verify(&self, leaf: Hash, root: Hash, n: usize) -> bool {
    let mut current = leaf;
    let mut pos = n + self.idx;
    for &sib in &self.siblings {
      current = if pos % 2 == 0 {
        hash_inner(&current, &sib)
      } else {
        hash_inner(&sib, &current)
      };
      pos /= 2;
    }
    current == root
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn leaf_hash(data: &[u8]) -> Hash {
    *blake3::hash(data).as_bytes()
  }

  #[test]
  fn build_and_root_deterministic() {
    let leaves: Vec<Hash> = (0u8..4).map(|i| leaf_hash(&[i])).collect();
    let t1 = MerkleTree::build(&leaves);
    let t2 = MerkleTree::build(&leaves);
    assert_eq!(t1.root(), t2.root());
  }

  #[test]
  fn prove_verify_all_leaves() {
    let leaves: Vec<Hash> = (0u8..8).map(|i| leaf_hash(&[i])).collect();
    let tree = MerkleTree::build(&leaves);
    let root = tree.root();
    for (i, &lf) in leaves.iter().enumerate() {
      let proof = tree.prove(i);
      assert!(proof.verify(lf, root, tree.n), "leaf {i} should verify");
    }
  }

  #[test]
  fn proof_rejects_wrong_leaf() {
    let leaves: Vec<Hash> = (0u8..4).map(|i| leaf_hash(&[i])).collect();
    let tree = MerkleTree::build(&leaves);
    let root = tree.root();
    let proof = tree.prove(0);
    let wrong_leaf = leaf_hash(&[0xFF]);
    assert!(!proof.verify(wrong_leaf, root, tree.n));
  }

  #[test]
  fn proof_rejects_wrong_root() {
    let leaves: Vec<Hash> = (0u8..4).map(|i| leaf_hash(&[i])).collect();
    let tree = MerkleTree::build(&leaves);
    let proof = tree.prove(1);
    let wrong_root = [0xAB; 32];
    assert!(!proof.verify(leaves[1], wrong_root, tree.n));
  }

  #[test]
  fn non_power_of_two_leaves() {
    // 3 leaves → padded to 4
    let leaves: Vec<Hash> = (0u8..3).map(|i| leaf_hash(&[i])).collect();
    let tree = MerkleTree::build(&leaves);
    assert_eq!(tree.n, 4);
    let root = tree.root();
    for (i, &lf) in leaves.iter().enumerate() {
      let proof = tree.prove(i);
      assert!(proof.verify(lf, root, tree.n), "leaf {i} should verify");
    }
  }

  #[test]
  fn single_leaf() {
    let leaves = vec![leaf_hash(b"only")];
    let tree = MerkleTree::build(&leaves);
    assert_eq!(tree.n, 1);
    let proof = tree.prove(0);
    assert!(proof.verify(leaves[0], tree.root(), tree.n));
    assert_eq!(proof.siblings.len(), 0);
  }

  #[test]
  fn two_leaves_sibling_swap() {
    let leaves = vec![leaf_hash(b"L"), leaf_hash(b"R")];
    let tree = MerkleTree::build(&leaves);
    let root = tree.root();
    // Proof for index 0 should have sibling = leaves[1] hash
    let p0 = tree.prove(0);
    let p1 = tree.prove(1);
    assert!(p0.verify(leaves[0], root, tree.n));
    assert!(p1.verify(leaves[1], root, tree.n));
    // Cross-verify must fail
    assert!(!p0.verify(leaves[1], root, tree.n));
    assert!(!p1.verify(leaves[0], root, tree.n));
  }

  #[test]
  fn proof_depth_is_log2() {
    let leaves: Vec<Hash> = (0u8..16).map(|i| leaf_hash(&[i])).collect();
    let tree = MerkleTree::build(&leaves);
    let proof = tree.prove(5);
    // log2(16) = 4 siblings
    assert_eq!(proof.siblings.len(), 4);
  }
}
