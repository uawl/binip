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
    pub fn root(&self) -> Hash { self.nodes[1] }

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
