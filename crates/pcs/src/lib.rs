//! BaseFold PCS — multilinear polynomial commitment over GF(2^128).
//!
//! ## Protocol
//!
//! For an n-variable MLE `P : {0,1}^n → GF(2^128)` with `2^n` evaluations:
//!
//! **Commit**
//! 1. Hash each consecutive pair `(evals[2j], evals[2j+1])` into a Merkle leaf.
//! 2. Build a Merkle tree over the `2^{n-1}` pair-hashes → root.
//! 3. Absorb the root into the Fiat-Shamir transcript.
//!
//! **Open at r = (r_0, r_1, …, r_{n-1})**
//! 1. Iteratively fold the evaluation table: at round `i`, apply
//!    `table'[j] = table[2j] · (1 + r_i) + table[2j+1] · r_i`.
//! 2. At each intermediate round, commit the folded table (Merkle root)
//!    and absorb it into the transcript.
//! 3. After all `n` rounds, a single scalar remains = `P(r)`.
//! 4. Squeeze `n_queries` random leaf positions; for each, reveal the
//!    pair of elements + Merkle proofs at every fold round so the verifier
//!    can re-check each fold step.
//!
//! **Verify**
//! For each query position: walk through the fold rounds, recompute the fold
//! from the revealed pairs, verify Merkle proofs, and check that the final
//! scalar equals the claimed evaluation.
//!
//! ## Key properties
//! - **No encoding**: commit is Merkle-hash of raw evaluations → O(n · 2^n)
//!   field work eliminated.
//! - **Fold = MLE partial evaluation**: same operation used in sumcheck.
//! - **Soundness**: ~(1/2)^n_queries proximity gap per query.

mod eq;
mod merkle;

use field::{FieldElem, GF2_128};
use rayon::prelude::*;
use transcript::Transcript;

pub use merkle::{Hash, MerkleProof};

// ─── params ──────────────────────────────────────────────────────────────────

/// Configuration for the BaseFold PCS.
#[derive(Clone, Debug)]
pub struct PcsParams {
  /// Number of variables in the committed MLE.  Must be ≥ 2.
  pub n_vars: u32,
  /// Number of query positions for proximity testing.
  /// ~40 gives ≥100-bit soundness.
  pub n_queries: usize,
}

// ─── public types ────────────────────────────────────────────────────────────

/// The Merkle-root commitment returned by `commit`.
#[derive(Clone, Debug, bincode::Encode, bincode::Decode)]
pub struct Commitment {
  pub root: Hash,
  pub n_vars: u32,
}

/// Internal prover state needed to answer opening queries.
pub struct PcsState {
  pub commitment: Commitment,
  params: PcsParams,
  /// The original evaluation table (length 2^n_vars).
  evals: Vec<GF2_128>,
  /// Merkle tree over pair-hashed evaluations.
  tree: merkle::MerkleTree,
}

/// Data revealed for one query at one fold round.
///
/// At each round the verifier needs the *pair* of elements that were
/// folded together. One element can be derived from the previous round's
/// fold result (or is given explicitly at round 0), so we store the other
/// one as `sibling` plus a Merkle proof for the pair hash.
#[derive(Clone, Debug, bincode::Encode, bincode::Decode)]
pub struct RoundQuery {
  /// The left element of the pair at this round.
  pub left: GF2_128,
  /// The right element of the pair at this round.
  pub right: GF2_128,
  /// Merkle proof for the pair hash at this round's tree.
  pub merkle_proof: MerkleProof,
}

/// A query path through all fold rounds for one queried position.
#[derive(Clone, Debug, bincode::Encode, bincode::Decode)]
pub struct QueryPath {
  /// One `RoundQuery` per fold round (n_vars entries).
  pub rounds: Vec<RoundQuery>,
}

/// The full opening proof produced by `open`.
#[derive(Clone, Debug, bincode::Encode, bincode::Decode)]
pub struct OpenProof {
  /// Merkle roots of each intermediate folded table (rounds 1..n-1).
  /// Length = n_vars - 1.  Round 0 uses the commitment root.
  pub round_roots: Vec<Hash>,
  /// One query path per queried position.
  pub query_paths: Vec<QueryPath>,
}

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Hash a pair of field elements into a Merkle leaf.
fn hash_pair(a: GF2_128, b: GF2_128) -> Hash {
  let mut h = blake3::Hasher::new();
  h.update(b"binip:pcs:pair:");
  h.update(&a.lo.to_le_bytes());
  h.update(&a.hi.to_le_bytes());
  h.update(&b.lo.to_le_bytes());
  h.update(&b.hi.to_le_bytes());
  *h.finalize().as_bytes()
}

/// Build a Merkle tree by hashing consecutive pairs of a table.
fn build_pair_tree(table: &[GF2_128]) -> merkle::MerkleTree {
  assert!(table.len() >= 2 && table.len().is_power_of_two());
  let n_pairs = table.len() / 2;
  let leaves: Vec<Hash> = (0..n_pairs)
    .map(|j| hash_pair(table[2 * j], table[2 * j + 1]))
    .collect();
  merkle::MerkleTree::build(&leaves)
}

/// Parallel variant of [`build_pair_tree`].
fn build_pair_tree_par(table: &[GF2_128]) -> merkle::MerkleTree {
  assert!(table.len() >= 2 && table.len().is_power_of_two());
  let n_pairs = table.len() / 2;
  let leaves: Vec<Hash> = table
    .par_chunks(2)
    .map(|pair| hash_pair(pair[0], pair[1]))
    .collect();
  debug_assert_eq!(leaves.len(), n_pairs);
  merkle::MerkleTree::build_par(&leaves)
}

/// Fold a table by one variable: `out[j] = table[2j]*(1+r) + table[2j+1]*r`.
fn fold_table(table: &[GF2_128], r: GF2_128) -> Vec<GF2_128> {
  let half = table.len() / 2;
  let one_plus_r = GF2_128::one() + r;
  (0..half)
    .map(|j| table[2 * j] * one_plus_r + table[2 * j + 1] * r)
    .collect()
}

/// Parallel variant of [`fold_table`].
fn fold_table_par(table: &[GF2_128], r: GF2_128) -> Vec<GF2_128> {
  let one_plus_r = GF2_128::one() + r;
  table
    .par_chunks_exact(2)
    .map(|pair| pair[0] * one_plus_r + pair[1] * r)
    .collect()
}

/// Evaluate an MLE at a point by folding the evaluation table.
fn eval_mle(evals: &[GF2_128], r: &[GF2_128]) -> GF2_128 {
  let mut table = evals.to_vec();
  for &ri in r {
    let one_plus_r = GF2_128::one() + ri;
    let half = table.len() / 2;
    for i in 0..half {
      table[i] = table[i] * one_plus_r + table[i + half] * ri;
    }
    table.truncate(half);
  }
  table[0]
}

// ─── commit ───────────────────────────────────────────────────────────────────

/// Commit to an n-variable MLE given as its `2^n` evaluations in lex order.
///
/// Absorbs the commitment root into `transcript`.
pub fn commit<T: Transcript>(
  evals: &[GF2_128],
  params: &PcsParams,
  transcript: &mut T,
) -> (Commitment, PcsState) {
  let n = params.n_vars as usize;
  let size = 1usize << n;
  assert_eq!(evals.len(), size, "evals.len() must equal 2^n_vars");

  let tree = build_pair_tree(evals);
  let root = tree.root();
  transcript.absorb_bytes(&root);

  let commitment = Commitment {
    root,
    n_vars: params.n_vars,
  };
  let state = PcsState {
    commitment: commitment.clone(),
    params: params.clone(),
    evals: evals.to_vec(),
    tree,
  };
  (commitment, state)
}

/// Parallel variant of [`commit`].
pub fn commit_par<T: Transcript>(
  evals: &[GF2_128],
  params: &PcsParams,
  transcript: &mut T,
) -> (Commitment, PcsState) {
  let n = params.n_vars as usize;
  let size = 1usize << n;
  assert_eq!(evals.len(), size, "evals.len() must equal 2^n_vars");

  let tree = build_pair_tree_par(evals);
  let root = tree.root();
  transcript.absorb_bytes(&root);

  let commitment = Commitment {
    root,
    n_vars: params.n_vars,
  };
  let state = PcsState {
    commitment: commitment.clone(),
    params: params.clone(),
    evals: evals.to_vec(),
    tree,
  };
  (commitment, state)
}

// ─── open ─────────────────────────────────────────────────────────────────────

/// Produce an opening proof for evaluation at `r`.
///
/// Returns `(claimed_eval, proof)`.
/// Precondition: `transcript` is in the state left by `commit`.
pub fn open<T: Transcript>(
  state: &PcsState,
  r: &[GF2_128],
  transcript: &mut T,
) -> (GF2_128, OpenProof) {
  let n = state.params.n_vars as usize;
  assert_eq!(r.len(), n);

  // Build all intermediate tables by folding.
  // tables[0] = original evals (length 2^n)
  // fold_table collapses adjacent pairs (bit 0), so we consume r in
  // reverse order: r[n-1] first (lowest-order variable), up to r[0] last,
  // matching the mle_eval convention where r[0] folds halves.
  let mut tables: Vec<Vec<GF2_128>> = Vec::with_capacity(n + 1);
  tables.push(state.evals.clone());

  let mut round_roots: Vec<Hash> = Vec::with_capacity(n.saturating_sub(1));
  let mut round_trees: Vec<merkle::MerkleTree> = Vec::with_capacity(n.saturating_sub(1));

  for i in 0..n {
    let folded = fold_table(&tables[i], r[n - 1 - i]);
    if i < n - 1 {
      // Commit the folded table (length ≥ 2 since i < n-1).
      let tree_i = build_pair_tree(&folded);
      let root_i = tree_i.root();
      round_roots.push(root_i);
      transcript.absorb_bytes(&root_i);
      round_trees.push(tree_i);
    }
    tables.push(folded);
  }

  let claim = tables[n][0];
  transcript.absorb_field(claim);

  // Squeeze query positions (pair-level indices into round-0 tree).
  // Deduplicate to avoid information loss from repeated queries (M-1).
  let n_pairs_round0 = 1usize << (n - 1);
  let effective_queries = state.params.n_queries.min(n_pairs_round0);
  let mut seen = std::collections::HashSet::with_capacity(effective_queries);
  let mut queries = Vec::with_capacity(effective_queries);
  while queries.len() < effective_queries {
    let q = (transcript.squeeze_challenge().lo as usize) % n_pairs_round0;
    if seen.insert(q) {
      queries.push(q);
    }
  }

  // For each query, build a path through all rounds.
  let query_paths: Vec<QueryPath> = queries
    .iter()
    .map(|&q0| {
      let mut pair_idx = q0;
      let mut rounds = Vec::with_capacity(n);

      for i in 0..n {
        let table = &tables[i];
        let left = table[2 * pair_idx];
        let right = table[2 * pair_idx + 1];

        // Merkle proof at this round's tree.
        let tree_ref = if i == 0 {
          &state.tree
        } else {
          &round_trees[i - 1]
        };
        let merkle_proof = tree_ref.prove(pair_idx);

        rounds.push(RoundQuery {
          left,
          right,
          merkle_proof,
        });

        // Next round: the fold result sits at position `pair_idx` in
        // the folded table. The pair containing it is `pair_idx / 2`.
        pair_idx /= 2;
      }

      QueryPath { rounds }
    })
    .collect();

  (
    claim,
    OpenProof {
      round_roots,
      query_paths,
    },
  )
}

/// Parallel variant of [`open`].
///
/// Three-phase approach for maximum parallelism:
/// 1. Fold rounds — sequential across rounds, but each fold is parallel internally.
/// 2. Merkle tree builds — all (n-1) intermediate trees built concurrently.
/// 3. Query path construction — parallel across queries.
pub fn open_par<T: Transcript>(
  state: &PcsState,
  r: &[GF2_128],
  transcript: &mut T,
) -> (GF2_128, OpenProof) {
  let n = state.params.n_vars as usize;
  assert_eq!(r.len(), n);

  // ── Phase 1: compute all fold rounds ──────────────────────────────────
  // Each round depends on the previous, but fold_table_par parallelises
  // the field arithmetic within each round.  We avoid cloning state.evals
  // by referencing it directly during query-path construction.
  let mut fold_results: Vec<Vec<GF2_128>> = Vec::with_capacity(n);
  {
    let first = if state.evals.len() >= 2048 {
      fold_table_par(&state.evals, r[n - 1])
    } else {
      fold_table(&state.evals, r[n - 1])
    };
    fold_results.push(first);
    for i in 1..n {
      let prev = &fold_results[i - 1];
      let folded = if prev.len() >= 2048 {
        fold_table_par(prev, r[n - 1 - i])
      } else {
        fold_table(prev, r[n - 1 - i])
      };
      fold_results.push(folded);
    }
  }

  let claim = fold_results[n - 1][0];

  // ── Phase 2: build intermediate Merkle trees in parallel ──────────────
  // Trees for rounds 0..n-2 are independent — each only reads from its
  // respective folded table.  Rayon distributes the work across cores.
  let round_trees: Vec<merkle::MerkleTree> = if n > 1 {
    (0..n - 1)
      .into_par_iter()
      .map(|j| {
        let table = &fold_results[j];
        if table.len() >= 2048 {
          build_pair_tree_par(table)
        } else {
          build_pair_tree(table)
        }
      })
      .collect()
  } else {
    Vec::new()
  };

  // ── Phase 3: absorb roots in order + squeeze queries ──────────────────
  let round_roots: Vec<Hash> = round_trees.iter().map(|t| t.root()).collect();
  for &root in &round_roots {
    transcript.absorb_bytes(&root);
  }
  transcript.absorb_field(claim);

  let n_pairs_round0 = 1usize << (n - 1);
  let effective_queries = state.params.n_queries.min(n_pairs_round0);
  let mut seen = std::collections::HashSet::with_capacity(effective_queries);
  let mut queries = Vec::with_capacity(effective_queries);
  while queries.len() < effective_queries {
    let q = (transcript.squeeze_challenge().lo as usize) % n_pairs_round0;
    if seen.insert(q) {
      queries.push(q);
    }
  }

  // ── Phase 4: build query paths in parallel ────────────────────────────
  let query_paths: Vec<QueryPath> = queries
    .par_iter()
    .map(|&q0| {
      let mut pair_idx = q0;
      let mut rounds = Vec::with_capacity(n);

      for i in 0..n {
        let table: &[GF2_128] = if i == 0 {
          &state.evals
        } else {
          &fold_results[i - 1]
        };
        let left = table[2 * pair_idx];
        let right = table[2 * pair_idx + 1];

        let tree_ref = if i == 0 {
          &state.tree
        } else {
          &round_trees[i - 1]
        };
        let merkle_proof = tree_ref.prove(pair_idx);

        rounds.push(RoundQuery {
          left,
          right,
          merkle_proof,
        });
        pair_idx /= 2;
      }
      QueryPath { rounds }
    })
    .collect();

  (
    claim,
    OpenProof {
      round_roots,
      query_paths,
    },
  )
}

// ─── verify ───────────────────────────────────────────────────────────────────

/// Verify an opening proof.
///
/// Precondition: `transcript` is in the state produced by `commit` (i.e., the
/// caller has already absorbed `commitment.root`).
pub fn verify<T: Transcript>(
  commitment: &Commitment,
  r: &[GF2_128],
  claim: GF2_128,
  proof: &OpenProof,
  params: &PcsParams,
  transcript: &mut T,
) -> bool {
  let n = params.n_vars as usize;
  if r.len() != n {
    return false;
  }
  if proof.round_roots.len() != n.saturating_sub(1) {
    return false;
  }
  let n_pairs_round0 = 1usize << (n - 1);
  let effective_queries = params.n_queries.min(n_pairs_round0);
  if proof.query_paths.len() != effective_queries {
    return false;
  }

  // Absorb intermediate round roots (mirrors prover).
  for root in &proof.round_roots {
    transcript.absorb_bytes(root);
  }
  transcript.absorb_field(claim);

  // Re-squeeze query positions (deduplicated, matching prover).
  let mut seen = std::collections::HashSet::with_capacity(effective_queries);
  let mut queries = Vec::with_capacity(effective_queries);
  while queries.len() < effective_queries {
    let q = (transcript.squeeze_challenge().lo as usize) % n_pairs_round0;
    if seen.insert(q) {
      queries.push(q);
    }
  }

  // Verify each query path.
  for (qi, &q0) in queries.iter().enumerate() {
    let path = &proof.query_paths[qi];
    if path.rounds.len() != n {
      return false;
    }

    let mut pair_idx = q0;

    for i in 0..n {
      let rq = &path.rounds[i];

      // Root for this round.
      let root = if i == 0 {
        commitment.root
      } else {
        proof.round_roots[i - 1]
      };

      // Verify the pair's Merkle proof.
      let leaf = hash_pair(rq.left, rq.right);
      let n_pairs = 1usize << (n - 1 - i);
      let tree_n = n_pairs.next_power_of_two().max(1);
      if !rq.merkle_proof.verify(leaf, root, tree_n) {
        return false;
      }

      // Fold consistency: check that the fold result from this pair
      // matches the element used in the next round.
      // Use reversed r order matching prover: round i uses r[n-1-i].
      let ri = r[n - 1 - i];
      let one_plus_r = GF2_128::one() + ri;
      let folded = rq.left * one_plus_r + rq.right * ri;

      if i < n - 1 {
        // The folded value should appear as one of the two elements
        // in the next round's pair.
        let next_rq = &path.rounds[i + 1];
        let _next_pair_idx = pair_idx / 2;
        // Which element within the next pair: pair_idx % 2.
        let expected = if pair_idx % 2 == 0 {
          next_rq.left
        } else {
          next_rq.right
        };
        if folded != expected {
          return false;
        }
      } else {
        // Last round: fold result should equal the claimed evaluation.
        if folded != claim {
          return false;
        }
      }

      pair_idx /= 2;
    }
  }

  true
}

// ═══════════════════════════════════════════════════════════════════════════════
// Batch BaseFold PCS
// ═══════════════════════════════════════════════════════════════════════════════
//
// Commits multiple MLEs (all with the same `n_vars`) into a single Merkle tree.
// The evaluations are interleaved: for `n_entries` polynomials of size `2^n`,
// each Merkle leaf hashes a "super-pair" of `2 * n_entries` elements
// (the pair from each entry at the same position).
//
// `batch_open` opens several (entry, point) queries sharing one set of
// random query positions — amortising Merkle proof cost.

// ─── batch types ─────────────────────────────────────────────────────────────

/// Batch PCS builder — accumulate MLEs then call [`BatchBaseFold::commit`].
pub struct BatchBaseFold {
  params: PcsParams,
  entries: Vec<Vec<GF2_128>>,
}

// Keep the old name as an alias so external code compiles unchanged.
pub type BatchTensorPCS = BatchBaseFold;

/// Merkle-root commitment for a batch of MLEs.
#[derive(Clone, Debug, bincode::Encode, bincode::Decode)]
pub struct BatchCommitment {
  pub root: Hash,
  pub n_vars: u32,
  pub n_entries: usize,
}

/// Internal prover state for answering batch opening queries.
pub struct BatchPcsState {
  pub commitment: BatchCommitment,
  params: PcsParams,
  /// Per-entry evaluation tables.
  entry_evals: Vec<Vec<GF2_128>>,
  /// Shared Merkle tree.
  tree: merkle::MerkleTree,
}

/// A single opening query within a batch.
#[derive(Clone, Debug)]
pub struct BatchOpenQuery {
  /// Which polynomial (entry index) to open.
  pub entry: usize,
  /// Evaluation point (length = `n_vars`).
  pub point: Vec<GF2_128>,
}

/// Shared round-0 query data at one sampled position.
///
/// At round 0 the batch Merkle tree commits all entries' pairs together
/// as "super-pair" leaves.  The verifier needs every entry's pair to
/// reconstruct the leaf hash.
#[derive(Clone, Debug, bincode::Encode, bincode::Decode)]
pub struct BatchRound0Query {
  /// All entries' pairs at this position.  Length = n_entries.
  pub pairs: Vec<(GF2_128, GF2_128)>,
  /// Merkle proof in the batch commitment tree.
  pub merkle_proof: MerkleProof,
}

/// Per-query evaluation proof.
///
/// Each query folds its entry's evaluations with the query's own evaluation
/// point (like single-point PCS), producing per-query intermediate Merkle
/// trees for rounds 1 through n-1.
#[derive(Clone, Debug, bincode::Encode, bincode::Decode)]
pub struct BatchEvalProof {
  /// Intermediate Merkle roots for this query's fold chain.
  /// `round_roots[j]` is the root for the entry's table after `j+1` folds.
  /// Length = n_vars − 1.
  pub round_roots: Vec<Hash>,
  /// Per-position inner-round data.
  /// `inner_paths[pos][j]` gives the pair + Merkle proof for round `j+1`
  /// at sampled position `pos`.
  /// Outer length = number of sampled positions.
  /// Inner length = n_vars − 1.
  pub inner_paths: Vec<Vec<RoundQuery>>,
}

/// Batch opening proof.
#[derive(Clone, Debug, bincode::Encode, bincode::Decode)]
pub struct BatchOpenProof {
  /// Shared round-0 data (one per sampled position).
  pub round0: Vec<BatchRound0Query>,
  /// Per-query evaluation proofs.
  pub eval_proofs: Vec<BatchEvalProof>,
}

// ─── batch helpers ───────────────────────────────────────────────────────────

/// Hash a "super-pair" — all entries' (left, right) at one pair index.
fn hash_super_pair(entries: &[Vec<GF2_128>], pair_idx: usize) -> Hash {
  let mut h = blake3::Hasher::new();
  h.update(b"binip:pcs:batch:");
  for e in entries {
    let l = e[2 * pair_idx];
    let r = e[2 * pair_idx + 1];
    h.update(&l.lo.to_le_bytes());
    h.update(&l.hi.to_le_bytes());
    h.update(&r.lo.to_le_bytes());
    h.update(&r.hi.to_le_bytes());
  }
  *h.finalize().as_bytes()
}

/// Build a Merkle tree from interleaved entries.
fn build_batch_tree(entries: &[Vec<GF2_128>]) -> merkle::MerkleTree {
  let size = entries[0].len();
  let n_pairs = size / 2;
  let leaves: Vec<Hash> = (0..n_pairs)
    .map(|j| hash_super_pair(entries, j))
    .collect();
  merkle::MerkleTree::build(&leaves)
}

fn build_batch_tree_par(entries: &[Vec<GF2_128>]) -> merkle::MerkleTree {
  let size = entries[0].len();
  let n_pairs = size / 2;
  let leaves: Vec<Hash> = (0..n_pairs)
    .into_par_iter()
    .map(|j| hash_super_pair(entries, j))
    .collect();
  merkle::MerkleTree::build_par(&leaves)
}

// ─── batch commit ────────────────────────────────────────────────────────────

impl BatchBaseFold {
  pub fn new(params: PcsParams) -> Self {
    Self {
      params,
      entries: Vec::new(),
    }
  }

  pub fn add_poly(&mut self, evals: Vec<GF2_128>) -> usize {
    let expected = 1usize << self.params.n_vars;
    assert_eq!(evals.len(), expected, "evals length must be 2^n_vars");
    let idx = self.entries.len();
    self.entries.push(evals);
    idx
  }

  pub fn commit<T: Transcript>(&self, transcript: &mut T) -> (BatchCommitment, BatchPcsState) {
    assert!(!self.entries.is_empty());
    let tree = build_batch_tree(&self.entries);
    let root = tree.root();
    transcript.absorb_bytes(&root);

    let commitment = BatchCommitment {
      root,
      n_vars: self.params.n_vars,
      n_entries: self.entries.len(),
    };
    let state = BatchPcsState {
      commitment: commitment.clone(),
      params: self.params.clone(),
      entry_evals: self.entries.clone(),
      tree,
    };
    (commitment, state)
  }

  pub fn commit_par<T: Transcript>(&self, transcript: &mut T) -> (BatchCommitment, BatchPcsState) {
    assert!(!self.entries.is_empty());
    let tree = build_batch_tree_par(&self.entries);
    let root = tree.root();
    transcript.absorb_bytes(&root);

    let commitment = BatchCommitment {
      root,
      n_vars: self.params.n_vars,
      n_entries: self.entries.len(),
    };
    let state = BatchPcsState {
      commitment: commitment.clone(),
      params: self.params.clone(),
      entry_evals: self.entries.clone(),
      tree,
    };
    (commitment, state)
  }
}

// ─── batch open ──────────────────────────────────────────────────────────────

/// Open multiple (entry, point) queries against the same batch commitment.
///
/// Each query folds the entry's evaluations with its own evaluation point
/// (like single-point BaseFold), producing per-query intermediate Merkle
/// trees.  Round 0 uses the shared batch commitment tree.
///
/// Returns `(claims, proof)`.
pub fn batch_open<T: Transcript>(
  state: &BatchPcsState,
  queries: &[BatchOpenQuery],
  transcript: &mut T,
) -> (Vec<GF2_128>, BatchOpenProof) {
  let n = state.params.n_vars as usize;
  let n_entries = state.entry_evals.len();

  // 1. Compute claims and absorb.
  let mut claims = Vec::with_capacity(queries.len());
  for q in queries {
    assert!(q.entry < n_entries);
    assert_eq!(q.point.len(), n);
    let claim = eval_mle(&state.entry_evals[q.entry], &q.point);
    claims.push(claim);
    transcript.absorb_field(claim);
  }

  // 2. Per-query fold chains using each query's evaluation point.
  //    tables_per_q[qi][round] = folded table after `round` folds.
  //    trees_per_q[qi][j]     = Merkle tree for tables_per_q[qi][j+1].
  //    roots_per_q[qi][j]     = root of trees_per_q[qi][j].
  let mut tables_per_q: Vec<Vec<Vec<GF2_128>>> = Vec::with_capacity(queries.len());
  let mut trees_per_q: Vec<Vec<merkle::MerkleTree>> = Vec::with_capacity(queries.len());

  for (qi, q) in queries.iter().enumerate() {
    let entry_evals = &state.entry_evals[q.entry];
    let point = &q.point;

    let mut tables: Vec<Vec<GF2_128>> = Vec::with_capacity(n + 1);
    tables.push(entry_evals.clone());

    let mut round_trees = Vec::with_capacity(n.saturating_sub(1));

    for i in 0..n {
      let folded = fold_table(&tables[i], point[n - 1 - i]);
      if i < n - 1 {
        let tree_i = build_pair_tree(&folded);
        let root_i = tree_i.root();
        transcript.absorb_bytes(&root_i);
        round_trees.push(tree_i);
      }
      tables.push(folded);
    }

    debug_assert_eq!(tables[n][0], claims[qi]);

    tables_per_q.push(tables);
    trees_per_q.push(round_trees);
  }

  // 3. Squeeze shared query positions (deduplicated).
  let n_pairs = 1usize << (n - 1);
  let effective_queries = state.params.n_queries.min(n_pairs);
  let mut seen = std::collections::HashSet::with_capacity(effective_queries);
  let mut positions = Vec::with_capacity(effective_queries);
  while positions.len() < effective_queries {
    let q = (transcript.squeeze_challenge().lo as usize) % n_pairs;
    if seen.insert(q) {
      positions.push(q);
    }
  }

  // 4. Build round-0 proofs (shared batch tree).
  let round0: Vec<BatchRound0Query> = positions
    .iter()
    .map(|&q0| {
      let pairs: Vec<(GF2_128, GF2_128)> = (0..n_entries)
        .map(|e| {
          let t = &state.entry_evals[e];
          (t[2 * q0], t[2 * q0 + 1])
        })
        .collect();
      let merkle_proof = state.tree.prove(q0);
      BatchRound0Query {
        pairs,
        merkle_proof,
      }
    })
    .collect();

  // 5. Build per-query eval proofs (rounds 1..n-1).
  let eval_proofs: Vec<BatchEvalProof> = (0..queries.len())
    .map(|qi| {
      let round_roots: Vec<Hash> = trees_per_q[qi].iter().map(|t| t.root()).collect();

      let inner_paths: Vec<Vec<RoundQuery>> = positions
        .iter()
        .map(|&q0| {
          let mut pair_idx = q0 / 2; // After round 0.
          let mut rounds = Vec::with_capacity(n.saturating_sub(1));
          for j in 1..n {
            let table = &tables_per_q[qi][j];
            let left = table[2 * pair_idx];
            let right = table[2 * pair_idx + 1];
            let merkle_proof = trees_per_q[qi][j - 1].prove(pair_idx);
            rounds.push(RoundQuery {
              left,
              right,
              merkle_proof,
            });
            pair_idx /= 2;
          }
          rounds
        })
        .collect();

      BatchEvalProof {
        round_roots,
        inner_paths,
      }
    })
    .collect();

  (claims, BatchOpenProof { round0, eval_proofs })
}

/// Parallel variant of [`batch_open`].
///
/// Phases:
/// 1. Per-query fold chains — parallel across queries, each fold is sequential.
/// 2. Per-query Merkle tree builds — parallel across queries × rounds.
/// 3. Absorb roots sequentially, squeeze positions.
/// 4. Build query paths — parallel across positions.
pub fn batch_open_par<T: Transcript>(
  state: &BatchPcsState,
  queries: &[BatchOpenQuery],
  transcript: &mut T,
) -> (Vec<GF2_128>, BatchOpenProof) {
  let n = state.params.n_vars as usize;
  let n_entries = state.entry_evals.len();

  // 1. Compute claims.
  let mut claims = Vec::with_capacity(queries.len());
  for q in queries {
    assert!(q.entry < n_entries);
    assert_eq!(q.point.len(), n);
    let claim = eval_mle(&state.entry_evals[q.entry], &q.point);
    claims.push(claim);
    transcript.absorb_field(claim);
  }

  // ── Phase 1: per-query fold chains (parallel across queries) ──────────
  let tables_per_q: Vec<Vec<Vec<GF2_128>>> = queries
    .par_iter()
    .map(|q| {
      let entry_evals = &state.entry_evals[q.entry];
      let point = &q.point;
      let mut tables: Vec<Vec<GF2_128>> = Vec::with_capacity(n + 1);
      tables.push(entry_evals.clone());
      for i in 0..n {
        let folded = if tables[i].len() >= 2048 {
          fold_table_par(&tables[i], point[n - 1 - i])
        } else {
          fold_table(&tables[i], point[n - 1 - i])
        };
        tables.push(folded);
      }
      tables
    })
    .collect();

  // ── Phase 2: build per-query intermediate Merkle trees ────────────────
  // trees_per_q[qi][j] = Merkle tree for tables_per_q[qi][j+1].
  let trees_per_q: Vec<Vec<merkle::MerkleTree>> = tables_per_q
    .par_iter()
    .map(|tables| {
      if n > 1 {
        (1..n)
          .map(|j| {
            let t = &tables[j];
            if t.len() >= 2048 {
              build_pair_tree_par(t)
            } else {
              build_pair_tree(t)
            }
          })
          .collect()
      } else {
        Vec::new()
      }
    })
    .collect();

  // ── Phase 3: absorb roots in transcript order, squeeze positions ──────
  for qi in 0..queries.len() {
    for tree in &trees_per_q[qi] {
      transcript.absorb_bytes(&tree.root());
    }
  }

  let n_pairs = 1usize << (n - 1);
  let effective_queries = state.params.n_queries.min(n_pairs);
  let mut seen = std::collections::HashSet::with_capacity(effective_queries);
  let mut positions = Vec::with_capacity(effective_queries);
  while positions.len() < effective_queries {
    let q = (transcript.squeeze_challenge().lo as usize) % n_pairs;
    if seen.insert(q) {
      positions.push(q);
    }
  }

  // ── Phase 4: build proof paths (parallel across positions) ────────────
  let round0: Vec<BatchRound0Query> = positions
    .par_iter()
    .map(|&q0| {
      let pairs: Vec<(GF2_128, GF2_128)> = (0..n_entries)
        .map(|e| {
          let t = &state.entry_evals[e];
          (t[2 * q0], t[2 * q0 + 1])
        })
        .collect();
      let merkle_proof = state.tree.prove(q0);
      BatchRound0Query {
        pairs,
        merkle_proof,
      }
    })
    .collect();

  let eval_proofs: Vec<BatchEvalProof> = (0..queries.len())
    .map(|qi| {
      let round_roots: Vec<Hash> = trees_per_q[qi].iter().map(|t| t.root()).collect();
      let inner_paths: Vec<Vec<RoundQuery>> = positions
        .par_iter()
        .map(|&q0| {
          let mut pair_idx = q0 / 2;
          let mut rounds = Vec::with_capacity(n.saturating_sub(1));
          for j in 1..n {
            let table = &tables_per_q[qi][j];
            let left = table[2 * pair_idx];
            let right = table[2 * pair_idx + 1];
            let merkle_proof = trees_per_q[qi][j - 1].prove(pair_idx);
            rounds.push(RoundQuery {
              left,
              right,
              merkle_proof,
            });
            pair_idx /= 2;
          }
          rounds
        })
        .collect();
      BatchEvalProof {
        round_roots,
        inner_paths,
      }
    })
    .collect();

  (claims, BatchOpenProof { round0, eval_proofs })
}

// ─── batch verify ────────────────────────────────────────────────────────────

/// Verify a batch opening proof.
///
/// Each query is verified with its own fold chain using the query's
/// evaluation point, exactly like single-point BaseFold.  Round 0 uses
/// the shared batch commitment tree; rounds 1+ use per-query intermediate
/// trees whose roots are absorbed into the transcript.
pub fn batch_verify<T: Transcript>(
  commitment: &BatchCommitment,
  queries: &[BatchOpenQuery],
  claims: &[GF2_128],
  proof: &BatchOpenProof,
  params: &PcsParams,
  transcript: &mut T,
) -> bool {
  let n = params.n_vars as usize;
  let n_entries = commitment.n_entries;

  if queries.len() != claims.len() {
    return false;
  }
  if proof.eval_proofs.len() != queries.len() {
    return false;
  }

  // Mirror prover: absorb claims.
  for &c in claims {
    transcript.absorb_field(c);
  }

  // Absorb per-query round roots (must match prover order).
  for ep in &proof.eval_proofs {
    if ep.round_roots.len() != n.saturating_sub(1) {
      return false;
    }
    for root in &ep.round_roots {
      transcript.absorb_bytes(root);
    }
  }

  // Re-squeeze shared query positions (deduplicated).
  let n_pairs_round0 = 1usize << (n - 1);
  let effective_queries = params.n_queries.min(n_pairs_round0);
  let mut seen = std::collections::HashSet::with_capacity(effective_queries);
  let mut positions = Vec::with_capacity(effective_queries);
  while positions.len() < effective_queries {
    let q = (transcript.squeeze_challenge().lo as usize) % n_pairs_round0;
    if seen.insert(q) {
      positions.push(q);
    }
  }

  if proof.round0.len() != positions.len() {
    return false;
  }

  // Verify each query at each sampled position.
  for (pos_idx, &q0) in positions.iter().enumerate() {
    let r0data = &proof.round0[pos_idx];

    // Validate round-0 data length.
    if r0data.pairs.len() != n_entries {
      return false;
    }

    // Verify batch Merkle proof at round 0.
    let leaf = {
      let mut h = blake3::Hasher::new();
      h.update(b"binip:pcs:batch:");
      for &(l, r) in &r0data.pairs {
        h.update(&l.lo.to_le_bytes());
        h.update(&l.hi.to_le_bytes());
        h.update(&r.lo.to_le_bytes());
        h.update(&r.hi.to_le_bytes());
      }
      *h.finalize().as_bytes()
    };
    let tree_n0 = n_pairs_round0.next_power_of_two().max(1);
    if !r0data.merkle_proof.verify(leaf, commitment.root, tree_n0) {
      return false;
    }

    // For every query, verify the fold chain starting from round 0.
    for (qi, q) in queries.iter().enumerate() {
      if q.point.len() != n || q.entry >= n_entries {
        return false;
      }
      let ep = &proof.eval_proofs[qi];
      if ep.inner_paths.len() != positions.len() {
        return false;
      }
      let inner_path = &ep.inner_paths[pos_idx];
      if inner_path.len() != n.saturating_sub(1) {
        return false;
      }

      let point = &q.point;
      let entry = q.entry;
      let mut pair_idx = q0;

      // Round 0: extract entry's pair from the batch super-leaf.
      let (left0, right0) = r0data.pairs[entry];
      let ri0 = point[n - 1];
      let folded0 = left0 * (GF2_128::one() + ri0) + right0 * ri0;

      if n == 1 {
        // Single variable: fold result is the evaluation.
        if folded0 != claims[qi] {
          return false;
        }
        continue;
      }

      // Check fold consistency with round 1.
      let next_rq = &inner_path[0];
      let expected = if pair_idx % 2 == 0 {
        next_rq.left
      } else {
        next_rq.right
      };
      if folded0 != expected {
        return false;
      }
      pair_idx /= 2;

      // Rounds 1..n-1: per-query intermediate trees.
      for j in 1..n {
        let rq = &inner_path[j - 1];

        // Verify Merkle proof against this query's round root.
        let root_j = ep.round_roots[j - 1];
        let leaf_j = hash_pair(rq.left, rq.right);
        let n_pairs_j = 1usize << (n - 1 - j);
        let tree_n_j = n_pairs_j.next_power_of_two().max(1);
        if !rq.merkle_proof.verify(leaf_j, root_j, tree_n_j) {
          return false;
        }

        // Fold.
        let ri = point[n - 1 - j];
        let folded = rq.left * (GF2_128::one() + ri) + rq.right * ri;

        if j < n - 1 {
          // Fold consistency with next round.
          let next_rq = &inner_path[j];
          let expected = if pair_idx % 2 == 0 {
            next_rq.left
          } else {
            next_rq.right
          };
          if folded != expected {
            return false;
          }
        } else {
          // Last round: fold result must equal the claim.
          if folded != claims[qi] {
            return false;
          }
        }

        pair_idx /= 2;
      }
    }
  }

  true
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use super::*;
  use field::FieldElem;
  use transcript::VisionTranscript;

  fn fixed_evals(n_vars: u32) -> Vec<GF2_128> {
    (0u64..1u64 << n_vars)
      .map(|i| GF2_128::from(i.wrapping_mul(6364136223846793005).wrapping_add(1)))
      .collect()
  }

  fn fixed_point(n_vars: u32, seed: u64) -> Vec<GF2_128> {
    (0..n_vars as u64)
      .map(|i| GF2_128::from(seed.wrapping_add(i + 1).wrapping_mul(2654435761)))
      .collect()
  }

  /// Direct MLE evaluation (for comparison).
  fn mle_eval(evals: &[GF2_128], r: &[GF2_128]) -> GF2_128 {
    let mut table = evals.to_vec();
    for &ri in r {
      let one_minus_r = GF2_128::one() + ri;
      let half = table.len() / 2;
      for i in 0..half {
        table[i] = table[i] * one_minus_r + table[i + half] * ri;
      }
      table.truncate(half);
    }
    table[0]
  }

  fn eval_mle_test(evals: &[GF2_128], r: &[GF2_128]) -> GF2_128 {
    mle_eval(evals, r)
  }

  #[test]
  fn commit_open_verify_n4() {
    let n_vars = 4u32;
    let params = PcsParams {
      n_vars,
      n_queries: 8,
    };
    let evals = fixed_evals(n_vars);
    let r = fixed_point(n_vars, 42);

    let mut tp = VisionTranscript::new();
    let (commitment, state) = commit(&evals, &params, &mut tp);
    let (claim, proof) = open(&state, &r, &mut tp);

    let direct = mle_eval(&evals, &r);
    assert_eq!(claim, direct, "claim must equal direct MLE evaluation");

    let mut tv = VisionTranscript::new();
    tv.absorb_bytes(&commitment.root);
    assert!(verify(&commitment, &r, claim, &proof, &params, &mut tv));
  }

  #[test]
  fn commit_open_verify_n6() {
    let n_vars = 6u32;
    let params = PcsParams {
      n_vars,
      n_queries: 10,
    };
    let evals = fixed_evals(n_vars);
    let r = fixed_point(n_vars, 137);

    let mut tp = VisionTranscript::new();
    let (commitment, state) = commit(&evals, &params, &mut tp);
    let (claim, proof) = open(&state, &r, &mut tp);

    let direct = mle_eval(&evals, &r);
    assert_eq!(claim, direct);

    let mut tv = VisionTranscript::new();
    tv.absorb_bytes(&commitment.root);
    assert!(verify(&commitment, &r, claim, &proof, &params, &mut tv));
  }

  #[test]
  fn verify_rejects_wrong_claim() {
    let n_vars = 4u32;
    let params = PcsParams {
      n_vars,
      n_queries: 8,
    };
    let evals = fixed_evals(n_vars);
    let r = fixed_point(n_vars, 99);

    let mut tp = VisionTranscript::new();
    let (commitment, state) = commit(&evals, &params, &mut tp);
    let (claim, proof) = open(&state, &r, &mut tp);

    let wrong_claim = claim + GF2_128::one();

    let mut tv = VisionTranscript::new();
    tv.absorb_bytes(&commitment.root);
    assert!(!verify(
      &commitment,
      &r,
      wrong_claim,
      &proof,
      &params,
      &mut tv
    ));
  }

  #[test]
  fn verify_rejects_tampered_pair() {
    let n_vars = 4u32;
    let params = PcsParams {
      n_vars,
      n_queries: 8,
    };
    let evals = fixed_evals(n_vars);
    let r = fixed_point(n_vars, 23);

    let mut tp = VisionTranscript::new();
    let (commitment, state) = commit(&evals, &params, &mut tp);
    let (claim, mut proof) = open(&state, &r, &mut tp);

    // Tamper with the first query's first round left element.
    if let Some(path) = proof.query_paths.first_mut() {
      if let Some(rq) = path.rounds.first_mut() {
        rq.left = rq.left + GF2_128::one();
      }
    }

    let mut tv = VisionTranscript::new();
    tv.absorb_bytes(&commitment.root);
    assert!(!verify(&commitment, &r, claim, &proof, &params, &mut tv));
  }

  #[test]
  fn commit_par_matches_commit() {
    let n_vars = 6u32;
    let params = PcsParams {
      n_vars,
      n_queries: 10,
    };
    let evals = fixed_evals(n_vars);

    let mut t1 = VisionTranscript::new();
    let (c1, _) = commit(&evals, &params, &mut t1);

    let mut t2 = VisionTranscript::new();
    let (c2, _) = commit_par(&evals, &params, &mut t2);

    assert_eq!(c1.root, c2.root);
  }

  #[test]
  fn open_par_matches_open() {
    let n_vars = 6u32;
    let params = PcsParams {
      n_vars,
      n_queries: 10,
    };
    let evals = fixed_evals(n_vars);
    let r = fixed_point(n_vars, 99);

    let mut t1 = VisionTranscript::new();
    let (_, s1) = commit(&evals, &params, &mut t1);
    let (c1, p1) = open(&s1, &r, &mut t1);

    let mut t2 = VisionTranscript::new();
    let (_, s2) = commit_par(&evals, &params, &mut t2);
    let (c2, p2) = open_par(&s2, &r, &mut t2);

    assert_eq!(c1, c2);
    assert_eq!(p1.round_roots, p2.round_roots);
    assert_eq!(p1.query_paths.len(), p2.query_paths.len());
    for (a, b) in p1.query_paths.iter().zip(&p2.query_paths) {
      for (ar, br) in a.rounds.iter().zip(&b.rounds) {
        assert_eq!(ar.left, br.left);
        assert_eq!(ar.right, br.right);
      }
    }
  }

  // ── batch PCS tests ──────────────────────────────────────────────────

  #[test]
  fn batch_single_entry_roundtrip() {
    let n_vars = 4u32;
    let params = PcsParams {
      n_vars,
      n_queries: 8,
    };
    let evals = fixed_evals(n_vars);
    let r = fixed_point(n_vars, 42);

    let mut pcs = BatchBaseFold::new(params.clone());
    pcs.add_poly(evals.clone());

    let mut tp = VisionTranscript::new();
    let (bc, state) = pcs.commit(&mut tp);

    let queries = vec![BatchOpenQuery {
      entry: 0,
      point: r.clone(),
    }];
    let (claims, proof) = batch_open(&state, &queries, &mut tp);

    let direct = mle_eval(&evals, &r);
    assert_eq!(claims[0], direct);

    let mut tv = VisionTranscript::new();
    tv.absorb_bytes(&bc.root);
    assert!(batch_verify(
      &bc, &queries, &claims, &proof, &params, &mut tv
    ));
  }

  #[test]
  fn batch_multi_entry_roundtrip() {
    let n_vars = 4u32;
    let params = PcsParams {
      n_vars,
      n_queries: 10,
    };

    let evals_a = fixed_evals(n_vars);
    let evals_b: Vec<GF2_128> = (0u64..1u64 << n_vars)
      .map(|i| GF2_128::from(i.wrapping_mul(1442695040888963407).wrapping_add(3)))
      .collect();
    let evals_c: Vec<GF2_128> = (0u64..1u64 << n_vars)
      .map(|i| GF2_128::from(i ^ 0xCAFE))
      .collect();

    let mut pcs = BatchBaseFold::new(params.clone());
    pcs.add_poly(evals_a.clone());
    pcs.add_poly(evals_b.clone());
    pcs.add_poly(evals_c.clone());

    let mut tp = VisionTranscript::new();
    let (bc, state) = pcs.commit(&mut tp);

    let r_a = fixed_point(n_vars, 10);
    let r_b = fixed_point(n_vars, 20);
    let r_c = fixed_point(n_vars, 30);

    let queries = vec![
      BatchOpenQuery {
        entry: 0,
        point: r_a.clone(),
      },
      BatchOpenQuery {
        entry: 1,
        point: r_b.clone(),
      },
      BatchOpenQuery {
        entry: 2,
        point: r_c.clone(),
      },
    ];
    let (claims, proof) = batch_open(&state, &queries, &mut tp);

    assert_eq!(claims[0], mle_eval(&evals_a, &r_a));
    assert_eq!(claims[1], mle_eval(&evals_b, &r_b));
    assert_eq!(claims[2], mle_eval(&evals_c, &r_c));

    let mut tv = VisionTranscript::new();
    tv.absorb_bytes(&bc.root);
    assert!(batch_verify(
      &bc, &queries, &claims, &proof, &params, &mut tv
    ));
  }

  #[test]
  fn batch_same_entry_two_points() {
    let n_vars = 4u32;
    let params = PcsParams {
      n_vars,
      n_queries: 8,
    };
    let evals = fixed_evals(n_vars);

    let mut pcs = BatchBaseFold::new(params.clone());
    pcs.add_poly(evals.clone());

    let mut tp = VisionTranscript::new();
    let (bc, state) = pcs.commit(&mut tp);

    let r1 = fixed_point(n_vars, 77);
    let r2 = fixed_point(n_vars, 88);

    let queries = vec![
      BatchOpenQuery {
        entry: 0,
        point: r1.clone(),
      },
      BatchOpenQuery {
        entry: 0,
        point: r2.clone(),
      },
    ];
    let (claims, proof) = batch_open(&state, &queries, &mut tp);

    assert_eq!(claims[0], mle_eval(&evals, &r1));
    assert_eq!(claims[1], mle_eval(&evals, &r2));

    let mut tv = VisionTranscript::new();
    tv.absorb_bytes(&bc.root);
    assert!(batch_verify(
      &bc, &queries, &claims, &proof, &params, &mut tv
    ));
  }

  #[test]
  fn batch_rejects_wrong_claim() {
    let n_vars = 4u32;
    let params = PcsParams {
      n_vars,
      n_queries: 8,
    };

    let mut pcs = BatchBaseFold::new(params.clone());
    pcs.add_poly(fixed_evals(n_vars));
    pcs.add_poly(fixed_evals(n_vars));

    let mut tp = VisionTranscript::new();
    let (bc, state) = pcs.commit(&mut tp);

    let queries = vec![
      BatchOpenQuery {
        entry: 0,
        point: fixed_point(n_vars, 1),
      },
      BatchOpenQuery {
        entry: 1,
        point: fixed_point(n_vars, 2),
      },
    ];
    let (mut claims, proof) = batch_open(&state, &queries, &mut tp);
    claims[1] = claims[1] + GF2_128::one();

    let mut tv = VisionTranscript::new();
    tv.absorb_bytes(&bc.root);
    assert!(!batch_verify(
      &bc, &queries, &claims, &proof, &params, &mut tv
    ));
  }

  #[test]
  fn batch_rejects_tampered_pair() {
    let n_vars = 4u32;
    let params = PcsParams {
      n_vars,
      n_queries: 8,
    };

    let mut pcs = BatchBaseFold::new(params.clone());
    pcs.add_poly(fixed_evals(n_vars));

    let mut tp = VisionTranscript::new();
    let (bc, state) = pcs.commit(&mut tp);

    let queries = vec![BatchOpenQuery {
      entry: 0,
      point: fixed_point(n_vars, 55),
    }];
    let (claims, mut proof) = batch_open(&state, &queries, &mut tp);

    // Tamper with the first round-0 query's first entry pair.
    if let Some(r0q) = proof.round0.first_mut() {
      if let Some(pair) = r0q.pairs.first_mut() {
        pair.0 = pair.0 + GF2_128::one();
      }
    }

    let mut tv = VisionTranscript::new();
    tv.absorb_bytes(&bc.root);
    assert!(!batch_verify(
      &bc, &queries, &claims, &proof, &params, &mut tv
    ));
  }

  #[test]
  fn batch_n6_five_entries() {
    let n_vars = 6u32;
    let params = PcsParams {
      n_vars,
      n_queries: 12,
    };

    let mut pcs = BatchBaseFold::new(params.clone());
    let all_evals: Vec<Vec<GF2_128>> = (0..5u64)
      .map(|seed| {
        (0u64..1u64 << n_vars)
          .map(|i| {
            GF2_128::from(i.wrapping_mul(seed.wrapping_add(1).wrapping_mul(6364136223846793005)))
          })
          .collect()
      })
      .collect();

    for e in &all_evals {
      pcs.add_poly(e.clone());
    }

    let mut tp = VisionTranscript::new();
    let (bc, state) = pcs.commit(&mut tp);

    let queries: Vec<BatchOpenQuery> = (0..5)
      .map(|i| BatchOpenQuery {
        entry: i,
        point: fixed_point(n_vars, (i as u64 + 1) * 100),
      })
      .collect();
    let (claims, proof) = batch_open(&state, &queries, &mut tp);

    for (i, q) in queries.iter().enumerate() {
      assert_eq!(claims[i], mle_eval(&all_evals[q.entry], &q.point));
    }

    let mut tv = VisionTranscript::new();
    tv.absorb_bytes(&bc.root);
    assert!(batch_verify(
      &bc, &queries, &claims, &proof, &params, &mut tv
    ));
  }
}
