//! Top-level proof types for the GF(2) ZK-STARK system.
//!
//! The [`Proof`] struct bundles every artefact the verifier needs:
//!
//! | Component           | Purpose                                  |
//! |---------------------|------------------------------------------|
//! | `type_cert`         | Structural type-derivation certificate   |
//! | `batch_commit`      | PCS Merkle root over all witness MLEs    |
//! | `constraint_sum`    | Claimed sum of the constraint MLE (== 0) |
//! | `shard_batch`       | Shard-level sumcheck proofs              |
//! | `recursive_proof`   | Recursive aggregation proof              |
//! | `pcs_open`          | PCS opening proof at the challenge point |

use evm_types::TypeCert;
use field::GF2_128;
use pcs::{Commitment, OpenProof};
use recursive::RecursiveProof;
use shard::{RecursiveConfig, ShardProofBatch};
use sumcheck::SumcheckProof;

use circuit::lookup::{LookupCommitments, LookupProofs};
use circuit::mpt::StorageProof;

/// Cryptographic binding for the boundary-constraint MLE.
///
/// When Seq junctions exist in the proof tree, this sub-proof ensures
/// that the boundary MLE was honestly committed and sums to zero.
#[derive(Debug, Clone, bincode::Encode, bincode::Decode)]
pub struct BoundaryPcsProof {
  /// PCS commitment over the blinded boundary MLE.
  pub commit: Commitment,
  /// Sumcheck proof that `Σ boundary_blinded(x) = 0`.
  pub sumcheck: SumcheckProof,
  /// PCS opening at the sumcheck challenge point.
  pub pcs_open: OpenProof,
  /// Evaluation of the boundary MLE at the challenge point.
  pub open_eval: GF2_128,
  /// Number of MLE variables used for the boundary PCS.
  pub n_vars: u32,
}

/// Cryptographic binding for the reconstruction-constraint MLE.
///
/// Proves that the byte-decomposition columns faithfully reconstruct
/// the main trace operands, binding the STARK and LUT witnesses.
/// Without this, a malicious prover could self-report
/// `reconstruction_sum = 0` without actually satisfying the constraint.
#[derive(Debug, Clone, bincode::Encode, bincode::Decode)]
pub struct ReconstructionPcsProof {
  /// PCS commitment over the blinded reconstruction MLE.
  pub commit: Commitment,
  /// Sumcheck proof that `Σ reconstruction_blinded(x) = 0`.
  pub sumcheck: SumcheckProof,
  /// PCS opening at the sumcheck challenge point.
  pub pcs_open: OpenProof,
  /// Evaluation of the reconstruction MLE at the challenge point.
  pub open_eval: GF2_128,
  /// Number of MLE variables used for the reconstruction PCS.
  pub n_vars: u32,
}

/// Top-level ZK-STARK proof.
#[derive(Debug, Clone, bincode::Encode, bincode::Decode)]
pub struct Proof {
  // ── structural layer ──────────────────────────────────────────────────
  /// Type-derivation certificate (Proof Tree shape commitment).
  pub type_cert: TypeCert,

  /// Whether the proof tree contains Seq/Branch boundaries.
  ///
  /// Absorbed into the Fiat-Shamir transcript so the verifier knows
  /// whether to expect a [`BoundaryPcsProof`] without receiving the
  /// full proof tree.
  pub has_seq_boundaries: bool,

  // ── PCS layer ─────────────────────────────────────────────────────────
  /// PCS commitment over the constraint MLE evaluations.
  pub batch_commit: Commitment,

  // ── sumcheck layer ────────────────────────────────────────────────────
  /// Batching challenge used to combine per-row constraints.
  pub beta: GF2_128,
  /// Claimed sum of the constraint MLE (must be zero for a valid proof).
  pub constraint_sum: GF2_128,

  // ── boundary constraint layer ─────────────────────────────────────────
  /// Claimed sum of the boundary-constraint MLE (must be zero).
  ///
  /// This proves that every `Seq` boundary in the proof tree has
  /// matching left-post / right-pre states (PC, gas, stack depth, …).
  pub boundary_constraint_sum: GF2_128,

  /// PCS + sumcheck binding for the boundary MLE.
  ///
  /// Present when the proof tree contains at least one `Seq` node.
  /// Cryptographically proves `boundary_constraint_sum` is honest.
  pub boundary_pcs: Option<BoundaryPcsProof>,

  // ── decomposition / lookup layer ──────────────────────────────────────
  /// Batching challenge for the reconstruction constraint (STARK ↔ LUT binding).
  pub gamma: GF2_128,

  /// Claimed sum of the reconstruction MLE (must be zero).
  ///
  /// This proves that committed byte-decomposition columns match the
  /// main trace operands, binding the STARK and LUT witnesses together.
  pub reconstruction_sum: GF2_128,

  /// PCS + sumcheck binding for the reconstruction MLE.
  ///
  /// Cryptographically proves `reconstruction_sum` is honest — without
  /// this, a malicious prover could simply set the sum to zero.
  pub reconstruction_pcs: ReconstructionPcsProof,

  /// Per-table LogUp proofs for byte-level lookup arguments.
  pub lookup_proofs: LookupProofs,

  /// Succinct witness commitments (blake3 digests) for lookup tables.
  pub lookup_commits: LookupCommitments,

  // ── shard layer ───────────────────────────────────────────────────────
  /// Shard-level proofs (independent sumcheck per shard).
  pub shard_batch: ShardProofBatch,

  // ── recursive layer ───────────────────────────────────────────────────
  /// Recursive aggregation proof (levels of fan-in sumcheck proofs).
  pub recursive_proof: RecursiveProof,

  // ── PCS opening ───────────────────────────────────────────────────────
  /// PCS opening at the evaluation point derived from sumcheck challenges.
  pub pcs_open: OpenProof,
  /// The evaluation point `r` used for PCS opening (= sumcheck challenges).
  pub open_point: Vec<GF2_128>,
  /// Claimed evaluation of the constraint MLE at `open_point`.
  pub open_eval: GF2_128,

  // ── Phase C: storage state binding ─────────────────────────────────
  /// Sparse Merkle Tree inclusion proofs for storage slots.
  ///
  /// Present when the execution touches persistent storage (SLOAD/SSTORE).
  /// Binds `RwSummaries.storage` initial/final values to pre/post state roots.
  pub storage_proof: Option<StorageProof>,

  // ── config ────────────────────────────────────────────────────────────
  /// The recursion configuration used during proving.
  pub config: RecursiveConfig,
}

// ─── Compressed Proof ──────────────────────────────────────────────────────

/// Recursively compressed ZK-STARK proof.
///
/// The full [`Proof`] contains all shard proofs, recursive level proofs,
/// multiple PCS openings, and lookup proofs (often >1 MiB). The
/// compression step verifies the full proof internally, hashes all
/// intermediate data into a binding digest, and produces a single
/// aggregation sumcheck that reduces everything to one claim.
///
/// # Size budget
///
/// | Field                    | Size (bytes)        |
/// |--------------------------|---------------------|
/// | `type_cert`              | 68                  |
/// | `batch_commit`           | 36                  |
/// | `config`                 | 12                  |
/// | `inner_digest`           | 32                  |
/// | `compression_sumcheck`   | ~n_claims × 48      |
/// | `root_eval`, challenges  | ~16 × n_vars        |
/// | `pcs_open` (reduced)     | ~n_queries × n_vars × 80 |
/// | `storage_roots`          | 64 (if present)     |
///
/// With n_queries=20 and ~10 aggregation claims, typically 20–60 KiB.
#[derive(Debug, Clone, bincode::Encode, bincode::Decode)]
pub struct CompressedProof {
  // ── public inputs ─────────────────────────────────────────────────────
  /// Type-derivation certificate (structural shape commitment).
  pub type_cert: TypeCert,
  /// Whether boundaries exist — determines transcript construction.
  pub has_seq_boundaries: bool,
  /// The main PCS Merkle root.  Binds the committed trace polynomial.
  pub batch_commit: Commitment,
  /// Recursion/shard configuration (needed for verification).
  pub config: RecursiveConfig,

  // ── transcript reconstruction ─────────────────────────────────────────
  /// Boundary PCS commitment root (if `has_seq_boundaries`).
  /// Needed to rebuild the Fiat-Shamir transcript state for PCS verification.
  pub boundary_commit_root: Option<[u8; 32]>,
  /// Reconstruction PCS commitment root.
  /// Needed to rebuild the Fiat-Shamir transcript state for PCS verification.
  pub reconstruction_commit_root: [u8; 32],

  // ── binding digest ────────────────────────────────────────────────────
  /// Blake3 hash of the full serialised inner [`Proof`].
  ///
  /// This cryptographically binds the compressed proof to a specific
  /// inner proof.  Any change to the inner proof (shard proofs,
  /// recursive proofs, lookups, PCS openings) changes this digest.
  pub inner_digest: [u8; 32],

  // ── compression layer ─────────────────────────────────────────────────
  /// Aggregation sumcheck: proves the verification claims (evaluations,
  /// root claim consistency, lookup sums) reduce correctly.
  ///
  /// The aggregation MLE contains all intermediate claims that the
  /// verifier would check.  A single sumcheck proves their aggregate.
  pub compression_sumcheck: SumcheckProof,

  /// Evaluation claims collected from the inner proof, in deterministic
  /// order.  The verifier recomputes the aggregation MLE from these
  /// plus the public-input-derived challenges and checks the sumcheck.
  pub inner_claims: Vec<GF2_128>,

  // ── single PCS opening (reduced queries) ──────────────────────────────
  /// PCS opening at the same evaluation point as the inner proof.
  ///
  /// Uses fewer queries than the inner proof (20 vs 40) to save space.
  /// The moderate reduction is safe for the compression layer because
  /// it is cryptographically bound to the inner digest.
  pub pcs_open: OpenProof,
  /// The evaluation point for the PCS opening.
  pub open_point: Vec<GF2_128>,
  /// Claimed evaluation at `open_point`.
  pub open_eval: GF2_128,

  // ── storage layer (public output) ─────────────────────────────────────
  /// Pre-execution storage state root (if persistent storage is used).
  pub pre_state_root: Option<[u8; 32]>,
  /// Post-execution storage state root (if persistent storage is used).
  pub post_state_root: Option<[u8; 32]>,
}

/// Parameters for proof generation and verification.
#[derive(Debug, Clone)]
pub struct StarkParams {
  /// Recursion / sharding configuration.
  pub config: RecursiveConfig,
  /// PCS parameters (n_vars, n_queries).
  pub pcs_params: pcs::PcsParams,
}

impl StarkParams {
  /// Sensible defaults for a given trace size.
  ///
  /// `n_vars` is `ceil(log2(n_rows))` — the number of MLE variables
  /// needed to represent the trace.
  pub fn for_n_vars(n_vars: u32) -> Self {
    let config = RecursiveConfig::for_n_vars(n_vars);
    Self {
      config,
      pcs_params: pcs::PcsParams {
        n_vars,
        n_queries: 40,
      },
    }
  }
}
