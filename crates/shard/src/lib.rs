//! Shard-level MLE splitting and independent sumcheck proving/verification.
//!
//! This crate partitions a large MLE polynomial into smaller **shards**,
//! each proven independently via sumcheck with a forked Blake3 transcript.
//! The recursive layer above aggregates shard proofs.
//!
//! # Pipeline
//!
//! 1. [`prover::split_mle`] — partition MLE into `2^(total_vars - shard_vars)` sub-MLEs.
//! 2. [`prover::prove_all`] — prove each shard with `Blake3Transcript::fork("shard", i)`.
//! 3. [`verifier::verify_all`] — verify each shard + total sum consistency.

pub mod config;
pub mod proof;
pub mod prover;
pub mod verifier;

pub use config::RecursiveConfig;
pub use proof::{Hash, ShardProof, ShardProofBatch};
pub use prover::{prove_all, prove_all_par, prove_shard, split_mle};
pub use verifier::{ShardVerifyResult, verify_all, verify_shard};
