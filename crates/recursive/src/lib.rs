//! Recursive aggregation for shard proofs.
//!
//! Aggregates `fan_in` child claims at each level into a single root
//! claim using iterated sumcheck proofs.

pub mod circuit;
pub mod proof;
pub mod prover;
pub mod verifier;

pub use proof::{LevelProof, RecursiveProof};
pub use prover::prove_recursive;
pub use prover::prove_recursive_par;
pub use verifier::{RecursiveVerifyError, verify_recursive};
