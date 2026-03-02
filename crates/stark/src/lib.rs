//! Top-level ZK-STARK prover and verifier.
//!
//! This crate orchestrates the full proof pipeline:
//!
//! 1. Type-check the Proof Tree → [`TypeCert`](evm_types::TypeCert).
//! 2. Encode the VM trace → constraint MLE.
//! 3. PCS commit → Merkle root.
//! 4. Shard the constraint MLE → independent sumcheck proofs.
//! 5. Recursive aggregation → single root claim.
//! 6. PCS open at the challenge point.
//! 7. Package into a [`Proof`].

pub mod proof;
pub mod prover;
pub mod verifier;

pub use proof::{Proof, StarkParams};
pub use prover::{ProveError, prove_cpu, prove_cpu_par};
pub use verifier::{VerifyError, verify};
