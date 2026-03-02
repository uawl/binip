//! VM trace → Boolean circuit encoding for ZK proving.
//!
//! The circuit crate converts a sequence of [`vm::Row`]s (the execution trace)
//! into multilinear extension (MLE) polynomials over GF(2^128) and enforces
//! per-opcode constraints via the sumcheck protocol.
//!
//! # Pipeline
//!
//! 1. **Encode**: each `Row` → 8 field elements (one per column).
//! 2. **TraceTable**: collect rows into column-major `Vec<GF2_128>` arrays.
//! 3. **Constrain**: for each op-tag, build a polynomial identity that must
//!    vanish on every valid row.
//! 4. **MLE**: convert column vectors → `MlePoly` (padded to power-of-2).
//! 5. Feed the constraint MLE to [`sumcheck::prove`].

pub mod constraint;
pub mod decomp;
pub mod encoder;
pub mod lookup;
pub mod state_constraint;
pub mod trace;

pub use constraint::{ConstraintError, eval_constraint};
pub use decomp::{DecompRow, eval_reconstruction};
pub use encoder::{
  COL_ADVICE, COL_FLAGS, COL_IN0, COL_IN1, COL_IN2, COL_OP, COL_OUT, COL_PC, NUM_COLS, encode_row,
};
pub use state_constraint::{BoundaryRow, BoundaryTraceTable, extract_boundaries};
pub use trace::TraceTable;
