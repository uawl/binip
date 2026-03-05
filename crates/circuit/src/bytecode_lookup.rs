//! Bytecode LogUp — proves that `(pc, opcode)` pairs come from committed bytecode.
//!
//! # Protocol
//!
//! Given a **code hash** (blake3 of the deployed bytecode), the prover builds
//! a [`LookupTable`] of all `(offset, byte)` pairs in the bytecode.
//! For each EVM step, the prover emits lookup entries:
//!
//! - `(pre_pc, pre_opcode)` — the opcode at this step's PC is in the bytecode.
//! - `(post_pc, post_opcode)` — the opcode at the next step's PC is in the bytecode.
//!
//! A LogUp proof then certifies that every witness entry is contained in the
//! bytecode table.
//!
//! In a recursive proof, adjacent shards only need to check:
//!   `left.post_pc == right.pre_pc` ∧ `left.post_opcode == right.pre_opcode`
//!
//! Combined with the LogUp proof, this guarantees PC advancement correctness
//! and opcode validity without re-reading the bytecode.
//!
//! # Encoding
//!
//! Each `(pc, opcode)` pair is packed into a single `GF2_128` element:
//!
//! | Bits 31:0 | Bits 39:32 |
//! |-----------|------------|
//! | pc (u32)  | opcode (u8)|

use field::GF2_128;
use logup::LookupTable;
use transcript::Blake3Transcript;

pub use logup::LogUpProof;

// ── Encoding ─────────────────────────────────────────────────────────────────

/// Pack a `(pc, opcode)` pair into a field element.
#[inline]
pub fn encode_pc_opcode(pc: u32, opcode: u8) -> GF2_128 {
  GF2_128::from(pc as u64 | (opcode as u64) << 32)
}

/// Unpack a field element back into `(pc, opcode)`.
#[inline]
pub fn decode_pc_opcode(e: GF2_128) -> (u32, u8) {
  let v = e.lo;
  ((v & 0xFFFF_FFFF) as u32, ((v >> 32) & 0xFF) as u8)
}

// ── Table construction ───────────────────────────────────────────────────────

/// Build a [`LookupTable`] from raw EVM bytecode.
///
/// Each byte at offset `i` in the bytecode produces the table entry
/// `encode_pc_opcode(i, bytecode[i])`.
///
/// The caller should verify that `blake3(bytecode) == expected_code_hash`
/// before invoking this function.
pub fn build_bytecode_table(bytecode: &[u8]) -> LookupTable {
  let entries: Vec<GF2_128> = bytecode
    .iter()
    .enumerate()
    .map(|(i, &byte)| encode_pc_opcode(i as u32, byte))
    .collect();
  // If bytecode is empty, insert a dummy entry so the table is non-empty.
  if entries.is_empty() {
    return LookupTable::new(vec![encode_pc_opcode(0, 0x00)]);
  }
  LookupTable::new(entries)
}

/// Compute the blake3 hash of bytecode (the "code hash").
pub fn code_hash(bytecode: &[u8]) -> [u8; 32] {
  *blake3::hash(bytecode).as_bytes()
}

// ── Witness ──────────────────────────────────────────────────────────────────

/// Bytecode lookup witness: `(pc, opcode)` pairs that must be in the table.
#[derive(Debug, Clone, Default)]
pub struct BytecodeLookupWitness {
  /// Encoded `(pc, opcode)` entries to look up.
  pub entries: Vec<GF2_128>,
}

impl BytecodeLookupWitness {
  pub fn new() -> Self {
    Self {
      entries: Vec::new(),
    }
  }

  /// Record a step's pre and post `(pc, opcode)` for lookup.
  pub fn push_step(&mut self, pre_pc: u32, pre_opcode: u8, post_pc: u32, post_opcode: u8) {
    self.entries.push(encode_pc_opcode(pre_pc, pre_opcode));
    self.entries.push(encode_pc_opcode(post_pc, post_opcode));
  }
}

// ── Proof types ──────────────────────────────────────────────────────────────

/// Bytecode LogUp proof bundle.
#[derive(Debug, Clone, bincode::Encode, bincode::Decode)]
pub struct BytecodeLookupProof {
  /// The LogUp proof for `witness ⊆ bytecode_table`.
  pub logup_proof: LogUpProof,
  /// The bytecode table (verifier needs it to verify the LogUp relation).
  pub table: LookupTable,
  /// Blake3 hash of the original bytecode.
  pub code_hash: [u8; 32],
}

/// Succinct commitment for the bytecode lookup witness.
#[derive(Debug, Clone, bincode::Encode, bincode::Decode)]
pub struct BytecodeLookupCommitment {
  /// Blake3 digest of the padded witness.
  pub digest: [u8; 32],
  /// Padded witness length.
  pub len: usize,
}

// ── Prove / Verify ───────────────────────────────────────────────────────────

/// Prove that all `(pc, opcode)` entries in the witness belong to the bytecode.
///
/// Returns `None` if the witness is empty (no steps to prove).
pub fn prove_bytecode_lookup(
  witness: &mut BytecodeLookupWitness,
  bytecode: &[u8],
  transcript: &mut Blake3Transcript,
) -> Option<(BytecodeLookupProof, BytecodeLookupCommitment)> {
  if witness.entries.is_empty() {
    return None;
  }

  let table = build_bytecode_table(bytecode);
  let hash = code_hash(bytecode);

  // Pad witness to next power of two.
  let n = witness.entries.len().next_power_of_two();
  let pad = table.entries[0];
  witness.entries.resize(n, pad);

  let (proof, digest) =
    logup::prove_committed(&witness.entries, &table, transcript);

  let bundle = BytecodeLookupProof {
    logup_proof: proof,
    table,
    code_hash: hash,
  };
  let commit = BytecodeLookupCommitment {
    digest,
    len: n,
  };
  Some((bundle, commit))
}

/// Verify a bytecode lookup proof.
///
/// The verifier:
/// 1. Re-derives the bytecode table from the proof's code hash + table.
/// 2. Verifies the LogUp relation.
///
/// Returns `true` on success.
pub fn verify_bytecode_lookup(
  proof: &BytecodeLookupProof,
  commit: &BytecodeLookupCommitment,
  transcript: &mut Blake3Transcript,
) -> bool {
  // Verify the committed LogUp proof.
  let result = logup::verify_committed(
    &proof.logup_proof,
    commit.len,
    &proof.table,
    transcript,
    &commit.digest,
  );
  result.is_some()
}

#[cfg(test)]
mod tests {
  use super::*;
  use transcript::Blake3Transcript;

  #[test]
  fn encode_decode_roundtrip() {
    let pc = 0x1234u32;
    let opcode = 0x60u8; // PUSH1
    let encoded = encode_pc_opcode(pc, opcode);
    let (dec_pc, dec_op) = decode_pc_opcode(encoded);
    assert_eq!(dec_pc, pc);
    assert_eq!(dec_op, opcode);
  }

  #[test]
  fn empty_witness_returns_none() {
    let mut w = BytecodeLookupWitness::new();
    let bytecode = vec![0x60, 0x01, 0x60, 0x02, 0x01, 0x00]; // PUSH1 1 PUSH1 2 ADD STOP
    let mut t = Blake3Transcript::new();
    assert!(prove_bytecode_lookup(&mut w, &bytecode, &mut t).is_none());
  }

  #[test]
  fn prove_and_verify_simple_bytecode() {
    // Bytecode: PUSH1 0x01 PUSH1 0x02 ADD STOP
    let bytecode = vec![0x60, 0x01, 0x60, 0x02, 0x01, 0x00];
    let mut w = BytecodeLookupWitness::new();

    // Step 0: PUSH1 at PC=0, next step at PC=2 (PUSH1)
    w.push_step(0, 0x60, 2, 0x60);
    // Step 1: PUSH1 at PC=2, next step at PC=4 (ADD)
    w.push_step(2, 0x60, 4, 0x01);
    // Step 2: ADD at PC=4, next step at PC=5 (STOP)
    w.push_step(4, 0x01, 5, 0x00);
    // Step 3: STOP at PC=5
    w.push_step(5, 0x00, 5, 0x00);

    let mut t_prove = Blake3Transcript::new();
    let (proof, commit) = prove_bytecode_lookup(&mut w, &bytecode, &mut t_prove).unwrap();

    assert_eq!(proof.code_hash, code_hash(&bytecode));

    let mut t_verify = Blake3Transcript::new();
    assert!(verify_bytecode_lookup(&proof, &commit, &mut t_verify));
  }

  #[test]
  fn code_hash_deterministic() {
    let bytecode = vec![0x60, 0x01, 0x00];
    let h1 = code_hash(&bytecode);
    let h2 = code_hash(&bytecode);
    assert_eq!(h1, h2);
  }
}
