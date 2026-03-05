//! LUT-backed lookup arguments for trace soundness.
//!
//! # Architecture
//!
//! 1. [`collect_witnesses`] walks the execution trace and emits byte-level
//!    lookup entries for each operation that needs enforcement beyond the
//!    existing algebraic constraint.
//!
//! 2. [`prove_lookups`] runs LogUp for each non-empty witness table.
//!
//! 3. [`verify_lookups`] pads witnesses identically, then verifies all
//!    LogUp proofs.  [`verify_deterministic_checks`] handles conditions
//!    that don't need lookups (boolean range, upper-byte zeros).
//!
//! ## STARK ↔ LUT binding
//!
//! To prevent a prover from committing different values to the STARK trace
//! vs. the LUT witness, each operand of a LUT-backed opcode is:
//!
//! 1. **Decomposed** into 16 committed byte columns ([`crate::decomp`]).
//! 2. **Reconstructed** algebraically: `v = Σ byte[k] · x^{8k}`.
//! 3. **Range-checked** via the Range LUT (each byte ∈ [0, 256)).
//! 4. **Operation-checked** via the relevant op LUT.
//!
//! ## Covered operations
//!
//! | Tag | Op         | LUT tables | Purpose                                 |
//! |-----|------------|------------|-----------------------------------------|
//! | 0   | Add128     | Add        | Byte carry-chain for integer addition   |
//! | 1   | Mul128     | Mul        | Byte×byte schoolbook products           |
//! | 2   | And128     | And        | Byte-level bitwise AND                  |
//! | 8   | Chi128     | And        | AND component of Keccak χ               |
//! | 15  | RangeCheck | Add        | Carry-chain `v + pad = mask`, carry=0   |

use field::{FieldElem, GF2_128};
use logup::LookupTable;
use rayon::prelude::*;
use transcript::{Blake3Transcript, Transcript};
use vm::Row;

use crate::bytecode_lookup::{
  prove_bytecode_lookup, verify_bytecode_lookup, BytecodeLookupCommitment, BytecodeLookupProof,
  BytecodeLookupWitness,
};
use crate::decomp;

pub use logup::{LogUpClaims, LogUpProof};

// ── R/W Log types ────────────────────────────────────────────────────────────

/// A single memory read/write operation recorded from the execution trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, bincode::Encode, bincode::Decode)]
pub struct RwEntry {
  /// Address or key of the access (low 128 bits).
  pub addr: u128,
  /// Value read or written (low 128 bits).
  pub value: u128,
  /// Global position in the execution trace (monotonically increasing).
  pub counter: u32,
  /// `true` for writes, `false` for reads.
  pub is_write: bool,
}

/// Memory type classification for R/W consistency rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemType {
  /// VM internal register memory — zero-initialized.
  VmMem,
  /// EVM byte-addressable memory — zero-initialized.
  EvmMem,
  /// Persistent contract storage — initial values from world state (external).
  Storage,
  /// EIP-1153 transient storage — zero-initialized per transaction.
  Transient,
}

/// Per-address R/W summary extracted after sorting the log.
///
/// The recursive proof layer (Phase B) uses these summaries to chain
/// adjacent shards; Phase C binds `initial_value` / `final_value` to
/// Merkle roots for persistent storage.
#[derive(Debug, Clone, PartialEq, Eq, bincode::Encode, bincode::Decode)]
pub struct RwSummary {
  pub addr: u128,
  pub initial_value: u128,
  pub final_value: u128,
}

/// R/W logs partitioned by memory type.
#[derive(Debug, Clone, Default)]
pub struct RwLog {
  /// VM internal memory (Load/Store, tags 16/17).
  pub mem: Vec<RwEntry>,
  /// EVM memory (MLoad/MStore/MStore8, tags 22/23/24).
  pub emem: Vec<RwEntry>,
  /// Persistent storage (SLoad/SStore, tags 25/26).
  pub storage: Vec<RwEntry>,
  /// Transient storage (TLoad/TStore, tags 27/28).
  pub transient: Vec<RwEntry>,
}

/// LogUp proof for a single R/W log permutation argument.
#[derive(Debug, Clone, bincode::Encode, bincode::Decode)]
pub struct RwProof {
  /// LogUp proof: execution-order entries ⊆ sorted entries.
  pub logup: LogUpProof,
  /// The sorted table (committed to transcript via LogUp).
  pub sorted_table: LookupTable,
  /// Raw sorted entries — the verifier re-checks ordering, read-after-write
  /// consistency, and RLC encoding against `sorted_table`.
  pub sorted_entries: Vec<RwEntry>,
  /// Number of entries before padding.
  pub n_entries: usize,
  /// RLC challenge used to encode entries as field elements.
  pub alpha: GF2_128,
  /// Blake3 digest of the padded execution-order witness.
  pub exec_digest: [u8; 32],
}

/// Per-address summaries extracted from the R/W log.
#[derive(Debug, Clone, Default, bincode::Encode, bincode::Decode)]
pub struct RwSummaries {
  pub mem: Vec<RwSummary>,
  pub emem: Vec<RwSummary>,
  pub storage: Vec<RwSummary>,
  pub transient: Vec<RwSummary>,
}

// ── Core lookup types ────────────────────────────────────────────────────────

/// Lookup witnesses partitioned by LUT table type.
#[derive(Debug, Clone)]
pub struct LookupWitness {
  /// Range table: `lut::encode_range(byte)`.
  pub range: Vec<GF2_128>,
  /// And table: `lut::encode_binary(a, b, a&b)`.
  pub and_op: Vec<GF2_128>,
  /// Add table: `lut::encode_add(a, b, sum, carry)`.
  pub add_op: Vec<GF2_128>,
  /// Mul table: `lut::encode_mul(a, b, lo, hi)`.
  pub mul_op: Vec<GF2_128>,
  /// R/W logs for memory consistency (Phase A).
  pub rw: RwLog,
  /// Bytecode `(pc, opcode)` lookup witness.
  pub bytecode: BytecodeLookupWitness,
  /// Raw bytecode bytes (needed by the prover to build the table).
  /// Set by the caller; `None` if no bytecode proof is needed.
  pub bytecode_raw: Option<Vec<u8>>,
}

/// LogUp proofs, one per active LUT table.
#[derive(Debug, Clone, bincode::Encode, bincode::Decode)]
pub struct LookupProofs {
  pub range: Option<(LogUpProof, LookupTable)>,
  pub and_op: Option<(LogUpProof, LookupTable)>,
  pub add_op: Option<(LogUpProof, LookupTable)>,
  pub mul_op: Option<(LogUpProof, LookupTable)>,
  /// R/W consistency proofs per memory type.
  pub rw_mem: Option<RwProof>,
  pub rw_emem: Option<RwProof>,
  pub rw_storage: Option<RwProof>,
  pub rw_transient: Option<RwProof>,
  /// Per-address R/W summaries (Phase B output).
  ///
  /// Extracted during proving; the verifier re-derives from sorted entries
  /// and checks they match.
  pub rw_summaries: RwSummaries,
  /// Bytecode `(pc, opcode)` LogUp proof.
  pub bytecode: Option<BytecodeLookupProof>,
}

/// Succinct witness commitments for the lookup tables.
///
/// Replaces [`LookupWitness`] in the proof: O(1) instead of O(n).
/// Each active table stores a 32-byte blake3 digest of the padded
/// witness and its padded length (needed for transcript replay).
#[derive(Debug, Clone, bincode::Encode, bincode::Decode)]
pub struct LookupCommitments {
  pub range: Option<([u8; 32], usize)>,
  pub and_op: Option<([u8; 32], usize)>,
  pub add_op: Option<([u8; 32], usize)>,
  pub mul_op: Option<([u8; 32], usize)>,
  /// R/W log commitments (exec-order witness digest + count).
  pub rw_mem: Option<([u8; 32], usize)>,
  pub rw_emem: Option<([u8; 32], usize)>,
  pub rw_storage: Option<([u8; 32], usize)>,
  pub rw_transient: Option<([u8; 32], usize)>,
  /// Bytecode lookup witness commitment.
  pub bytecode: Option<BytecodeLookupCommitment>,
}

#[inline]
fn bytes_of(v: u128) -> [u8; 16] {
  v.to_le_bytes()
}

// ── Witness collection ───────────────────────────────────────────────────────

/// Collect byte-level lookup witnesses from every trace row.
pub fn collect_witnesses(rows: &[Row]) -> LookupWitness {
  let mut w = LookupWitness {
    range: Vec::new(),
    and_op: Vec::new(),
    add_op: Vec::new(),
    mul_op: Vec::new(),
    rw: RwLog::default(),
    bytecode: BytecodeLookupWitness::new(),
    bytecode_raw: None,
  };
  for (i, row) in rows.iter().enumerate() {
    emit_decomp_ranges(&mut w, row);
    emit_rw_entry(&mut w.rw, row, i as u32);
    match row.op as u32 {
      0 => emit_add128(&mut w, row),
      1 => {
        emit_mul128(&mut w, row);
        emit_mul_accum(&mut w, row);
      }
      2 => emit_and128(&mut w, row),
      8 => emit_chi128(&mut w, row),
      12 => {
        emit_check_div(&mut w, row);
        emit_mul_accum(&mut w, row);
      }
      13 => {
        emit_check_mul(&mut w, row);
        emit_mul_accum(&mut w, row);
      }
      14 => {
        emit_check_inv(&mut w, row);
        emit_mul_accum(&mut w, row);
      }
      15 => emit_range_check(&mut w, row),
      33 => emit_cmp_lt(&mut w, row),
      _ => {}
    }
  }
  w
}

/// Parallel variant of [`collect_witnesses`] — per-row witnesses computed
/// via rayon, then concatenated.
///
/// Uses per-chunk accumulation with pre-sized buffers to reduce allocation
/// overhead: instead of 786k tiny Vecs, we get ~12 chunk-sized Vecs.
pub fn collect_witnesses_par(rows: &[Row]) -> LookupWitness {
  let n_threads = rayon::current_num_threads().max(1);
  let chunk_size = (rows.len() + n_threads - 1) / n_threads;

  // Compute global counter offsets for each chunk so that RW entry
  // counters are globally monotone across the merged result.
  let chunk_starts: Vec<usize> = {
    let mut starts = Vec::with_capacity(
      (rows.len() + chunk_size.max(1) - 1) / chunk_size.max(1),
    );
    let mut offset = 0usize;
    for c in rows.chunks(chunk_size.max(1)) {
      starts.push(offset);
      offset += c.len();
    }
    starts
  };

  let chunks: Vec<LookupWitness> = rows
    .par_chunks(chunk_size.max(1))
    .enumerate()
    .map(|(ci, chunk)| {
      // Pre-allocate based on average entries per row:
      // ~50 range + ~16 and + ~32 add + ~256 mul (conservative upper bound)
      let n = chunk.len();
      let base_counter = chunk_starts[ci] as u32;
      let mut w = LookupWitness {
        range: Vec::with_capacity(n * 50),
        and_op: Vec::with_capacity(n * 4),
        add_op: Vec::with_capacity(n * 8),
        mul_op: Vec::with_capacity(n * 8),
        rw: RwLog::default(),
        bytecode: BytecodeLookupWitness::new(),
        bytecode_raw: None,
      };
      for (j, row) in chunk.iter().enumerate() {
        emit_decomp_ranges(&mut w, row);
        emit_rw_entry(&mut w.rw, row, base_counter + j as u32);
        match row.op as u32 {
          0 => emit_add128(&mut w, row),
          1 => {
            emit_mul128(&mut w, row);
            emit_mul_accum(&mut w, row);
          }
          2 => emit_and128(&mut w, row),
          8 => emit_chi128(&mut w, row),
          12 => {
            emit_check_div(&mut w, row);
            emit_mul_accum(&mut w, row);
          }
          13 => {
            emit_check_mul(&mut w, row);
            emit_mul_accum(&mut w, row);
          }
          14 => {
            emit_check_inv(&mut w, row);
            emit_mul_accum(&mut w, row);
          }
          15 => emit_range_check(&mut w, row),
          33 => emit_cmp_lt(&mut w, row),
          _ => {}
        }
      }
      w
    })
    .collect();

  // Merge ~12 chunk results: pre-compute sizes, single allocation for final buffers.
  let total_range: usize = chunks.iter().map(|w| w.range.len()).sum();
  let total_and: usize = chunks.iter().map(|w| w.and_op.len()).sum();
  let total_add: usize = chunks.iter().map(|w| w.add_op.len()).sum();
  let total_mul: usize = chunks.iter().map(|w| w.mul_op.len()).sum();
  let total_mem: usize = chunks.iter().map(|w| w.rw.mem.len()).sum();
  let total_emem: usize = chunks.iter().map(|w| w.rw.emem.len()).sum();
  let total_storage: usize = chunks.iter().map(|w| w.rw.storage.len()).sum();
  let total_transient: usize = chunks.iter().map(|w| w.rw.transient.len()).sum();

  let mut merged = LookupWitness {
    range: Vec::with_capacity(total_range),
    and_op: Vec::with_capacity(total_and),
    add_op: Vec::with_capacity(total_add),
    mul_op: Vec::with_capacity(total_mul),
    rw: RwLog {
      mem: Vec::with_capacity(total_mem),
      emem: Vec::with_capacity(total_emem),
      storage: Vec::with_capacity(total_storage),
      transient: Vec::with_capacity(total_transient),
    },
    bytecode: BytecodeLookupWitness::new(),
    bytecode_raw: None,
  };
  for w in chunks {
    merged.range.extend(w.range);
    merged.and_op.extend(w.and_op);
    merged.add_op.extend(w.add_op);
    merged.mul_op.extend(w.mul_op);
    merged.rw.mem.extend(w.rw.mem);
    merged.rw.emem.extend(w.rw.emem);
    merged.rw.storage.extend(w.rw.storage);
    merged.rw.transient.extend(w.rw.transient);
  }
  merged
}

/// Emit Range LUT entries for all byte-decomposed operands (STARK ↔ LUT
/// binding).  Each committed byte column must be proven ∈ [0, 256).
fn emit_decomp_ranges(w: &mut LookupWitness, row: &Row) {
  let tag = row.op as u32;
  let mask = decomp::decomp_mask(tag);
  if mask.in0 {
    for b in row.in0.to_le_bytes() {
      w.range.push(lut::encode_range(b));
    }
  }
  if mask.in1 {
    for b in row.in1.to_le_bytes() {
      w.range.push(lut::encode_range(b));
    }
  }
  if mask.in2 {
    for b in row.in2.to_le_bytes() {
      w.range.push(lut::encode_range(b));
    }
  }
  if mask.out {
    for b in row.out.to_le_bytes() {
      w.range.push(lut::encode_range(b));
    }
  }
  if mask.flags {
    for b in row.flags.to_le_bytes() {
      w.range.push(lut::encode_range(b));
    }
  }
}

/// RangeCheck (tag 15): prove `v + pad = mask` via byte-level Add LUT.
///
/// Row layout: in0 = v, in1 = mask = 2^bits − 1, advice = pad = mask − v,
/// out = mask.  The carry chain proves integer addition; an extra entry
/// forces carry_out = 0, so `v + pad = mask` without overflow, implying
/// `v ≤ mask`, i.e., `v < 2^bits`.
fn emit_range_check(w: &mut LookupWitness, row: &Row) {
  let v = row.in0;
  let pad = row.advice;
  let mask = row.out; // = in1 = 2^bits - 1

  let v_bs = bytes_of(v);
  let p_bs = bytes_of(pad);
  let m_bs = bytes_of(mask);

  let mut carry: u8 = 0;
  for k in 0..16 {
    // Step 1: v[k] + pad[k] → (partial, c1)
    let sum1 = v_bs[k] as u16 + p_bs[k] as u16;
    let p = (sum1 & 0xFF) as u8;
    let c1 = (sum1 >> 8) as u8;
    w.add_op.push(lut::encode_add(v_bs[k], p_bs[k], p, c1));

    // Step 2: partial + carry_in → (result_byte, c2)
    let sum2 = p as u16 + carry as u16;
    let s = (sum2 & 0xFF) as u8;
    let c2 = (sum2 >> 8) as u8;
    w.add_op.push(lut::encode_add(p, carry, s, c2));

    debug_assert_eq!(s, m_bs[k]);
    carry = c1 ^ c2;
  }

  // Extra entry: force carry_out = 0.
  // encode_add(0, 0, 0, 0) is valid (0+0=0).
  // encode_add(1, 0, 0, 0) would mean 1+0=0, which is NOT in the table.
  w.add_op.push(lut::encode_add(carry, 0, 0, 0));
}

/// CmpLt (tag 33): prove unsigned 128-bit less-than via Add LUT.
///
/// Row: in0 = a, in1 = b, out = result ∈ {0,1}, advice = pad.
/// - result = 1: proves `a + pad + 1 = b` (carry_out = 0) → pad = b−a−1 ≥ 0 → a < b.
/// - result = 0: proves `b + pad = a`     (carry_out = 0) → pad = a−b   ≥ 0 → a ≥ b.
fn emit_cmp_lt(w: &mut LookupWitness, row: &Row) {
  let a = row.in0;
  let b = row.in1;
  let result = row.out;
  let pad = row.advice;

  let (x_bs, y_bs, expected_bs, initial_carry) = if result == 1 {
    (bytes_of(a), bytes_of(pad), bytes_of(b), 1u8)
  } else {
    (bytes_of(b), bytes_of(pad), bytes_of(a), 0u8)
  };

  let mut carry = initial_carry;
  for k in 0..16 {
    let sum1 = x_bs[k] as u16 + y_bs[k] as u16;
    let p = (sum1 & 0xFF) as u8;
    let c1 = (sum1 >> 8) as u8;
    w.add_op.push(lut::encode_add(x_bs[k], y_bs[k], p, c1));

    let sum2 = p as u16 + carry as u16;
    let s = (sum2 & 0xFF) as u8;
    let c2 = (sum2 >> 8) as u8;
    w.add_op.push(lut::encode_add(p, carry, s, c2));

    debug_assert_eq!(s, expected_bs[k]);
    carry = c1 ^ c2;
  }

  // carry_out must be 0
  w.add_op.push(lut::encode_add(carry, 0, 0, 0));
}

/// Mul128 (tag 1): schoolbook byte×byte products via Mul LUT.
///
/// Emits 256 entries: one `encode_mul(a_i, b_j, lo, hi)` for each (i,j) pair.
/// Column-sum accumulation is verified deterministically in
/// [`verify_deterministic_checks`].
fn emit_mul128(w: &mut LookupWitness, row: &Row) {
  let a = bytes_of(row.in0);
  let b = bytes_of(row.in1);
  for i in 0..16 {
    for j in 0..16 {
      let prod = a[i] as u16 * b[j] as u16;
      let lo = (prod & 0xFF) as u8;
      let hi = ((prod >> 8) & 0xFF) as u8;
      w.mul_op.push(lut::encode_mul(a[i], b[j], lo, hi));
    }
  }
}

/// And128 (tag 2): for each byte position emit an And LUT entry.
fn emit_and128(w: &mut LookupWitness, row: &Row) {
  let a = bytes_of(row.in0);
  let b = bytes_of(row.in1);
  let c = bytes_of(row.out);
  for k in 0..16 {
    w.and_op.push(lut::encode_binary(a[k], b[k], c[k]));
  }
}

/// Chi128 (tag 8): `out = in0 ^ ((~in1) & in2)`.
///
/// The AND component `(~in1_byte) & in2_byte` is proven via the And LUT.
/// XOR with `in0` is already algebraically sound.
fn emit_chi128(w: &mut LookupWitness, row: &Row) {
  let in0 = bytes_of(row.in0);
  let in1 = bytes_of(row.in1);
  let in2 = bytes_of(row.in2);
  let out = bytes_of(row.out);
  for k in 0..16 {
    let not_b = !in1[k];
    let and_result = out[k] ^ in0[k]; // = (~in1) & in2
    w.and_op.push(lut::encode_binary(not_b, in2[k], and_result));
  }
}

/// Add128 (tag 0): byte-level carry chain via two Add LUT lookups per byte.
///
/// Carry-in is recovered from the trace values: `result ∈ {a+b, a+b+1}`.
fn emit_add128(w: &mut LookupWitness, row: &Row) {
  let a = row.in0;
  let b = row.in1;
  let result = row.out;

  let cin = recover_cin(a, b, result);

  let a_bs = bytes_of(a);
  let b_bs = bytes_of(b);
  let r_bs = bytes_of(result);

  let mut carry = cin;
  for k in 0..16 {
    // Step 1: a[k] + b[k] → (partial, c1)
    let sum1 = a_bs[k] as u16 + b_bs[k] as u16;
    let p = (sum1 & 0xFF) as u8;
    let c1 = (sum1 >> 8) as u8;
    w.add_op.push(lut::encode_add(a_bs[k], b_bs[k], p, c1));

    // Step 2: partial + carry_in → (result_byte, c2)
    let sum2 = p as u16 + carry as u16;
    let s = (sum2 & 0xFF) as u8;
    let c2 = (sum2 >> 8) as u8;
    w.add_op.push(lut::encode_add(p, carry, s, c2));

    debug_assert_eq!(s, r_bs[k]);
    carry = c1 ^ c2; // at most 1, proven by non-overlap
  }
}

fn recover_cin(a: u128, b: u128, result: u128) -> u8 {
  if result == a.wrapping_add(b) {
    0
  } else if result == a.wrapping_add(b).wrapping_add(1) {
    1
  } else {
    panic!("Add128: result is neither a+b nor a+b+1");
  }
}

// ── Integer multiplication helpers (tags 1, 12, 13, 14) ──────────────────────

/// Resolve the (a, b) operands and 32-byte result for multiplication rows.
///
/// Returns `(a_bytes, b_bytes, result_bytes_32)` where result_bytes_32 is
/// the expected 256-bit product laid out as 32 little-endian bytes.
fn mul_operands(row: &Row) -> ([u8; 16], [u8; 16], [u8; 32]) {
  let tag = row.op as u32;
  match tag {
    // Mul128: in0 × in1 → out (lo) || flags (hi)
    1 => {
      let mut result = [0u8; 32];
      result[..16].copy_from_slice(&row.out.to_le_bytes());
      result[16..].copy_from_slice(&row.flags.to_le_bytes());
      (bytes_of(row.in0), bytes_of(row.in1), result)
    }
    // CheckDiv: advice (divisor) × in0 (quot) → product (dividend − rem)
    // Result = in2 (dividend), but carry chain proves product only.
    // The product + rem = dividend is checked separately via Add LUT.
    12 => {
      // Product bytes are recomputed deterministically (not stored in trace)
      let (lo, hi) = widening_mul(row.advice, row.in0);
      let mut result = [0u8; 32];
      result[..16].copy_from_slice(&lo.to_le_bytes());
      result[16..].copy_from_slice(&hi.to_le_bytes());
      (bytes_of(row.advice), bytes_of(row.in0), result)
    }
    // CheckMul: in2 × advice → in1 (hi) || in0 (lo)
    13 => {
      let mut result = [0u8; 32];
      result[..16].copy_from_slice(&row.in0.to_le_bytes());
      result[16..].copy_from_slice(&row.in1.to_le_bytes());
      (bytes_of(row.in2), bytes_of(row.advice), result)
    }
    // CheckInv: in0 × in1 ≡ 1 mod 2^128, full product low = 1, high = k
    14 => {
      let (lo, hi) = widening_mul(row.in0, row.in1);
      let mut result = [0u8; 32];
      result[..16].copy_from_slice(&lo.to_le_bytes());
      result[16..].copy_from_slice(&hi.to_le_bytes());
      (bytes_of(row.in0), bytes_of(row.in1), result)
    }
    _ => unreachable!("mul_operands called with non-mul tag {}", tag),
  }
}

/// Emit byte×byte schoolbook products for any integer multiplication row.
///
/// Works for Mul128, CheckMul, CheckDiv, CheckInv — the (a, b) operands
/// are resolved by [`mul_operands`].
fn emit_mul_products(w: &mut LookupWitness, a: &[u8; 16], b: &[u8; 16]) {
  for i in 0..16 {
    for j in 0..16 {
      let prod = a[i] as u16 * b[j] as u16;
      let lo = (prod & 0xFF) as u8;
      let hi = ((prod >> 8) & 0xFF) as u8;
      w.mul_op.push(lut::encode_mul(a[i], b[j], lo, hi));
    }
  }
}

/// Emit the accumulation Add LUT entries that prove the schoolbook
/// multiplication column sums match the committed result bytes + carries.
///
/// For each output byte position `k` (0..32), verifies:
///
/// ```text
/// col_sum_lo[k] + carry[k-1] = result[k] + c1 * 256    (Add LUT)
/// col_sum_hi[k] + c1         = carry[k]                 (Add LUT)
/// ```
///
/// Where `col_sum[k] = Σ lo[i][j] for i+j=k  +  Σ hi[i][j] for i+j=k−1`
/// from the schoolbook byte products, and `result[k]`/`carry[k]` are
/// committed via the decomposition system.
fn emit_mul_accum(w: &mut LookupWitness, row: &Row) {
  let (a_bs, b_bs, result_bs) = mul_operands(row);
  let d = decomp::DecompRow::compute(row);

  let mut carry: u8 = 0;
  for k in 0u32..32 {
    // Sum contributions: lo bytes from products at position k,
    // hi bytes from products at position k-1.
    let mut col_sum: u32 = 0;
    let i_min = k.saturating_sub(15) as usize;
    let i_max = (k as usize).min(15);
    for i in i_min..=i_max {
      let j = k as usize - i;
      if j < 16 {
        let prod = a_bs[i] as u32 * b_bs[j] as u32;
        col_sum += prod & 0xFF;
      }
    }
    if k > 0 {
      let pk = k - 1;
      let pi_min = pk.saturating_sub(15) as usize;
      let pi_max = (pk as usize).min(15);
      for i in pi_min..=pi_max {
        let j = pk as usize - i;
        if j < 16 {
          let prod = a_bs[i] as u32 * b_bs[j] as u32;
          col_sum += (prod >> 8) & 0xFF;
        }
      }
    }

    let col_sum_lo = (col_sum & 0xFF) as u8;
    let col_sum_hi = ((col_sum >> 8) & 0xFF) as u8;

    // Step 1: col_sum_lo + carry_in → (result[k], c1)
    let full1 = col_sum_lo as u16 + carry as u16;
    let c1 = (full1 >> 8) as u8;
    w.add_op.push(lut::encode_add(
      col_sum_lo,
      carry,
      result_bs[k as usize],
      c1,
    ));

    // Step 2: col_sum_hi + c1 → (carry_out, 0)
    let new_carry = col_sum_hi.wrapping_add(c1);
    w.add_op.push(lut::encode_add(col_sum_hi, c1, new_carry, 0));

    debug_assert_eq!(result_bs[k as usize], (full1 & 0xFF) as u8);
    debug_assert_eq!(new_carry, d.mul_carries[k as usize]);
    carry = new_carry;
  }
}

/// CheckDiv (tag 12): divisor × quot byte products + product+rem=dividend via Add LUT.
///
/// Columns: in0=quot, in1=rem, in2=dividend, advice=divisor.
fn emit_check_div(w: &mut LookupWitness, row: &Row) {
  // 1. Byte products for divisor × quot.
  let a = bytes_of(row.advice);
  let b = bytes_of(row.in0);
  emit_mul_products(w, &a, &b);

  // 2. product + rem = dividend via byte-level Add carry chain.
  //    product_lo = (advice * in0) & mask128, rem = in1, dividend = in2.
  let (product_lo, _product_hi) = widening_mul(row.advice, row.in0);
  let p_bs = bytes_of(product_lo);
  let rem_bs = bytes_of(row.in1);
  let div_bs = bytes_of(row.in2);

  let mut carry: u8 = 0;
  for k in 0..16 {
    // Step 1: product_byte + rem_byte → (partial, c1)
    let sum1 = p_bs[k] as u16 + rem_bs[k] as u16;
    let p = (sum1 & 0xFF) as u8;
    let c1 = (sum1 >> 8) as u8;
    w.add_op.push(lut::encode_add(p_bs[k], rem_bs[k], p, c1));

    // Step 2: partial + carry_in → (dividend_byte, c2)
    let sum2 = p as u16 + carry as u16;
    let s = (sum2 & 0xFF) as u8;
    let c2 = (sum2 >> 8) as u8;
    w.add_op.push(lut::encode_add(p, carry, s, c2));

    debug_assert_eq!(s, div_bs[k]);
    carry = c1 ^ c2;
  }
}

/// CheckMul (tag 13): a × b byte products.
///
/// Columns: in0=q_lo, in1=q_hi, in2=a, advice=b.
fn emit_check_mul(w: &mut LookupWitness, row: &Row) {
  let a = bytes_of(row.in2);
  let b = bytes_of(row.advice);
  emit_mul_products(w, &a, &b);
}

/// CheckInv (tag 14): a × a_inv byte products.
///
/// Columns: in0=a, in1=a_inv.
fn emit_check_inv(w: &mut LookupWitness, row: &Row) {
  let a = bytes_of(row.in0);
  let b = bytes_of(row.in1);
  emit_mul_products(w, &a, &b);
}

// ── R/W log emission ─────────────────────────────────────────────────────────

/// Emit an R/W log entry for memory operations (tags 16–28).
fn emit_rw_entry(rw: &mut RwLog, row: &Row, counter: u32) {
  match row.op as u32 {
    // VM Load: addr=in0, value=out (read)
    16 => rw.mem.push(RwEntry {
      addr: row.in0,
      value: row.out,
      counter,
      is_write: false,
    }),
    // VM Store: addr=in0, value=in1 (write)
    17 => rw.mem.push(RwEntry {
      addr: row.in0,
      value: row.in1,
      counter,
      is_write: true,
    }),
    // MLoad: offset=in0, value=out (read)
    22 => rw.emem.push(RwEntry {
      addr: row.in0,
      value: row.out,
      counter,
      is_write: false,
    }),
    // MStore: offset=in0, value=in1 (write)
    23 => rw.emem.push(RwEntry {
      addr: row.in0,
      value: row.in1,
      counter,
      is_write: true,
    }),
    // MStore8: offset=in0, value=in1 (write, single byte)
    24 => rw.emem.push(RwEntry {
      addr: row.in0,
      value: row.in1,
      counter,
      is_write: true,
    }),
    // SLoad: key=in0, value=out (read)
    25 => rw.storage.push(RwEntry {
      addr: row.in0,
      value: row.out,
      counter,
      is_write: false,
    }),
    // SStore: key=in0, value=in1 (write)
    26 => rw.storage.push(RwEntry {
      addr: row.in0,
      value: row.in1,
      counter,
      is_write: true,
    }),
    // TLoad: key=in0, value=out (read)
    27 => rw.transient.push(RwEntry {
      addr: row.in0,
      value: row.out,
      counter,
      is_write: false,
    }),
    // TStore: key=in0, value=in1 (write)
    28 => rw.transient.push(RwEntry {
      addr: row.in0,
      value: row.in1,
      counter,
      is_write: true,
    }),
    _ => {}
  }
}

// ── R/W log encoding (RLC) ───────────────────────────────────────────────────

/// Encode an R/W entry as a single field element via random linear combination.
///
/// ```text
/// entry = addr + α · value + α² · (counter | is_write << 32)
/// ```
///
/// The RLC uses challenge `α` squeezed from the transcript, ensuring that
/// distinct logical entries produce distinct encodings with overwhelming
/// probability.
#[inline]
fn encode_rw(entry: &RwEntry, alpha: GF2_128, alpha2: GF2_128) -> GF2_128 {
  let a = GF2_128::new(entry.addr as u64, (entry.addr >> 64) as u64);
  let v = GF2_128::new(entry.value as u64, (entry.value >> 64) as u64);
  let meta = GF2_128::from(entry.counter as u64 | ((entry.is_write as u64) << 32));
  a + alpha * v + alpha2 * meta
}

/// Encode a slice of R/W entries into field elements.
fn encode_rw_entries(entries: &[RwEntry], alpha: GF2_128) -> Vec<GF2_128> {
  let alpha2 = alpha * alpha;
  entries.iter().map(|e| encode_rw(e, alpha, alpha2)).collect()
}

// ── R/W sorted-trace verification ────────────────────────────────────────────

/// R/W log verification error.
#[derive(Debug, Clone, thiserror::Error)]
pub enum RwError {
  #[error("read from zero-initialized address {addr:#x} returned non-zero value {value:#x}")]
  InitialReadNonZero { addr: u128, value: u128 },
  #[error("read at counter {counter} addr {addr:#x}: expected {expected:#x}, got {got:#x}")]
  ReadMismatch {
    addr: u128,
    counter: u32,
    expected: u128,
    got: u128,
  },
  #[error("sorted log not sorted at index {index}: ({addr_a:#x}, {counter_a}) >= ({addr_b:#x}, {counter_b})")]
  NotSorted {
    index: usize,
    addr_a: u128,
    counter_a: u32,
    addr_b: u128,
    counter_b: u32,
  },
  #[error("RLC encoding mismatch at index {index}")]
  EncodingMismatch { index: usize },
  #[error("bridge mismatch at addr {addr:#x}: left final {left_final:#x} != right initial {right_initial:#x}")]
  BridgeMismatch {
    addr: u128,
    left_final: u128,
    right_initial: u128,
  },
}

/// Verify that already-sorted R/W entries satisfy ordering and read-after-write
/// consistency, and extract per-address summaries.
///
/// Does NOT sort — entries must already be sorted by `(addr, counter)`.
/// Use [`sort_and_verify_rw`] when the prover needs to sort in-place.
pub fn verify_rw_consistency(
  entries: &[RwEntry],
  mem_type: MemType,
) -> Result<Vec<RwSummary>, RwError> {
  if entries.is_empty() {
    return Ok(vec![]);
  }

  // Verify sorted order.
  for i in 0..entries.len() - 1 {
    let a = &entries[i];
    let b = &entries[i + 1];
    if (a.addr, a.counter) >= (b.addr, b.counter) {
      return Err(RwError::NotSorted {
        index: i,
        addr_a: a.addr,
        counter_a: a.counter,
        addr_b: b.addr,
        counter_b: b.counter,
      });
    }
  }

  let zero_init = matches!(mem_type, MemType::VmMem | MemType::EvmMem | MemType::Transient);
  let mut summaries = Vec::new();
  let mut i = 0;

  while i < entries.len() {
    let addr = entries[i].addr;

    // Determine initial value for this address.
    let initial_value = if entries[i].is_write {
      // First access is a write — initial state is zero for zero-init types,
      // unknown for storage (will be bound externally in Phase C).
      if zero_init { 0u128 } else { 0u128 }
    } else {
      // First access is a read — the value IS the initial state.
      let val = entries[i].value;
      if zero_init && val != 0 {
        return Err(RwError::InitialReadNonZero { addr, value: val });
      }
      val
    };

    // Walk all entries for this address, verifying read consistency.
    let mut current_value = initial_value;
    while i < entries.len() && entries[i].addr == addr {
      if entries[i].is_write {
        current_value = entries[i].value;
      } else {
        // Read must return the most recently written value.
        if entries[i].value != current_value {
          return Err(RwError::ReadMismatch {
            addr,
            counter: entries[i].counter,
            expected: current_value,
            got: entries[i].value,
          });
        }
      }
      i += 1;
    }

    summaries.push(RwSummary {
      addr,
      initial_value,
      final_value: current_value,
    });
  }

  Ok(summaries)
}

/// Sort R/W entries by `(addr, counter)`, verify read-after-write consistency,
/// and extract per-address summaries.
///
/// For `mem_type` ∈ {VmMem, EvmMem, Transient}: the initial value of every
/// address is 0.  For `Storage`: the initial value of the first read is
/// accepted as-is (it will be bound to a Merkle root in Phase C).
///
/// Returns sorted entries (in-place) and per-address `(initial, final)` summaries.
pub fn sort_and_verify_rw(
  entries: &mut [RwEntry],
  mem_type: MemType,
) -> Result<Vec<RwSummary>, RwError> {
  entries.sort_unstable_by_key(|e| (e.addr, e.counter));
  verify_rw_consistency(entries, mem_type)
}

// ── Deterministic checks ─────────────────────────────────────────────────────

/// Verify conditions that don't need lookups:
///
/// - **Mul128**: full 256-bit product matches out||flags.
///
/// RangeCheck soundness is now fully enforced by the Add LUT carry
/// chain + carry_out = 0 entry + the algebraic `out = in1` constraint.
pub fn verify_deterministic_checks(rows: &[Row]) -> bool {
  for row in rows {
    match row.op as u32 {
      // Mul128: verify full 256-bit product matches out||flags.
      1 => {
        let (expected_lo, expected_hi) = widening_mul(row.in0, row.in1);
        if row.out != expected_lo || row.flags != expected_hi {
          return false;
        }
      }
      _ => {}
    }
  }
  true
}

/// 128×128 → 256-bit schoolbook multiplication (u64-limb decomposition).
fn widening_mul(a: u128, b: u128) -> (u128, u128) {
  let al = a as u64 as u128;
  let ah = (a >> 64) as u64 as u128;
  let bl = b as u64 as u128;
  let bh = (b >> 64) as u64 as u128;

  let p0 = al * bl;
  let p1 = al * bh;
  let p2 = ah * bl;
  let p3 = ah * bh;

  let (mid, mid_overflow) = p1.overflowing_add(p2);
  let mid_lo = (mid as u64) as u128;
  let mid_hi = mid >> 64;

  let (lo, lo_carry) = p0.overflowing_add(mid_lo << 64);
  let mut hi = p3.wrapping_add(mid_hi).wrapping_add(lo_carry as u128);
  if mid_overflow {
    hi = hi.wrapping_add(1u128 << 64);
  }
  (lo, hi)
}

// ── R/W LogUp permutation argument ──────────────────────────────────────────

/// Prove a single R/W log: sorted-trace permutation via LogUp.
///
/// 1. Squeeze RLC challenge α from the transcript.
/// 2. Sort entries by (addr, counter), verify read-after-write consistency.
/// 3. Encode both execution-order and sorted entries with α.
/// 4. Build a [`LookupTable`] from sorted encoded entries.
/// 5. Run LogUp: exec_encoded ⊆ sorted_table.
///
/// Returns (proof, summaries) or None if the log is empty.
fn prove_rw_single(
  entries: &mut Vec<RwEntry>,
  mem_type: MemType,
  label: &[u8],
  transcript: &mut Blake3Transcript,
) -> Option<(RwProof, Vec<RwSummary>)> {
  if entries.is_empty() {
    return None;
  }

  transcript.absorb_bytes(label);
  let n_entries = entries.len();

  // 1. RLC challenge
  transcript.absorb_bytes(&(n_entries as u64).to_le_bytes());
  let alpha: GF2_128 = transcript.squeeze_challenge();

  // 2. Encode execution-order entries
  let exec_encoded = encode_rw_entries(entries, alpha);

  // 3. Sort and verify read-after-write consistency
  let summaries = sort_and_verify_rw(entries, mem_type)
    .expect("R/W consistency check must pass for honest prover");

  // 4. Snapshot sorted entries (verifier will re-check ordering + consistency)
  let sorted_entries = entries.to_vec();

  // 5. Encode sorted entries
  let sorted_encoded = encode_rw_entries(entries, alpha);

  // 6. Build table from sorted entries, run LogUp
  let table = LookupTable::new(sorted_encoded);
  let mut witness = exec_encoded;
  let n = witness.len().next_power_of_two();
  witness.resize(n, table.entries[0]);

  let (logup_proof, exec_digest) = logup::prove_committed(&witness, &table, transcript);

  // 7. Commit summaries to transcript (Phase B binding)
  absorb_summaries(&summaries, transcript);

  Some((
    RwProof {
      logup: logup_proof,
      sorted_table: table,
      sorted_entries,
      n_entries,
      alpha,
      exec_digest,
    },
    summaries,
  ))
}

/// Verify a single R/W LogUp proof and re-derive per-address summaries.
///
/// The verifier:
/// 1. Re-derives α from the transcript.
/// 2. Verifies the LogUp permutation argument (exec ⊆ sorted).
/// 3. **Phase B**: Re-checks that `sorted_entries` are truly sorted by
///    `(addr, counter)`, satisfy read-after-write consistency, and re-encode
///    to the values in `sorted_table`.
/// 4. Absorbs re-derived summaries into the transcript (matching prover).
/// 5. Returns the verified summaries.
fn verify_rw_single(
  proof: &RwProof,
  label: &[u8],
  mem_type: MemType,
  transcript: &mut Blake3Transcript,
) -> Option<Vec<RwSummary>> {
  transcript.absorb_bytes(label);

  // 1. Re-derive α
  transcript.absorb_bytes(&(proof.n_entries as u64).to_le_bytes());
  let alpha: GF2_128 = transcript.squeeze_challenge();

  if alpha != proof.alpha {
    return None;
  }

  // 2. Verify LogUp (committed variant uses digest stored in proof)
  let n_padded = proof.n_entries.next_power_of_two();
  logup::verify_committed(
    &proof.logup,
    n_padded,
    &proof.sorted_table,
    transcript,
    &proof.exec_digest,
  )?;

  // ── Phase B: verify sorted entries ────────────────────────────────────
  //
  // The LogUp argument proves exec-order ⊂ sorted table (as encoded field
  // elements). We now verify the *raw* sorted entries are:
  //   a) correctly encoded — each raw entry, encoded with α, matches the
  //      corresponding sorted_table entry;
  //   b) properly ordered by (addr, counter);
  //   c) read-after-write consistent; and
  //   d) zero-init compliant for applicable memory types.

  // 3a. Length check
  if proof.sorted_entries.len() != proof.n_entries {
    return None;
  }

  // 3b. Re-encode sorted entries and verify against sorted_table
  let alpha2 = alpha * alpha;
  for (i, entry) in proof.sorted_entries.iter().enumerate() {
    let encoded = encode_rw(entry, alpha, alpha2);
    if encoded != proof.sorted_table.entries[i] {
      return None;
    }
  }
  // Verify that padding entries (n_entries .. sorted_table.len) are all
  // equal to the first table entry (LookupTable pads with entries[0]).
  if proof.n_entries < proof.sorted_table.entries.len() {
    let pad_val = proof.sorted_table.entries[0];
    for entry in &proof.sorted_table.entries[proof.n_entries..] {
      if *entry != pad_val {
        return None;
      }
    }
  }

  // 3c. Verify ordering + read-after-write consistency → extract summaries
  let summaries = verify_rw_consistency(&proof.sorted_entries, mem_type).ok()?;

  // 4. Absorb summaries into transcript (matches prover)
  absorb_summaries(&summaries, transcript);

  Some(summaries)
}

// ── Phase B helpers ──────────────────────────────────────────────────────────

/// Deterministically absorb R/W summaries into the transcript.
fn absorb_summaries(summaries: &[RwSummary], transcript: &mut Blake3Transcript) {
  transcript.absorb_bytes(&(summaries.len() as u64).to_le_bytes());
  for s in summaries {
    transcript.absorb_bytes(&s.addr.to_le_bytes());
    transcript.absorb_bytes(&s.initial_value.to_le_bytes());
    transcript.absorb_bytes(&s.final_value.to_le_bytes());
  }
}

/// Compose R/W summaries from two adjacent trace segments.
///
/// For addresses in both segments, verifies the bridge constraint:
/// `left.final_value == right.initial_value`, then merges:
/// `(left.initial, right.final)`.
///
/// Both inputs must be sorted by `addr` (as produced by [`verify_rw_consistency`]).
pub fn compose_summaries(
  left: &[RwSummary],
  right: &[RwSummary],
) -> Result<Vec<RwSummary>, RwError> {
  let mut result = Vec::with_capacity(left.len().max(right.len()));
  let (mut l, mut r) = (0, 0);

  while l < left.len() && r < right.len() {
    use std::cmp::Ordering;
    match left[l].addr.cmp(&right[r].addr) {
      Ordering::Less => {
        result.push(left[l].clone());
        l += 1;
      }
      Ordering::Greater => {
        result.push(right[r].clone());
        r += 1;
      }
      Ordering::Equal => {
        if left[l].final_value != right[r].initial_value {
          return Err(RwError::BridgeMismatch {
            addr: left[l].addr,
            left_final: left[l].final_value,
            right_initial: right[r].initial_value,
          });
        }
        result.push(RwSummary {
          addr: left[l].addr,
          initial_value: left[l].initial_value,
          final_value: right[r].final_value,
        });
        l += 1;
        r += 1;
      }
    }
  }
  result.extend_from_slice(&left[l..]);
  result.extend_from_slice(&right[r..]);
  Ok(result)
}

/// Compose all four memory-type summaries from two adjacent segments.
pub fn compose_rw_summaries(
  left: &RwSummaries,
  right: &RwSummaries,
) -> Result<RwSummaries, RwError> {
  Ok(RwSummaries {
    mem: compose_summaries(&left.mem, &right.mem)?,
    emem: compose_summaries(&left.emem, &right.emem)?,
    storage: compose_summaries(&left.storage, &right.storage)?,
    transient: compose_summaries(&left.transient, &right.transient)?,
  })
}

// ── Prove / Verify ───────────────────────────────────────────────────────────

/// Extract the witness digest from an RwProof.
fn rw_exec_digest(proof: &RwProof) -> [u8; 32] {
  proof.exec_digest
}

/// Prove all lookup arguments.
///
/// Witnesses in `w` are padded to powers of two after this call.
pub fn prove_lookups(w: &mut LookupWitness, transcript: &mut Blake3Transcript) -> LookupProofs {
  transcript.absorb_bytes(b"circuit:lookups");

  let range = if !w.range.is_empty() {
    transcript.absorb_bytes(b"lut:range");
    let (proof, table) = lut::prove(lut::Op8::Range, &mut w.range, transcript);
    Some((proof, table))
  } else {
    None
  };

  let and_op = if !w.and_op.is_empty() {
    transcript.absorb_bytes(b"lut:and");
    let (proof, table) = lut::prove(lut::Op8::And, &mut w.and_op, transcript);
    Some((proof, table))
  } else {
    None
  };

  let add_op = if !w.add_op.is_empty() {
    transcript.absorb_bytes(b"lut:add");
    let (proof, table) = lut::prove(lut::Op8::Add, &mut w.add_op, transcript);
    Some((proof, table))
  } else {
    None
  };

  let mul_op = if !w.mul_op.is_empty() {
    transcript.absorb_bytes(b"lut:mul");
    let (proof, table) = lut::prove(lut::Op8::Mul, &mut w.mul_op, transcript);
    Some((proof, table))
  } else {
    None
  };

  // R/W log proofs (sequentially after ALU lookups)
  let mut t_rw = transcript.clone();
  t_rw.absorb_bytes(b"rw:logs");
  let rw_mem = prove_rw_single(&mut w.rw.mem, MemType::VmMem, b"rw:mem", &mut t_rw);
  let rw_emem = prove_rw_single(&mut w.rw.emem, MemType::EvmMem, b"rw:emem", &mut t_rw);
  let rw_storage = prove_rw_single(&mut w.rw.storage, MemType::Storage, b"rw:storage", &mut t_rw);
  let rw_transient = prove_rw_single(
    &mut w.rw.transient,
    MemType::Transient,
    b"rw:transient",
    &mut t_rw,
  );

  let rw_summaries = RwSummaries {
    mem: rw_mem.as_ref().map_or(vec![], |(_, s)| s.clone()),
    emem: rw_emem.as_ref().map_or(vec![], |(_, s)| s.clone()),
    storage: rw_storage.as_ref().map_or(vec![], |(_, s)| s.clone()),
    transient: rw_transient.as_ref().map_or(vec![], |(_, s)| s.clone()),
  };

  LookupProofs {
    range,
    and_op,
    add_op,
    mul_op,
    rw_mem: rw_mem.map(|(p, _)| p),
    rw_emem: rw_emem.map(|(p, _)| p),
    rw_storage: rw_storage.map(|(p, _)| p),
    rw_transient: rw_transient.map(|(p, _)| p),
    rw_summaries,
    bytecode: None,
  }
}

/// Sequential variant of [`prove_lookups`] that also returns commitments.
///
/// Uses forked transcripts (same as [`prove_lookups_par`]) for identical
/// output, but executes each table proof sequentially.
pub fn prove_lookups_committed(
  w: &mut LookupWitness,
  transcript: &mut Blake3Transcript,
) -> (LookupProofs, LookupCommitments) {
  transcript.absorb_bytes(b"circuit:lookups");

  let mut t_range = transcript.fork("lut:range", 0);
  let mut t_and = transcript.fork("lut:and", 1);
  let mut t_add = transcript.fork("lut:add", 2);
  let mut t_mul = transcript.fork("lut:mul", 3);

  let range_res = if w.range.is_empty() {
    None
  } else {
    let (p, t, d) = lut::prove_committed(lut::Op8::Range, &mut w.range, &mut t_range);
    Some((p, t, d, w.range.len()))
  };
  let and_res = if w.and_op.is_empty() {
    None
  } else {
    let (p, t, d) = lut::prove_committed(lut::Op8::And, &mut w.and_op, &mut t_and);
    Some((p, t, d, w.and_op.len()))
  };
  let add_res = if w.add_op.is_empty() {
    None
  } else {
    let (p, t, d) = lut::prove_committed(lut::Op8::Add, &mut w.add_op, &mut t_add);
    Some((p, t, d, w.add_op.len()))
  };
  let mul_res = if w.mul_op.is_empty() {
    None
  } else {
    let (p, t, d) = lut::prove_committed(lut::Op8::Mul, &mut w.mul_op, &mut t_mul);
    Some((p, t, d, w.mul_op.len()))
  };

  // R/W log proofs
  let mut t_rw = transcript.fork("rw:logs", 4);
  let mut t_bytecode = transcript.fork("lut:bytecode", 5);
  let rw_mem = prove_rw_single(&mut w.rw.mem, MemType::VmMem, b"rw:mem", &mut t_rw);
  let rw_emem = prove_rw_single(&mut w.rw.emem, MemType::EvmMem, b"rw:emem", &mut t_rw);
  let rw_storage = prove_rw_single(&mut w.rw.storage, MemType::Storage, b"rw:storage", &mut t_rw);
  let rw_transient = prove_rw_single(
    &mut w.rw.transient,
    MemType::Transient,
    b"rw:transient",
    &mut t_rw,
  );

  // Bytecode LogUp proof.
  let bytecode_res = if let Some(ref bytecode) = w.bytecode_raw {
    prove_bytecode_lookup(&mut w.bytecode, bytecode, &mut t_bytecode)
  } else {
    None
  };

  let proofs = LookupProofs {
    range: range_res.as_ref().map(|(p, t, _, _)| (p.clone(), t.clone())),
    and_op: and_res.as_ref().map(|(p, t, _, _)| (p.clone(), t.clone())),
    add_op: add_res.as_ref().map(|(p, t, _, _)| (p.clone(), t.clone())),
    mul_op: mul_res.as_ref().map(|(p, t, _, _)| (p.clone(), t.clone())),
    rw_mem: rw_mem.as_ref().map(|(p, _)| p.clone()),
    rw_emem: rw_emem.as_ref().map(|(p, _)| p.clone()),
    rw_storage: rw_storage.as_ref().map(|(p, _)| p.clone()),
    rw_transient: rw_transient.as_ref().map(|(p, _)| p.clone()),
    rw_summaries: RwSummaries {
      mem: rw_mem.as_ref().map_or(vec![], |(_, s)| s.clone()),
      emem: rw_emem.as_ref().map_or(vec![], |(_, s)| s.clone()),
      storage: rw_storage.as_ref().map_or(vec![], |(_, s)| s.clone()),
      transient: rw_transient.as_ref().map_or(vec![], |(_, s)| s.clone()),
    },
    bytecode: bytecode_res.as_ref().map(|(p, _)| p.clone()),
  };

  let commits = LookupCommitments {
    range: range_res.map(|(_, _, d, n)| (d, n)),
    and_op: and_res.map(|(_, _, d, n)| (d, n)),
    add_op: add_res.map(|(_, _, d, n)| (d, n)),
    mul_op: mul_res.map(|(_, _, d, n)| (d, n)),
    rw_mem: rw_mem.map(|(p, _)| (rw_exec_digest(&p), p.n_entries)),
    rw_emem: rw_emem.map(|(p, _)| (rw_exec_digest(&p), p.n_entries)),
    rw_storage: rw_storage.map(|(p, _)| (rw_exec_digest(&p), p.n_entries)),
    rw_transient: rw_transient.map(|(p, _)| (rw_exec_digest(&p), p.n_entries)),
    bytecode: bytecode_res.map(|(_, c)| c),
  };

  (proofs, commits)
}

/// Parallel variant of [`prove_lookups`] — each table gets a forked
/// transcript so the four LogUp proofs can run concurrently via rayon.
///
/// Returns `(proofs, commitments)` where commitments contain succinct
/// 32-byte digests of each padded witness table.
pub fn prove_lookups_par(
  w: &mut LookupWitness,
  transcript: &mut Blake3Transcript,
) -> (LookupProofs, LookupCommitments) {
  transcript.absorb_bytes(b"circuit:lookups");

  // Fork transcripts — independent and deterministic per table.
  let mut t_range = transcript.fork("lut:range", 0);
  let mut t_and = transcript.fork("lut:and", 1);
  let mut t_add = transcript.fork("lut:add", 2);
  let mut t_mul = transcript.fork("lut:mul", 3);
  let mut t_bytecode = transcript.fork("lut:bytecode", 5);

  let ((range_res, and_res), (add_res, mul_res)) = rayon::join(
    || {
      rayon::join(
        || {
          if w.range.is_empty() {
            None
          } else {
            let (p, t, d) = lut::prove_committed(lut::Op8::Range, &mut w.range, &mut t_range);
            Some((p, t, d, w.range.len()))
          }
        },
        || {
          if w.and_op.is_empty() {
            None
          } else {
            let (p, t, d) = lut::prove_committed(lut::Op8::And, &mut w.and_op, &mut t_and);
            Some((p, t, d, w.and_op.len()))
          }
        },
      )
    },
    || {
      rayon::join(
        || {
          if w.add_op.is_empty() {
            None
          } else {
            let (p, t, d) = lut::prove_committed(lut::Op8::Add, &mut w.add_op, &mut t_add);
            Some((p, t, d, w.add_op.len()))
          }
        },
        || {
          if w.mul_op.is_empty() {
            None
          } else {
            let (p, t, d) = lut::prove_committed(lut::Op8::Mul, &mut w.mul_op, &mut t_mul);
            Some((p, t, d, w.mul_op.len()))
          }
        },
      )
    },
  );

  // Bytecode LogUp proof (uses its own forked transcript).
  let bytecode_res = if let Some(ref bytecode) = w.bytecode_raw {
    prove_bytecode_lookup(&mut w.bytecode, bytecode, &mut t_bytecode)
  } else {
    None
  };

  // R/W log proofs (sequential — sorting is not parallelizable here because
  // prove_rw_single mutates entries in-place).
  let mut t_rw = transcript.fork("rw:logs", 4);
  let rw_mem = prove_rw_single(&mut w.rw.mem, MemType::VmMem, b"rw:mem", &mut t_rw);
  let rw_emem = prove_rw_single(&mut w.rw.emem, MemType::EvmMem, b"rw:emem", &mut t_rw);
  let rw_storage = prove_rw_single(&mut w.rw.storage, MemType::Storage, b"rw:storage", &mut t_rw);
  let rw_transient = prove_rw_single(
    &mut w.rw.transient,
    MemType::Transient,
    b"rw:transient",
    &mut t_rw,
  );

  let proofs = LookupProofs {
    range: range_res
      .as_ref()
      .map(|(p, t, _, _)| (p.clone(), t.clone())),
    and_op: and_res.as_ref().map(|(p, t, _, _)| (p.clone(), t.clone())),
    add_op: add_res.as_ref().map(|(p, t, _, _)| (p.clone(), t.clone())),
    mul_op: mul_res.as_ref().map(|(p, t, _, _)| (p.clone(), t.clone())),
    rw_mem: rw_mem.as_ref().map(|(p, _)| p.clone()),
    rw_emem: rw_emem.as_ref().map(|(p, _)| p.clone()),
    rw_storage: rw_storage.as_ref().map(|(p, _)| p.clone()),
    rw_transient: rw_transient.as_ref().map(|(p, _)| p.clone()),
    rw_summaries: RwSummaries {
      mem: rw_mem.as_ref().map_or(vec![], |(_, s)| s.clone()),
      emem: rw_emem.as_ref().map_or(vec![], |(_, s)| s.clone()),
      storage: rw_storage.as_ref().map_or(vec![], |(_, s)| s.clone()),
      transient: rw_transient.as_ref().map_or(vec![], |(_, s)| s.clone()),
    },
    bytecode: bytecode_res.as_ref().map(|(p, _)| p.clone()),
  };

  let commits = LookupCommitments {
    range: range_res.map(|(_, _, d, n)| (d, n)),
    and_op: and_res.map(|(_, _, d, n)| (d, n)),
    add_op: add_res.map(|(_, _, d, n)| (d, n)),
    mul_op: mul_res.map(|(_, _, d, n)| (d, n)),
    rw_mem: rw_mem.map(|(p, _)| (rw_exec_digest(&p), p.n_entries)),
    rw_emem: rw_emem.map(|(p, _)| (rw_exec_digest(&p), p.n_entries)),
    rw_storage: rw_storage.map(|(p, _)| (rw_exec_digest(&p), p.n_entries)),
    rw_transient: rw_transient.map(|(p, _)| (rw_exec_digest(&p), p.n_entries)),
    bytecode: bytecode_res.map(|(_, c)| c),
  };

  (proofs, commits)
}

/// Verify all lookup proofs.
///
/// Witnesses in `w` are padded to match the prover's padding.
/// Returns verified [`RwSummaries`] on success.
pub fn verify_lookups(
  proofs: &LookupProofs,
  w: &mut LookupWitness,
  transcript: &mut Blake3Transcript,
) -> Option<RwSummaries> {
  transcript.absorb_bytes(b"circuit:lookups");

  if let Some((proof, table)) = &proofs.range {
    transcript.absorb_bytes(b"lut:range");
    let n = w.range.len().next_power_of_two();
    w.range.resize(n, table.entries[0]);
    if lut::verify(proof, &w.range, table, transcript).is_none() {
      return None;
    }
  } else if !w.range.is_empty() {
    return None;
  }

  if let Some((proof, table)) = &proofs.and_op {
    transcript.absorb_bytes(b"lut:and");
    let n = w.and_op.len().next_power_of_two();
    w.and_op.resize(n, table.entries[0]);
    if lut::verify(proof, &w.and_op, table, transcript).is_none() {
      return None;
    }
  } else if !w.and_op.is_empty() {
    return None;
  }

  if let Some((proof, table)) = &proofs.add_op {
    transcript.absorb_bytes(b"lut:add");
    let n = w.add_op.len().next_power_of_two();
    w.add_op.resize(n, table.entries[0]);
    if lut::verify(proof, &w.add_op, table, transcript).is_none() {
      return None;
    }
  } else if !w.add_op.is_empty() {
    return None;
  }

  if let Some((proof, table)) = &proofs.mul_op {
    transcript.absorb_bytes(b"lut:mul");
    let n = w.mul_op.len().next_power_of_two();
    w.mul_op.resize(n, table.entries[0]);
    if lut::verify(proof, &w.mul_op, table, transcript).is_none() {
      return None;
    }
  } else if !w.mul_op.is_empty() {
    return None;
  }

  // R/W consistency proofs + Phase B summary extraction
  let mut t_rw = transcript.clone();
  t_rw.absorb_bytes(b"rw:logs");
  let mem = if let Some(proof) = &proofs.rw_mem {
    verify_rw_single(proof, b"rw:mem", MemType::VmMem, &mut t_rw)?
  } else {
    vec![]
  };
  let emem = if let Some(proof) = &proofs.rw_emem {
    verify_rw_single(proof, b"rw:emem", MemType::EvmMem, &mut t_rw)?
  } else {
    vec![]
  };
  let storage = if let Some(proof) = &proofs.rw_storage {
    verify_rw_single(proof, b"rw:storage", MemType::Storage, &mut t_rw)?
  } else {
    vec![]
  };
  let transient = if let Some(proof) = &proofs.rw_transient {
    verify_rw_single(proof, b"rw:transient", MemType::Transient, &mut t_rw)?
  } else {
    vec![]
  };

  Some(RwSummaries { mem, emem, storage, transient })
}

/// Parallel variant of [`verify_lookups`] — uses forked transcripts
/// matching [`prove_lookups_par`].
///
/// Accepts [`LookupCommitments`] (succinct digests) instead of raw
/// witness data.  Returns verified [`RwSummaries`] on success.
pub fn verify_lookups_par(
  proofs: &LookupProofs,
  commits: &LookupCommitments,
  transcript: &mut Blake3Transcript,
) -> Option<RwSummaries> {
  transcript.absorb_bytes(b"circuit:lookups");

  let mut t_range = transcript.fork("lut:range", 0);
  let mut t_and = transcript.fork("lut:and", 1);
  let mut t_add = transcript.fork("lut:add", 2);
  let mut t_mul = transcript.fork("lut:mul", 3);
  let mut t_bytecode = transcript.fork("lut:bytecode", 5);

  // Check presence consistency first (proof ↔ commit non-emptiness).
  if proofs.range.is_none() != commits.range.is_none() {
    return None;
  }
  if proofs.and_op.is_none() != commits.and_op.is_none() {
    return None;
  }
  if proofs.add_op.is_none() != commits.add_op.is_none() {
    return None;
  }
  if proofs.mul_op.is_none() != commits.mul_op.is_none() {
    return None;
  }
  // R/W presence consistency
  if proofs.rw_mem.is_none() != commits.rw_mem.is_none() {
    return None;
  }
  if proofs.rw_emem.is_none() != commits.rw_emem.is_none() {
    return None;
  }
  if proofs.rw_storage.is_none() != commits.rw_storage.is_none() {
    return None;
  }
  if proofs.rw_transient.is_none() != commits.rw_transient.is_none() {
    return None;
  }
  // Bytecode presence consistency
  if proofs.bytecode.is_none() != commits.bytecode.is_none() {
    return None;
  }

  let ((r1, r2), (r3, r4)) = rayon::join(
    || {
      rayon::join(
        || {
          proofs.range.as_ref().map_or(true, |(proof, table)| {
            let (digest, n) = commits.range.as_ref().unwrap();
            lut::verify_committed(proof, *n, table, &mut t_range, digest).is_some()
          })
        },
        || {
          proofs.and_op.as_ref().map_or(true, |(proof, table)| {
            let (digest, n) = commits.and_op.as_ref().unwrap();
            lut::verify_committed(proof, *n, table, &mut t_and, digest).is_some()
          })
        },
      )
    },
    || {
      rayon::join(
        || {
          proofs.add_op.as_ref().map_or(true, |(proof, table)| {
            let (digest, n) = commits.add_op.as_ref().unwrap();
            lut::verify_committed(proof, *n, table, &mut t_add, digest).is_some()
          })
        },
        || {
          proofs.mul_op.as_ref().map_or(true, |(proof, table)| {
            let (digest, n) = commits.mul_op.as_ref().unwrap();
            lut::verify_committed(proof, *n, table, &mut t_mul, digest).is_some()
          })
        },
      )
    },
  );

  if !(r1 && r2 && r3 && r4) {
    return None;
  }

  // R/W log verification (sequential — mirrors prover's sequential rw fork).
  // Phase B: collect verified summaries.
  let mut t_rw = transcript.fork("rw:logs", 4);
  let mem = if let Some(proof) = &proofs.rw_mem {
    verify_rw_single(proof, b"rw:mem", MemType::VmMem, &mut t_rw)?
  } else {
    vec![]
  };
  let emem = if let Some(proof) = &proofs.rw_emem {
    verify_rw_single(proof, b"rw:emem", MemType::EvmMem, &mut t_rw)?
  } else {
    vec![]
  };
  let storage = if let Some(proof) = &proofs.rw_storage {
    verify_rw_single(proof, b"rw:storage", MemType::Storage, &mut t_rw)?
  } else {
    vec![]
  };
  let transient = if let Some(proof) = &proofs.rw_transient {
    verify_rw_single(proof, b"rw:transient", MemType::Transient, &mut t_rw)?
  } else {
    vec![]
  };

  // Bytecode LogUp verification.
  if let (Some(proof), Some(commit)) = (&proofs.bytecode, &commits.bytecode) {
    if !verify_bytecode_lookup(proof, commit, &mut t_bytecode) {
      return None;
    }
  }

  Some(RwSummaries { mem, emem, storage, transient })
}

// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
  use super::*;
  use transcript::Blake3Transcript;

  fn make_row(op: u32, in0: u128, in1: u128, in2: u128, out: u128, flags: u128) -> Row {
    Row {
      pc: 0,
      op: op as u128,
      in0,
      in1,
      in2,
      out,
      flags,
      advice: 0,
    }
  }

  /// Helper: collect → prove → verify.
  fn run_lookup(rows: &[Row]) -> bool {
    let mut w_prover = collect_witnesses(rows);
    let mut tp = Blake3Transcript::new();
    let proofs = prove_lookups(&mut w_prover, &mut tp);

    let mut w_verifier = collect_witnesses(rows);
    let mut tv = Blake3Transcript::new();
    let ok = verify_lookups(&proofs, &mut w_verifier, &mut tv).is_some();
    ok && verify_deterministic_checks(rows)
  }

  // ── RangeCheck ──────────────────────────────────────────────────────

  /// Helper: build a valid RangeCheck row for given value and bit width.
  fn range_row(val: u128, bits: u32) -> Row {
    let mask = if bits >= 128 { u128::MAX } else { (1u128 << bits) - 1 };
    let pad = mask - val;
    Row { pc: 0, op: 15, in0: val, in1: mask, in2: 0, out: mask, flags: 0, advice: pad }
  }

  #[test]
  fn range_check_8bit() {
    let rows = vec![
      range_row(0, 8),
      range_row(42, 8),
      range_row(255, 8),
      range_row(0, 8), // pad to pow2
    ];
    assert!(run_lookup(&rows));
  }

  #[test]
  fn range_check_1bit_boolean() {
    // bits=1: mask=1, proves v ∈ {0,1} via Add LUT.
    let rows = vec![range_row(0, 1), range_row(1, 1)];
    assert!(run_lookup(&rows));
  }

  #[test]
  #[should_panic]
  fn range_check_1bit_fails_for_2() {
    // v=2. mask=1, pad=1-2 underflows → carry_out=1 → LUT entry invalid.
    let mask: u128 = 1;
    let pad = mask.wrapping_sub(2); // wrapping: u128::MAX
    let row = Row { pc: 0, op: 15, in0: 2, in1: mask, in2: 0, out: mask, flags: 0, advice: pad };
    run_lookup(&[row]);
  }

  #[test]
  #[should_panic]
  fn range_check_upper_bytes_fail() {
    // value=256 with bits=8 → mask=255, pad wraps, carry_out=1 → fails.
    let mask: u128 = 255;
    let val: u128 = 256;
    let pad = mask.wrapping_sub(val);
    let row = Row { pc: 0, op: 15, in0: val, in1: mask, in2: 0, out: mask, flags: 0, advice: pad };
    run_lookup(&[row]);
  }

  // ── And128 ──────────────────────────────────────────────────────────

  #[test]
  fn and128_correct() {
    let a: u128 = 0xFF00_FF00_FF00_FF00_FF00_FF00_FF00_FF00;
    let b: u128 = 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
    let rows = vec![make_row(2, a, b, 0, a & b, 0)];
    assert!(run_lookup(&rows));
  }

  #[test]
  fn and128_all_ones() {
    let a = u128::MAX;
    let b = 0x1234_5678_9ABC_DEF0_1234_5678_9ABC_DEF0u128;
    let rows = vec![make_row(2, a, b, 0, a & b, 0)];
    assert!(run_lookup(&rows));
  }

  // ── Add128 ──────────────────────────────────────────────────────────

  #[test]
  fn add128_no_carry() {
    let a: u128 = 100;
    let b: u128 = 200;
    let result = a.wrapping_add(b);
    let rows = vec![make_row(0, a, b, 0, result, 0)];
    assert!(run_lookup(&rows));
  }

  #[test]
  fn add128_with_carry() {
    let a = u128::MAX;
    let b: u128 = 1;
    let result = a.wrapping_add(b); // wraps to 0
    let rows = vec![make_row(0, a, b, 0, result, 1)];
    assert!(run_lookup(&rows));
  }

  #[test]
  fn add128_byte_boundary_carries() {
    // 0xFF + 0x01 = 0x100 → carry at byte boundary
    let a: u128 = 0xFF;
    let b: u128 = 0x01;
    let rows = vec![make_row(0, a, b, 0, a + b, 0)];
    assert!(run_lookup(&rows));
  }

  #[test]
  fn add128_large_values() {
    let a: u128 = 0xDEAD_BEEF_CAFE_BABE_1234_5678_9ABC_DEF0;
    let b: u128 = 0x1111_1111_1111_1111_1111_1111_1111_1111;
    let result = a.wrapping_add(b);
    let carry = if a.checked_add(b).is_none() { 1 } else { 0 };
    let rows = vec![make_row(0, a, b, 0, result, carry)];
    assert!(run_lookup(&rows));
  }

  // ── Chi128 ──────────────────────────────────────────────────────────

  #[test]
  fn chi128_correct() {
    let a: u128 = 0xAAAA_BBBB_CCCC_DDDD_EEEE_FFFF_0000_1111;
    let b: u128 = 0x1234_5678_9ABC_DEF0_1234_5678_9ABC_DEF0;
    let c: u128 = 0xFEDC_BA98_7654_3210_FEDC_BA98_7654_3210;
    let result = a ^ ((!b) & c);
    let rows = vec![make_row(8, a, b, c, result, 0)];
    assert!(run_lookup(&rows));
  }

  // ── Mul128 ──────────────────────────────────────────────────────────

  /// Helper: compute 128×128→256-bit product for test rows.
  fn mul_row(a: u128, b: u128) -> Row {
    let (lo, hi) = widening_mul(a, b);
    make_row(1, a, b, 0, lo, hi)
  }

  #[test]
  fn mul128_small() {
    let rows = vec![mul_row(7, 11)];
    assert!(run_lookup(&rows));
  }

  #[test]
  fn mul128_large() {
    let a: u128 = 0xDEAD_BEEF_CAFE_BABE_1234_5678_9ABC_DEF0;
    let b: u128 = 0x1111_1111_1111_1111_1111_1111_1111_1111;
    let rows = vec![mul_row(a, b)];
    assert!(run_lookup(&rows));
  }

  #[test]
  fn mul128_max() {
    let rows = vec![mul_row(u128::MAX, u128::MAX)];
    assert!(run_lookup(&rows));
  }

  #[test]
  fn mul128_by_zero() {
    let rows = vec![mul_row(0xABCD, 0)];
    assert!(run_lookup(&rows));
  }

  #[test]
  fn mul128_deterministic_wrong_result() {
    // Correct product lo, but tamper flags (hi).
    let a: u128 = 100;
    let b: u128 = 200;
    let (lo, _hi) = widening_mul(a, b);
    let rows = vec![make_row(1, a, b, 0, lo, 999)];
    assert!(!verify_deterministic_checks(&rows));
  }

  // ── Mixed ops ──────────────────────────────────────────────────────

  #[test]
  fn mixed_operations() {
    let rows = vec![
      // And128
      make_row(2, 0xFF, 0x0F, 0, 0x0F, 0),
      // Add128
      make_row(0, 100, 200, 0, 300, 0),
      // Mul128
      mul_row(6, 7),
      // RangeCheck bits=8
      range_row(42, 8),
      // Xor128 (tag 3 — no lookup needed, just padding)
      make_row(3, 5, 3, 0, 6, 0),
    ];
    assert!(run_lookup(&rows));
  }

  // ── Soundness: tampered traces ─────────────────────────────────────

  #[test]
  #[should_panic(expected = "not found in table")]
  fn and128_wrong_result_fails() {
    let a: u128 = 0xFF;
    let b: u128 = 0x0F;
    // Correct AND = 0x0F, but we put 0xFF (wrong).
    let rows = vec![make_row(2, a, b, 0, 0xFF, 0)];
    // This should fail at prove time because (0xFF, 0x0F, 0xFF)
    // is not a valid AND triple and won't be in the table.
    run_lookup(&rows);
  }

  #[test]
  #[should_panic(expected = "neither a+b nor a+b+1")]
  fn add128_wrong_result_panics() {
    let rows = vec![make_row(0, 100, 200, 0, 999, 0)];
    run_lookup(&rows);
  }

  /// Parallel collect → prove → verify matches sequential.
  fn run_lookup_par(rows: &[Row]) -> bool {
    let mut w_prover = collect_witnesses_par(rows);
    let mut tp = Blake3Transcript::new();
    let (proofs, commits) = prove_lookups_par(&mut w_prover, &mut tp);

    let mut tv = Blake3Transcript::new();
    verify_lookups_par(&proofs, &commits, &mut tv).is_some() && verify_deterministic_checks(rows)
  }

  #[test]
  fn par_mixed_operations() {
    let rows = vec![
      make_row(2, 0xFF, 0x0F, 0, 0x0F, 0),
      make_row(0, 100, 200, 0, 300, 0),
      mul_row(6, 7),
      range_row(42, 8),
      make_row(3, 5, 3, 0, 6, 0),
    ];
    assert!(run_lookup_par(&rows));
  }

  #[test]
  fn par_witnesses_match_sequential() {
    let rows = vec![
      make_row(2, 0xFF00, 0x0F0F, 0, 0xFF00 & 0x0F0F, 0),
      make_row(0, 100, 200, 0, 300, 0),
      mul_row(3, 5),
    ];
    let seq = collect_witnesses(&rows);
    let par = collect_witnesses_par(&rows);
    assert_eq!(seq.range, par.range);
    assert_eq!(seq.and_op, par.and_op);
    assert_eq!(seq.add_op, par.add_op);
    assert_eq!(seq.mul_op, par.mul_op);
  }

  // ── R/W log: emit_rw_entry routing ─────────────────────────────────

  #[test]
  fn emit_rw_entry_routes_vm_load_store() {
    // Load (16): addr=in0, value=out → mem, is_write=false
    // Store (17): addr=in0, value=in1 → mem, is_write=true
    let rows = vec![
      make_row(16, 42, 0, 0, 99, 0),  // VM Load
      make_row(17, 42, 77, 0, 0, 0),   // VM Store
    ];
    let w = collect_witnesses(&rows);
    assert_eq!(w.rw.mem.len(), 2);
    assert!(!w.rw.mem[0].is_write);
    assert_eq!(w.rw.mem[0].addr, 42);
    assert_eq!(w.rw.mem[0].value, 99);
    assert!(w.rw.mem[1].is_write);
    assert_eq!(w.rw.mem[1].value, 77);
    assert!(w.rw.emem.is_empty());
    assert!(w.rw.storage.is_empty());
    assert!(w.rw.transient.is_empty());
  }

  #[test]
  fn emit_rw_entry_routes_evm_memory() {
    let rows = vec![
      make_row(22, 0x20, 0, 0, 0xAB, 0),  // MLoad
      make_row(23, 0x40, 0xCD, 0, 0, 0),   // MStore
      make_row(24, 0x60, 0xEF, 0, 0, 0),   // MStore8
    ];
    let w = collect_witnesses(&rows);
    assert_eq!(w.rw.emem.len(), 3);
    assert!(!w.rw.emem[0].is_write); // MLoad → read
    assert!(w.rw.emem[1].is_write);  // MStore → write
    assert!(w.rw.emem[2].is_write);  // MStore8 → write
    assert!(w.rw.mem.is_empty());
  }

  #[test]
  fn emit_rw_entry_routes_storage_and_transient() {
    let rows = vec![
      make_row(25, 1, 0, 0, 100, 0), // SLoad
      make_row(26, 1, 200, 0, 0, 0),  // SStore
      make_row(27, 2, 0, 0, 300, 0),  // TLoad
      make_row(28, 2, 400, 0, 0, 0),  // TStore
    ];
    let w = collect_witnesses(&rows);
    assert_eq!(w.rw.storage.len(), 2);
    assert_eq!(w.rw.transient.len(), 2);
    assert!(!w.rw.storage[0].is_write);
    assert!(w.rw.storage[1].is_write);
    assert!(!w.rw.transient[0].is_write);
    assert!(w.rw.transient[1].is_write);
  }

  #[test]
  fn emit_rw_entry_counters_sequential() {
    let rows = vec![
      make_row(16, 1, 0, 0, 10, 0),
      make_row(17, 1, 20, 0, 0, 0),
      make_row(16, 1, 0, 0, 20, 0),
    ];
    let w = collect_witnesses(&rows);
    assert_eq!(w.rw.mem[0].counter, 0);
    assert_eq!(w.rw.mem[1].counter, 1);
    assert_eq!(w.rw.mem[2].counter, 2);
  }

  // ── R/W log: sort_and_verify_rw ────────────────────────────────────

  #[test]
  fn sort_verify_write_then_read() {
    let mut entries = vec![
      RwEntry { addr: 10, value: 42, counter: 0, is_write: true },
      RwEntry { addr: 10, value: 42, counter: 1, is_write: false },
    ];
    let summaries = sort_and_verify_rw(&mut entries, MemType::VmMem).unwrap();
    assert_eq!(summaries.len(), 1);
    assert_eq!(summaries[0].addr, 10);
    assert_eq!(summaries[0].initial_value, 0); // zero-init, first is write
    assert_eq!(summaries[0].final_value, 42);
  }

  #[test]
  fn sort_verify_multiple_addresses() {
    let mut entries = vec![
      RwEntry { addr: 20, value: 1, counter: 0, is_write: true },
      RwEntry { addr: 10, value: 2, counter: 1, is_write: true },
      RwEntry { addr: 20, value: 1, counter: 2, is_write: false },
      RwEntry { addr: 10, value: 2, counter: 3, is_write: false },
    ];
    let summaries = sort_and_verify_rw(&mut entries, MemType::VmMem).unwrap();
    assert_eq!(summaries.len(), 2);
    // Sorted by addr: 10 first, then 20.
    assert_eq!(summaries[0].addr, 10);
    assert_eq!(summaries[1].addr, 20);
  }

  #[test]
  fn sort_verify_read_mismatch_error() {
    let mut entries = vec![
      RwEntry { addr: 10, value: 42, counter: 0, is_write: true },
      RwEntry { addr: 10, value: 99, counter: 1, is_write: false }, // wrong!
    ];
    let result = sort_and_verify_rw(&mut entries, MemType::VmMem);
    assert!(matches!(result, Err(RwError::ReadMismatch { .. })));
  }

  #[test]
  fn sort_verify_zero_init_initial_read_must_be_zero() {
    // For VmMem: first access is read with value=0 → OK.
    let mut entries_ok = vec![
      RwEntry { addr: 10, value: 0, counter: 0, is_write: false },
    ];
    assert!(sort_and_verify_rw(&mut entries_ok, MemType::VmMem).is_ok());

    // For VmMem: first access is read with value≠0 → error.
    let mut entries_bad = vec![
      RwEntry { addr: 10, value: 5, counter: 0, is_write: false },
    ];
    assert!(matches!(
      sort_and_verify_rw(&mut entries_bad, MemType::VmMem),
      Err(RwError::InitialReadNonZero { .. })
    ));
  }

  #[test]
  fn sort_verify_storage_initial_read_any_value() {
    // For Storage: first access is read with non-zero → OK (external binding).
    let mut entries = vec![
      RwEntry { addr: 10, value: 999, counter: 0, is_write: false },
    ];
    assert!(sort_and_verify_rw(&mut entries, MemType::Storage).is_ok());
  }

  #[test]
  fn sort_verify_empty_entries() {
    let mut entries: Vec<RwEntry> = vec![];
    let summaries = sort_and_verify_rw(&mut entries, MemType::VmMem).unwrap();
    assert!(summaries.is_empty());
  }

  #[test]
  fn sort_verify_write_overwrite_read() {
    let mut entries = vec![
      RwEntry { addr: 10, value: 1, counter: 0, is_write: true },
      RwEntry { addr: 10, value: 2, counter: 1, is_write: true },
      RwEntry { addr: 10, value: 2, counter: 2, is_write: false },
    ];
    let summaries = sort_and_verify_rw(&mut entries, MemType::VmMem).unwrap();
    assert_eq!(summaries[0].final_value, 2);
  }

  // ── R/W log: encode_rw ─────────────────────────────────────────────

  #[test]
  fn encode_rw_distinct_entries() {
    let alpha = GF2_128::new(0x1234, 0x5678);
    let alpha2 = alpha * alpha;
    let a = RwEntry { addr: 1, value: 2, counter: 0, is_write: false };
    let b = RwEntry { addr: 1, value: 2, counter: 1, is_write: false };
    let c = RwEntry { addr: 1, value: 3, counter: 0, is_write: false };
    // Same addr+value but different counter → different encoding.
    assert_ne!(encode_rw(&a, alpha, alpha2), encode_rw(&b, alpha, alpha2));
    // Same addr+counter but different value → different encoding.
    assert_ne!(encode_rw(&a, alpha, alpha2), encode_rw(&c, alpha, alpha2));
  }

  // ── R/W log: prove/verify round trip ───────────────────────────────

  #[test]
  fn rw_prove_verify_vm_mem() {
    // Store to addr 5, then load from addr 5 — valid trace.
    let rows = vec![
      make_row(17, 5, 42, 0, 0, 0),  // Store addr=5, val=42
      make_row(16, 5, 0, 0, 42, 0),  // Load addr=5 → 42
    ];
    assert!(run_lookup(&rows));
  }

  #[test]
  fn rw_prove_verify_evm_memory() {
    let rows = vec![
      make_row(23, 0x20, 0xFF, 0, 0, 0), // MStore offset=0x20, val=0xFF
      make_row(22, 0x20, 0, 0, 0xFF, 0),  // MLoad offset=0x20 → 0xFF
    ];
    assert!(run_lookup(&rows));
  }

  #[test]
  fn rw_prove_verify_storage_roundtrip() {
    // SLoad key=1 val=100 (first read, initial state), then SStore key=1 val=200.
    let rows = vec![
      make_row(25, 1, 0, 0, 100, 0),  // SLoad key=1 → 100
      make_row(26, 1, 200, 0, 0, 0),  // SStore key=1 := 200
      make_row(25, 1, 0, 0, 200, 0),  // SLoad key=1 → 200
    ];
    assert!(run_lookup(&rows));
  }

  #[test]
  fn rw_prove_verify_transient() {
    let rows = vec![
      make_row(28, 3, 50, 0, 0, 0),  // TStore key=3 := 50
      make_row(27, 3, 0, 0, 50, 0),  // TLoad key=3 → 50
    ];
    assert!(run_lookup(&rows));
  }

  #[test]
  fn rw_prove_verify_par() {
    // Parallel path with mixed rw ops.
    let rows = vec![
      make_row(17, 1, 10, 0, 0, 0),  // VM Store
      make_row(16, 1, 0, 0, 10, 0),  // VM Load
      make_row(23, 0, 20, 0, 0, 0),  // MStore
      make_row(22, 0, 0, 0, 20, 0),  // MLoad
    ];
    assert!(run_lookup_par(&rows));
  }

  #[test]
  #[should_panic(expected = "R/W consistency check must pass")]
  fn rw_read_mismatch_panics_at_prove() {
    // Store 42 to addr 5, but load returns 99 — inconsistent.
    let rows = vec![
      make_row(17, 5, 42, 0, 0, 0),  // Store addr=5, val=42
      make_row(16, 5, 0, 0, 99, 0),  // Load addr=5 → 99 (WRONG)
    ];
    run_lookup(&rows);
  }

  #[test]
  fn rw_empty_log_prove_verify() {
    // No memory ops — should still pass.
    let rows = vec![
      make_row(0, 100, 200, 0, 300, 0), // Add128
    ];
    assert!(run_lookup(&rows));
  }

  #[test]
  fn rw_par_witnesses_match_sequential() {
    let rows = vec![
      make_row(17, 1, 10, 0, 0, 0),
      make_row(16, 1, 0, 0, 10, 0),
      make_row(23, 0, 20, 0, 0, 0),
      make_row(22, 0, 0, 0, 20, 0),
      make_row(26, 5, 30, 0, 0, 0),
      make_row(25, 5, 0, 0, 30, 0),
    ];
    let seq = collect_witnesses(&rows);
    let par = collect_witnesses_par(&rows);
    // R/W entries should have same data (possibly different counters for par).
    assert_eq!(seq.rw.mem.len(), par.rw.mem.len());
    assert_eq!(seq.rw.emem.len(), par.rw.emem.len());
    assert_eq!(seq.rw.storage.len(), par.rw.storage.len());
    assert_eq!(seq.rw.transient.len(), par.rw.transient.len());
  }

  // ── L-1 indirect coverage: compose_summaries bridge enforcement ────
  //
  // The R/W consistency argument enforces value continuity at Seq
  // boundaries via compose_summaries.  When two adjacent shards share
  // an address, left.final_value must equal right.initial_value.
  // These tests verify the bridge constraint catches mismatches.

  #[test]
  fn compose_bridge_match_merges_correctly() {
    let left = vec![RwSummary { addr: 10, initial_value: 0, final_value: 42 }];
    let right = vec![RwSummary { addr: 10, initial_value: 42, final_value: 99 }];
    let merged = compose_summaries(&left, &right).unwrap();
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].addr, 10);
    assert_eq!(merged[0].initial_value, 0);
    assert_eq!(merged[0].final_value, 99);
  }

  #[test]
  fn compose_bridge_mismatch_detected() {
    // left writes final_value=42, right expects initial_value=99 → mismatch.
    let left = vec![RwSummary { addr: 10, initial_value: 0, final_value: 42 }];
    let right = vec![RwSummary { addr: 10, initial_value: 99, final_value: 200 }];
    let result = compose_summaries(&left, &right);
    assert!(
      matches!(result, Err(RwError::BridgeMismatch { addr: 10, left_final: 42, right_initial: 99 })),
      "compose_summaries must reject Seq boundary value mismatch: {result:?}"
    );
  }

  #[test]
  fn compose_bridge_disjoint_addresses_preserved() {
    // Non-overlapping addresses: both are preserved without bridge check.
    let left = vec![RwSummary { addr: 5, initial_value: 1, final_value: 2 }];
    let right = vec![RwSummary { addr: 20, initial_value: 3, final_value: 4 }];
    let merged = compose_summaries(&left, &right).unwrap();
    assert_eq!(merged.len(), 2);
    assert_eq!(merged[0].addr, 5);
    assert_eq!(merged[1].addr, 20);
  }

  #[test]
  fn compose_bridge_multi_addr_partial_overlap() {
    // Two shared addrs + one unique per side.
    let left = vec![
      RwSummary { addr: 1, initial_value: 0, final_value: 10 },
      RwSummary { addr: 3, initial_value: 0, final_value: 30 },
      RwSummary { addr: 5, initial_value: 0, final_value: 50 },
    ];
    let right = vec![
      RwSummary { addr: 2, initial_value: 0, final_value: 20 },
      RwSummary { addr: 3, initial_value: 30, final_value: 33 },
      RwSummary { addr: 5, initial_value: 50, final_value: 55 },
    ];
    let merged = compose_summaries(&left, &right).unwrap();
    assert_eq!(merged.len(), 4); // addrs 1, 2, 3, 5
    // addr 3: initial=0, final=33 (bridged through 30)
    let a3 = merged.iter().find(|s| s.addr == 3).unwrap();
    assert_eq!(a3.initial_value, 0);
    assert_eq!(a3.final_value, 33);
  }

  #[test]
  fn compose_rw_summaries_bridge_mismatch_any_type() {
    // Bridge mismatch in storage type → compose_rw_summaries fails.
    let left = RwSummaries {
      mem: vec![],
      emem: vec![],
      storage: vec![RwSummary { addr: 1, initial_value: 0, final_value: 100 }],
      transient: vec![],
    };
    let right = RwSummaries {
      mem: vec![],
      emem: vec![],
      storage: vec![RwSummary { addr: 1, initial_value: 999, final_value: 200 }],
      transient: vec![],
    };
    assert!(compose_rw_summaries(&left, &right).is_err());
  }

  // ── L-1 indirect coverage: storage key preservation ────────────────

  #[test]
  fn rw_storage_write_then_read_preserves_key() {
    // Write to addr 10, then read from addr 10 — key survives.
    let mut entries = vec![
      RwEntry { addr: 10, value: 42, counter: 0, is_write: true },
      RwEntry { addr: 10, value: 42, counter: 1, is_write: false },
    ];
    let summaries = sort_and_verify_rw(&mut entries, MemType::VmMem).unwrap();
    assert_eq!(summaries.len(), 1);
    assert_eq!(summaries[0].final_value, 42);
  }

  #[test]
  fn rw_multiple_keys_all_preserved() {
    // Multiple keys: all survive across the sorted log.
    let mut entries = vec![
      RwEntry { addr: 1, value: 10, counter: 0, is_write: true },
      RwEntry { addr: 2, value: 20, counter: 1, is_write: true },
      RwEntry { addr: 3, value: 30, counter: 2, is_write: true },
      RwEntry { addr: 1, value: 10, counter: 3, is_write: false },
      RwEntry { addr: 2, value: 20, counter: 4, is_write: false },
      RwEntry { addr: 3, value: 30, counter: 5, is_write: false },
    ];
    let summaries = sort_and_verify_rw(&mut entries, MemType::VmMem).unwrap();
    assert_eq!(summaries.len(), 3);
  }
}
