//! CPU prover profiling binary.
//!
//! Usage:
//!   cargo flamegraph -p bench --bin profile_cpu -- [N_STEPS]
//!
//! Default: 64 steps.

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::hint::black_box;

use e2e::{EvmStep, build_witness, compress, prove_cpu_par, verify, verify_compressed};
use revm::primitives::{Address, B256, U256};

/// Enable large (2 MiB) OS pages for all mimalloc allocations.
fn enable_large_pages() {
  unsafe {
    libmimalloc_sys::mi_option_set_enabled(libmimalloc_sys::mi_option_large_os_pages, true);
  }
}

/// Create a mixed EVM trace of `n` steps — **O(n) time and memory**.
///
/// Alternates PUSH0 and a binary op (ADD/SUB/AND cycle) so that the
/// stack depth stays constant at 2, satisfying the type checker's
/// Seq-continuity rule (`left.post_depth == right.pre_depth`).
///
/// ```text
///   depth 2 → PUSH0 → depth 3 → ADD → depth 2 → PUSH0 → depth 3 → SUB → …
/// ```
///
/// Each step stores only the operands it touches (≤3 elements), so
/// total memory is O(n) regardless of trace length.
fn mixed_trace(n: usize) -> Vec<EvmStep> {
  use revm::bytecode::opcode::{ADD, AND, PUSH32, SUB};
  let bin_ops = [ADD, SUB, AND];
  let mut steps = Vec::with_capacity(n);

  // Persistent "top of stack" values; depth stays at 2.
  let mut top0 = U256::from(100u64);
  let top1 = U256::from(101u64);
  let mut bin_idx = 0usize;
  let mut last_gas = 1_000_000_000u64;
  let mut pc = 0u32;

  for i in 0..n {
    if i % 2 == 0 {
      // ── PUSH32: depth 2 → 3 ────────────────────────────────────
      let next_pc = pc + 1 + 32;
      let next_op = if i + 1 < n { bin_ops[bin_idx % 3] } else { PUSH32 };
      steps.push(EvmStep {
        pre_opcode: PUSH32,
        post_opcode: next_op,
        pre_pc: pc,
        post_pc: next_pc,
        gas_before: last_gas,
        gas_after: last_gas - 3,
        pre_stack: vec![top0, top1],
        post_stack: vec![U256::from_le_bytes([0xAA; 32]), top0, top1],
        pre_push_data: Some(vec![0xAA; 32]),
        post_push_data: None,
        call_depth: 0,
        address: Address::ZERO,
        caller: Address::ZERO,
        balance: U256::ZERO,
        nonce: 0,
        code_hash: B256::ZERO,
      });
      last_gas -= 3;
      pc = next_pc;
    } else {
      // ── Binary op: depth 3 → 2 ────────────────────────────────
      // Operands are the top two: [U256::ZERO, top0] from previous PUSH32.
      let op = bin_ops[bin_idx % 3];
      bin_idx += 1;
      let a = U256::from_le_bytes([0xAA; 32]);
      let b = top0;
      let result = match op {
        ADD => a.overflowing_add(b).0,
        SUB => a.overflowing_sub(b).0,
        AND => a & b,
        _ => unreachable!(),
      };
      let next_pc = pc + 1;
      steps.push(EvmStep {
        pre_opcode: op,
        post_opcode: PUSH32,
        pre_pc: pc,
        post_pc: next_pc,
        gas_before: last_gas,
        gas_after: last_gas - 3,
        pre_stack: vec![a, b, top1],
        post_stack: vec![result, top1],
        pre_push_data: None,
        post_push_data: Some(vec![0xAA; 32]),
        call_depth: 0,
        address: Address::ZERO,
        caller: Address::ZERO,
        balance: U256::ZERO,
        nonce: 0,
        code_hash: B256::ZERO,
      });
      // Update running state.
      top0 = result;
      last_gas -= 3;
      pc = next_pc;
    }
  }
  steps
}

/// Create a heavy-advice EVM trace of `n` steps.
///
/// Alternates PUSH0 → MUL (or DIV/EXP) so that every other step uses advice.
/// Stack depth oscillates: 2 → PUSH0 → 3 → MUL → 2.
fn advice_trace(n: usize) -> Vec<EvmStep> {
  use revm::bytecode::opcode::{EXP, PUSH0};
  let mut steps = Vec::with_capacity(n);

  let mut s0 = U256::from(7u64);
  let s1 = U256::from(3u64);
  let mut last_gas = 1_000_000_000u64;

  for i in 0..n {
    if i % 2 == 0 {
      let cost = 2u64;
      let next_op = if i + 1 < n { EXP } else { PUSH0 };
      steps.push(EvmStep {
        pre_opcode: PUSH0,
        post_opcode: next_op,
        pre_pc: i as u32,
        post_pc: i as u32 + 1,
        gas_before: last_gas,
        gas_after: last_gas - cost,
        pre_stack: vec![s0, s1],
        post_stack: vec![U256::ZERO, s0, s1],
        pre_push_data: None,
        post_push_data: None,
        call_depth: 0,
        address: Address::ZERO,
        caller: Address::ZERO,
        balance: U256::ZERO,
        nonce: 0,
        code_hash: B256::ZERO,
      });
      last_gas -= cost;
    } else {
      let a = U256::ZERO;
      let b = s0;
      let result = a.overflowing_mul(b).0;
      let cost = 5u64;
      let next_op = if i + 1 < n { PUSH0 } else { EXP };
      steps.push(EvmStep {
        pre_opcode: EXP,
        post_opcode: next_op,
        pre_pc: i as u32,
        post_pc: i as u32 + 1,
        gas_before: last_gas,
        gas_after: last_gas - cost,
        pre_stack: vec![a, b, s1],
        post_stack: vec![result, s1],
        pre_push_data: None,
        post_push_data: None,
        call_depth: 0,
        address: Address::ZERO,
        caller: Address::ZERO,
        balance: U256::ZERO,
        nonce: 0,
        code_hash: B256::ZERO,
      });
      s0 = result;
      last_gas -= cost;
    }
  }
  steps
}

fn bench_trace(label: &str, steps: &[EvmStep]) {
  let witness = build_witness(steps, None).unwrap();
  eprintln!(
    "[{}] {} steps → {} rows",
    label,
    steps.len(),
    witness.rows.len()
  );

  let t0 = std::time::Instant::now();
  let (proof, params) = black_box(prove_cpu_par(black_box(&witness))).unwrap();
  let t1 = std::time::Instant::now();

  let res = verify(&proof, &params);
  let t2 = std::time::Instant::now();

  let proof_vec = bincode::encode_to_vec(&proof, bincode::config::standard()).unwrap();
  eprintln!("[{}] proof size: {:.2} KiB", label, proof_vec.len() as f64 / 1024.0);

  // Recursive compression
  let t3 = std::time::Instant::now();
  let compressed = compress(&proof, &params).unwrap();
  let t4 = std::time::Instant::now();
  let comp_vec = bincode::encode_to_vec(&compressed, bincode::config::standard()).unwrap();
  eprintln!("[{}] compressed size: {:.2} KiB ({:.1}x reduction)",
    label, comp_vec.len() as f64 / 1024.0,
    proof_vec.len() as f64 / comp_vec.len() as f64);

  let t5 = std::time::Instant::now();
  let comp_res = verify_compressed(&compressed);
  let t6 = std::time::Instant::now();

  eprintln!("[{}] parallel {:.2?}", label, t1 - t0);
  eprintln!("[{}] compress {:.2?}", label, t4 - t3);
  if let Err(err) = res {
    eprintln!("[{}] verification FAILED: {:?}", label, err);
  } else {
    eprintln!("[{}] verification ok in {:.2?}", label, t3 - t2);
  }
  if let Err(err) = comp_res {
    eprintln!("[{}] compressed verification FAILED: {:?}", label, err);
  } else {
    eprintln!("[{}] compressed verification ok in {:.2?}", label, t6 - t5);
  }
}

fn main() {
  enable_large_pages();

  let n: usize = std::env::args()
    .nth(1)
    .and_then(|s| s.parse().ok())
    .unwrap_or(1 << 18);

  eprintln!("=== Mixed trace (ALU-only) ===");
  bench_trace("mixed", &mixed_trace(n));

  eprintln!();
  eprintln!("=== Advice trace (MUL/DIV/EXP) ===");
  bench_trace("advice", &advice_trace(n));

  eprintln!("\nDone.");
}
