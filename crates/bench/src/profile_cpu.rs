//! CPU prover profiling binary.
//!
//! Usage:
//!   cargo flamegraph -p bench --bin profile_cpu -- [N_STEPS]
//!
//! Default: 64 steps.

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::hint::black_box;

use e2e::{EvmStep, build_witness, prove_cpu, prove_cpu_par, verify};
use revm::primitives::U256;

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
  use revm::bytecode::opcode::{ADD, AND, PUSH0, SUB};
  let bin_ops = [ADD, SUB, AND];
  let mut steps = Vec::with_capacity(n);

  // Persistent "top of stack" values; depth stays at 2.
  let mut top0 = U256::from(100u64);
  let top1 = U256::from(101u64);
  let mut bin_idx = 0usize;
  let mut last_gas = 1_000_000_000u64;

  for i in 0..n {
    if i % 2 == 0 {
      // ── PUSH0: depth 2 → 3 ────────────────────────────────────
      steps.push(EvmStep {
        opcode: PUSH0,
        pc: i as u32,
        gas_before: last_gas,
        gas_after: last_gas - 2,
        pre_stack: vec![top0, top1],
        post_stack: vec![U256::ZERO, top0, top1],
      });
      last_gas -= 2;
    } else {
      // ── Binary op: depth 3 → 2 ────────────────────────────────
      // Operands are the top two: [U256::ZERO, top0] from previous PUSH0.
      let op = bin_ops[bin_idx % 3];
      bin_idx += 1;
      let a = U256::ZERO;
      let b = top0;
      let result = match op {
        ADD => a.overflowing_add(b).0,
        SUB => a.overflowing_sub(b).0,
        AND => a & b,
        _ => unreachable!(),
      };
      steps.push(EvmStep {
        opcode: op,
        pc: i as u32,
        gas_before: last_gas,
        gas_after: last_gas - 3,
        pre_stack: vec![a, b, top1],
        post_stack: vec![result, top1],
      });
      // Update running state.
      top0 = result;
      last_gas -= 3;
    }
  }
  steps
}

/// Create a heavy-advice EVM trace of `n` steps.
///
/// Alternates PUSH0 → MUL (or DIV/EXP) so that every other step uses advice.
/// Stack depth oscillates: 2 → PUSH0 → 3 → MUL → 2.
fn advice_trace(n: usize) -> Vec<EvmStep> {
  use revm::bytecode::opcode::{MUL, PUSH0};
  let mut steps = Vec::with_capacity(n);

  // Stack: depth 2 → PUSH0 → depth 3 → MUL → depth 2
  // MUL(0, x) = 0, so after first cycle all MULs are MUL(0,0) = 0.
  // This is fine for profiling — exercises Advice2 + CheckMul path.
  let mut s0 = U256::from(7u64);
  let s1 = U256::from(3u64);
  let mut last_gas = 1_000_000_000u64;

  for i in 0..n {
    if i % 2 == 0 {
      let cost = 2u64;
      steps.push(EvmStep {
        opcode: PUSH0,
        pc: i as u32,
        gas_before: last_gas,
        gas_after: last_gas - cost,
        pre_stack: vec![s0, s1],
        post_stack: vec![U256::ZERO, s0, s1],
      });
      last_gas -= cost;
    } else {
      let a = U256::ZERO;
      let b = s0;
      let result = a.overflowing_mul(b).0;
      let cost = 5u64;
      steps.push(EvmStep {
        opcode: MUL,
        pc: i as u32,
        gas_before: last_gas,
        gas_after: last_gas - cost,
        pre_stack: vec![a, b, s1],
        post_stack: vec![result, s1],
      });
      s0 = result;
      last_gas -= cost;
    }
  }
  steps
}

fn bench_trace(label: &str, steps: &[EvmStep]) {
  let witness = build_witness(steps).unwrap();
  eprintln!(
    "[{}] {} steps → {} rows",
    label,
    steps.len(),
    witness.rows.len()
  );

  let t0 = std::time::Instant::now();
  let _ = black_box(prove_cpu_par(black_box(&witness))).unwrap();
  let t1 = std::time::Instant::now();

  let (proof, params) = black_box(prove_cpu(black_box(&witness))).unwrap();
  let t2 = std::time::Instant::now();

  let res = verify(&proof, &params);
  let t3 = std::time::Instant::now();

  eprintln!("[{}] parallel {:.2?}  seq {:.2?}", label, t1 - t0, t2 - t1);
  if let Err(err) = res {
    eprintln!("[{}] verification FAILED: {:?}", label, err);
  } else {
    eprintln!("[{}] verification ok in {:.2?}", label, t3 - t2);
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
