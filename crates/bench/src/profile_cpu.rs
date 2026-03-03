//! CPU prover profiling binary.
//!
//! Usage:
//!   cargo flamegraph -p bench --bin profile_cpu -- [N_STEPS]
//!
//! Default: 64 steps.

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::hint::black_box;

use e2e::{EvmStep, build_witness, prove_cpu_par, verify};
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
  let mut top1 = U256::from(101u64);
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
      top1 = top1.overflowing_add(U256::from(1u64)).0;
      last_gas -= 3;
    }
  }
  steps
}

fn main() {
  enable_large_pages();

  let n: usize = std::env::args()
    .nth(1)
    .and_then(|s| s.parse().ok())
    .unwrap_or(1 << 18);

  eprintln!("Building witness for {} steps...", n);
  let steps = mixed_trace(n);
  let witness = build_witness(&steps).unwrap();

  eprintln!(
    "Running prove_cpu ({} steps, {} rows)...",
    n,
    witness.rows.len()
  );

  let t0 = std::time::Instant::now();

  let (proof, params) = black_box(prove_cpu_par(black_box(&witness))).unwrap();

  let t1 = std::time::Instant::now();

  let res = verify(&proof, &witness.tree, &params);

  let t2 = std::time::Instant::now();

  eprintln!("Prover took {:.2?}", t1 - t0);

  if let Err(_err) = res {
    eprintln!("Verification failed in {:.2?}", t2 - t1);
  } else {
    eprintln!("Verification succeeded in {:.2?}", t2 - t1);
  }

  eprintln!("Done.");
}
