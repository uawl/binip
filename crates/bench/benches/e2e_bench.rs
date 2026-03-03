//! E2E benchmarks: witness building, proving, and verification.
//!
//! Run with:
//! ```sh
//! cargo bench -p bench
//! ```

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use e2e::{EvmStep, build_witness, prove_cpu, prove_cpu_par};
use revm::primitives::U256;

// ─────────────────────────────────────────────────────────────────────────────
// Step generators
// ─────────────────────────────────────────────────────────────────────────────

/// Create a synthetic ADD step.
fn add_step(pc: u32) -> EvmStep {
  use revm::bytecode::opcode::ADD;
  EvmStep {
    opcode: ADD,
    pc,
    gas_before: 1_000_000,
    gas_after: 999_997,
    pre_stack: vec![U256::from(pc as u64 + 1), U256::from(pc as u64 + 2)],
    post_stack: vec![U256::from(pc as u64 * 2 + 3)],
  }
}

/// Create a synthetic DIV step (advice-heavy: 16 limbs).
fn div_step(pc: u32) -> EvmStep {
  use revm::bytecode::opcode::DIV;
  let a = U256::from(1000u64 + pc as u64);
  let b = U256::from(7u64);
  EvmStep {
    opcode: DIV,
    pc,
    gas_before: 1_000_000,
    gas_after: 999_995,
    pre_stack: vec![a, b],
    post_stack: vec![a / b],
  }
}

/// Create a synthetic LT step (boolean advice: 1 limb).
fn lt_step(pc: u32) -> EvmStep {
  use revm::bytecode::opcode::LT;
  let a = U256::from(pc as u64);
  let b = U256::from(pc as u64 + 10);
  EvmStep {
    opcode: LT,
    pc,
    gas_before: 1_000_000,
    gas_after: 999_997,
    pre_stack: vec![a, b],
    post_stack: vec![U256::from(1u64)],
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

  for i in 0..n {
    if i % 2 == 0 {
      // ── PUSH0: depth 2 → 3 ────────────────────────────────────
      steps.push(EvmStep {
        opcode: PUSH0,
        pc: i as u32,
        gas_before: 1_000_000,
        gas_after: 999_998,
        pre_stack: vec![top0, top1],
        post_stack: vec![U256::ZERO, top0, top1],
      });
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
        gas_before: 1_000_000,
        gas_after: 999_997,
        pre_stack: vec![a, b, top1],
        post_stack: vec![result, top1],
      });
      // Update running state.
      top0 = result;
      top1 = top1.overflowing_add(U256::from(1u64)).0;
    }
  }
  steps
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmarks
// ─────────────────────────────────────────────────────────────────────────────

/// Benchmark: `build_witness` for varying step counts.
fn bench_build_witness(c: &mut Criterion) {
  let mut group = c.benchmark_group("build_witness");
  for n in [1, 4, 16, 64] {
    let steps = mixed_trace(n);
    group.bench_with_input(BenchmarkId::from_parameter(n), &steps, |b, steps| {
      b.iter(|| build_witness(black_box(steps)).unwrap());
    });
  }
  group.finish();
}

/// Benchmark: `prove_cpu` (witness → proof) for varying step counts.
fn bench_prove(c: &mut Criterion) {
  let mut group = c.benchmark_group("prove_cpu");
  for n in [1, 4, 16, 64, 256, 1024] {
    let steps = mixed_trace(n);
    let witness = build_witness(&steps).unwrap();
    group.bench_with_input(BenchmarkId::from_parameter(n), &witness, |b, w| {
      b.iter(|| prove_cpu(black_box(w)).unwrap());
    });
  }
  group.finish();
}

/// Benchmark: `prove_cpu_par` — shard-level parallelism within a single proof.
///
/// Unlike `bench_prove_par` which runs multiple independent proofs in parallel,
/// this measures a single call to `prove_cpu_par` which internally parallelises
/// shard proving via rayon.
fn bench_prove_shard_par(c: &mut Criterion) {
  let mut group = c.benchmark_group("prove_cpu_par");
  let powers_of_4 = (0..=9).map(|i| 1 << (2 * i));
  for n in powers_of_4 {
    let steps = mixed_trace(n);
    let witness = build_witness(&steps).unwrap();
    group.bench_with_input(BenchmarkId::from_parameter(n), &witness, |b, w| {
      b.iter(|| prove_cpu_par(black_box(w)).unwrap());
    });
  }
  group.finish();
}

/// Benchmark: `verify` for varying step counts.
///
/// Uses the proof generated by `prove` which may have a non-zero constraint
/// sum (synthetic witnesses don't satisfy the full circuit).  We measure the
/// computational cost of the verify pipeline by tolerating `ConstraintSumNonZero`
/// and only asserting the "happy path" parts succeed.
fn bench_verify(c: &mut Criterion) {
  let mut group = c.benchmark_group("verify");
  for n in [1, 4, 16, 64] {
    let steps = mixed_trace(n);
    let witness = build_witness(&steps).unwrap();
    let (proof, params) = prove_cpu(&witness).unwrap();
    group.bench_with_input(
      BenchmarkId::from_parameter(n),
      &(proof, witness.tree, params),
      |b, (proof, tree, params)| {
        b.iter(|| {
          let _ = e2e::verify(black_box(proof), black_box(tree), black_box(params));
        });
      },
    );
  }
  group.finish();
}

/// Benchmark: full `prove_and_verify` pipeline.
///
/// Same caveat as `bench_verify` — synthetic witnesses may produce a
/// non-zero constraint sum, so we tolerate verify failures here and
/// measure total throughput.
fn bench_prove_and_verify(c: &mut Criterion) {
  let mut group = c.benchmark_group("prove_and_verify");
  for n in [1, 4, 16, 64] {
    let steps = mixed_trace(n);
    group.bench_with_input(BenchmarkId::from_parameter(n), &steps, |b, steps| {
      b.iter(|| {
        let witness = build_witness(black_box(steps)).unwrap();
        let (proof, params) = prove_cpu(&witness).unwrap();
        let _ = e2e::verify(&proof, &witness.tree, &params);
      });
    });
  }
  group.finish();
}

/// Benchmark: opcode-specific witness building.
fn bench_opcode_witness(c: &mut Criterion) {
  let mut group = c.benchmark_group("opcode_witness");

  // ADD (no advice)
  let add_steps: Vec<_> = (0..16).map(add_step).collect();
  group.bench_function("ADD_x16", |b| {
    b.iter(|| build_witness(black_box(&add_steps)).unwrap());
  });

  // DIV (16-limb advice)
  let div_steps: Vec<_> = (0..16).map(div_step).collect();
  group.bench_function("DIV_x16", |b| {
    b.iter(|| build_witness(black_box(&div_steps)).unwrap());
  });

  // LT (1-limb advice)
  let lt_steps: Vec<_> = (0..16).map(lt_step).collect();
  group.bench_function("LT_x16", |b| {
    b.iter(|| build_witness(black_box(&lt_steps)).unwrap());
  });

  group.finish();
}

criterion_group!(
  benches,
  bench_build_witness,
  bench_prove,
  bench_prove_shard_par,
  bench_verify,
  bench_prove_and_verify,
  bench_opcode_witness,
);

criterion_main!(benches);
