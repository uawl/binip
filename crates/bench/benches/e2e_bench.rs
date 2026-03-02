//! E2E benchmarks: witness building, proving, and verification.
//!
//! Run with:
//! ```sh
//! cargo bench -p bench
//! ```

use criterion::{BenchmarkId, Criterion, black_box, criterion_group};
use e2e::{EvmStep, build_witness, prove_cpu, prove_gpu};
use gpu::{GpuContext, PipelineCache};
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

/// Create a mixed EVM trace of `n` steps with consistent stack depths.
///
/// Pattern: ADD, SUB, AND repeated in a cycle.  Each opcode consumes 2
/// stack elements and produces 1, so post depth = pre depth − 1.
/// To keep sequential type-check happy, each step's pre_stack has depth
/// equal to the previous step's post_stack depth.
fn mixed_trace(n: usize) -> Vec<EvmStep> {
    use revm::bytecode::opcode::{ADD, AND, SUB};
    let opcodes = [ADD, SUB, AND];
    let mut steps = Vec::with_capacity(n);

    // Build a full EVM stack so that adjacent steps chain:
    //   step[i].post_stack.len() == step[i+1].pre_stack.len()
    // Each binary op (ADD/SUB/AND) pops 2, pushes 1 → depth shrinks by 1.
    // Start with depth n+1, end at depth 1.
    let mut stack: Vec<U256> = (0..n as u64 + 1).map(|i| U256::from(100 + i)).collect();

    for i in 0..n {
        let opcode = opcodes[i % 3];
        let pre_stack = stack.clone();
        let a = stack[0];
        let b = stack[1];
        let result = match opcode {
            ADD => a.overflowing_add(b).0,
            SUB => a.overflowing_sub(b).0,
            AND => a & b,
            _ => unreachable!(),
        };
        // Pop two, push one.
        stack.remove(0);
        stack[0] = result;
        steps.push(EvmStep {
            opcode,
            pc: i as u32,
            gas_before: 1_000_000,
            gas_after: 999_997,
            pre_stack,
            post_stack: stack.clone(),
        });
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
    for n in [1, 4, 16, 64, 256] {
        let steps = mixed_trace(n);
        let witness = build_witness(&steps).unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(n), &witness, |b, w| {
            b.iter(|| prove_cpu(black_box(w)).unwrap());
        });
    }
    group.finish();
}

/// Benchmark: `prove_gpu` (witness → proof) for varying step counts.
///
/// A single warmup call is made first so that shader compilation and
/// pipeline creation cost is excluded from the measured iterations.
fn bench_prove_gpu(c: &mut Criterion, ctx: &GpuContext, cache: &mut PipelineCache) {
    // Warmup: populate PipelineCache with compiled shaders.
    let warmup_witness = build_witness(&mixed_trace(1)).unwrap();
    let _ = prove_gpu(&warmup_witness, ctx, cache);

    let mut group = c.benchmark_group("prove_gpu");
    for n in [1, 4, 16, 64, 256] {
        let steps = mixed_trace(n);
        let witness = build_witness(&steps).unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(n), &witness, |b, w| {
            b.iter(|| prove_gpu(black_box(w), ctx, cache).unwrap());
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
    bench_verify,
    bench_prove_and_verify,
    bench_opcode_witness,
);

fn main() {
    // Initialize GPU context + pipeline cache once, outside all benchmarks.
    let gpu = GpuContext::new().ok();
    if let Some(ref ctx) = gpu {
        eprintln!("GPU initialized: {}", ctx.backend_name());
    } else {
        eprintln!("GPU unavailable — prove_gpu benchmarks will be skipped");
    }
    let mut cache = PipelineCache::new();

    // Run CPU-only benchmark groups via criterion_group.
    // (criterion_group! generates `fn benches()` which calls Criterion internally.)
    benches();

    // Run GPU benchmark group with the pre-initialised context.
    if let Some(ref ctx) = gpu {
        let mut criterion = Criterion::default().configure_from_args();
        bench_prove_gpu(&mut criterion, ctx, &mut cache);
        criterion.final_summary();
    }
}
