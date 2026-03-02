//! CPU prover profiling binary.
//!
//! Usage:
//!   cargo flamegraph -p bench --bin profile_cpu -- [N_STEPS]
//!
//! Default: 64 steps.

use e2e::{EvmStep, build_witness, prove_cpu};
use revm::primitives::U256;

fn mixed_trace(n: usize) -> Vec<EvmStep> {
    use revm::bytecode::opcode::{ADD, AND, SUB};
    let opcodes = [ADD, SUB, AND];
    let mut steps = Vec::with_capacity(n);

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

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);

    eprintln!("Building witness for {} steps...", n);
    let steps = mixed_trace(n);
    let witness = build_witness(&steps).unwrap();

    eprintln!("Running prove_cpu ({} steps, {} rows)...", n, witness.rows.len());

    // Run multiple iterations for better sampling
    for i in 0..1000 {
        let (proof, _params) = prove_cpu(&witness).unwrap();
        // Prevent optimizing away
        std::hint::black_box(&proof);
        if i == 0 {
            eprintln!("  first proof done");
        }
    }

    eprintln!("Done.");
}
