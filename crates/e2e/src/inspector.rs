//! Tracing EVM inspector — captures opcode-level execution steps.

use revm::interpreter::{Interpreter, interpreter::EthInterpreter};
use revm::interpreter::interpreter_types::Jumps;
use revm::primitives::U256;

/// One EVM opcode execution step captured by the [`TracingInspector`].
#[derive(Debug, Clone)]
pub struct EvmStep {
    /// EVM opcode byte (e.g. `0x01` = ADD).
    pub opcode: u8,
    /// Program counter before execution.
    pub pc: u32,
    /// Gas remaining before execution.
    pub gas_before: u64,
    /// Gas remaining after execution.
    pub gas_after: u64,
    /// Stack snapshot **before** executing this opcode (TOS = index 0).
    pub pre_stack: Vec<U256>,
    /// Stack snapshot **after** executing this opcode (TOS = index 0).
    pub post_stack: Vec<U256>,
}

/// Pending step being captured — pre-state recorded in `step`, post-state
/// completed in `step_end`.
#[derive(Debug, Clone)]
struct Pending {
    opcode: u8,
    pc: u32,
    gas_before: u64,
    pre_stack: Vec<U256>,
}

/// A revm [`Inspector`](revm::Inspector) that records every EVM step.
///
/// Use with `build_mainnet_with_inspector` to capture a full execution trace.
#[derive(Debug, Default)]
pub struct TracingInspector {
    pending: Option<Pending>,
    /// Completed execution steps.
    pub steps: Vec<EvmStep>,
}

impl TracingInspector {
    pub fn new() -> Self {
        Self { pending: None, steps: Vec::new() }
    }
}

/// Helper: read the interpreter stack as a `Vec<U256>` (TOS = index 0).
fn read_stack(interp: &Interpreter<EthInterpreter>) -> Vec<U256> {
    let data = interp.stack.data();
    // revm stores stack with TOS at the highest index; we reverse for TOS=0
    data.iter().rev().copied().collect()
}

impl<CTX> revm::Inspector<CTX> for TracingInspector {
    fn step(&mut self, interp: &mut Interpreter<EthInterpreter>, _context: &mut CTX) {
        let opcode = interp.bytecode.opcode();
        let pc = interp.bytecode.pc() as u32;
        let gas_before = interp.gas.remaining();
        let pre_stack = read_stack(interp);
        self.pending = Some(Pending { opcode, pc, gas_before, pre_stack });
    }

    fn step_end(&mut self, interp: &mut Interpreter<EthInterpreter>, _context: &mut CTX) {
        if let Some(pending) = self.pending.take() {
            let gas_after = interp.gas.remaining();
            let post_stack = read_stack(interp);
            self.steps.push(EvmStep {
                opcode: pending.opcode,
                pc: pending.pc,
                gas_before: pending.gas_before,
                gas_after,
                pre_stack: pending.pre_stack,
                post_stack,
            });
        }
    }
}
