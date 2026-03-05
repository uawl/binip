//! Tracing EVM inspector — captures opcode-level execution steps.

use revm::interpreter::interpreter_types::{Immediates, Jumps};
use revm::interpreter::{CallInputs, CallOutcome, CreateInputs, CreateOutcome, Interpreter, interpreter::EthInterpreter};
use revm::primitives::{Address, B256, U256};

/// One EVM opcode execution step captured by the [`TracingInspector`].
///
/// Each step carries **pre** and **post** snapshots of opcode, PC, and push
/// data so that recursive proof boundaries only need to check
/// `left.post_* == right.pre_*`.
#[derive(Debug, Clone)]
pub struct EvmStep {
  /// EVM opcode byte **of this step** (e.g. `0x01` = ADD).
  pub pre_opcode: u8,
  /// EVM opcode byte that the **next step** will execute.
  /// For the last step (STOP / RETURN / …) this equals `pre_opcode`.
  pub post_opcode: u8,
  /// Program counter **before** executing this opcode.
  pub pre_pc: u32,
  /// Program counter **after** executing this opcode (= next step's PC).
  pub post_pc: u32,
  /// Gas remaining before execution.
  pub gas_before: u64,
  /// Gas remaining after execution.
  pub gas_after: u64,
  /// Stack snapshot **before** executing this opcode (TOS = index 0).
  pub pre_stack: Vec<U256>,
  /// Stack snapshot **after** executing this opcode (TOS = index 0).
  pub post_stack: Vec<U256>,
  /// Raw PUSH data bytes for this step (PUSH1..PUSH32).
  /// `None` for all other opcodes (including PUSH0).
  pub pre_push_data: Option<Vec<u8>>,
  /// Raw PUSH data bytes for the **next** step.
  /// `None` when the next opcode is not a PUSH.
  pub post_push_data: Option<Vec<u8>>,
  /// Current call depth (0 = top-level call).
  pub call_depth: u32,
  /// Address of the contract whose bytecode is executing.
  pub address: Address,
  /// Caller of the current execution context.
  pub caller: Address,
  /// Account balance of the executing contract (in wei).
  pub balance: U256,
  /// Account nonce of the executing contract.
  pub nonce: u64,
  /// Code hash of the executing contract's bytecode.
  pub code_hash: B256,
}

/// Pending step being captured — pre-state recorded in `step`, post-state
/// completed in `step_end`.
#[derive(Debug, Clone)]
struct Pending {
  opcode: u8,
  pc: u32,
  gas_before: u64,
  pre_stack: Vec<U256>,
  push_data: Option<Vec<u8>>,
  call_depth: u32,
  address: Address,
  caller: Address,
  balance: U256,
  nonce: u64,
  code_hash: B256,
}

/// A call/create frame boundary recorded by the inspector.
#[derive(Debug, Clone)]
pub struct CallFrame {
  /// Call depth of this frame (0 = top-level).
  pub depth: u32,
  /// Address of the contract being called/created.
  pub address: Address,
  /// The caller that initiated this frame.
  pub caller: Address,
  /// Whether this is a CREATE/CREATE2 (vs CALL/DELEGATECALL/STATICCALL).
  pub is_create: bool,
  /// Whether the call succeeded.
  pub success: bool,
  /// Account balance at frame entry.
  pub balance: U256,
  /// Account nonce at frame entry.
  pub nonce: u64,
  /// Code hash of the contract bytecode.
  pub code_hash: B256,
}

/// Compute the post-PC for a given opcode/pc.
///
/// PUSH1..PUSH32 advance PC by `1 + data_len`; all others advance by 1.
/// Terminal opcodes (STOP, RETURN, …) keep PC unchanged.
fn advance_pc(opcode: u8, pc: u32) -> u32 {
  match opcode {
    0x00 | 0xf3 | 0xfd | 0xff | 0xfe => pc, // STOP/RETURN/REVERT/SELFDESTRUCT/INVALID
    0x60..=0x7f => pc + 1 + (opcode as u32 - 0x5f), // PUSH1..PUSH32
    _ => pc + 1,
  }
}

/// Read PUSH data for the given opcode from an interpreter, if applicable.
fn read_push_data(interp: &Interpreter<EthInterpreter>, opcode: u8) -> Option<Vec<u8>> {
  if (0x60..=0x7f).contains(&opcode) {
    let data_len = (opcode - 0x5f) as usize;
    let slice = interp.bytecode.read_slice(1 + data_len);
    Some(slice[1..].to_vec())
  } else {
    None
  }
}

/// A revm [`Inspector`](revm::Inspector) that records every EVM step.
///
/// Use with `build_mainnet_with_inspector` to capture a full execution trace.
#[derive(Debug, Default)]
pub struct TracingInspector {
  pending: Option<Pending>,
  /// Completed execution steps.
  pub steps: Vec<EvmStep>,
  /// Stack of active call frames (depth 0 = outermost).
  frame_stack: Vec<CallFrame>,
  /// Completed call frame boundaries (in order of completion).
  pub call_frames: Vec<CallFrame>,
  /// Current call depth tracker.
  current_depth: u32,
  /// Current executing contract address.
  current_address: Address,
  /// Current caller address.
  current_caller: Address,
  /// Current account balance.
  current_balance: U256,
  /// Current account nonce.
  current_nonce: u64,
  /// Current code hash.
  current_code_hash: B256,
}

impl TracingInspector {
  pub fn new() -> Self {
    Self {
      pending: None,
      steps: Vec::new(),
      frame_stack: Vec::new(),
      call_frames: Vec::new(),
      current_depth: 0,
      current_address: Address::ZERO,
      current_caller: Address::ZERO,
      current_balance: U256::ZERO,
      current_nonce: 0,
      current_code_hash: B256::ZERO,
    }
  }

  /// Set the top-level call context before execution.
  pub fn set_context(&mut self, address: Address, caller: Address) {
    self.current_address = address;
    self.current_caller = caller;
  }

  /// Set the account state for the top-level execution context.
  pub fn set_account(&mut self, balance: U256, nonce: u64, code_hash: B256) {
    self.current_balance = balance;
    self.current_nonce = nonce;
    self.current_code_hash = code_hash;
  }

  /// Maximum call depth observed during execution.
  pub fn max_depth(&self) -> u32 {
    self
      .steps
      .iter()
      .map(|s| s.call_depth)
      .max()
      .unwrap_or(0)
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
    let push_data = read_push_data(interp, opcode);

    self.pending = Some(Pending {
      opcode,
      pc,
      gas_before,
      pre_stack,
      push_data,
      call_depth: self.current_depth,
      address: self.current_address,
      caller: self.current_caller,
      balance: self.current_balance,
      nonce: self.current_nonce,
      code_hash: self.current_code_hash,
    });
  }

  fn step_end(&mut self, interp: &mut Interpreter<EthInterpreter>, _context: &mut CTX) {
    if let Some(pending) = self.pending.take() {
      let gas_after = interp.gas.remaining();
      let post_stack = read_stack(interp);

      // Peek at the next opcode/pc/push_data for the post fields.
      let post_opcode = interp.bytecode.opcode();
      let post_pc = interp.bytecode.pc() as u32;
      let post_push_data = read_push_data(interp, post_opcode);

      self.steps.push(EvmStep {
        pre_opcode: pending.opcode,
        post_opcode,
        pre_pc: pending.pc,
        post_pc,
        gas_before: pending.gas_before,
        gas_after,
        pre_stack: pending.pre_stack,
        post_stack,
        pre_push_data: pending.push_data,
        post_push_data,
        call_depth: pending.call_depth,
        address: pending.address,
        caller: pending.caller,
        balance: pending.balance,
        nonce: pending.nonce,
        code_hash: pending.code_hash,
      });
    }
  }

  fn call(&mut self, _context: &mut CTX, inputs: &mut CallInputs) -> Option<CallOutcome> {
    let frame = CallFrame {
      depth: self.current_depth + 1,
      address: inputs.bytecode_address,
      caller: inputs.caller,
      is_create: false,
      success: false, // updated in call_end
      balance: U256::ZERO,
      nonce: 0,
      code_hash: B256::ZERO,
    };
    // Save current state onto the frame stack before entering the callee.
    self.frame_stack.push(frame);
    self.current_depth += 1;
    self.current_address = inputs.bytecode_address;
    self.current_caller = inputs.caller;
    // Account state for callee will be populated by a higher-level pipeline
    // that has DB access (the generic CTX doesn't expose it).
    self.current_balance = U256::ZERO;
    self.current_nonce = 0;
    self.current_code_hash = B256::ZERO;
    None
  }

  fn call_end(
    &mut self,
    _context: &mut CTX,
    _inputs: &CallInputs,
    outcome: &mut CallOutcome,
  ) {
    if let Some(mut frame) = self.frame_stack.pop() {
      frame.success = outcome.result.result.is_ok();
      self.call_frames.push(frame);
      self.current_depth -= 1;
      // Restore parent context.
      if let Some(parent) = self.frame_stack.last() {
        self.current_address = parent.address;
        self.current_caller = parent.caller;
        self.current_balance = parent.balance;
        self.current_nonce = parent.nonce;
        self.current_code_hash = parent.code_hash;
      } else {
        // Back to top-level — state set by set_context / set_account.
      }
    }
  }

  fn create(
    &mut self,
    _context: &mut CTX,
    inputs: &mut CreateInputs,
  ) -> Option<CreateOutcome> {
    let frame = CallFrame {
      depth: self.current_depth + 1,
      address: Address::ZERO, // unknown until create_end
      caller: inputs.caller(),
      is_create: true,
      success: false,
      balance: U256::ZERO,
      nonce: 0,
      code_hash: B256::ZERO,
    };
    self.frame_stack.push(frame);
    self.current_depth += 1;
    self.current_caller = inputs.caller();
    self.current_balance = U256::ZERO;
    self.current_nonce = 0;
    self.current_code_hash = B256::ZERO;
    None
  }

  fn create_end(
    &mut self,
    _context: &mut CTX,
    _inputs: &CreateInputs,
    outcome: &mut CreateOutcome,
  ) {
    if let Some(mut frame) = self.frame_stack.pop() {
      frame.success = outcome.result.result.is_ok();
      if let Some(addr) = outcome.address {
        frame.address = addr;
      }
      self.call_frames.push(frame);
      self.current_depth -= 1;
      if let Some(parent) = self.frame_stack.last() {
        self.current_address = parent.address;
        self.current_caller = parent.caller;
        self.current_balance = parent.balance;
        self.current_nonce = parent.nonce;
        self.current_code_hash = parent.code_hash;
      }
    }
  }
}
