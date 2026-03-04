# binip zkEVM Security Audit Report

**Date**: 2026-03-04
**Scope**: 15 crates (field, poly, lut, logup, sumcheck, pcs, transcript, stark, evm-types, circuit, vm, e2e, recursive, shard, gpu)
**Field**: GF(2^128), irreducible polynomial x^128 + x^7 + x^2 + x + 1

---

## Executive Summary

| Severity | Count | Fixed |
|----------|-------|-------|
| CRITICAL | 5 | 3 (C-1, C-2, C-3) |
| HIGH     | 12 | 0 |
| MEDIUM   | 18 | 1 (M-15, C-1과 동시) |
| **Total** | **35** | **4** |

이 시스템은 **계층적 위임 아키텍처**를 사용합니다:

- Arithmetic constraint → LUT (byte-level decomposition + LogUp)
- Memory/Storage → R/W consistency argument
- Shard integrity → Recursive aggregation
- Proof binding → PCS + Fiat-Shamir transcript

각 계층이 정확해야만 전체가 sound한 구조에서 여러 계층에 동시에 gap이 존재하며, chain of delegation에서 하나라도 빠지면 전체 soundness가 무너집니다.

---

## CRITICAL Issues (5)

### C-1: `batch_verify()` — evaluation claim 미검증 ✅ FIXED

- **Crate**: pcs
- **File**: `crates/pcs/src/lib.rs`
- **Severity**: CRITICAL
- **Status**: **Fixed** — per-query fold chains 구현 완료

**기존 문제**: `batch_verify()`가 shared fold challenges로 proximity test만 수행하고, 각 query의 claim이 committed polynomial의 실제 evaluation과 일치하는지 검증하지 않고 `true`를 반환.

**수정 내용**: batch PCS를 per-query fold chain 방식으로 완전히 재설계:

1. **새 증명 구조**: `BatchRound0Query` (round-0 shared batch tree) + `BatchEvalProof` (per-query intermediate Merkle trees)
2. **batch_open / batch_open_par**: 각 query가 자신의 evaluation point로 fold chain 수행 (single-point PCS와 동일 방식). Round 0는 batch commitment tree를 공유하되, rounds 1..n-1은 per-query Merkle tree 생성.
3. **batch_verify**: Round 0에서 batch Merkle 검증 + entry pair 추출 → rounds 1..n-1에서 per-query Merkle 검증 + fold consistency → 마지막 round에서 `folded == claim` 검증. **Evaluation binding이 완전히 보장됨.**
4. **Query position dedup**: `HashSet` 기반 중복 제거로 M-15도 동시에 수정.
5. **Transcript order**: claims → per-query round_roots → squeeze positions (prover/verifier 동일).

모든 기존 테스트 + 6개 batch 테스트 통과 (reject_wrong_claim, reject_tampered_pair 포함).

---

### C-2: 17개 constraint tag가 무조건 `zero()` 반환 ✅ FIXED

- **Crate**: circuit, stark
- **File**: `crates/circuit/src/constraint.rs` line 64+, `crates/stark/src/proof.rs`, `prover.rs`, `verifier.rs`
- **Severity**: CRITICAL
- **Status**: **Fixed** — reconstruction PCS binding 구현 완료

**기존 문제**: 다음 tags가 algebraic constraint를 전혀 검사하지 않고 `Ok(GF2_128::zero())`를 반환:

| Tags | Operations | 위임 대상 |
|------|-----------|----------|
| 0, 1, 2, 8 | Add128, Mul128, And128, Chi128 | Byte-level LUT |
| 12, 13, 14 | CheckDiv, CheckMul, CheckInv | Mul/Add LUT |
| 17, 23, 24, 26, 28 | Store, MStore, MStore8, SStore, TStore | R/W consistency argument |
| 18, 19, 20 | KeccakLeaf, Compose, TypeCheck | Proof-tree / sub-proof |
| 29, 30 | Advice2, Advice4 | Downstream Check* ops |

**수정 내용**: reconstruction MLE에 대한 PCS + sumcheck binding 추가:

1. **`ReconstructionPcsProof`** 구조체 신설 (`proof.rs`): `BoundaryPcsProof`와 동일 패턴 (commit root, sumcheck proof, PCS opening proof, claimed sum).
2. **Prover** (`prover.rs`): `prepare()` 및 `prepare_par()` 양쪽에서 reconstruction MLE 블라인딩 → PCS commit → sumcheck prove → PCS open 수행. `Proof`에 `reconstruction_pcs` 필드 추가.
3. **Verifier** (`verifier.rs`): Section 4a에서 reconstruction PCS 검증 — commit root 흡수 → claimed_sum == 0 확인 → sumcheck 검증 → PCS opening 검증. 실패 시 `ReconstructionSumcheckFailed`, `ReconstructionSumcheckNonZero`, `ReconstructionPcsOpenFailed` 에러.

이로써 committed decomposition bytes와 trace columns 간의 reconstruction binding이 cryptographic하게 보장됨.

---

### C-3: 30+ EVM opcode에 무검증 advice 주입 ✅ FIXED

- **Crate**: vm
- **File**: `crates/vm/src/compile.rs`, `crates/e2e/src/witness.rs`
- **Severity**: CRITICAL
- **Status**: **Fixed** — 산술 opcode별 verified compiler 구현 완료

**기존 문제**: EXP, SHL, SHR, SAR, SIGNEXTEND, SDIV, SMOD 등이 `compile_advice_u256()`로 무검증 advice 주입.

**수정 내용**: 각 opcode에 대해 circuit-verified 연산 체인 구현:

| Opcode | 검증 방식 | 상세 |
|--------|----------|------|
| **SDIV** | `Mul128` LUT 검증 | q_lo × b_lo 분해+재구성 byte-level LUT 바인딩 |
| **SMOD** | `Mul128` LUT 검증 | SDIV과 동일 패턴 |
| **SHL** | `Shl128` 직접 연산 | shift_amount < 128: Shl128 체인, ≥ 128: lo→hi 이동, ≥ 256: zero |
| **SHR** | `Shr128` 직접 연산 | SHL과 대칭 패턴 |
| **SAR** | `Shr128` + sign extension | SHR + MSB 기반 부호 확장 마스크 |
| **SIGNEXTEND** | mask 연산 | byte_idx별 sign bit 추출 + Xor/And 마스크 체인 |
| **EXP** | `CheckMul` 이분법 (binary exponentiation) | 아래 상세 설명 참조 |

**EXP 이분법 (square-and-multiply) 구현**:

`compile_exp(base, exponent)` — 컴파일 타임에 exponent 비트를 알고, 각 단계를 `CheckMul`로 검증:

1. 특수 케이스: exp=0 → 1, exp=1 → identity, base=0 → 0, base=1 → 1 (advice 불필요)
2. 일반 케이스: base를 slot(2)에 복사하여 보존
3. exponent의 MSB부터 LSB까지 순회:
   - **Squaring**: `Advice2(result_lo, result_hi) + AdviceLoad(mul_hi) + CheckMul(q_lo, q_hi, a, a) + 2×Mov`
   - **Multiply-by-base** (bit=1일 때): 동일 패턴, `b = slot(2)` (보존된 base)
4. 각 단계에서 128×128 → 256 widening multiplication이 `CheckMul`로 검증됨
5. `compute_exp_advice()`: witness 빌더에서 동일 알고리즘 실행, 단계별 `[result_lo, result_hi, mul_hi]` 3-limb advice 생성

**참고**: ADDRESS, BALANCE, CALLER 등 환경 쿼리 opcode는 EVM inspector에서 결과가 결정되므로, 현재 `compile_advice_u256()` 유지 (type-checking layer에서 검증 위임).

---

### C-4: Shard boundary에 대한 cryptographic binding 부재

- **Crate**: shard
- **Files**: `crates/shard/src/prover.rs` line 24, `crates/shard/src/verifier.rs` line 75
- **Severity**: CRITICAL

두 가지 문제:

1. **shard_idx 순서 미검증**: verifier가 `shard_proofs[i].shard_idx == i`를 확인하지 않음.
2. **경계 commitment 부재**: `ShardProof`에 원본 MLE의 어느 부분(contiguous chunk)인지에 대한 commitment이 없음. sumcheck proof와 `shard_idx`만 포함.

```rust
// verifier.rs — shard_idx 검증 없이 순회
for (proof, sub) in batch.shard_proofs.iter().zip(&sub_mles) {
  // proof.shard_idx is NEVER checked against iteration index
  let oracle_eval = sub.evaluate(&challenges);
  let result = verify_shard(proof, oracle_eval, root_transcript)?;
}
```

**영향**: prover가 shard 순서를 바꾸거나, 다른 다항식에 대한 sumcheck proof를 제출 가능.

**권장**: shard_idx 순서 검증 + shard boundary commitment 추가.

---

### C-5: Recursive aggregation에 `fan_in` domain separation 누락

- **Crate**: recursive
- **File**: `crates/recursive/src/prover.rs` line 93
- **Severity**: CRITICAL

Transcript fork에 `fan_in` 값이 포함되지 않아, 서로 다른 configuration에서 동일한 challenge가 생성됩니다:

```rust
// 현재 (VULNERABLE):
let mut t = root_transcript.fork("recursive", level * 0x1_0000 + node_idx as u32);
t.absorb_bytes(&level.to_le_bytes());
t.absorb_bytes(&(node_idx as u32).to_le_bytes());
// fan_in 미포함 → config A(fan_in=2)와 config B(fan_in=4)가 동일 challenge 생성
```

**영향**: 한 configuration에서 만든 proof를 다른 configuration에서 재사용 가능.

**권장**:
```rust
t.absorb_bytes(&config.fan_in.to_le_bytes());  // fork 직후 추가
```

---

## HIGH Issues (12)

### H-1: LogUp `debug_assert_eq` — release 빌드에서 제거됨

- **Crate**: logup
- **File**: `crates/logup/src/prover.rs` line 221-222
- **Severity**: HIGH

```rust
debug_assert_eq!(open_claims[0], witness_proof.final_eval);
debug_assert_eq!(open_claims[1], table_proof.final_eval);
```

PCS opening claim과 sumcheck final_eval의 일치 확인이 `debug_assert`로만 되어 있어 release 빌드에서 제거됩니다.

**권장**: `assert_eq!`로 변경.

---

### H-2: STARK prover `.expect()` — 에러 전파 미흡

- **Crate**: stark
- **Files**: `crates/stark/src/prover.rs` line 110, 215, 467
- **Severity**: HIGH

```rust
let bnd_challenges = sumcheck::verify(&bnd_sumcheck, bnd_sumcheck.final_eval, &mut bnd_sc_t2)
  .expect("own boundary sumcheck should verify");
```

내부 sumcheck 검증 실패 시 panic 대신 에러 전파가 필요합니다.

**권장**: `.ok_or(ProveError::SumcheckVerificationFailed)?` 패턴 사용.

---

### H-3: `derive_open_point()` — verifier 독립 검증 없음

- **Crate**: stark
- **File**: `crates/stark/src/prover.rs` line 450-485
- **Severity**: HIGH

Prover가 sumcheck replay로 challenge를 도출하여 PCS opening point를 구성하지만, verifier는 이를 독립적으로 도출/검증하지 않습니다. Proof soundness가 bugless `derive_open_point()`에 의존합니다.

**권장**: Verifier에서 동일 로직 실행 후 비교하거나, derived challenges를 transcript에 commit.

---

### H-4: `constraint_sum ↔ shard total_sum` binding 취약

- **Crate**: stark
- **File**: `crates/stark/src/verifier.rs` line 213-219
- **Severity**: HIGH

Verifier가 `proof.constraint_sum != proof.shard_batch.total_sum`만 확인하며, sharding decomposition의 정확성까지 보장하지 않습니다.

**권장**: Shard batch structure 검증 강화 (shard count, 크기 일치 등).

---

### H-5: evm-types — ADDMOD/MULMOD/SHL/SHR/SAR 의미론 검증 누락

- **Crate**: evm-types
- **File**: `crates/evm-types/src/consistency.rs` line 645+
- **Severity**: HIGH

`check_opcode_semantics()`에 ADDMOD (0x06), MULMOD (0x07), SHL (0x1b), SHR (0x1c), SAR (0x1d), SDIV, SMOD, SIGNEXTEND, BYTE의 분기가 없습니다. 이 opcode들의 결과값이 검증되지 않습니다.

**권장**: 각 opcode에 대해 pre/post stack 관계 검증 분기 추가.

---

### H-6: e2e witness — PC 증가값 하드코딩 (`+1`)

- **Crate**: e2e
- **File**: `crates/e2e/src/witness.rs` line 89
- **Severity**: HIGH

```rust
pc: step.pc + 1,  // PUSH1~PUSH32는 2~33 bytes 진행
```

PUSH 명령어는 1보다 큰 폭으로 PC가 증가해야 하나, `+1`로 하드코딩되어 proof tree의 post-state PC가 잘못됩니다.

**권장**: `opcode_advance(step.opcode)` 함수 사용.

---

### H-7: e2e witness — memory/storage 상태 미전달

- **Crate**: e2e
- **File**: `crates/e2e/src/witness.rs` line 85-105
- **Severity**: HIGH

```rust
let post_state = EvmState {
  stack: step.post_stack.clone(),
  memory: Vec::new(),              // ← 항상 empty
  storage: Default::default(),     // ← 항상 empty
  transient_storage: Default::default(),
  jumpdest_table: Default::default(),
};
```

각 step의 post-state에 memory/storage가 비어 있어 multi-step transaction의 상태 연속성이 없습니다.

**권장**: 이전 step의 memory/storage를 carry-forward.

---

### H-8: e2e witness — step 간 gas/stack continuity 미검증

- **Crate**: e2e
- **File**: `crates/e2e/src/witness.rs` line 50-105
- **Severity**: HIGH

각 `EvmStep`이 독립적으로 처리되며, `step[i].gas_after == step[i+1].gas_before` 및 stack 값 연속성 검증이 없습니다.

**권장**: `build_witness`에 cross-step validation 추가.

---

### H-9: e2e witness — ADDMOD/MULMOD mod 0 잘못된 결과

- **Crate**: e2e
- **File**: `crates/e2e/src/witness.rs` line 267-290
- **Severity**: HIGH

```rust
ADDMOD => {
  let n = step.pre_stack.get(2).copied().unwrap_or(U256::ZERO);
  if n.is_zero() {
    (U256::ZERO, U256::ZERO)  // ← EVM spec: ADDMOD(a,b,0) = 0이지만 remainder 처리 오류
  }
}
```

**권장**: EVM yellow paper에 맞게 수정: `ADDMOD(a, b, 0) = 0`.

---

### H-10: GPU staging buffer 에러 시 메모리 누수

- **Crate**: gpu
- **File**: `crates/gpu/src/buffer.rs` line 85-120
- **Severity**: HIGH

`poll()` 실패 또는 `rx.recv()` 실패 시 staging buffer가 GPU 메모리에서 해제되지 않습니다.

**권장**: RAII guard 또는 explicit drop 사용.

---

### H-11: GPU channel recv unwrap — panic

- **Crate**: gpu
- **File**: `crates/gpu/src/buffer.rs` line 112
- **Severity**: HIGH

```rust
rx.recv().unwrap().map_err(|_| GpuError::MapFailed)?;
```

Channel 실패 시 panic 대신 graceful 에러 반환이 필요합니다.

**권장**: `.map_err(|_| GpuError::ChannelClosed)?` 사용.

---

### H-12: `blind()` RNG에 `CryptoRng` trait bound 미적용

- **Crate**: poly
- **File**: `crates/poly/src/poly.rs` line 156-165
- **Severity**: HIGH

`blind()` 및 `blind_par()`에 전달되는 RNG에 `CryptoRng` 요구사항이 없습니다. 약한 RNG 사용 시 zero-knowledge 보장이 깨집니다.

**권장**: trait bound에 `R: Rng + CryptoRng` 추가 또는 최소한 문서화.

---

## MEDIUM Issues (18)

### M-1: `GF2_128::inv()` — zero 입력 시 panic

- **Crate**: field
- **File**: `crates/field/src/gf2_128.rs` line 282
- **Severity**: MEDIUM

```rust
fn inv(self) -> Self {
  assert!(!self.is_zero(), "GF2_128::inv called on zero");
```

Untrusted 입력이 도달할 경우 DoS 벡터.

**권장**: `Option<Self>` 반환 또는 caller-side validation.

---

### M-2: `batch_inv()` — zero 원소 포함 시 panic

- **Crate**: field
- **File**: `crates/field/src/lib.rs` line 8-10
- **Severity**: MEDIUM

Zero 원소가 포함된 slice에서 panic 발생.

**권장**: `Option<Vec<F>>` 반환 또는 zero 필터링.

---

### M-3: `MlePoly::zero()` — shift overflow

- **Crate**: poly
- **File**: `crates/poly/src/poly.rs` line 49
- **Severity**: MEDIUM

```rust
fn zero(n_vars: u32) -> Self {
  vec![GF2_128::zero(); 1 << n_vars]  // n_vars > 63이면 overflow
}
```

**권장**: `assert!(n_vars <= 62)` 추가.

---

### M-4: `MlePoly::evaluate()` — 전체 테이블 clone 시 OOM

- **Crate**: poly
- **File**: `crates/poly/src/poly.rs` line 65
- **Severity**: MEDIUM

n_vars=28이면 2^28 × 16 bytes = 4GB clone. 악의적 입력으로 메모리 고갈 가능.

**권장**: In-place reduction 또는 folding buffer 사용.

---

### M-5: `MlePoly::fold_last()` — constant polynomial에서 panic

- **Crate**: poly
- **File**: `crates/poly/src/poly.rs` line 82
- **Severity**: MEDIUM

```rust
assert!(self.n_vars > 0);  // n_vars=0이면 panic
```

**권장**: `Option<Self>` 반환.

---

### M-6: LUT `prove()` — witness in-place 변환 후 원본 크기 미반환

- **Crate**: lut
- **File**: `crates/lut/src/lib.rs` line 273, 289
- **Severity**: MEDIUM

```rust
pub fn prove(op: Op8, witness: &mut Vec<GF2_128>, ...) {
  let n = witness.len().next_power_of_two();
  witness.resize(n, pad);  // 원본 위에 덮어씀
}
```

`prove_committed()` 호출 후 `witness.len()`이 변경되며, caller가 이를 알아야 `verify_committed()`에 전달 가능.

**권장**: padded length 반환 또는 문서화.

---

### M-7: Sumcheck verifier `oracle_eval` — PCS binding 위임

- **Crate**: sumcheck
- **File**: `crates/sumcheck/src/verifier.rs` line 40-47
- **Severity**: MEDIUM

Verifier가 `oracle_eval`을 caller로부터 받지만, 이것이 PCS opening에서 온 것인지 검증하지 않습니다. Caller 책임이나 미문서화.

**권장**: 함수 docstring에 "oracle_eval MUST come from a valid PCS opening proof" 명시.

---

### M-8: Transcript sponge — Rate = Capacity (50/50)

- **Crate**: transcript
- **File**: `crates/transcript/src/sponge.rs` line 12-13
- **Severity**: MEDIUM

```rust
const RATE_BYTES: usize = 32;  // 2 × u128
// Capacity = 2 × u128 = 32 bytes
```

State 512bit 중 rate 256bit, capacity 256bit. 유효 보안 = capacity/2 = 128bit. GF(2^128) 필드에는 충분하나, 256-bit 보안 목표 시 불충분.

**권장**: 256-bit 보안 필요 시 state 크기 증가 또는 rate 감소.

---

### M-9: `absorb_fields()` — unsafe slice cast, big-endian 비안전

- **Crate**: transcript
- **File**: `crates/transcript/src/blake3_transcript.rs` line 87-97
- **Severity**: MEDIUM

`#[repr(C)]` GF2_128를 `&[u8]`로 transmute하여 Blake3에 전달. `#[cfg(target_endian = "little")]` guard가 있으나, 필드 타입 변경 시 불안전.

**권장**: 명시적 serialization 또는 `bytemuck::Pod` derive.

---

### M-10: RangeCheck (tag 15) — algebraic constraint가 tautology

- **Crate**: circuit
- **File**: `crates/circuit/src/constraint.rs` line 149-163
- **Severity**: MEDIUM

```rust
15 => Ok(out + in1),  // out = mask, in1 = mask → 항상 0
```

실제 range 관계 `v + pad = mask (carry=0)`는 전적으로 Add LUT carry chain에 의존. LUT에 버그 시 range check 무효화.

**권장**: 최소한 `v + pad + out != 0` 형태의 보조 constraint 추가.

---

### M-11: `row_tag()` — 범위 외 tag 처리

- **Crate**: circuit
- **File**: `crates/circuit/src/trace.rs` line 293-302
- **Severity**: MEDIUM

`op.lo >= 34`이면 `u32::MAX` 반환 후 caller가 에러 처리. 악의적 prover가 중간 값(예: 5.5)을 commit해도 trace에는 포함되고 evaluation 시에만 reject.

---

### M-12: VM register 범위 검증 없음

- **Crate**: vm
- **File**: `crates/vm/src/exec.rs` line 810-895
- **Severity**: MEDIUM

MLoad/SLoad/TLoad에서 `*dst + 1`이 16 이상이면 panic. 2-register 연산에 대한 ISA-level 범위 검증이 없음.

**권장**: `assert!(dst + 1 < 16)` 추가.

---

### M-13: DUP6+ — slot 1 pre-loading 전제 미검증

- **Crate**: vm
- **File**: `crates/vm/src/compile.rs` line 440-455
- **Severity**: MEDIUM

```rust
let src_slot = if (n as usize) <= 5 {
  (n - 1) as usize
} else {
  1  // caller가 slot 1을 미리 채워야 함
};
```

DUP6 이상에서 caller가 slot 1에 올바른 stack 원소를 pre-load해야 하나, 이 contract이 검증/문서화되지 않음.

---

### M-14: Recursive prover — parallel variant `.expect()` panic

- **Crate**: recursive
- **File**: `crates/recursive/src/prover.rs` line 304, 311
- **Severity**: MEDIUM

```rust
results[root_id].take().expect("all nodes must complete");
assert_eq!(last_level.len(), 1);
```

Task 실패 시 graceful 에러 대신 panic.

**권장**: `Result` 기반 에러 처리.

---

### M-15: Batch PCS — query deduplication 없음 ✅ FIXED (C-1과 동시 수정)

- **Crate**: pcs
- **File**: `crates/pcs/src/lib.rs`
- **Severity**: MEDIUM
- **Status**: **Fixed** — C-1 batch PCS 재설계 시 `HashSet` 기반 dedup 포함

Single-point `open`은 `HashSet`으로 중복 query를 제거하지만, `batch_open`/`batch_verify`는 하지 않음. 중복 query 시 soundness가 `(1/2)^n_queries`에서 `(1/2)^k` (k = unique queries)로 저하.

**권장**: batch 경로에도 deduplication 로직 추가.

---

### M-16: GPU buffer 크기 계산 integer overflow

- **Crate**: gpu
- **File**: `crates/gpu/src/buffer.rs` line 128-132
- **Severity**: MEDIUM

```rust
let bytes = (len * mem::size_of::<T>()).max(4);  // len이 크면 overflow
```

**권장**: `len.checked_mul(mem::size_of::<T>()).expect("buffer size overflow")`.

---

### M-17: GPU workgroup 크기 truncation

- **Crate**: gpu
- **File**: `crates/gpu/src/pipeline.rs` line 90
- **Severity**: MEDIUM

`n / 64` 정수 나눗셈으로 `n % 64 ≠ 0`이면 마지막 원소들이 처리되지 않음.

**권장**: `(n + 63) / 64` (ceiling division) 사용.

---

### M-18: GPU `ManuallyDrop<Device>` — 의도적 메모리 누수

- **Crate**: gpu
- **File**: `crates/gpu/src/context.rs` line 8-15
- **Severity**: MEDIUM

```rust
pub device: ManuallyDrop<wgpu::Device>,
pub queue: ManuallyDrop<wgpu::Queue>,
// dzn driver SIGSEGV 회피용 — OS 프로세스 종료 시 해제에 의존
```

문서화된 workaround이나, 장기 실행 프로세스에서 GPU 리소스 누수.

---

## Crate Summary

| Crate | Rating | C | H | M | Core Risk |
|-------|--------|---|---|---|-----------|
| **field** | ✅ PASS | 0 | 0 | 2 | `inv(0)` panic |
| **poly** | ⚠ CONDITIONAL | 0 | 1 | 3 | OOM, CryptoRng 미적용 |
| **lut** | ✅ PASS | 0 | 0 | 1 | Witness mutation API |
| **logup** | ⚠ CAUTION | 0 | 1 | 0 | `debug_assert` 릴리즈 누락 |
| **sumcheck** | ✅ PASS | 0 | 0 | 1 | `oracle_eval` PCS 위임 (의도적) |
| **pcs** | ✅ PASS | ~~1~~ 0 | 0 | ~~1~~ 0 | ~~`batch_verify` 미완성~~ C-1 수정 완료, M-15 동시 수정 |
| **transcript** | ⚠ CONDITIONAL | 0 | 0 | 2 | 128-bit effective security |
| **stark** | ⚠ CAUTION | 0 | 3 | 0 | `derive_open_point` 미검증 |
| **evm-types** | ⚠ CAUTION | 0 | 1 | 0 | Opcode 의미론 일부 누락 |
| **circuit** | ✅ PASS | ~~1~~ 0 | 0 | 2 | ~~17/34 tags 무조건 zero~~ C-2 reconstruction PCS binding 추가 |
| **vm** | ✅ PASS | ~~1~~ 0 | 0 | 2 | ~~Advice 무검증 주입~~ C-3 opcode별 verified compiler 구현 |
| **e2e** | 🔴 FAIL | 0 | 4 | 0 | State continuity 전무 |
| **recursive** | 🔴 FAIL | 1 | 1 | 0 | `fan_in` domain sep 누락 |
| **shard** | 🔴 FAIL | 1 | 0 | 0 | Shard binding 없음 |
| **gpu** | ⚠ CAUTION | 0 | 2 | 3 | Buffer 누수, overflow |

---

## Architecture Soundness Analysis

### Delegation Chain

```
VM (advice oracle) → Circuit (constraint eval) → LUT (LogUp) → PCS (commitment)
                                                                     ↓
Shard (partitioning) → Recursive (aggregation) → STARK (composition) → Verifier
```

### Gap Analysis

1. ~~**VM → Circuit gap**: VM이 advice를 무검증 주입 (C-3) → circuit의 Advice2/Advice4 tags도 `zero()` 반환 (C-2) → **산술 결과 위조 가능**~~ **✅ CLOSED**: C-3 (opcode별 CheckMul/Mul128/Shl128 검증) + C-2 (reconstruction PCS binding)로 해결. EXP는 이분법(square-and-multiply)으로 각 단계가 CheckMul 검증됨.
2. ~~**Circuit → LUT → PCS gap**: Circuit이 LUT에 위임 → LUT는 PCS binding 의존 → `batch_verify` 미완성 (C-1) → **LUT soundness 미보장**~~ **✅ CLOSED**: C-1 (per-query fold chain)으로 evaluation binding 완전 보장.
3. **Shard → Recursive gap**: Shard index 미검증 (C-4) → recursive fan_in domain sep 누락 (C-5) → **proof 재사용/재정렬 가능**
4. **e2e → STARK gap**: Witness builder가 state continuity를 보장하지 않음 (H-6~H-8) → 잘못된 witness가 STARK에 전달될 수 있음

### Positive Aspects

- ✅ Sumcheck 프로토콜 구현이 수학적으로 정확
- ✅ Fiat-Shamir transcript의 fork() 설계가 건전
- ✅ Single-point PCS (commit/open/verify) 구현이 sound
- ✅ GF(2^128) 필드 연산이 정확 (Karatsuba, PCLMULQDQ/PMULL)
- ✅ LogUp γ-weighting이 char-2 cancellation을 올바르게 방지
- ✅ V1.1 soundness fixes 완료: C-1 (batch PCS per-query fold), C-2 (reconstruction PCS binding), C-3 (opcode별 verified compilers + EXP binary exponentiation)
- ✅ Succinct verifier O(√n·log n) 달성
- ✅ 500+ 테스트 통과, profile_cpu 10000 steps prove+verify 정상

---

## Remediation Priority

### Immediate (production blocker)

| Priority | Issue | Fix | Status |
|----------|-------|-----|--------|
| ~~1~~ | ~~C-1: `batch_verify` evaluation binding~~ | ~~Per-query eval check 구현~~ | ✅ FIXED |
| 2 | C-4: Shard index 미검증 | `proof.shard_idx == i` assert 추가 | ⬜ |
| 3 | C-5: Recursive fan_in domain sep | `absorb_bytes(&fan_in.to_le_bytes())` 추가 | ⬜ |
| 4 | H-1: `debug_assert` → `assert` | 1-line 수정 | ⬜ |

### Short-term (soundness hardening)

| Priority | Issue | Fix | Status |
|----------|-------|-----|--------|
| ~~5~~ | ~~C-3: Advice 무검증 주입~~ | ~~Opcode별 Check 연산 설계/구현~~ | ✅ FIXED |
| ~~6~~ | ~~C-2: Constraint zero delegation~~ | ~~Reconstruction binding constraint 추가~~ | ✅ FIXED |
| 7 | H-2: `.expect()` → 에러 전파 | `Result` 패턴 전환 | ⬜ |
| 8 | H-3: `derive_open_point` 검증 | Verifier에서 독립 도출 | ⬜ |

### Medium-term (robustness)

| Priority | Issue | Fix |
|----------|-------|-----|
| 9 | H-5: Opcode 의미론 추가 | consistency.rs에 분기 추가 |
| 10 | H-6~H-9: e2e witness builder | State carry-forward + PC advance + validation |
| 11 | H-12: CryptoRng bound | Trait bound 추가 |
| 12 | M-1~M-5: Panic → Result 변환 | 각 함수 시그니처 변경 |
| 13 | M-15: Batch dedup | HashSet 로직 이식 |
| 14 | H-10~H-11: GPU buffer safety | RAII guard + recv 에러 처리 |

---

## Methodology

- 각 crate의 모든 소스 파일을 전수 검토
- Constraint tag 완전성 분석 (34개 tag 각각의 soundness 검증)
- Fiat-Shamir transcript binding 흐름 추적 (absorb → squeeze 순서 일관성)
- PCS commit → open → verify 체인의 soundness 검증
- Delegation chain의 각 연결점에서 gap 분석
- EVM opcode별 compile → execute → constrain 경로 추적
- Unsafe 코드 및 SIMD intrinsic 사용 검토
- Parallel variant의 data race / atomicity 검증
