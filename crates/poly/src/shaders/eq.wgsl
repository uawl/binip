// ── eq 다항식 GPU 커널 ────────────────────────────────────────────────────────
// NOTE: GF(2^128) 산술 함수는 gpu::GF128_WGSL 로부터 앞에 연결됨.
//
// eq(r, e) = Π_k ( r[k]*b[k] + (1+r[k])*(1+b[k]) )
// 여기서 b[k] = (e >> k) & 1.
//
// 전략: n_vars 회의 커널 런치로 점진적으로 확장.
// 각 런치에서 현재 배열 크기를 2배로 늘린다.
//
// 초기: evals[0] = 1
// 라운드 i (r[i] 처리):
//   evals[2*j + 0] = evals[j] * (1 + r[i])
//   evals[2*j + 1] = evals[j] * r[i]
// 이 커널은 한 라운드의 확장을 수행한다.

struct EqExpandParams {
    cur_len: u32,     // 확장 전 현재 원소 수 (2^i)
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    r_i: vec4<u32>,   // 현재 라운드의 r[i]
}

@group(0) @binding(0) var<uniform>             eq_params: EqExpandParams;
@group(0) @binding(1) var<storage, read_write> eq_evals:  array<vec4<u32>>;

@compute @workgroup_size(256)
fn eq_expand(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    let cur = eq_params.cur_len;
    if j >= cur { return; }

    let ri = eq_params.r_i;
    let one_plus_ri = gf128_add(GF128_ONE, ri);

    // 역순 처리: 큰 j부터 작은 j로 (덮어쓰기 방지)
    // → 이 커널은 역순 인덱스로 호출된다: j → cur - 1 - gid.x
    let idx = cur - 1u - j;
    let v = eq_evals[idx];
    eq_evals[2u * idx + 1u] = gf128_mul(v, ri);
    eq_evals[2u * idx]      = gf128_mul(v, one_plus_ri);
}
