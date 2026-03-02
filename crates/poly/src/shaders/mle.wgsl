// ── MLE fold 커널 ─────────────────────────────────────────────────────────────
// NOTE: GF(2^128) 산술 함수는 gpu::GF128_WGSL 로부터 앞에 연결됨.
// evals'[i] = evals[i] * (1 + r) + evals[i + half] * r
//
// 유니폼: half (u32), r (vec4<u32>)
// 버퍼:   evals (read-write, vec4<u32> 배열)

struct FoldParams {
    half: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    r: vec4<u32>,
}

@group(0) @binding(0) var<uniform>             params: FoldParams;
@group(0) @binding(1) var<storage, read_write> evals:  array<vec4<u32>>;

@compute @workgroup_size(256)
fn mle_fold(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.half { return; }

    let r         = params.r;
    let one_minus_r = vec4<u32>(1u, 0u, 0u, 0u) ^ r;  // 1 XOR r in GF(2^128)

    let a = evals[i];
    let b = evals[i + params.half];

    // a * (1+r) + b * r
    evals[i] = gf128_add(gf128_mul(a, one_minus_r), gf128_mul(b, r));
}
