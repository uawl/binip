// ─── Sumcheck 커널 ────────────────────────────────────────────────────────────
// NOTE: GF(2^128) 산술 함수는 gpu::GF128_WGSL 로부터 앞에 연결됨.
//
// 라운드마다 한 변수에 대해 g(0), g(1), g(α) 를 계산한다.
//
// 테이블 레이아웃: [lo_half | hi_half], 각 half 크기 = `half`
//   g(0) = Σ_{i=0..half-1} table[i]        (변수 = 0)
//   g(1) = Σ_{i=0..half-1} table[i + half] (변수 = 1)
//   g(α) = g(0)*(1+α) + g(1)*α              (α = GF2_128::from(2) = (2,0,0,0))
//
// 출력 버퍼: [g0(4 u32), g1(4 u32), ga(4 u32)]  = 12 u32 (48 bytes)
//
// Workgroup 구성: workgroup_size = 256 threads
// 스레드 i는 i 번째 element를 처리, 부분합을 shared memory에 누적 후
// tree reduction으로 합산.
//
// half 가 workgroup_size 의 최대 배수가 아닐 수 있으므로, range check 필수.

struct RoundParams {
    // .x = half, .yzw = unused padding  (vec4 satisfies uniform 16-byte alignment)
    data: vec4<u32>,
}

@group(0) @binding(0) var<uniform>             params:  RoundParams;
@group(0) @binding(1) var<storage, read>       table:   array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> out_buf: array<vec4<u32>>; // 3 elements

// Shared memory: two arrays of 256 vec4 each (lo-half partial sums, hi-half partial sums)
var<workgroup> wg_g0: array<vec4<u32>, 256>;
var<workgroup> wg_g1: array<vec4<u32>, 256>;

@compute @workgroup_size(256)
fn sumcheck_round(
    @builtin(global_invocation_id)  gid: vec3<u32>,
    @builtin(local_invocation_id)   lid: vec3<u32>,
    @builtin(workgroup_id)          wgid: vec3<u32>,
    @builtin(num_workgroups)        nwg:  vec3<u32>,
) {
    let local_idx = lid.x;
    let half      = params.data.x;

    // Each thread accumulates its slice contribution
    var acc_g0 = vec4<u32>(0u, 0u, 0u, 0u);
    var acc_g1 = vec4<u32>(0u, 0u, 0u, 0u);

    // Stride loop: thread `local_idx` covers indices local_idx, local_idx+256, ...
    var idx = gid.x;
    loop {
        if idx >= half { break; }
        acc_g0 = gf128_add(acc_g0, table[idx]);
        acc_g1 = gf128_add(acc_g1, table[idx + half]);
        idx = idx + 256u * nwg.x;
    }

    wg_g0[local_idx] = acc_g0;
    wg_g1[local_idx] = acc_g1;
    workgroupBarrier();

    // Tree reduction within workgroup (256 → 1)
    var stride = 128u;
    loop {
        if stride == 0u { break; }
        if local_idx < stride {
            wg_g0[local_idx] = gf128_add(wg_g0[local_idx], wg_g0[local_idx + stride]);
            wg_g1[local_idx] = gf128_add(wg_g1[local_idx], wg_g1[local_idx + stride]);
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    // Thread 0 of each workgroup writes partial result
    // For multi-workgroup launches we need an intermediate buffer.
    // Simple case (single workgroup): write final result.
    if local_idx == 0u {
        // If this is the only workgroup, finalize g(α) = g0*(1+α) + g1*α
        if nwg.x == 1u {
            let g0 = wg_g0[0];
            let g1 = wg_g1[0];
            // α = (2, 0, 0, 0) in vec4<u32>
            let alpha     = vec4<u32>(2u, 0u, 0u, 0u);
            let one       = vec4<u32>(1u, 0u, 0u, 0u);
            let one_plus_a = gf128_add(one, alpha);   // 1 XOR α
            let ga = gf128_add(gf128_mul(g0, one_plus_a), gf128_mul(g1, alpha));
            out_buf[0] = g0;
            out_buf[1] = g1;
            out_buf[2] = ga;
        } else {
            // Multi-workgroup: write partial sums to workgroup slot
            // Caller must launch a second pass to reduce across workgroups.
            let slot = wgid.x;
            out_buf[slot * 2u]      = wg_g0[0];
            out_buf[slot * 2u + 1u] = wg_g1[0];
        }
    }
}

// ─── Fold 커널 (mle.wgsl 동일 — GPU prover 내부에서 재사용) ──────────────────
struct FoldParams {
    // .x = half, .yzw = unused padding
    pad: vec4<u32>,
    r: vec4<u32>,
}

@group(0) @binding(0) var<uniform>             fold_params: FoldParams;
@group(0) @binding(1) var<storage, read_write> fold_evals:  array<vec4<u32>>;

@compute @workgroup_size(256)
fn sumcheck_fold(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i    = gid.x;
    let half = fold_params.pad.x;
    if i >= half { return; }

    let r           = fold_params.r;
    let one_minus_r = gf128_add(vec4<u32>(1u, 0u, 0u, 0u), r);

    let a = fold_evals[i];
    let b = fold_evals[i + half];
    fold_evals[i] = gf128_add(gf128_mul(a, one_minus_r), gf128_mul(b, r));
}
