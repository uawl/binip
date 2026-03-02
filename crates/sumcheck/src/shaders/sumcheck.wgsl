// ─── GF(2^64) = GF(2^32)[t]/(t^2+t+β32), β32 = GF2_32(2) = 2u ────────────────
// Matches redesigned `GF2_64` in crates/field/src/gf2_64.rs.
// Element: vec2<u32> = (lo, hi)  where lo,hi ∈ GF(2^32).
//
// ─── GF(2^128) = GF(2^64)[t]/(t^2+t+β64), β64 = GF2_64(2) = vec2(2u,0u) ─────
// Matches `GF2_128` in crates/field/src/gf2_128.rs.
// Element: vec4<u32> = (lo.x, lo.y, hi.x, hi.y).
// ─────────────────────────────────────────────────────────────────────────────

fn clmul32(a: u32, b: u32) -> vec2<u32> {
    var lo: u32 = 0u;
    var hi: u32 = 0u;
    for (var i = 0u; i < 32u; i = i + 1u) {
        if ((a >> i) & 1u) != 0u {
            lo ^= b << i;
            if i > 0u { hi ^= b >> (32u - i); }
        }
    }
    return vec2<u32>(lo, hi);
}

// GF(2^32) mod p32 = x^32+x^7+x^3+x^2+1
fn reduce32(lo: u32, hi: u32) -> u32 {
    let tail     = hi ^ (hi << 7u) ^ (hi << 3u) ^ (hi << 2u);
    let overflow = (hi >> 25u) ^ (hi >> 29u) ^ (hi >> 30u);
    let extra    = overflow ^ (overflow << 7u) ^ (overflow << 3u) ^ (overflow << 2u);
    return lo ^ tail ^ extra;
}

fn gf32_mul(a: u32, b: u32) -> u32 {
    let p = clmul32(a, b);
    return reduce32(p.x, p.y);
}

// β32 = 2u — constant in GF(2^64) irred poly t^2+t+β32
const BETA32: u32 = 2u;

// GF(2^64) tower mul: (a0+a1·t)(b0+b1·t) = (a0b0+a1b1·β32) + (a0b1+a1b0+a1b1)·t
fn gf64_mul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let a0b0      = gf32_mul(a.x, b.x);
    let a1b1      = gf32_mul(a.y, b.y);
    let a0b1_a1b0 = gf32_mul(a.x, b.y) ^ gf32_mul(a.y, b.x);
    return vec2<u32>(
        a0b0 ^ gf32_mul(a1b1, BETA32),
        a0b1_a1b0 ^ a1b1,
    );
}

fn gf64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> { return a ^ b; }

fn gf128_add(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> { return a ^ b; }

// β64 = GF2_64(2) = lo=GF2_32(2), hi=GF2_32(0) = vec2<u32>(2u, 0u)
fn gf128_mul(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    let al = a.xy; let ah = a.zw;
    let bl = b.xy; let bh = b.zw;
    let a0b0  = gf64_mul(al, bl);
    let a1b1  = gf64_mul(ah, bh);
    let cross = gf64_add(gf64_mul(al, bh), gf64_mul(ah, bl));
    let beta64 = vec2<u32>(2u, 0u);
    let rl = a0b0 ^ gf64_mul(a1b1, beta64);
    let rh = cross ^ a1b1;
    return vec4<u32>(rl.x, rl.y, rh.x, rh.y);
}

// ─── Sumcheck round 커널 ──────────────────────────────────────────────────────
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
