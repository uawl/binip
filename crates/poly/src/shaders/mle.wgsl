// ─── GF(2^128) 타워 산술 ────────────────────────────────────────────────────
//
// 표현: vec4<u32>  =  (lo_lo, lo_hi, hi_lo, hi_hi)
//   GF(2^128) = GF(2^64)[t] / (t^2 + t + β64)   β64 = (0, 2, 0, 0) 위치 원소
//   GF(2^64)  = [lo_lo, lo_hi] 쌍
//
// 덧셈: XOR
// 곱셈: 2단계 Karatsuba (GF(2^64) 수준에서 한 번, GF(2^32) 수준에서 한 번)

// ── GF(2^32) CLMUL (u32 × u32 → u64 as vec2<u32>) ───────────────────────────
fn clmul32(a: u32, b: u32) -> vec2<u32> {
    var lo: u32 = 0u;
    var hi: u32 = 0u;
    for (var i = 0u; i < 32u; i = i + 1u) {
        if ((a >> i) & 1u) != 0u {
            lo ^= b << i;
            if i > 0u {
                hi ^= b >> (32u - i);
            }
        }
    }
    return vec2<u32>(lo, hi);
}

// ── GF(2^32) 감약: x^32 ≡ x^7 + x^3 + x^2 + 1 (mod p32) ───────────────────
// irreducible poly: x^32 + x^7 + x^3 + x^2 + 1
//   tail = bits 32..63 of 64-bit product → reduce
fn reduce32(lo: u32, hi: u32) -> u32 {
    // hi bits: position k means coefficient of x^(32+k)
    // x^(32+k) ≡ x^(k+7) + x^(k+3) + x^(k+2) + x^k
    let tail = hi ^ (hi << 7u) ^ (hi << 3u) ^ (hi << 2u);
    // overflow from tail into hi-range (bits >= 32): recurse one step
    let overflow = (hi >> 25u) ^ (hi >> 29u) ^ (hi >> 30u);
    let extra = overflow ^ (overflow << 7u) ^ (overflow << 3u) ^ (overflow << 2u);
    return lo ^ tail ^ extra;
}

// ── GF(2^32) 곱셈 ─────────────────────────────────────────────────────────────
fn gf32_mul(a: u32, b: u32) -> u32 {
    let p = clmul32(a, b);
    return reduce32(p.x, p.y);
}

// ── GF(2^64) = vec2<u32> 곱셈 (Schoolbook, 4 GF32 muls) ─────────────────────
// β32 = 2u (= x in GF(2^32)) — the constant in GF(2^64) irred poly t^2+t+β32
const BETA32: u32 = 2u;

fn gf64_mul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let a0b0 = gf32_mul(a.x, b.x);
    let a1b1 = gf32_mul(a.y, b.y);
    let a0b1_a1b0 = gf32_mul(a.x, b.y) ^ gf32_mul(a.y, b.x);
    return vec2<u32>(
        a0b0 ^ gf32_mul(a1b1, BETA32),   // lo = a0b0 + a1b1*β32
        a0b1_a1b0 ^ a1b1,               // hi = cross + a1b1
    );
}

fn gf64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return a ^ b;
}

// ── GF(2^128) = vec4<u32>  (lo=.xy, hi=.zw) ─────────────────────────────────
// β64 = vec2<u32>(0u, 2u) — the constant in GF(2^128) irred poly t^2+t+β64

fn gf128_add(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    return a ^ b;
}

fn gf128_mul(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    let a_lo = a.xy;
    let a_hi = a.zw;
    let b_lo = b.xy;
    let b_hi = b.zw;

    let a0b0   = gf64_mul(a_lo, b_lo);
    let a1b1   = gf64_mul(a_hi, b_hi);
    let cross  = gf64_add(gf64_mul(a_lo, b_hi), gf64_mul(a_hi, b_lo));
    let beta64 = vec2<u32>(2u, 0u);  // β64 = GF2_64(2) = lo=GF2_32(2), hi=0

    let res_lo = gf64_add(a0b0, gf64_mul(a1b1, beta64));
    let res_hi = gf64_add(cross, a1b1);

    return vec4<u32>(res_lo.x, res_lo.y, res_hi.x, res_hi.y);
}

// ─────────────────────────────────────────────────────────────────────────────

// ── MLE fold 커널 ─────────────────────────────────────────────────────────────
// evals'[i] = evals[i] * (1 + r) + evals[i + half] * r
//
// 유니폼: half (u32), r (vec4<u32>)
// 버퍼:   evals (read-write, vec4<u32> 배열)

struct FoldParams {
    half: u32,
    _pad: array<u32, 3>,
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
