// ─── GF(2^128) 타워 산술 (공유 모듈) ──────────────────────────────────────────
//
// 표현: vec4<u32>  =  (lo_lo, lo_hi, hi_lo, hi_hi)
//   GF(2^128) = GF(2^64)[t] / (t^2 + t + β64)   β64 = GF2_64(2) = vec2(2,0)
//   GF(2^64)  = GF(2^32)[t] / (t^2 + t + β32)   β32 = GF2_32(2) = 2u
//   GF(2^32)  = GF(2)[x]   / (x^32+x^7+x^3+x^2+1)
//
// 덧셈: XOR
// 곱셈: 2단계 Schoolbook (GF(2^64) 수준, GF(2^32) 수준)

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
fn reduce32(lo: u32, hi: u32) -> u32 {
    let tail = hi ^ (hi << 7u) ^ (hi << 3u) ^ (hi << 2u);
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
// β32 = 2u = x in GF(2^32), the constant in GF(2^64) irred poly t^2+t+β32
const BETA32: u32 = 2u;

fn gf64_mul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let a0b0 = gf32_mul(a.x, b.x);
    let a1b1 = gf32_mul(a.y, b.y);
    let a0b1_a1b0 = gf32_mul(a.x, b.y) ^ gf32_mul(a.y, b.x);
    return vec2<u32>(
        a0b0 ^ gf32_mul(a1b1, BETA32),
        a0b1_a1b0 ^ a1b1,
    );
}

fn gf64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return a ^ b;
}

// ── GF(2^128) = vec4<u32>  (lo=.xy, hi=.zw) ─────────────────────────────────
// β64 = GF2_64(2) = vec2<u32>(2u, 0u)

fn gf128_add(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    return a ^ b;
}

fn gf128_mul(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    let a_lo = a.xy;
    let a_hi = a.zw;
    let b_lo = b.xy;
    let b_hi = b.zw;

    let a0b0  = gf64_mul(a_lo, b_lo);
    let a1b1  = gf64_mul(a_hi, b_hi);
    let cross = gf64_add(gf64_mul(a_lo, b_hi), gf64_mul(a_hi, b_lo));
    let beta64 = vec2<u32>(2u, 0u);

    let res_lo = gf64_add(a0b0, gf64_mul(a1b1, beta64));
    let res_hi = gf64_add(cross, a1b1);

    return vec4<u32>(res_lo.x, res_lo.y, res_hi.x, res_hi.y);
}

fn gf128_sqr(a: vec4<u32>) -> vec4<u32> {
    return gf128_mul(a, a);
}

// ── GF(2^128) a^(2^n) : n 회 제곱 ───────────────────────────────────────────
fn gf128_pow2n(a: vec4<u32>, n: u32) -> vec4<u32> {
    var t = a;
    for (var i = 0u; i < n; i = i + 1u) {
        t = gf128_sqr(t);
    }
    return t;
}

// ── GF(2^128) Fermat 역원: a^(2^128-2) ──────────────────────────────────────
// Addition chain: 2^127-1 = all-ones 127비트 Mersenne수
// 1) a^(2^k - 1) 계열을 k = 1,2,4,8,16,32,64 로 build
// 2) 부분 조립으로 a^(2^127-1) 계산
// 3) 한 번 더 제곱하면 a^(2^128-2)
fn gf128_inv(a: vec4<u32>) -> vec4<u32> {
    // a^(2^k - 1) 를 doubling 으로 구축
    let t1  = a;                                              // a^(2^1 - 1)
    let t2  = gf128_mul(gf128_sqr(t1), t1);                  // a^(2^2 - 1)
    let t4  = gf128_mul(gf128_pow2n(t2, 2u), t2);            // a^(2^4 - 1)
    let t8  = gf128_mul(gf128_pow2n(t4, 4u), t4);            // a^(2^8 - 1)
    let t16 = gf128_mul(gf128_pow2n(t8, 8u), t8);            // a^(2^16 - 1)
    let t32 = gf128_mul(gf128_pow2n(t16, 16u), t16);         // a^(2^32 - 1)
    let t64 = gf128_mul(gf128_pow2n(t32, 32u), t32);         // a^(2^64 - 1)

    // 비멱 부분: a^(2^(2^k - 1) - 1) 를 조립
    let t3   = gf128_mul(gf128_sqr(t2), t1);                 // a^(2^3 - 1)
    let t7   = gf128_mul(gf128_pow2n(t4, 3u), t3);           // a^(2^7 - 1)
    let t15  = gf128_mul(gf128_pow2n(t8, 7u), t7);           // a^(2^15 - 1)
    let t31  = gf128_mul(gf128_pow2n(t16, 15u), t15);        // a^(2^31 - 1)
    let t63  = gf128_mul(gf128_pow2n(t32, 31u), t31);        // a^(2^63 - 1)
    let t127 = gf128_mul(gf128_pow2n(t64, 63u), t63);        // a^(2^127 - 1)

    return gf128_sqr(t127);                                   // a^(2^128 - 2)
}

// ── 상수 ──────────────────────────────────────────────────────────────────────
const GF128_ZERO: vec4<u32> = vec4<u32>(0u, 0u, 0u, 0u);
const GF128_ONE:  vec4<u32> = vec4<u32>(1u, 0u, 0u, 0u);
const GF128_ALPHA: vec4<u32> = vec4<u32>(2u, 0u, 0u, 0u); // α = GF2_128::from(2)
