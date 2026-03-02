//! GF(2^128) arithmetic using the GCM/GHASH irreducible polynomial
//! `p(x) = x^128 + x^7 + x^2 + x + 1`.
//!
//! Used internally by the Vision-4 hash permutation.
//! Elements are represented as `u128` (bit 0 = coefficient of x^0).

// ─── Carry-less multiply ──────────────────────────────────────────────────────

/// Carry-less multiply two 128-bit values → 256-bit result `(lo, hi)`.
///
/// Uses a 4-bit windowed Horner scheme: precompute `a * {0..15}`, then
/// process `b` in 32 nibbles from MSB to LSB.  This gives ~4× fewer
/// iterations than the naive bit-by-bit loop.
#[inline]
pub fn clmul(a: u128, b: u128) -> (u128, u128) {
    // Precompute a * nibble_value for all 4-bit values.
    // Results are at most 131 bits (128 + 3).
    let mut t_lo = [0u128; 16];
    let mut t_hi = [0u128; 16];
    t_lo[1] = a;
    t_lo[2] = a << 1;  t_hi[2] = a >> 127;
    t_lo[4] = a << 2;  t_hi[4] = a >> 126;
    t_lo[8] = a << 3;  t_hi[8] = a >> 125;
    // Fill composites by XOR of power-of-two entries.
    for i in [3u8, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15] {
        let msb = 1u8 << (7 - i.leading_zeros());
        let rest = i ^ msb;
        t_lo[i as usize] = t_lo[msb as usize] ^ t_lo[rest as usize];
        t_hi[i as usize] = t_hi[msb as usize] ^ t_hi[rest as usize];
    }

    let mut lo = 0u128;
    let mut hi = 0u128;
    // Horner: process nibbles of b from MSB (nibble 31) to LSB (nibble 0).
    for nib in (0..32u32).rev() {
        // Shift 256-bit accumulator left by 4.
        hi = (hi << 4) | (lo >> 124);
        lo <<= 4;
        // XOR with a * current_nibble.
        let idx = ((b >> (nib * 4)) & 0xF) as usize;
        lo ^= t_lo[idx];
        hi ^= t_hi[idx];
    }
    (lo, hi)
}

// ─── Reduction mod p(x) = x^128 + x^7 + x^2 + x + 1 ─────────────────────────

/// Reduce a 256-bit value `(lo, hi)` modulo `p(x)`.
///
/// Uses x^128 ≡ x^7 + x^2 + x + 1 (mod p).
/// hi contributes bits at x^128..x^255; each hi-bit at x^{128+i} is replaced
/// by (x^7+x^2+x+1) * x^i.
#[inline]
pub fn reduce(lo: u128, hi: u128) -> u128 {
    // m_lo = low 128 bits of hi*(x^7+x^2+x+1)
    let m_lo = (hi << 7) ^ (hi << 2) ^ (hi << 1) ^ hi;
    // Overflow bits beyond bit-127 from the four shifts:
    let t = (hi >> 121) ^ (hi >> 126) ^ (hi >> 127);  // ≤ 7 bits
    // Reduce the overflow (≤ 7 bits): t*(x^7+x^2+x+1) fits in 14 bits, no more overflow.
    lo ^ m_lo ^ (t << 7) ^ (t << 2) ^ (t << 1) ^ t
}

// ─── Public field operations ──────────────────────────────────────────────────

/// GCM field multiplication.
#[inline]
pub fn gcm_mul(a: u128, b: u128) -> u128 {
    let (lo, hi) = clmul(a, b);
    reduce(lo, hi)
}

// ─── Spread table for fast squaring ───────────────────────────────────────────

/// Map a single byte to its "spread" form: bit i → bit 2i (16-bit result).
const fn spread_byte(b: u8) -> u16 {
    let mut r = 0u16;
    let mut i = 0u32;
    while i < 8 {
        if b & (1 << i) != 0 {
            r |= 1 << (2 * i);
        }
        i += 1;
    }
    r
}

const SPREAD: [u16; 256] = {
    let mut t = [0u16; 256];
    let mut i = 0usize;
    while i < 256 {
        t[i] = spread_byte(i as u8);
        i += 1;
    }
    t
};

/// GCM field squaring (Frobenius: spread bits, then reduce).
///
/// Uses a compile-time spread table for O(16) lookups instead of O(128) bit tests.
#[inline]
pub fn gcm_sq(a: u128) -> u128 {
    let bytes = a.to_le_bytes();
    // Low 8 bytes (bits 0..63) → even positions 0..126 of lo.
    let mut lo = 0u128;
    for i in 0..8 {
        lo |= (SPREAD[bytes[i] as usize] as u128) << (16 * i);
    }
    // High 8 bytes (bits 64..127) → even positions 0..126 of hi.
    let mut hi = 0u128;
    for i in 0..8 {
        hi |= (SPREAD[bytes[i + 8] as usize] as u128) << (16 * i);
    }
    reduce(lo, hi)
}

/// GCM field inversion via Fermat: a^(2^128-2).
/// Returns 0 for input 0.
#[inline]
pub fn gcm_inv(a: u128) -> u128 {
    if a == 0 {
        return 0;
    }
    // Exponent = 2^128-2 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFE
    // Binary: 127 ones then a zero.
    // Square-and-multiply from bit 127 down to bit 0.
    let mut r = a;
    // Start from bit 126 (bit 127 = 1 already handled by r=a).
    for _i in 0..126 {
        r = gcm_sq(r);
        r = gcm_mul(r, a);
    }
    // Final squaring for the trailing zero bit:
    gcm_sq(r)
}

/// Batch-invert 4 elements in place (Montgomery's trick, cost ≈ 1 inv + 8 mul).
/// Zeros are mapped to zeros.
#[inline]
pub fn batch_inv4(s: &mut [u128; 4]) {
    // Prefix products
    let p01 = gcm_mul(s[0], s[1]);
    let p23 = gcm_mul(s[2], s[3]);
    let p0123 = gcm_mul(p01, p23);

    let inv_p0123 = gcm_inv(p0123);

    // Handle the all-zero degenerate case (extremely unlikely in practice).
    if inv_p0123 == 0 {
        for x in s.iter_mut() {
            *x = gcm_inv(*x);
        }
        return;
    }

    let inv_p01 = gcm_mul(inv_p0123, p23);
    let inv_p23 = gcm_mul(inv_p0123, p01);

    let inv0 = gcm_mul(inv_p01, s[1]);
    let inv1 = gcm_mul(inv_p01, s[0]);
    let inv2 = gcm_mul(inv_p23, s[3]);
    let inv3 = gcm_mul(inv_p23, s[2]);

    s[0] = inv0;
    s[1] = inv1;
    s[2] = inv2;
    s[3] = inv3;
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduce_x128_is_x7_x2_x1_1() {
        // x^128 → hi=1, lo=0
        let r = reduce(0, 1);
        assert_eq!(r, (1 << 7) | (1 << 2) | (1 << 1) | 1, "x^128 should reduce to x^7+x^2+x+1");
    }

    #[test]
    fn mul_identity() {
        let a = 0xdeadbeefcafebabe1234567890abcdefu128;
        assert_eq!(gcm_mul(a, 1), a);
        assert_eq!(gcm_mul(1, a), a);
    }

    #[test]
    fn mul_commutative() {
        let a = 0x1234567890abcdef1122334455667788u128;
        let b = 0xffeeddccbbaa99887766554433221100u128;
        assert_eq!(gcm_mul(a, b), gcm_mul(b, a));
    }

    #[test]
    fn sq_matches_mul() {
        let a = 0xabcdef1234567890u128;
        assert_eq!(gcm_sq(a), gcm_mul(a, a));
    }

    #[test]
    fn inv_roundtrip() {
        let a = 0x0102030405060708090a0b0c0d0e0f10u128;
        assert_eq!(gcm_mul(a, gcm_inv(a)), 1);
    }

    #[test]
    fn inv_zero_is_zero() {
        assert_eq!(gcm_inv(0), 0);
    }

    #[test]
    fn batch_inv_matches_single() {
        let orig = [
            0x1u128,
            0xdeadbeef00000000u128,
            0xffffffffffffffffffffffffffffffff,
            0xaabbccddeeff0011u128,
        ];
        let mut b = orig;
        batch_inv4(&mut b);
        for i in 0..4 {
            assert_eq!(gcm_mul(orig[i], b[i]), 1, "batch_inv4 failed at index {i}");
        }
    }

    #[test]
    fn clmul_zero() {
        assert_eq!(clmul(0, 0), (0, 0));
        assert_eq!(clmul(0xdeadbeef, 0), (0, 0));
        assert_eq!(clmul(0, 0xdeadbeef), (0, 0));
    }

    #[test]
    fn clmul_by_one() {
        let a = 0x0102030405060708090a0b0c0d0e0f10u128;
        assert_eq!(clmul(a, 1), (a, 0));
        assert_eq!(clmul(1, a), (a, 0));
    }

    #[test]
    fn clmul_commutative() {
        let a = 0x1234567890abcdef1122334455667788u128;
        let b = 0xffeeddccbbaa99887766554433221100u128;
        assert_eq!(clmul(a, b), clmul(b, a));
    }

    #[test]
    fn sq_spread_matches_mul_large() {
        // Verify spread-table squaring against multiplication-based squaring
        // for several values spanning the full 128-bit range.
        let vals = [
            1u128,
            0xdeadbeefcafebabe1234567890abcdefu128,
            0xffffffffffffffffffffffffffffffffu128,
            0x80000000000000000000000000000001u128,
        ];
        for &a in &vals {
            assert_eq!(gcm_sq(a), gcm_mul(a, a), "sq != mul for {a:#x}");
        }
    }

    #[test]
    fn mul_associative() {
        let a = 0x111u128;
        let b = 0x222u128;
        let c = 0x333u128;
        assert_eq!(gcm_mul(gcm_mul(a, b), c), gcm_mul(a, gcm_mul(b, c)));
    }
}
