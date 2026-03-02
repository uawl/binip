//! GF(2^128) arithmetic using the GCM/GHASH irreducible polynomial
//! `p(x) = x^128 + x^7 + x^2 + x + 1`.
//!
//! Used internally by the Vision-4 hash permutation.
//! Elements are represented as `u128` (bit 0 = coefficient of x^0).

// ─── Carry-less multiply ──────────────────────────────────────────────────────

/// Carry-less multiply two 128-bit values → 256-bit result `(lo, hi)`.
#[inline]
pub fn clmul(a: u128, b: u128) -> (u128, u128) {
    let mut lo: u128 = 0;
    let mut hi: u128 = 0;
    for i in 0..128u32 {
        if (b >> i) & 1 == 1 {
            lo ^= a << i;
            if i > 0 {
                hi ^= a >> (128 - i);
            }
        }
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

/// GCM field squaring (Frobenius: spread bits, then reduce).
#[inline]
pub fn gcm_sq(a: u128) -> u128 {
    // Bit i of a → bit 2i of the 256-bit square.
    // Low half (from bits 0..63 → bits 0..127):
    let mut lo: u128 = 0;
    for i in 0u32..64 {
        if (a >> i) & 1 == 1 {
            lo |= 1u128 << (2 * i);
        }
    }
    // High half (from bits 64..127 → bits 128..255):
    let mut hi: u128 = 0;
    for i in 0u32..64 {
        if (a >> (i + 64)) & 1 == 1 {
            hi |= 1u128 << (2 * i);
        }
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
}
