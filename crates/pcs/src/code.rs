//! Rate-1/4 systematic linear code over GF(2^128).
//!
//! For input length `k`:
//!   - positions 0..k   : systematic (identity)
//!   - positions k..4k  : `G · x` where G is a k×3k random matrix over GF(2^128)
//!
//! G is derived deterministically from a Blake3 XOF keyed on `k`, so both
//! prover and verifier recover the identical generator matrix.

use field::{gf2_64::GF2_64, gf2_128::GF2_128, FieldElem};

pub struct LinearCode {
    pub(crate) k: usize,
    pub(crate) n_red: usize,        // = 3k
    pub(crate) generated: Vec<GF2_128>,   // k × n_red, row-major: generated[i * n_red + j]
}

impl LinearCode {
    /// Build the code for input length `k`.  O(k²) construction.
    pub fn new(k: usize) -> Self {
        let n_red = 3 * k;
        let mut domain = b"binip:pcs:gen:k=".to_vec();
        domain.extend_from_slice(&(k as u64).to_le_bytes());
        let generated = xof_gf128(k * n_red, &domain);
        Self { k, n_red, generated }
    }

    /// Output length.
    pub fn n_enc(&self) -> usize { 4 * self.k }

    /// Encode `x` (length k) → encoded vector (length 4k).
    pub fn encode(&self, x: &[GF2_128]) -> Vec<GF2_128> {
        assert_eq!(x.len(), self.k);
        let mut out = Vec::with_capacity(4 * self.k);
        // systematic part
        out.extend_from_slice(x);
        // redundant part: for each column j, compute Σ_i x[i] * gen[i*n_red+j]
        for j in 0..self.n_red {
            let mut v = GF2_128::zero();
            for i in 0..self.k {
                v = v + x[i] * self.generated[i * self.n_red + j];
            }
            out.push(v);
        }
        out
    }
}

/// Expand a Blake3 XOF into `n` pseudo-random GF(2^128) elements.
fn xof_gf128(n: usize, domain: &[u8]) -> Vec<GF2_128> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    let mut reader = hasher.finalize_xof();
    let mut bytes = vec![0u8; n * 16];
    reader.fill(&mut bytes);
    bytes
        .chunks_exact(16)
        .map(|b| {
            let lo = u64::from_le_bytes(b[0..8].try_into().unwrap());
            let hi = u64::from_le_bytes(b[8..16].try_into().unwrap());
            GF2_128::new(GF2_64(lo), GF2_64(hi))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use field::FieldElem;

    #[test]
    fn encode_output_length() {
        for k in [1, 2, 4, 8] {
            let code = LinearCode::new(k);
            assert_eq!(code.n_enc(), 4 * k);
            let x: Vec<GF2_128> = (1..=k as u64).map(GF2_128::from).collect();
            let enc = code.encode(&x);
            assert_eq!(enc.len(), 4 * k);
        }
    }

    #[test]
    fn systematic_prefix() {
        let k = 4;
        let code = LinearCode::new(k);
        let x: Vec<GF2_128> = (10..10 + k as u64).map(GF2_128::from).collect();
        let enc = code.encode(&x);
        // first k elements must be identical to input
        assert_eq!(&enc[..k], &x);
    }

    #[test]
    fn linearity() {
        // encode(a + b) == encode(a) + encode(b)
        let k = 4;
        let code = LinearCode::new(k);
        let a: Vec<GF2_128> = (1..=k as u64).map(GF2_128::from).collect();
        let b: Vec<GF2_128> = (100..100 + k as u64).map(GF2_128::from).collect();
        let ab: Vec<GF2_128> = a.iter().zip(&b).map(|(&x, &y)| x + y).collect();

        let enc_a = code.encode(&a);
        let enc_b = code.encode(&b);
        let enc_ab = code.encode(&ab);

        let sum: Vec<GF2_128> = enc_a.iter().zip(&enc_b).map(|(&x, &y)| x + y).collect();
        assert_eq!(enc_ab, sum, "encode must be GF(2^128)-linear");
    }

    #[test]
    fn scalar_homogeneity() {
        // encode(c * x) == c * encode(x)
        let k = 4;
        let code = LinearCode::new(k);
        let c = GF2_128::from(0xDEAD_BEEF_u64);
        let x: Vec<GF2_128> = (1..=k as u64).map(GF2_128::from).collect();
        let cx: Vec<GF2_128> = x.iter().map(|&v| c * v).collect();

        let enc_x = code.encode(&x);
        let enc_cx = code.encode(&cx);

        let scaled: Vec<GF2_128> = enc_x.iter().map(|&v| c * v).collect();
        assert_eq!(enc_cx, scaled);
    }

    #[test]
    fn deterministic_generator() {
        // Two codes with the same k must produce identical encodings.
        let k = 4;
        let code1 = LinearCode::new(k);
        let code2 = LinearCode::new(k);
        let x: Vec<GF2_128> = (0..k as u64).map(GF2_128::from).collect();
        assert_eq!(code1.encode(&x), code2.encode(&x));
    }

    #[test]
    fn encode_zero_is_zero() {
        let k = 4;
        let code = LinearCode::new(k);
        let zero = vec![GF2_128::zero(); k];
        let enc = code.encode(&zero);
        assert!(enc.iter().all(|&v| v == GF2_128::zero()));
    }

    #[test]
    fn different_k_different_generator() {
        // Codes with different k should produce different generators.
        let code4 = LinearCode::new(4);
        let code8 = LinearCode::new(8);
        assert_ne!(code4.generated.len(), code8.generated.len());
    }
}
