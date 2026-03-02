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
    k: usize,
    n_red: usize,        // = 3k
    gen: Vec<GF2_128>,   // k × n_red, row-major: gen[i * n_red + j]
}

impl LinearCode {
    /// Build the code for input length `k`.  O(k²) construction.
    pub fn new(k: usize) -> Self {
        let n_red = 3 * k;
        let mut domain = b"binip:pcs:gen:k=".to_vec();
        domain.extend_from_slice(&(k as u64).to_le_bytes());
        let gen = xof_gf128(k * n_red, &domain);
        Self { k, n_red, gen }
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
                v = v + x[i] * self.gen[i * self.n_red + j];
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
