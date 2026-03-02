//! Multilinear equality polynomial utilities.
//!
//! Convention (matches MlePoly in `poly` crate):
//!   index `e` encodes binary vector **b** where `b[k] = (e >> k) & 1`.
//!
//!   eq(r, e) = Π_k  ( r[k] * b[k]  +  (1 + r[k]) * (1 + b[k]) )
//!
//! In char-2: 1 - x = 1 + x, so:
//!   eq_k(r_k, 0) = 1 + r_k
//!   eq_k(r_k, 1) = r_k

use field::{gf2_128::GF2_128, FieldElem};

/// Compute eq(r, ·) at all 2^n binary inputs.
/// Returns a vector of length 2^|r|.
pub fn eq_evals(r: &[GF2_128]) -> Vec<GF2_128> {
    let n = r.len();
    let mut evals = vec![GF2_128::zero(); 1usize << n];
    evals[0] = GF2_128::one();
    let mut cur = 1usize;
    for &ri in r {
        let one_minus_ri = GF2_128::one() + ri;
        // Expand existing entries in reverse to avoid overwrite
        for i in (0..cur).rev() {
            let v = evals[i];
            evals[2 * i + 1] = v * ri;
            evals[2 * i]     = v * one_minus_ri;
        }
        cur *= 2;
    }
    evals
}

/// Inner product: Σ_i a[i] * b[i] over GF(2^128).
pub fn inner_product(a: &[GF2_128], b: &[GF2_128]) -> GF2_128 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(GF2_128::zero(), |acc, (&x, &y)| acc + x * y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use field::FieldElem;

    #[test]
    fn eq_evals_sums_to_one() {
        // Σ_x eq(r, x) = 1 for any r (by the MLE identity Σ eq = 1).
        let r = vec![GF2_128::from(3u64), GF2_128::from(7u64), GF2_128::from(11u64)];
        let evals = eq_evals(&r);
        let sum = evals.iter().fold(GF2_128::zero(), |acc, &x| acc + x);
        assert_eq!(sum, GF2_128::one());
    }

    #[test]
    fn eq_evals_orthogonality() {
        // eq(r, e) == 1 when r is the "binary" point equal to e
        let r = vec![GF2_128::one(), GF2_128::zero(), GF2_128::one()]; // binary point b=5
        let evals = eq_evals(&r);
        let target = 0b101usize; // b[0]=1, b[1]=0, b[2]=1 → index 5
        for (i, &v) in evals.iter().enumerate() {
            if i == target {
                assert_eq!(v, GF2_128::one(), "eq(r, {i}) should be 1");
            } else {
                assert_eq!(v, GF2_128::zero(), "eq(r, {i}) should be 0");
            }
        }
    }
}
