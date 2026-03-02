//! Rate-1/4 `[4k, k, 3k+1]` linear code over GF(2^128) via additive FFT.
//!
//! Tensor PCS는 각 row를 선형 코드로 인코딩한다.  입력이 MLE evaluation이든
//! 아니든 상관없이, `encode`가 **GF(2^128)-선형 사상**이고 최소 거리가 충분하면 된다.
//!
//! 내부적으로 additive FFT를 사용한다:
//!   1. 입력 벡터 `x[0..k]`를 novel polynomial basis 계수로 *간주*하여
//!   2. GF(2)-선형 부분공간의 4k 점에서 평가한다.
//!
//! 이 사상은 선형이고, 생성되는 코드의 최소 거리는 `3k+1` (MDS)이다.
//! 인코딩 복잡도는 행당 `O(k log k)` — 이전 dense matrix `O(k²)` 대비 개선.

use field::{FieldElem, GF2_128};

pub struct LinearCode {
  pub(crate) k: usize,
  log_n: usize,        // log2(4k)
  gamma: Vec<GF2_128>, // gamma[i] = s_i(β_i), one per FFT level
}

impl LinearCode {
  /// Build the code for input length `k`.  `k` must be a power of 2.
  ///
  /// Construction is `O(m²)` where `m = log2(4k)` — negligible.
  pub fn new(k: usize) -> Self {
    assert!(k.is_power_of_two(), "k must be a power of 2");
    let log_n = (4 * k).trailing_zeros() as usize; // log2(4k)

    // Generate basis elements β_0..β_{m-1} deterministically via Blake3 XOF.
    let basis = generate_basis(log_n);

    // Compute γ_i = s_i(β_i) iteratively.
    //   s_0(x) = x                               → γ_0 = β_0
    //   s_{j+1}(x) = s_j(x)² + γ_j · s_j(x)    → γ_i via composition
    let mut gamma = Vec::with_capacity(log_n);
    for i in 0..log_n {
      let mut v = basis[i]; // s_0(β_i) = β_i
      for j in 0..i {
        v = v * v + gamma[j] * v; // s_{j+1}(β_i) = s_j(β_i)² + γ_j · s_j(β_i)
      }
      assert!(!v.is_zero(), "FFT basis element {i} is linearly dependent");
      gamma.push(v);
    }

    Self { k, log_n, gamma }
  }

  /// Output length.
  pub fn n_enc(&self) -> usize {
    4 * self.k
  }

  /// Encode `x` (length `k`) → codeword (length `4k`).
  ///
  /// `x`를 novel polynomial basis 계수로 간주, 4k 점에서의 평가값을 반환.
  /// **선형 사상**이므로 Tensor PCS의 row 인코딩으로 사용 가능.
  ///
  /// Complexity: `O(k log k)` field operations.
  pub fn encode(&self, x: &[GF2_128]) -> Vec<GF2_128> {
    assert_eq!(x.len(), self.k);
    let n = 1usize << self.log_n; // = 4k
    let mut data = vec![GF2_128::zero(); n];
    data[..self.k].copy_from_slice(x);

    fft_butterfly(&mut data, self.log_n, &self.gamma);
    data
  }
}

// ─── Additive FFT ────────────────────────────────────────────────────────────

/// In-place radix-2 additive FFT (novel polynomial basis).
///
/// Evaluates a polynomial given as `2^log_n` novel-basis coefficients
/// at all points of the linear subspace `V_{log_n} = span{β_0, …, β_{log_n-1}}`.
fn fft_butterfly(data: &mut [GF2_128], log_n: usize, gamma: &[GF2_128]) {
  // Iterative bottom-up: level 0 (size-2 butterflies) up to level log_n-1.
  for level in 0..log_n {
    let block = 1usize << (level + 1); // butterfly block size
    let half = block / 2;
    let g = gamma[level];

    let mut start = 0;
    while start < data.len() {
      for i in 0..half {
        let b = data[start + half + i];
        data[start + half + i] = data[start + i] + g * b;
      }
      start += block;
    }
  }
}

// ─── Basis generation ────────────────────────────────────────────────────────

/// Generate `m` elements of GF(2^128) deterministically via Blake3 XOF.
///
/// Linear independence over GF(2) is verified during γ computation (caller
/// asserts γ_i ≠ 0).
fn generate_basis(m: usize) -> Vec<GF2_128> {
  let mut hasher = blake3::Hasher::new();
  hasher.update(b"binip:fft:basis");
  let mut reader = hasher.finalize_xof();

  let mut basis = Vec::with_capacity(m);
  let mut buf = [0u8; 16];

  for _ in 0..m {
    reader.fill(&mut buf);
    let lo = u64::from_le_bytes(buf[0..8].try_into().unwrap());
    let hi = u64::from_le_bytes(buf[8..16].try_into().unwrap());
    basis.push(GF2_128::new(lo, hi));
  }
  basis
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
  fn deterministic_encoding() {
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
  fn nonzero_input_has_high_weight() {
    // A nonzero polynomial of degree < k evaluated at 4k points
    // should have at most k-1 zeros (MDS: weight ≥ 3k+1).
    let k = 8;
    let code = LinearCode::new(k);
    let mut x = vec![GF2_128::zero(); k];
    x[0] = GF2_128::one(); // constant polynomial
    let enc = code.encode(&x);
    let zeros = enc.iter().filter(|&&v| v == GF2_128::zero()).count();
    assert!(zeros <= k - 1, "too many zeros ({zeros}) for MDS code");
  }
}
