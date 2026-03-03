pub mod gf2_128;
pub mod traits;

pub use gf2_128::GF2_128;
pub use traits::FieldElem;

/// Batch-invert a slice of field elements using Montgomery's trick.
///
/// Returns `out[i] = elems[i]⁻¹` for all non-zero elements.
/// Panics if any element is zero.
///
/// Cost: 3·N multiplications + 1 inversion (vs. N inversions naively).
pub fn batch_inv<F: FieldElem>(elems: &[F]) -> Vec<F> {
  let n = elems.len();
  if n == 0 {
    return vec![];
  }
  // prefix[i] = elems[0] * elems[1] * … * elems[i]
  let mut prefix = Vec::with_capacity(n);
  prefix.push(elems[0]);
  for i in 1..n {
    prefix.push(prefix[i - 1] * elems[i]);
  }
  // Single inversion of the product of all elements.
  let mut inv_acc = prefix[n - 1].inv();
  // Allocate without zero-init: every slot is written exactly once.
  let mut out: Vec<F> = Vec::with_capacity(n);
  unsafe {
    let ptr = out.as_mut_ptr();
    for i in (1..n).rev() {
      ptr.add(i).write(inv_acc * prefix[i - 1]);
      inv_acc = inv_acc * elems[i];
    }
    ptr.write(inv_acc);
    out.set_len(n);
  }
  out
}
