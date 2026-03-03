//! Blake3-based Fiat-Shamir transcript for external (top-level) use.
//!
//! This transcript uses Blake3 in XOF mode for domain-separated challenge
//! derivation.  It supports [`fork()`](Blake3Transcript::fork) to create
//! independent sub-transcripts for parallel shard proofs.

use field::GF2_128;

use crate::transcript::Transcript;

/// Blake3-backed Fiat-Shamir transcript.
///
/// Used as the **external** (top-level) transcript.  In-circuit transcripts
/// use [`VisionTranscript`](crate::VisionTranscript) instead.
#[derive(Clone)]
pub struct Blake3Transcript {
  hasher: blake3::Hasher,
}

impl Blake3Transcript {
  /// Create a new transcript with the domain separator `"binip:transcript"`.
  pub fn new() -> Self {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"binip:transcript:");
    Self { hasher }
  }

  /// Create an independent sub-transcript for shard `idx` under `label`.
  ///
  /// The fork absorbs the current transcript state, the label, and the shard
  /// index, producing a child transcript whose challenge stream is
  /// cryptographically independent of (a) the parent and (b) every other
  /// fork with a different `(label, idx)` pair.
  ///
  /// ```text
  /// child_state = Blake3(parent_state ‖ "binip:fork:" ‖ label ‖ ":" ‖ idx_le)
  /// ```
  pub fn fork(&self, label: &str, idx: u32) -> Self {
    // Finalize current state into 32 bytes to snapshot the transcript.
    let parent_digest = self.hasher.finalize();

    let mut child = blake3::Hasher::new();
    child.update(parent_digest.as_bytes());
    child.update(b"binip:fork:");
    child.update(label.as_bytes());
    child.update(b":");
    child.update(&idx.to_le_bytes());
    Self { hasher: child }
  }
}

impl Default for Blake3Transcript {
  fn default() -> Self {
    Self::new()
  }
}

impl Transcript for Blake3Transcript {
  fn absorb_bytes(&mut self, data: &[u8]) {
    // Length-prefix to prevent ambiguous concatenation.
    self.hasher.update(&(data.len() as u64).to_le_bytes());
    self.hasher.update(data);
  }

  fn absorb_field(&mut self, v: GF2_128) {
    let mut buf = [0u8; 16];
    buf[..8].copy_from_slice(&v.lo.to_le_bytes());
    buf[8..].copy_from_slice(&v.hi.to_le_bytes());
    // Fixed-size; no length prefix needed.
    self.hasher.update(&buf);
  }

  fn absorb_fields(&mut self, elems: &[GF2_128]) {
    // GF2_128 is #[repr(C)] { lo: u64, hi: u64 }.  On little-endian the
    // in-memory byte layout matches absorb_field's encoding, so we can
    // feed the entire slice as one contiguous update.  This lets blake3
    // use its multi-chunk AVX2/AVX-512 paths instead of per-element
    // SSE4.1 compress_in_place.
    #[cfg(target_endian = "little")]
    {
      // SAFETY: GF2_128 is #[repr(C)] with two u64 fields, no padding.
      let bytes = unsafe {
        std::slice::from_raw_parts(
          elems.as_ptr() as *const u8,
          elems.len() * std::mem::size_of::<GF2_128>(),
        )
      };
      self.hasher.update(bytes);
    }
    #[cfg(not(target_endian = "little"))]
    {
      for &v in elems {
        self.absorb_field(v);
      }
    }
  }

  fn squeeze_challenge(&mut self) -> GF2_128 {
    // Finalize (non-destructive) → 32 bytes via XOF, take first 16.
    let digest = self.hasher.finalize();
    let bytes = digest.as_bytes();

    let lo = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    let hi = u64::from_le_bytes(bytes[8..16].try_into().unwrap());

    // Feed challenge back into the hasher so subsequent squeezes differ.
    self.hasher.update(digest.as_bytes());

    GF2_128::new(lo, hi)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn empty_squeeze_nonzero() {
    let mut t = Blake3Transcript::new();
    let c = t.squeeze_challenge();
    assert!(c.lo != 0 || c.hi != 0);
  }

  #[test]
  fn different_absorb_different_challenge() {
    let mut t1 = Blake3Transcript::new();
    t1.absorb_field(GF2_128::from(1u64));
    let c1 = t1.squeeze_challenge();

    let mut t2 = Blake3Transcript::new();
    t2.absorb_field(GF2_128::from(2u64));
    let c2 = t2.squeeze_challenge();

    assert_ne!(c1, c2);
  }

  #[test]
  fn same_absorb_same_challenge() {
    let mut t1 = Blake3Transcript::new();
    t1.absorb_bytes(b"hello");
    let c1 = t1.squeeze_challenge();

    let mut t2 = Blake3Transcript::new();
    t2.absorb_bytes(b"hello");
    let c2 = t2.squeeze_challenge();

    assert_eq!(c1, c2);
  }

  #[test]
  fn multiple_squeezes_distinct() {
    let mut t = Blake3Transcript::new();
    t.absorb_bytes(b"test");
    let c1 = t.squeeze_challenge();
    let c2 = t.squeeze_challenge();
    assert_ne!(c1, c2);
  }

  #[test]
  fn fork_produces_different_challenges() {
    let mut parent = Blake3Transcript::new();
    parent.absorb_bytes(b"root");

    let mut f0 = parent.fork("shard", 0);
    let mut f1 = parent.fork("shard", 1);

    let c0 = f0.squeeze_challenge();
    let c1 = f1.squeeze_challenge();

    // Different shard indices must yield different challenges.
    assert_ne!(c0, c1);
  }

  #[test]
  fn fork_different_labels_differ() {
    let mut parent = Blake3Transcript::new();
    parent.absorb_bytes(b"root");

    let mut fa = parent.fork("alpha", 0);
    let mut fb = parent.fork("beta", 0);

    assert_ne!(fa.squeeze_challenge(), fb.squeeze_challenge());
  }

  #[test]
  fn fork_does_not_mutate_parent() {
    let mut parent = Blake3Transcript::new();
    parent.absorb_bytes(b"data");

    let before = parent.squeeze_challenge();

    // Re-create an identical parent (fork is non-destructive).
    let mut parent2 = Blake3Transcript::new();
    parent2.absorb_bytes(b"data");
    let _ = parent2.fork("shard", 42);
    let after = parent2.squeeze_challenge();

    assert_eq!(before, after);
  }

  #[test]
  fn fork_deterministic() {
    let mut p1 = Blake3Transcript::new();
    p1.absorb_bytes(b"seed");
    let mut f1 = p1.fork("shard", 7);

    let mut p2 = Blake3Transcript::new();
    p2.absorb_bytes(b"seed");
    let mut f2 = p2.fork("shard", 7);

    assert_eq!(f1.squeeze_challenge(), f2.squeeze_challenge());
  }
}
