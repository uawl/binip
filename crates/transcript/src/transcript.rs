//! Fiat-Shamir transcript backed by the Vision-4 sponge.

use field::GF2_128;

use crate::sponge::VisionSponge;

/// Minimal trait expected by provers/verifiers.
pub trait Transcript {
  /// Feed arbitrary bytes into the transcript.
  fn absorb_bytes(&mut self, data: &[u8]);
  /// Feed a field element (16 bytes, little-endian lo||hi).
  fn absorb_field(&mut self, v: GF2_128);
  /// Derive a challenge field element from the current state.
  fn squeeze_challenge(&mut self) -> GF2_128;
}

/// Vision-4 based Fiat-Shamir transcript.
#[derive(Clone)]
pub struct VisionTranscript {
  sponge: VisionSponge,
}

impl VisionTranscript {
  pub fn new() -> Self {
    Self {
      sponge: VisionSponge::new(),
    }
  }

  /// Create a domain-separated child transcript.
  ///
  /// The child absorbs the parent’s current state plus a unique
  /// `(label, idx)` tag, producing a challenge stream that is
  /// cryptographically independent of other forks.
  pub fn fork(&self, label: &str, idx: u32) -> Self {
    let mut child = self.clone();
    child.sponge.absorb(b"binip:fork:");
    child.sponge.absorb(label.as_bytes());
    child.sponge.absorb(b":");
    child.sponge.absorb(&idx.to_le_bytes());
    child
  }
}

impl Default for VisionTranscript {
  fn default() -> Self {
    Self::new()
  }
}

impl Transcript for VisionTranscript {
  fn absorb_bytes(&mut self, data: &[u8]) {
    self.sponge.absorb(data);
  }

  fn absorb_field(&mut self, v: GF2_128) {
    // Serialize as lo (8 bytes LE) || hi (8 bytes LE)
    let mut buf = [0u8; 16];
    buf[..8].copy_from_slice(&v.lo.to_le_bytes());
    buf[8..].copy_from_slice(&v.hi.to_le_bytes());
    self.sponge.absorb(&buf);
  }

  fn squeeze_challenge(&mut self) -> GF2_128 {
    // Finalise current sponge, then create a fresh one seeded with the output.
    // This gives a clean challenge without consuming the transcript irreversibly.
    let saved_state = self.sponge.state;
    let saved_buf = self.sponge.buf;
    let saved_filled = self.sponge.filled;

    // Squeeze from a clone of the current sponge
    let clone = VisionSponge {
      state: saved_state,
      buf: saved_buf,
      filled: saved_filled,
    };
    let out = clone.squeeze();

    // Feed the challenge back so the transcript advances
    self.sponge.absorb(&out);

    let lo = u64::from_le_bytes(out[..8].try_into().unwrap());
    let hi = u64::from_le_bytes(out[8..16].try_into().unwrap());
    GF2_128::new(lo, hi)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_empty_squeeze_nonzero() {
    let mut t = VisionTranscript::new();
    let c = t.squeeze_challenge();
    // Challenge from empty transcript must be non-trivial
    assert!(c.lo != 0 || c.hi != 0);
  }

  #[test]
  fn test_different_absorb_different_challenge() {
    let mut t1 = VisionTranscript::new();
    t1.absorb_field(GF2_128::new(1, 0));
    let c1 = t1.squeeze_challenge();

    let mut t2 = VisionTranscript::new();
    t2.absorb_field(GF2_128::new(2, 0));
    let c2 = t2.squeeze_challenge();

    assert_ne!(c1, c2);
  }

  #[test]
  fn test_same_absorb_same_challenge() {
    let mut t1 = VisionTranscript::new();
    t1.absorb_bytes(b"hello");
    let c1 = t1.squeeze_challenge();

    let mut t2 = VisionTranscript::new();
    t2.absorb_bytes(b"hello");
    let c2 = t2.squeeze_challenge();

    assert_eq!(c1, c2);
  }

  #[test]
  fn test_multiple_challenges_distinct() {
    let mut t = VisionTranscript::new();
    t.absorb_bytes(b"test");
    let c1 = t.squeeze_challenge();
    let c2 = t.squeeze_challenge();
    assert_ne!(c1, c2);
  }

  #[test]
  fn test_fork_produces_different_challenges() {
    let mut parent = VisionTranscript::new();
    parent.absorb_bytes(b"setup");
    let mut f1 = parent.fork("shard", 0);
    let mut f2 = parent.fork("shard", 1);
    assert_ne!(f1.squeeze_challenge(), f2.squeeze_challenge());
  }

  #[test]
  fn test_fork_different_labels_differ() {
    let parent = VisionTranscript::new();
    let mut fa = parent.fork("alpha", 0);
    let mut fb = parent.fork("beta", 0);
    assert_ne!(fa.squeeze_challenge(), fb.squeeze_challenge());
  }

  #[test]
  fn test_fork_does_not_mutate_parent() {
    let mut t = VisionTranscript::new();
    t.absorb_bytes(b"data");
    let c_before = {
      let mut tmp = t.clone();
      tmp.squeeze_challenge()
    };
    let _child = t.fork("x", 42);
    let c_after = {
      let mut tmp = t.clone();
      tmp.squeeze_challenge()
    };
    assert_eq!(c_before, c_after);
  }

  #[test]
  fn test_fork_deterministic() {
    let parent = VisionTranscript::new();
    let mut f1 = parent.fork("test", 7);
    let mut f2 = parent.fork("test", 7);
    assert_eq!(f1.squeeze_challenge(), f2.squeeze_challenge());
  }
}
