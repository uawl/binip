//! Fiat-Shamir transcript backed by the Vision-4 sponge.

use field::{gf2_64::GF2_64, gf2_128::GF2_128};

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
pub struct VisionTranscript {
    sponge: VisionSponge,
}

impl VisionTranscript {
    pub fn new() -> Self {
        Self {
            sponge: VisionSponge::new(),
        }
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
        buf[..8].copy_from_slice(&v.lo.0.to_le_bytes());
        buf[8..].copy_from_slice(&v.hi.0.to_le_bytes());
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
        GF2_128::new(GF2_64(lo), GF2_64(hi))
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
        assert!(c.lo.0 != 0 || c.hi.0 != 0);
    }

    #[test]
    fn test_different_absorb_different_challenge() {
        let mut t1 = VisionTranscript::new();
        t1.absorb_field(GF2_128::new(GF2_64(1), GF2_64(0)));
        let c1 = t1.squeeze_challenge();

        let mut t2 = VisionTranscript::new();
        t2.absorb_field(GF2_128::new(GF2_64(2), GF2_64(0)));
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
}
