//! Vision-4 duplex sponge.
//!
//! Construction mirrors binius64 `digest.rs`:
//! - State: [u128; 4]
//! - Rate:  2 × u128 = 32 bytes  (elements 0 and 1)
//! - Cap:   2 × u128 = 32 bytes  (elements 2 and 3)
//! - Absorb: *overwrite* (not XOR) the rate elements from input, then permute
//! - Padding: Keccak-style — first byte ORed with 0x80, last byte ORed with 0x01
//! - Squeeze: copy rate elements from state as little-endian bytes

use crate::vision4::permutation;

const RATE_BYTES: usize = 32; // 2 × 16 bytes

pub struct VisionSponge {
    pub(crate) state: [u128; 4],
    pub(crate) buf: [u8; RATE_BYTES],
    pub(crate) filled: usize, // bytes written into buf
}

impl VisionSponge {
    pub fn new() -> Self {
        Self {
            state: [0u128; 4],
            buf: [0u8; RATE_BYTES],
            filled: 0,
        }
    }

    /// Absorb arbitrary bytes into the sponge.
    pub fn absorb(&mut self, mut data: &[u8]) {
        while !data.is_empty() {
            let space = RATE_BYTES - self.filled;
            let take = space.min(data.len());
            self.buf[self.filled..self.filled + take].copy_from_slice(&data[..take]);
            self.filled += take;
            data = &data[take..];

            if self.filled == RATE_BYTES {
                self.permute_no_pad();
            }
        }
    }

    /// Finalise (pad + permute) and return 32 squeezed bytes.
    pub fn squeeze(mut self) -> [u8; RATE_BYTES] {
        // Keccak-style padding: first byte |= 0x80, last byte |= 0x01
        self.buf[self.filled] |= 0x80;
        self.buf[RATE_BYTES - 1] |= 0x01;
        self.permute_no_pad();
        // Output rate elements as little-endian bytes
        let mut out = [0u8; RATE_BYTES];
        out[..16].copy_from_slice(&self.state[0].to_le_bytes());
        out[16..].copy_from_slice(&self.state[1].to_le_bytes());
        out
    }

    // Overwrite rate part of state from buf, then permute. Resets buf and filled.
    fn permute_no_pad(&mut self) {
        self.state[0] = u128::from_le_bytes(self.buf[..16].try_into().unwrap());
        self.state[1] = u128::from_le_bytes(self.buf[16..].try_into().unwrap());
        self.buf = [0u8; RATE_BYTES];
        self.filled = 0;
        permutation(&mut self.state);
    }
}

impl Default for VisionSponge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_message_deterministic() {
        let out1 = VisionSponge::new().squeeze();
        let out2 = VisionSponge::new().squeeze();
        assert_eq!(out1, out2);
        // Must not be all zeros
        assert_ne!(out1, [0u8; RATE_BYTES]);
    }

    #[test]
    fn test_different_inputs_different_outputs() {
        let mut s1 = VisionSponge::new();
        s1.absorb(&[0u8; 32]);
        let out1 = s1.squeeze();

        let mut s2 = VisionSponge::new();
        s2.absorb(&[1u8; 32]);
        let out2 = s2.squeeze();

        assert_ne!(out1, out2);
    }

    #[test]
    fn test_multi_block_same_as_single() {
        let data = [0xABu8; 64];

        let mut s1 = VisionSponge::new();
        s1.absorb(&data);
        let out1 = s1.squeeze();

        let mut s2 = VisionSponge::new();
        s2.absorb(&data[..32]);
        s2.absorb(&data[32..]);
        let out2 = s2.squeeze();

        assert_eq!(out1, out2);
    }
}
