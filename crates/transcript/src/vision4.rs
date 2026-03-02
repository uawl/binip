//! Vision-4 hash permutation over GF(2^128) / GHASH field.
//!
//! Implements the Vision-4 permutation from the Binius STARK system.
//! State: 4 × u128 elements; 8 full rounds (16 half-rounds).
//!
//! Each half-round:  batch_invert → B-polynomial → MDS → add_round_constant
//! Alternates B_inv (first half) and B_fwd (second half) per round.
//!
//! Reference: https://github.com/binius-zk/binius64/tree/92ec1563ffec1e1f5f667d07f620abbe57253c04/crates/hash/src/vision_4
//! Copyright (c) 2025 The Binius Developers
//! Copyright (c) 2025 Irreducible, Inc.
//! License: MIT

use super::ghash_field::{batch_inv4, gcm_mul, gcm_sq};

// ─── Layout constants ─────────────────────────────────────────────────────────

pub const M: usize = 4;
pub const NUM_ROUNDS: usize = 8;
/// Rate in u128 words (absorb/squeeze width).
pub const RATE: usize = 2;
/// Capacity in u128 words.
pub const CAP: usize = M - RATE;

// ─── B_fwd polynomial ────────────────────────────────────────────────────────
//
// B_fwd(x) = C0 + C1*x + C2*x^2 + C3*x^4
// (affine linearized polynomial over GF(2^128))

const B_FWD: [u128; 4] = [
  0xb0b849b207a0f1c74c29e4d892ca33dd,
  0xf2240891aac3c3a5855eeb8ce24c9523,
  0xb017e96797ef1fbe9d79908388dd768d,
  0x2bf9f0b94c2b2ceb00dbb86a2cb472bb,
];

// ─── B_inv polynomial ────────────────────────────────────────────────────────
//
// B_inv(x) = C0 + sum_{i=0}^{127} C_{i+1} * x^(2^i)
// (affine linearized polynomial: the inverse of B_fwd)
//
// 129 coefficients: index 0 is the affine constant, indices 1..=128 are Frobenius coefficients.

const B_INV: [u128; 129] = [
  0x1d124529af098017d515f025b2107a04,
  0xb2733aa187f5954a397b9985012327dc,
  0xf9ed22a697f45e3fac28266a3c9c8bf9,
  0xb2563578203d5df23ed37df5274f9b62,
  0x9f60ca3b6e570f9bb35f672c4cc71e78,
  0x2b7839e6c8c6f35c4cfe4dd6ed9988bf,
  0xd7e470d53c8a1441335c0e39f8e825ba,
  0xc2a0d87c2f6de7f3656d51ee118d93ca,
  0xb80cb728430ce418d8763ad073d43c72,
  0xf1424d16f8ff5ad20c172fa0bce84a66,
  0xf27af010f636dd9d8b47d92989d38c91,
  0xeebb7636921fdc8bb35172c4f2904488,
  0xf17407875a9069991c532b85716f8278,
  0x72032c6e2ef39011b80918ec4884682e,
  0xa93dd016787f32dafb792593a1c0664c,
  0x2a72f7f35410abe48bf86284d7153568,
  0x539206fa82bb0fd3fc257acfd85198cf,
  0x6325125ff725e1435f66efa328075c4b,
  0x73abb6e9157b3791bc1a05a15a0ba8e2,
  0x3447131e361d3a430e26535b49f37884,
  0xeeeb383e8b2e366b3b5d33b416320285,
  0xa6ef061627c8b7d6b5ea5ddf3b3ceeee,
  0x3f97a996898ed33fb4c0b023284209b7,
  0xbdf02af9a8b31ed299580c95f214d5a3,
  0xfded88f14050f0fc196849074ce0f9ad,
  0x6c8b4d18e1ba055c6b631b36bde8d60c,
  0x1027cad9488ab75d6741c4f5b9026779,
  0x9f1c5193f4762dcadac6b8979fd40e1d,
  0x7af3e53f8f00a454d215ee466e50388c,
  0x5e3227be2a589427f2870f655f41ef5b,
  0xd964ed03293469753b504cb624c60228,
  0x43c3a9698d3135080fe4f57716b03036,
  0x922b2fb49b13518ec3895e0eb37085b0,
  0xaa09df0da87a0fceb212105e5b297b5a,
  0x790c7ca5f5905488e82bd9d875c8715a,
  0x369411c407a9ef883133bc16aceb21a2,
  0xd577819f72b6158c1eca19e57d2e1e95,
  0xac4d176698af651f201efd2cfea05ab6,
  0x268f4c40a4553eb3a2afbde2b10df863,
  0x8749898084983a777debda3686b5e752,
  0xc723199815bae544773110e7054256a0,
  0x960b23bc674dd468d75ff82b164067b4,
  0xb94130b0a65d36009150e505b5e4a818,
  0x742b2c52bba058cc8c9a364c0985136b,
  0xdf298fbb15e6e58f9edb384f5c34673c,
  0x4fdbc95e71fe2bdbcc6bee273caad844,
  0x19d487689fe24e3a3eceabdefdcef9e5,
  0xb01bf42d575ca9a0df60e3e99d4a4450,
  0xe8a777b48d74755f5091a9d460f80660,
  0x9f435ce2ff626ba06cb1ee4478a1356f,
  0x6804034746f7cca00180116ebfe5b33a,
  0xaef4cc5aa39d8f6957c15f9b173601eb,
  0x97d15c1e09df82180a062c5268804182,
  0x3109fb44aa8caba5fb41f60a49ab9f99,
  0x67ff3e7094682c2c7ae9258280b00f7d,
  0xbf6a8f87c80acf09f9456b27b9c44a04,
  0x4cd2646e793fd4eab1319cb89fb72054,
  0x076e92092b1f43709e7b9b2856ed53a9,
  0xb987929f99110f41f4f44c08b778e929,
  0x09b1c58023e9c9c267451053eed4b989,
  0xb27b2fa5789c2749058ebed7e489d0a1,
  0xcf9a22da173d17c475c43a59ed45535f,
  0x77df29bf03c18bbd103431cf69e35840,
  0x2a746d8aec21318be6985ff6dfa5d116,
  0x3411ccb789ac3dfc5bbd1c8b967559f6,
  0xd6faab5c3e02915dc279b22dbbf81919,
  0x2b9a29fd7e4775061fbb90c2033f304d,
  0x8a78a5986d399576aade221e1a4a4759,
  0x8cc9cb6657270a712b2c776b7f231214,
  0xa4206fef28f9fad8b05707b515fa176b,
  0x71984ef63a6a49ddefa6b9cb6b8832c4,
  0x2347fdcc33e95ee2a1bb2c4edc8da6a9,
  0xc63f0aaf33ca8b99a03decedc48b2e94,
  0x3237cb90cca2283888cb5729283a8b4d,
  0xbe8d8bbd445bf3e3e2947257afff2516,
  0xe7524df886b9fd83a2d4bdcfdb252124,
  0x6cb5512f7bdd181f675d059cf05799ff,
  0xc56e3f62ddedbabb35a200087a5640ce,
  0xba0a5c1d5ce94e53f9efb731c6634b44,
  0x4280ae86e53b14eb0202f4c147a6d14a,
  0xb4dffae679af530c397b2d07ac62d79a,
  0xd0311963a7fafcafd3660af96cc461e1,
  0x7d4f16aea2ce0d4e4c8c545d4d0040dc,
  0xda8304eff41a3ae357f1ec94fe9e062b,
  0x1b7c9a9f3666986912070ec52f99f801,
  0x004ccc9ddf6ca951f20c1b0ac31fa008,
  0x14f6a26f68c0ffd76f8f5fd65aa79542,
  0xd31406666953d392634f0e80af43fe0e,
  0x020c264e630f8bf516200218fb1d1a0e,
  0xdd5a2efdde3cae83465d1315d7af0852,
  0x055702e27b6a93676389c452c5e4b41a,
  0xc4684448e291abeeaa305e6c0e6af873,
  0xd2c1419f07a837e617a10b6ac6d1f275,
  0x8c605a36b3ecdd16edfc7c499e550871,
  0xe034129a3ccbfd6590ebaa430dcad339,
  0x5cb6429235829cdcdfc6a221725c755c,
  0x5737eb97111ca259d691ae13f5763345,
  0x755cd247d4267e960075f86302a39702,
  0xb12a9aa970a91097fa8ad2f2bdfb1b20,
  0x0dc789e36c5547c0571e479d7fd9a75f,
  0x919900cb8cbe4e756f19ab79df7196ee,
  0xbebfc69ba65f89dfecd241385602e25f,
  0xcac2967113753aaa7f3d8b4c9df4b9aa,
  0xf97eff95e366110b9b47509e730fd04f,
  0xf939cdd3b4d28d6f3e30aeb60f0346a0,
  0x982a7c68b7e0ae1603e324d5453d2b1d,
  0x1a589c0f209aeca97224c0c9db79142b,
  0x5219c08df0734e2b98dd90d13bc83d8f,
  0x95f1c0b71be50da58a8f752a0724e53b,
  0x8c494d44cb21e64e58401414631f3666,
  0x07b5f194eaa7cf3c96287d6aaedd3051,
  0x12e5008f6b6b3dd1d640a0a72a6c6d42,
  0x3329d9133fa80c1e6aba028e9255256c,
  0xb859b9f70c7d3da6cb8bc0127316f7ad,
  0xc31f1a1e3e98194053a0c5759aba5125,
  0x6f7deb6cd260c52090fa62f4b170cc55,
  0x991089b8d80963ddcf856990b5b55a89,
  0x36ebdacdc017e6cd38a1ccedfa6799d6,
  0x6ee345c84ddb46a42c8139c52ef87729,
  0x31bc991954d18dfb00a0e9f1c8177027,
  0x576056edf9e7f6f349740dffa8adc7d5,
  0xd7861327edc091fa6b27b5447b05255e,
  0x1ad33492132898f582444df642286e8c,
  0xfabef4a4b044b301e6d8a3ed7fd754a2,
  0x03dbd4cca837cf33b42622b75fec035a,
  0xbb8da06ec7ec3a9c2867a57e2cde5cc6,
  0x7f0dfc7829c9ecef117778448e04cc6c,
  0x128da4c873150f44e7e43bf88921c090,
  0xd9bc6edaf1b4a689ba169b5554228af8,
];

// ─── Round constants ──────────────────────────────────────────────────────────
// 17 arrays of 4 u128 each (1 + 2*NUM_ROUNDS).

pub(crate) const ROUND_CONSTANTS: [[u128; 4]; 1 + 2 * NUM_ROUNDS] = [
  [
    0xd0e015190f5e0795f4a1d28234ffdf87,
    0x6111731acd9a89f6b93ec3a23ec7681b,
    0x15da8de707ee3f3918a34a96728b8d29,
    0x2ea920e89fbb13a1ed2a216b1d232bfb,
  ],
  [
    0x52c94b04b178f80fa586c37c798171a3,
    0xf0e02cc3459757dc8ec9259341831bcb,
    0x025c53bf6d6dee09307a55c008868354,
    0x54258ece50ceb1e9b5a8e38e7bbcaa3d,
  ],
  [
    0x77c332c3ebbfbaa73bffaec318055661,
    0x5abbaeac51c6925d2aff59f11d164328,
    0x9f8daadefaebb59d77ad096fb56e6462,
    0x322de9fd29fb135d569895b07925cacf,
  ],
  [
    0x95c521abe74b013381778cf1b46e6cbd,
    0x85fcfe5a641b67ada8700b95fd906f70,
    0x5d61e0714a000a6d491d8e0db8835a16,
    0x3bea1f3ff2c984d47f91ec35358f1e16,
  ],
  [
    0x66b6cd1b70d97c63163197d429553ad3,
    0xd1be1516a276fb5f590b628b4f8db8b9,
    0x60f3fccd56bfad8cb506e2f1adce50af,
    0x205a1e29bd8a90dac5f776c450e88d10,
  ],
  [
    0x2a02ee3e61560ba34c286e943b3d169f,
    0x73e4bfe44495bfbb54f6fd20adedcf8b,
    0x23e8322b8a8cf1d3b681c1b577cee348,
    0x0d892aeb52c9b2d7096f9698ef0951df,
  ],
  [
    0x8571aa085a872bfcd96c3a1ebbe26682,
    0x3ae1e8efdff55ee0cb6d91af315d614b,
    0xdcf3ec9dd466e76de5f9f9e9fd5dbdba,
    0x5242d748a66f9abe17d4ca014c92081f,
  ],
  [
    0x38176034b37ed0b51ea51bade772fd64,
    0xdc10902a22159fdc02bc8fc3c43ee732,
    0x54eafeed45ab9f55c2a668054e1433f2,
    0x18c196024042197b29bba5779e6b4419,
  ],
  [
    0xeeee77f02774996aff2fbdee3cafd5c6,
    0xc09b024ff9e932be0fc9d294001fa6a7,
    0x62c063613d99b026579fa6bf82e24cdb,
    0x87df42121a98610c661fb711f68ee06e,
  ],
  [
    0x40405fdefe20704cda9bccfaa1189228,
    0xb683ec6a7ff2f49ae429c9f0e0e3f518,
    0xa8e472626f047ffe07759441d3d2547f,
    0x8c89cca408b7c95407db5cb1cc572a63,
  ],
  [
    0xcaf9fccf7c70d8a7c2ec4142b9d5397e,
    0x414420fface1c14f77cf760b5f7f980f,
    0x0c8811f0d2dfb0dfe515f581db7c4d9d,
    0x6303211a8c2a68f6103e2f03f7957ca4,
  ],
  [
    0x10c51b39a067af1e593e9c46a1b77d37,
    0x2319f7d203bbf1d50fe342d17066c9d3,
    0x3dd51ba31a2018b8da50ec10ed923e6e,
    0x676c6481332d0803cdea11539da55247,
  ],
  [
    0xcd06f2d74ae291550c7cd3e34c3ca94d,
    0x2aaea85e1163205f340a28bf015b3488,
    0x48e0f19b5886ca1e9ce659ddf1dece57,
    0x4b2ea0d6191010f82491ac06b6cbf7ad,
  ],
  [
    0xb73fb2fc3f8a413ee8806392998452aa,
    0x470372c4c732a0a7e61f858edaaa7f97,
    0x647c2b2751c258135ccd956db51e7ffa,
    0x079e5c6d2ec255a3206f1d231109247b,
  ],
  [
    0x367538f6742ae83251d22a8112bf6fa1,
    0x08539d5a297209d419a8906fee11e0a3,
    0x3de8eec08b29b79911eb2420e4336284,
    0x2fc0e3b4d3ea4de849486e07be859117,
  ],
  [
    0xe25e1dbbcabb7885ae28db51bacbee5e,
    0x9916b64185652326fda0575231b69ed2,
    0x1ff4e4155cbf886110f5545da1f8722d,
    0x4dcd1260ab92fbe0cec3b014367f626e,
  ],
  [
    0x40ae8c84f958d259141595fed91e09cd,
    0xf8222584576d0b5fa41584b91cf5f526,
    0x5c021db63b4a3a27de7d38be3b26c52f,
    0x509c5cb38660fee6ccc20b9abda1189b,
  ],
];

// ─── Polynomial transforms ────────────────────────────────────────────────────

/// B_fwd(x) = C0 + C1*x + C2*x^2 + C3*x^4 (linearized degree-4 polynomial).
#[inline]
pub(crate) fn b_fwd(x: u128) -> u128 {
  let x2 = gcm_sq(x);
  let x4 = gcm_sq(x2);
  B_FWD[0] ^ gcm_mul(B_FWD[1], x) ^ gcm_mul(B_FWD[2], x2) ^ gcm_mul(B_FWD[3], x4)
}

/// B_inv(x) = C0 + sum_{i=0}^{127} C_{i+1} * x^(2^i) (Frobenius series, 128 terms).
#[inline]
pub(crate) fn b_inv(x: u128) -> u128 {
  let mut result = B_INV[0];
  let mut xp = x; // x^(2^i) — starts at x^1
  for i in 0..128 {
    result ^= gcm_mul(B_INV[i + 1], xp);
    xp = gcm_sq(xp);
  }
  result
}

// ─── MDS layer ────────────────────────────────────────────────────────────────
//
// Circulant MDS matrix [2,3,1,1] (and cyclic rotations):
//   a[0]' = 2*a[0] + 3*a[1] + a[2] + a[3]
//   a[1]' = a[0] + 2*a[1] + 3*a[2] + a[3]
//   a[2]' = a[0] + a[1] + 2*a[2] + 3*a[3]
//   a[3]' = 3*a[0] + a[1] + a[2] + 2*a[3]
//
// In GF(2^128): multiply by 2 = mul_x (shift x^{i} → x^{i+1}, then reduce).
// In GF(2^128): multiply by 3 = mul_x(a) XOR a = 2*a + a.

/// Multiply by x (= 2) in GCM GF(2^128): shift left 1 bit and reduce if needed.
#[inline]
fn mul_x(a: u128) -> u128 {
  let overflow = a >> 127;
  let shifted = a << 1;
  // If overflow bit set, XOR with the reduction of x^128 = x^7+x^2+x+1 = 0x87
  // (using the irred poly x^128+x^7+x^2+x+1, so reducing x^128 gives 0x87)
  shifted ^ (overflow.wrapping_neg() & 0x87u128)
}

#[inline]
pub(crate) fn mds(s: &mut [u128; M]) {
  let sum = s[0] ^ s[1] ^ s[2] ^ s[3];
  let a0 = s[0]; // save original s[0] for row-3
  // binius64: a[i] += sum + mul_x(a[i] + a[i+1])  (GF(2): += is ^=)
  // Row 0: 2*s0 + 3*s1 + s2 + s3
  s[0] ^= sum ^ mul_x(s[0] ^ s[1]);
  // Row 1: s0 + 2*s1 + 3*s2 + s3
  s[1] ^= sum ^ mul_x(s[1] ^ s[2]);
  // Row 2: s0 + s1 + 2*s2 + 3*s3
  s[2] ^= sum ^ mul_x(s[2] ^ s[3]);
  // Row 3: 3*s0 + s1 + s2 + 2*s3
  s[3] ^= sum ^ mul_x(s[3] ^ a0);
}

// ─── Full permutation ─────────────────────────────────────────────────────────

/// Apply the Vision-4 permutation to a 4-element state in-place.
pub fn permutation(s: &mut [u128; M]) {
  // Initial round constant addition
  for i in 0..M {
    s[i] ^= ROUND_CONSTANTS[0][i];
  }
  // 8 full rounds (each = 2 half-rounds)
  for r in 0..NUM_ROUNDS {
    // First half-round: invert → B_inv → MDS → constants
    batch_inv4(s);
    for x in s.iter_mut() {
      *x = b_inv(*x);
    }
    mds(s);
    let ci = 1 + 2 * r;
    for i in 0..M {
      s[i] ^= ROUND_CONSTANTS[ci][i];
    }
    // Second half-round: invert → B_fwd → MDS → constants
    batch_inv4(s);
    for x in s.iter_mut() {
      *x = b_fwd(*x);
    }
    mds(s);
    let ci2 = 1 + 2 * r + 1;
    for i in 0..M {
      s[i] ^= ROUND_CONSTANTS[ci2][i];
    }
  }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use super::*;

  /// Test vectors from binius64 crates/hash/src/vision_4/permutation.rs
  #[test]
  fn test_permutation_zero_input() {
    let mut s = [0u128; M];
    permutation(&mut s);
    assert_eq!(s[0], 0x5e9a7b63d8d1a93953d56ceb6dcf6a35);
    assert_eq!(s[1], 0xa3262c57f6cdd8c368639c1a4f01ab5a);
    assert_eq!(s[2], 0x1dc99e37723063c4f178826d2a6802e3);
    assert_eq!(s[3], 0xfdf935c9d9fae3d560a75026a049bf7c);
  }

  #[test]
  fn test_permutation_deadbeef() {
    let mut s = [0u128; M];
    s[0] = 0xdeadbeef;
    s[2] = 0xdeadbeef;
    permutation(&mut s);
    assert_eq!(s[0], 0x1d02eaf6cf48c108a2ae1d9e27812364);
    assert_eq!(s[1], 0xc9bae4f4c782d46ed28245525f04fb3c);
    assert_eq!(s[2], 0xf4fea518a1e62f97748266e86acac536);
    assert_eq!(s[3], 0x22b25c68a52fef4b855f8862bdd418c4);
  }

  #[test]
  fn test_b_fwd_inv_roundtrip() {
    // b_fwd(b_inv(x)) should equal x for arbitrary inputs.
    let vals = [
      0u128,
      1u128,
      0xdeadbeefcafebabe1234567890abcdefu128,
      0xffffffffffffffffffffffffffffffffu128,
      0x80000000000000000000000000000001u128,
    ];
    for &x in &vals {
      let roundtrip = b_fwd(b_inv(x));
      assert_eq!(roundtrip, x, "b_fwd(b_inv({x:#x})) != {x:#x}");
    }
  }

  #[test]
  fn test_b_inv_fwd_roundtrip() {
    // b_inv(b_fwd(x)) should also equal x (the polynomials are mutual inverses).
    let vals = [
      0u128,
      1,
      0xabcdef,
      0x1234_5678_9abc_def0_u128 << 64 | 0xcafe,
    ];
    for &x in &vals {
      let roundtrip = b_inv(b_fwd(x));
      assert_eq!(roundtrip, x, "b_inv(b_fwd({x:#x})) != {x:#x}");
    }
  }

  #[test]
  fn test_mds_not_identity() {
    let mut s = [1u128, 0, 0, 0];
    mds(&mut s);
    // MDS of [1,0,0,0] should NOT be [1,0,0,0].
    assert_ne!(s, [1, 0, 0, 0]);
  }

  #[test]
  fn test_mds_matches_matrix() {
    // MDS matrix [2,3,1,1] circulant.
    // Row 0: 2*a[0] + 3*a[1] + a[2] + a[3]
    let a = [0x11u128, 0x22, 0x33, 0x44];
    let mut s = a;
    mds(&mut s);
    // Verify row 0: 2*0x11 + 3*0x22 + 0x33 + 0x44
    let two_a0 = mul_x(a[0]);
    let three_a1 = mul_x(a[1]) ^ a[1];
    let expected_0 = two_a0 ^ three_a1 ^ a[2] ^ a[3];
    assert_eq!(s[0], expected_0, "MDS row 0 mismatch");
  }

  #[test]
  fn test_permutation_deterministic() {
    let mut s1 = [0x42u128; M];
    let mut s2 = [0x42u128; M];
    permutation(&mut s1);
    permutation(&mut s2);
    assert_eq!(s1, s2);
  }

  #[test]
  fn test_permutation_different_inputs_differ() {
    let mut s1 = [0u128; M];
    let mut s2 = [1u128, 0, 0, 0];
    permutation(&mut s1);
    permutation(&mut s2);
    assert_ne!(s1, s2);
  }
}
