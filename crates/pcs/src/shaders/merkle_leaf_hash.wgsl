// ── Merkle 리프 해시 커널 (Blake3 column hash) ────────────────────────────────
// NOTE: GF(2^128) 산술 함수는 gpu::GF128_WGSL 로부터 앞에 연결됨.
//
// 각 인코딩된 열(column j)에 대해 Blake3 해시를 계산한다.
// leaf_j = Blake3("binip:pcs:col:" || col_j[0] || col_j[1] || ... || col_j[n_rows-1])
//
// Blake3 compression function on GPU — 단순화된 단일-블록 버전.
// 열 데이터가 1블록(64 bytes) 이내인 경우만 처리.
// 대부분의 실용적 케이스(n_rows ≤ 4, elem = 16 bytes → 64 bytes per col)에 해당.
//
// NOTE: 대규모 열(> 64 bytes)에 대한 full Blake3는 CPU 폴백 사용.

struct MerkleLeafParams {
    n_rows: u32,
    n_enc_cols: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform>             ml_params:      MerkleLeafParams;
@group(0) @binding(1) var<storage, read>       ml_encoded:     array<vec4<u32>>; // [n_rows * n_enc_cols]
@group(0) @binding(2) var<storage, read_write> ml_leaf_hashes: array<array<u32, 8>>; // [n_enc_cols], 32-byte hashes

// Blake3 IV (first 8 words of fractional parts of sqrt of first 8 primes)
const BLAKE3_IV_0: u32 = 0x6A09E667u;
const BLAKE3_IV_1: u32 = 0xBB67AE85u;
const BLAKE3_IV_2: u32 = 0x3C6EF372u;
const BLAKE3_IV_3: u32 = 0xA54FF53Au;
const BLAKE3_IV_4: u32 = 0x510E527Fu;
const BLAKE3_IV_5: u32 = 0x9B05688Cu;
const BLAKE3_IV_6: u32 = 0x1F83D9ABu;
const BLAKE3_IV_7: u32 = 0x5BE0CD19u;

// Blake3 message permutation
const MSG_PERM: array<u32, 16> = array<u32, 16>(
    2u, 6u, 3u, 10u, 7u, 0u, 4u, 13u,
    1u, 11u, 12u, 5u, 9u, 14u, 15u, 8u
);

fn rotr32(x: u32, n: u32) -> u32 {
    return (x >> n) | (x << (32u - n));
}

fn blake3_g(state: ptr<function, array<u32, 16>>, a: u32, b: u32, c: u32, d: u32, mx: u32, my: u32) {
    (*state)[a] = (*state)[a] + (*state)[b] + mx;
    (*state)[d] = rotr32((*state)[d] ^ (*state)[a], 16u);
    (*state)[c] = (*state)[c] + (*state)[d];
    (*state)[b] = rotr32((*state)[b] ^ (*state)[c], 12u);
    (*state)[a] = (*state)[a] + (*state)[b] + my;
    (*state)[d] = rotr32((*state)[d] ^ (*state)[a], 8u);
    (*state)[c] = (*state)[c] + (*state)[d];
    (*state)[b] = rotr32((*state)[b] ^ (*state)[c], 7u);
}

fn blake3_round(state: ptr<function, array<u32, 16>>, m: ptr<function, array<u32, 16>>) {
    // Column step
    blake3_g(state, 0u, 4u,  8u, 12u, (*m)[0],  (*m)[1]);
    blake3_g(state, 1u, 5u,  9u, 13u, (*m)[2],  (*m)[3]);
    blake3_g(state, 2u, 6u, 10u, 14u, (*m)[4],  (*m)[5]);
    blake3_g(state, 3u, 7u, 11u, 15u, (*m)[6],  (*m)[7]);
    // Diagonal step
    blake3_g(state, 0u, 5u, 10u, 15u, (*m)[8],  (*m)[9]);
    blake3_g(state, 1u, 6u, 11u, 12u, (*m)[10], (*m)[11]);
    blake3_g(state, 2u, 7u,  8u, 13u, (*m)[12], (*m)[13]);
    blake3_g(state, 3u, 4u,  9u, 14u, (*m)[14], (*m)[15]);
}

fn blake3_permute(m: ptr<function, array<u32, 16>>) {
    var t: array<u32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        t[i] = (*m)[MSG_PERM[i]];
    }
    *m = t;
}

// 단일-블록 Blake3 compression (≤ 64 bytes message)
// domain: "binip:pcs:col:" (14 bytes) + data
// 실제로는 domain + data를 합쳐 message block으로 사용
fn blake3_compress_single(msg: ptr<function, array<u32, 16>>, msg_len: u32) -> array<u32, 8> {
    // flags: CHUNK_START | CHUNK_END | ROOT = 1|2|8 = 11
    let flags: u32 = 11u;
    let counter_lo: u32 = 0u;
    let counter_hi: u32 = 0u;
    let block_len: u32 = msg_len;

    var state: array<u32, 16> = array<u32, 16>(
        BLAKE3_IV_0, BLAKE3_IV_1, BLAKE3_IV_2, BLAKE3_IV_3,
        BLAKE3_IV_4, BLAKE3_IV_5, BLAKE3_IV_6, BLAKE3_IV_7,
        BLAKE3_IV_0, BLAKE3_IV_1, BLAKE3_IV_2, BLAKE3_IV_3,
        counter_lo, counter_hi, block_len, flags
    );

    // 7 rounds
    blake3_round(&state, msg);
    blake3_permute(msg);
    blake3_round(&state, msg);
    blake3_permute(msg);
    blake3_round(&state, msg);
    blake3_permute(msg);
    blake3_round(&state, msg);
    blake3_permute(msg);
    blake3_round(&state, msg);
    blake3_permute(msg);
    blake3_round(&state, msg);
    blake3_permute(msg);
    blake3_round(&state, msg);

    // XOR output: h[i] = state[i] ^ state[i+8]
    var out: array<u32, 8>;
    for (var i = 0u; i < 8u; i = i + 1u) {
        out[i] = state[i] ^ state[i + 8u];
    }
    return out;
}

@compute @workgroup_size(256)
fn merkle_leaf_hash(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col_j = gid.x;
    if col_j >= ml_params.n_enc_cols { return; }

    let n_rows = ml_params.n_rows;
    let n_enc = ml_params.n_enc_cols;

    // 도메인 "binip:pcs:col:" = 14 bytes → 3.5 u32s (패딩 포함 4 u32)
    // 각 GF(2^128) = 4 u32 = 16 bytes
    // 총 메시지: 14 + n_rows*16 bytes, 64 bytes 이내일 때만 GPU 처리
    var msg: array<u32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        msg[i] = 0u;
    }

    // "binip:pcs:col:" in little-endian u32s (ASCII)
    // b=0x62, i=0x69, n=0x6E, i=0x69, p=0x70, :=0x3A
    // p=0x70, c=0x63, s=0x73, :=0x3A
    // c=0x63, o=0x6F, l=0x6C, :=0x3A
    msg[0] = 0x696E6962u; // "bini" LE
    msg[1] = 0x63703A70u; // "p:pc" LE
    msg[2] = 0x6F633A73u; // "s:co" LE
    msg[3] = 0x00003A6Cu; // "l:\0\0" LE

    // 열 데이터 채우기 (n_rows ≤ 3 가정 → 14 + 48 = 62 ≤ 64 bytes)
    var byte_offset = 14u;
    for (var i = 0u; i < n_rows; i = i + 1u) {
        if i >= 3u { break; } // 안전 제한: 최대 3행
        let elem = ml_encoded[i * n_enc + col_j]; // column-major 접근
        // elem = vec4<u32> = [lo_lo, lo_hi, hi_lo, hi_hi]
        // little-endian 16 bytes
        let word_offset = byte_offset / 4u;
        let sub = byte_offset % 4u;
        if sub == 0u {
            msg[word_offset]      = elem.x;
            msg[word_offset + 1u] = elem.y;
            msg[word_offset + 2u] = elem.z;
            msg[word_offset + 3u] = elem.w;
        } else {
            // byte_offset=14 → word=3,sub=2: cross-word packing
            msg[word_offset]      = msg[word_offset] | (elem.x << (sub * 8u));
            msg[word_offset + 1u] = (elem.x >> ((4u - sub) * 8u)) | (elem.y << (sub * 8u));
            msg[word_offset + 2u] = (elem.y >> ((4u - sub) * 8u)) | (elem.z << (sub * 8u));
            msg[word_offset + 3u] = (elem.z >> ((4u - sub) * 8u)) | (elem.w << (sub * 8u));
            msg[word_offset + 4u] = elem.w >> ((4u - sub) * 8u);
        }
        byte_offset = byte_offset + 16u;
    }

    let hash = blake3_compress_single(&msg, byte_offset);
    ml_leaf_hashes[col_j] = hash;
}
