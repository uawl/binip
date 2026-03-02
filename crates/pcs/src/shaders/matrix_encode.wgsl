// ── 행렬 인코딩 커널 (rate-1/4 systematic code) ──────────────────────────────
// NOTE: GF(2^128) 산술 함수는 gpu::GF128_WGSL 로부터 앞에 연결됨.
//
// 각 행 row[0..k] 를 encoded[0..4k] 로 확장한다.
//   encoded[j]     = row[j]              (j < k, systematic)
//   encoded[k + j] = Σ_i row[i] * G[i * n_red + j]   (j < 3k, redundant)
//
// 2D dispatch: (output_col, row_idx)
//   gid.x = j (열 인덱스, 0..4k)
//   gid.y = row_idx (행 인덱스, 0..n_rows)

struct EncodeParams {
    k: u32,         // input cols per row
    n_red: u32,     // = 3k (redundant part cols)
    n_rows: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform>             enc_params:  EncodeParams;
@group(0) @binding(1) var<storage, read>       enc_rows:    array<vec4<u32>>; // [n_rows * k]
@group(0) @binding(2) var<storage, read>       enc_gen:     array<vec4<u32>>; // [k * n_red]
@group(0) @binding(3) var<storage, read_write> enc_out:     array<vec4<u32>>; // [n_rows * 4k]

@compute @workgroup_size(256)
fn matrix_encode(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;       // 출력 열 인덱스 (0..4k)
    let row = gid.y;     // 행 인덱스

    let k     = enc_params.k;
    let n_red = enc_params.n_red;
    let n_enc = k + n_red;  // = 4k

    if row >= enc_params.n_rows { return; }
    if j >= n_enc { return; }

    let row_base_in  = row * k;
    let row_base_out = row * n_enc;

    if j < k {
        // Systematic: 그대로 복사
        enc_out[row_base_out + j] = enc_rows[row_base_in + j];
    } else {
        // Redundant: Σ_i row[i] * G[i * n_red + (j - k)]
        let red_j = j - k;
        var acc = GF128_ZERO;
        for (var i = 0u; i < k; i = i + 1u) {
            acc = gf128_add(acc, gf128_mul(enc_rows[row_base_in + i], enc_gen[i * n_red + red_j]));
        }
        enc_out[row_base_out + j] = acc;
    }
}
