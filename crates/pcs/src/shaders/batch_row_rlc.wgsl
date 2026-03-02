// ── batch_row_rlc: t_raw[j] = Σ_i eq_hi[i] * M[i*k + j] ────────────────────
// NOTE: GF(2^128) 산술 함수는 gpu::GF128_WGSL 로부터 앞에 연결됨.
//
// 행렬-벡터 곱: n_rows × k 행렬 M 과 n_rows 벡터 eq_hi 의 선형 결합.
// 각 스레드가 하나의 출력 열 j 를 담당한다.
//
// 입력:
//   matrix[n_rows * k]  — row-major: M[i][j] = matrix[i * k + j]
//   eq_vec[n_rows]      — eq_hi 벡터
// 출력:
//   t_raw[k]            — t_raw[j] = Σ_i eq_vec[i] * matrix[i*k + j]

struct RowRlcParams {
    n_rows: u32,
    k: u32,       // n_raw_cols
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform>             rlc_params:  RowRlcParams;
@group(0) @binding(1) var<storage, read>       rlc_matrix:  array<vec4<u32>>;
@group(0) @binding(2) var<storage, read>       rlc_eq_vec:  array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> rlc_t_raw:   array<vec4<u32>>;

@compute @workgroup_size(256)
fn batch_row_rlc(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    let k = rlc_params.k;
    let n_rows = rlc_params.n_rows;
    if j >= k { return; }

    var acc = GF128_ZERO;
    for (var i = 0u; i < n_rows; i = i + 1u) {
        acc = gf128_add(acc, gf128_mul(rlc_eq_vec[i], rlc_matrix[i * k + j]));
    }
    rlc_t_raw[j] = acc;
}
