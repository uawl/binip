// ── Batch GF(2^128) 연산 커널 ─────────────────────────────────────────────────
// NOTE: GF(2^128) 산술 함수는 gpu::GF128_WGSL 로부터 앞에 연결됨.
//
// batch_mul: c[i] = a[i] * b[i]   (element-wise multiplication)
// batch_inv: out[i] = a[i]^(-1)   (Fermat inverse per element)

struct BatchParams {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// ── batch_mul ─────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<uniform>             batch_params: BatchParams;
@group(0) @binding(1) var<storage, read>       a_buf: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read>       b_buf: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> c_buf: array<vec4<u32>>;

@compute @workgroup_size(256)
fn batch_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= batch_params.len { return; }
    c_buf[i] = gf128_mul(a_buf[i], b_buf[i]);
}

// ── batch_inv ─────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<uniform>             inv_params: BatchParams;
@group(0) @binding(1) var<storage, read>       inv_in:  array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> inv_out: array<vec4<u32>>;

@compute @workgroup_size(256)
fn batch_inv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= inv_params.len { return; }
    inv_out[i] = gf128_inv(inv_in[i]);
}
