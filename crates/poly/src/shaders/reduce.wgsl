// ── 병렬 합산 커널 (sum_reduce) ───────────────────────────────────────────────
// NOTE: GF(2^128) 산술 함수는 gpu::GF128_WGSL 로부터 앞에 연결됨.
//
// 단일 워크그룹으로 배열의 GF(2^128) 합을 계산한다.
// Stride-loop + shared memory tree reduction.
// 결과: out[0] = Σ data[i]

struct ReduceParams {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform>             reduce_params: ReduceParams;
@group(0) @binding(1) var<storage, read>       reduce_data:   array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> reduce_out:    array<vec4<u32>>; // 1 element

var<workgroup> wg_sum: array<vec4<u32>, 256>;

@compute @workgroup_size(256)
fn sum_reduce(
    @builtin(global_invocation_id)  gid: vec3<u32>,
    @builtin(local_invocation_id)   lid: vec3<u32>,
    @builtin(num_workgroups)        nwg: vec3<u32>,
) {
    let local_idx = lid.x;
    let n = reduce_params.len;

    // Stride-loop accumulation
    var acc = GF128_ZERO;
    var idx = gid.x;
    loop {
        if idx >= n { break; }
        acc = gf128_add(acc, reduce_data[idx]);
        idx = idx + 256u * nwg.x;
    }

    wg_sum[local_idx] = acc;
    workgroupBarrier();

    // Tree reduction (256 → 1)
    var stride = 128u;
    loop {
        if stride == 0u { break; }
        if local_idx < stride {
            wg_sum[local_idx] = gf128_add(wg_sum[local_idx], wg_sum[local_idx + stride]);
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if local_idx == 0u {
        if nwg.x == 1u {
            reduce_out[0] = wg_sum[0];
        } else {
            // Multi-workgroup: write partial sum to slot
            let slot = gid.x / 256u;
            reduce_out[slot] = wg_sum[0];
        }
    }
}
