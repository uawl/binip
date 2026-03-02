// ── 샤드 병렬 Sumcheck 커널 ───────────────────────────────────────────────────
// NOTE: GF(2^128) 산술 함수는 gpu::GF128_WGSL 로부터 앞에 연결됨.
//
// 각 워크그룹이 독립적인 샤드의 sumcheck 라운드를 처리한다.
// workgroup_id.x = 샤드 인덱스
// local_invocation_id.x = 샤드 내 스레드 위치
//
// 입력: 전체 평가 테이블 (연속 배치된 샤드들)
//   shard i 의 데이터: table[i * shard_size .. (i+1) * shard_size]
//
// 출력: shard_out[shard_idx * 3 + 0..3] = [g0, g1, g(α)]

struct ShardRoundParams {
    half: u32,          // 샤드 내 half (shard_size / 2)
    shard_size: u32,    // 2^shard_vars
    n_shards: u32,      // 총 샤드 수
    _pad: u32,
}

@group(0) @binding(0) var<uniform>             shard_params: ShardRoundParams;
@group(0) @binding(1) var<storage, read>       shard_table:  array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> shard_out:    array<vec4<u32>>;

var<workgroup> wg_g0: array<vec4<u32>, 256>;
var<workgroup> wg_g1: array<vec4<u32>, 256>;

@compute @workgroup_size(256)
fn shard_sumcheck_round(
    @builtin(workgroup_id)          wgid: vec3<u32>,
    @builtin(local_invocation_id)   lid:  vec3<u32>,
) {
    let shard_idx  = wgid.x;
    let local_idx  = lid.x;
    let half       = shard_params.half;
    let shard_size = shard_params.shard_size;

    if shard_idx >= shard_params.n_shards { return; }

    let base = shard_idx * shard_size;

    // Stride-loop 부분합 누적
    var acc_g0 = GF128_ZERO;
    var acc_g1 = GF128_ZERO;

    var idx = local_idx;
    loop {
        if idx >= half { break; }
        acc_g0 = gf128_add(acc_g0, shard_table[base + idx]);
        acc_g1 = gf128_add(acc_g1, shard_table[base + idx + half]);
        idx = idx + 256u;
    }

    wg_g0[local_idx] = acc_g0;
    wg_g1[local_idx] = acc_g1;
    workgroupBarrier();

    // Tree reduction (256 → 1)
    var stride = 128u;
    loop {
        if stride == 0u { break; }
        if local_idx < stride {
            wg_g0[local_idx] = gf128_add(wg_g0[local_idx], wg_g0[local_idx + stride]);
            wg_g1[local_idx] = gf128_add(wg_g1[local_idx], wg_g1[local_idx + stride]);
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if local_idx == 0u {
        let g0 = wg_g0[0];
        let g1 = wg_g1[0];
        let one_plus_alpha = gf128_add(GF128_ONE, GF128_ALPHA);
        let ga = gf128_add(gf128_mul(g0, one_plus_alpha), gf128_mul(g1, GF128_ALPHA));

        let out_base = shard_idx * 3u;
        shard_out[out_base]      = g0;
        shard_out[out_base + 1u] = g1;
        shard_out[out_base + 2u] = ga;
    }
}

// ── 샤드 Fold 커널 ───────────────────────────────────────────────────────────
// 각 스레드가 하나의 원소를 fold 한다.
// 전체 테이블에서 shard 경계를 고려한 in-place fold.
// 각 shard는 독립 transcript에서 생성된 고유한 challenge를 사용한다.

struct ShardFoldParams {
    half: u32,
    shard_size: u32,
    n_shards: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform>             sf_params:     ShardFoldParams;
@group(0) @binding(1) var<storage, read_write> sf_table:      array<vec4<u32>>;
@group(0) @binding(2) var<storage, read>       sf_challenges: array<vec4<u32>>; // per-shard r

@compute @workgroup_size(256)
fn shard_fold(@builtin(global_invocation_id) gid: vec3<u32>) {
    // gid.x: 선형 인덱스 → (shard, within-shard position) 로 분해
    let total_work = sf_params.half * sf_params.n_shards;
    if gid.x >= total_work { return; }

    let shard_idx = gid.x / sf_params.half;
    let local_i   = gid.x % sf_params.half;
    let base = shard_idx * sf_params.shard_size;

    let r           = sf_challenges[shard_idx];
    let one_minus_r = gf128_add(GF128_ONE, r);

    let a = sf_table[base + local_i];
    let b = sf_table[base + local_i + sf_params.half];
    sf_table[base + local_i] = gf128_add(gf128_mul(a, one_minus_r), gf128_mul(b, r));
}
