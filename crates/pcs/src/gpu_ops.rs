//! GPU-accelerated PCS operations.
//!
//! Provides GPU dispatch for:
//! - `batch_row_rlc_gpu`: matrix-vector product for opening computation
//! - `matrix_encode_gpu`: rate-1/4 systematic encoding of all rows

use bytemuck::{Pod, Zeroable};
use field::{gf2_64::GF2_64, gf2_128::GF2_128};
use gpu::{GpuBuffer, GpuContext, Dispatcher, PipelineCache, shader_with_gf128};
use wgpu::{util::DeviceExt, BufferUsages};

const ROW_RLC_KERNEL_SRC: &str = include_str!("shaders/batch_row_rlc.wgsl");
const ENCODE_KERNEL_SRC: &str = include_str!("shaders/matrix_encode.wgsl");

// ─── GF2_128 ↔ [u32; 4] ──────────────────────────────────────────────────────

fn to_u32x4(v: GF2_128) -> [u32; 4] {
    let lo = v.lo.0;
    let hi = v.hi.0;
    [
        (lo & 0xFFFF_FFFF) as u32,
        (lo >> 32) as u32,
        (hi & 0xFFFF_FFFF) as u32,
        (hi >> 32) as u32,
    ]
}

fn from_u32x4(w: [u32; 4]) -> GF2_128 {
    GF2_128 {
        lo: GF2_64((w[0] as u64) | ((w[1] as u64) << 32)),
        hi: GF2_64((w[2] as u64) | ((w[3] as u64) << 32)),
    }
}

// ─── Uniform layouts ─────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct RowRlcParams {
    n_rows: u32,
    k: u32,
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct EncodeParams {
    k: u32,
    n_red: u32,
    n_rows: u32,
    _pad: u32,
}

// ─── batch_row_rlc ───────────────────────────────────────────────────────────

/// GPU: `t_raw[j] = Σ_i eq_vec[i] * matrix[i*k + j]` for j in 0..k.
///
/// `matrix` is row-major `[n_rows][k]`.
pub fn batch_row_rlc_gpu(
    ctx: &GpuContext,
    cache: &mut PipelineCache,
    matrix: &[Vec<GF2_128>],
    eq_vec: &[GF2_128],
) -> Vec<GF2_128> {
    let n_rows = matrix.len();
    let k = matrix[0].len();
    assert_eq!(eq_vec.len(), n_rows);

    let flat_matrix: Vec<u32> = matrix.iter()
        .flat_map(|row| row.iter().flat_map(|&e| to_u32x4(e)))
        .collect();
    let flat_eq: Vec<u32> = eq_vec.iter().flat_map(|&e| to_u32x4(e)).collect();

    let mat_buf = GpuBuffer::from_slice(ctx, &flat_matrix, BufferUsages::STORAGE);
    let eq_buf = GpuBuffer::from_slice(ctx, &flat_eq, BufferUsages::STORAGE);
    let out_buf = GpuBuffer::<u32>::zeroed(ctx, k * 4, BufferUsages::STORAGE | BufferUsages::COPY_SRC);

    let params = RowRlcParams { n_rows: n_rows as u32, k: k as u32, _pad: [0; 2] };
    let params_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rlc_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader_src = shader_with_gf128(ROW_RLC_KERNEL_SRC);
    let pipeline = cache.get_or_compile(ctx, "batch_row_rlc", &shader_src, "batch_row_rlc");

    let bg_layout = pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rlc_bg"),
        layout: &bg_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: mat_buf.inner.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: eq_buf.inner.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: out_buf.inner.as_entire_binding() },
        ],
    });

    let wg = (k as u32).div_ceil(256);
    Dispatcher::new(ctx).dispatch(pipeline, &[&bind_group], [wg, 1, 1]);

    let flat = out_buf.read(ctx).expect("batch_row_rlc_gpu: read failed");
    flat.chunks_exact(4).map(|w| from_u32x4([w[0], w[1], w[2], w[3]])).collect()
}

// ─── matrix_encode ───────────────────────────────────────────────────────────

/// GPU: encode all rows of the evaluation matrix with a rate-1/4 systematic code.
///
/// Returns `encoded_rows[n_rows][4k]`.
pub fn matrix_encode_gpu(
    ctx: &GpuContext,
    cache: &mut PipelineCache,
    rows: &[Vec<GF2_128>],
    gen_matrix: &[GF2_128], // [k * 3k] row-major
) -> Vec<Vec<GF2_128>> {
    let n_rows = rows.len();
    let k = rows[0].len();
    let n_red = 3 * k;
    let n_enc = 4 * k;

    let flat_rows: Vec<u32> = rows.iter()
        .flat_map(|row| row.iter().flat_map(|&e| to_u32x4(e)))
        .collect();
    let flat_gen: Vec<u32> = gen_matrix.iter().flat_map(|&e| to_u32x4(e)).collect();

    let rows_buf = GpuBuffer::from_slice(ctx, &flat_rows, BufferUsages::STORAGE);
    let gen_buf = GpuBuffer::from_slice(ctx, &flat_gen, BufferUsages::STORAGE);
    let out_buf = GpuBuffer::<u32>::zeroed(
        ctx,
        n_rows * n_enc * 4,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    );

    let params = EncodeParams {
        k: k as u32,
        n_red: n_red as u32,
        n_rows: n_rows as u32,
        _pad: 0,
    };
    let params_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("encode_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader_src = shader_with_gf128(ENCODE_KERNEL_SRC);
    let pipeline = cache.get_or_compile(ctx, "matrix_encode", &shader_src, "matrix_encode");

    let bg_layout = pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("encode_bg"),
        layout: &bg_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: rows_buf.inner.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: gen_buf.inner.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: out_buf.inner.as_entire_binding() },
        ],
    });

    // 2D dispatch: x = enc cols, y = rows
    let wg_x = (n_enc as u32).div_ceil(256);
    let wg_y = n_rows as u32;
    Dispatcher::new(ctx).dispatch(pipeline, &[&bind_group], [wg_x, wg_y, 1]);

    let flat = out_buf.read(ctx).expect("matrix_encode_gpu: read failed");
    (0..n_rows)
        .map(|row| {
            let base = row * n_enc * 4;
            (0..n_enc)
                .map(|j| {
                    let o = base + j * 4;
                    from_u32x4([flat[o], flat[o+1], flat[o+2], flat[o+3]])
                })
                .collect()
        })
        .collect()
}
