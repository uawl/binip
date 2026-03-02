//! GPU-accelerated MLE operations using `crates/gpu` infrastructure.
//!
//! Provides `fold_gpu` — a single fold step on the GPU — and `evaluate_gpu`
//! which chains `n_vars` folds to evaluate the MLE at a point.

use bytemuck::{Pod, Zeroable};
use field::GF2_128;
use gpu::{GpuBuffer, GpuContext, Dispatcher, PipelineCache, shader_with_gf128};
use wgpu::{util::DeviceExt, BufferUsages};

use crate::poly::MlePoly;

// ─── WGSL source ──────────────────────────────────────────────────────────────

const MLE_KERNEL_SRC: &str = include_str!("shaders/mle.wgsl");
const FOLD_ENTRY: &str = "mle_fold";

// ─── Uniform layout (must match `FoldParams` in mle.wgsl) ────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FoldParams {
  half: u32,
  _pad: [u32; 3],
  r: [u32; 4], // GF2_128 as 4 × u32
}

// ─── GF2_128 ↔ [u32; 4] conversion ───────────────────────────────────────────

/// CPU `GF2_128` stores `(lo: GF2_64(u64), hi: GF2_64(u64))`.
/// GPU expects `[lo_lo, lo_hi, hi_lo, hi_hi]` (little-endian limbs).
fn to_u32x4(v: GF2_128) -> [u32; 4] {
  let lo = v.lo.0;
  let hi = v.hi.0;
  [(lo & 0xFFFF_FFFF) as u32, (lo >> 32) as u32, (hi & 0xFFFF_FFFF) as u32, (hi >> 32) as u32]
}

fn from_u32x4(w: [u32; 4]) -> GF2_128 {
  use field::GF2_64;
  GF2_128 {
    lo: GF2_64((w[0] as u64) | ((w[1] as u64) << 32)),
    hi: GF2_64((w[2] as u64) | ((w[3] as u64) << 32)),
  }
}

// ─── GPU eval table (packed u32 buffer) ──────────────────────────────────────

/// Holds the MLE evaluation table in a GPU buffer (each element = 4 × u32).
pub struct GpuMlePoly {
  /// Packed evals: each GF2_128 = 4 consecutive u32 words.
  pub buf: GpuBuffer<u32>,
  pub n_vars: u32,
}

impl GpuMlePoly {
  /// Upload a CPU `MlePoly` to the GPU.
  pub fn upload(ctx: &GpuContext, poly: &MlePoly) -> Self {
    let flat: Vec<u32> = poly.evals.iter().flat_map(|&e| to_u32x4(e)).collect();
    let buf = GpuBuffer::from_slice(
      ctx,
      &flat,
      BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    );
    Self { buf, n_vars: poly.n_vars }
  }

  /// Download the evaluation table back to CPU.
  pub fn download(&self, ctx: &GpuContext) -> MlePoly {
    let flat = self.buf.read(ctx).expect("GpuMlePoly::download failed");
    let evals = flat.chunks_exact(4).map(|w| from_u32x4([w[0], w[1], w[2], w[3]])).collect();
    MlePoly { evals, n_vars: self.n_vars }
  }
}

// ─── Fold one variable on the GPU ─────────────────────────────────────────────

/// Apply one fold step in-place: fixes the last variable to `r`.
/// After this call `gpu_poly.n_vars` decreases by 1.
pub fn fold_gpu(
  ctx: &GpuContext,
  cache: &mut PipelineCache,
  gpu_poly: &mut GpuMlePoly,
  r: GF2_128,
) {
  let half = (1u32 << gpu_poly.n_vars) >> 1;

  let params = FoldParams { half, _pad: [0; 3], r: to_u32x4(r) };
  let params_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    label: Some("fold_params"),
    contents: bytemuck::bytes_of(&params),
    usage: wgpu::BufferUsages::UNIFORM,
  });

  let pipeline = cache.get_or_compile(ctx, FOLD_ENTRY, &shader_with_gf128(MLE_KERNEL_SRC), FOLD_ENTRY);

  let bg_layout = pipeline.get_bind_group_layout(0);
  let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
    label: Some("fold_bg"),
    layout: &bg_layout,
    entries: &[
      wgpu::BindGroupEntry {
        binding: 0,
        resource: params_buf.as_entire_binding(),
      },
      wgpu::BindGroupEntry {
        binding: 1,
        resource: gpu_poly.buf.inner.as_entire_binding(),
      },
    ],
  });

  let dispatcher = Dispatcher::new(ctx);
  let wg = half.div_ceil(256);
  dispatcher.dispatch(&pipeline, &[&bind_group], [wg, 1, 1]);
  gpu_poly.n_vars -= 1;
}

/// Evaluate the MLE on the GPU by folding all variables.
/// Returns a single `GF2_128` scalar.
pub fn evaluate_gpu(
  ctx: &GpuContext,
  cache: &mut PipelineCache,
  poly: &MlePoly,
  r: &[GF2_128],
) -> GF2_128 {
  assert_eq!(r.len() as u32, poly.n_vars);
  let mut gpu_poly = GpuMlePoly::upload(ctx, poly);
  for &ri in r {
    fold_gpu(ctx, cache, &mut gpu_poly, ri);
  }
  // Download single element.
  let flat = gpu_poly.buf.read(ctx).expect("evaluate_gpu: read failed");
  from_u32x4([flat[0], flat[1], flat[2], flat[3]])
}

// ─── Batch operations ─────────────────────────────────────────────────────────

const BATCH_KERNEL_SRC: &str = include_str!("shaders/batch_ops.wgsl");

/// Element-wise GF(2^128) multiplication on GPU: `c[i] = a[i] * b[i]`.
pub fn batch_mul_gpu(
  ctx: &GpuContext,
  cache: &mut PipelineCache,
  a: &[GF2_128],
  b: &[GF2_128],
) -> Vec<GF2_128> {
  assert_eq!(a.len(), b.len());
  let n = a.len();

  let flat_a: Vec<u32> = a.iter().flat_map(|&e| to_u32x4(e)).collect();
  let flat_b: Vec<u32> = b.iter().flat_map(|&e| to_u32x4(e)).collect();

  let a_buf = GpuBuffer::from_slice(ctx, &flat_a, BufferUsages::STORAGE);
  let b_buf = GpuBuffer::from_slice(ctx, &flat_b, BufferUsages::STORAGE);
  let c_buf = GpuBuffer::<u32>::zeroed(ctx, n * 4, BufferUsages::STORAGE | BufferUsages::COPY_SRC);

  let params = BatchParams { len: n as u32, _pad: [0; 3] };
  let params_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    label: Some("batch_mul_params"),
    contents: bytemuck::bytes_of(&params),
    usage: wgpu::BufferUsages::UNIFORM,
  });

  let shader_src = shader_with_gf128(BATCH_KERNEL_SRC);
  let pipeline = cache.get_or_compile(ctx, "batch_mul", &shader_src, "batch_mul");

  let bg_layout = pipeline.get_bind_group_layout(0);
  let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
    label: Some("batch_mul_bg"),
    layout: &bg_layout,
    entries: &[
      wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
      wgpu::BindGroupEntry { binding: 1, resource: a_buf.inner.as_entire_binding() },
      wgpu::BindGroupEntry { binding: 2, resource: b_buf.inner.as_entire_binding() },
      wgpu::BindGroupEntry { binding: 3, resource: c_buf.inner.as_entire_binding() },
    ],
  });

  let wg = (n as u32).div_ceil(256);
  Dispatcher::new(ctx).dispatch(pipeline, &[&bind_group], [wg, 1, 1]);

  let flat = c_buf.read(ctx).expect("batch_mul_gpu: read failed");
  flat.chunks_exact(4).map(|w| from_u32x4([w[0], w[1], w[2], w[3]])).collect()
}

/// Element-wise GF(2^128) Fermat inverse on GPU: `out[i] = a[i]^(-1)`.
pub fn batch_inv_gpu(
  ctx: &GpuContext,
  cache: &mut PipelineCache,
  a: &[GF2_128],
) -> Vec<GF2_128> {
  let n = a.len();

  let flat_a: Vec<u32> = a.iter().flat_map(|&e| to_u32x4(e)).collect();
  let a_buf = GpuBuffer::from_slice(ctx, &flat_a, BufferUsages::STORAGE);
  let out_buf = GpuBuffer::<u32>::zeroed(ctx, n * 4, BufferUsages::STORAGE | BufferUsages::COPY_SRC);

  let params = BatchParams { len: n as u32, _pad: [0; 3] };
  let params_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    label: Some("batch_inv_params"),
    contents: bytemuck::bytes_of(&params),
    usage: wgpu::BufferUsages::UNIFORM,
  });

  let shader_src = shader_with_gf128(BATCH_KERNEL_SRC);
  let pipeline = cache.get_or_compile(ctx, "batch_inv", &shader_src, "batch_inv");

  let bg_layout = pipeline.get_bind_group_layout(0);
  let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
    label: Some("batch_inv_bg"),
    layout: &bg_layout,
    entries: &[
      wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
      wgpu::BindGroupEntry { binding: 1, resource: a_buf.inner.as_entire_binding() },
      wgpu::BindGroupEntry { binding: 2, resource: out_buf.inner.as_entire_binding() },
    ],
  });

  let wg = (n as u32).div_ceil(256);
  Dispatcher::new(ctx).dispatch(pipeline, &[&bind_group], [wg, 1, 1]);

  let flat = out_buf.read(ctx).expect("batch_inv_gpu: read failed");
  flat.chunks_exact(4).map(|w| from_u32x4([w[0], w[1], w[2], w[3]])).collect()
}

// ─── Sum reduce ──────────────────────────────────────────────────────────────

const REDUCE_KERNEL_SRC: &str = include_str!("shaders/reduce.wgsl");

/// Sum all GF(2^128) elements on the GPU via parallel tree reduction.
pub fn sum_reduce_gpu(
  ctx: &GpuContext,
  cache: &mut PipelineCache,
  data: &[GF2_128],
) -> GF2_128 {
  let n = data.len();
  let flat: Vec<u32> = data.iter().flat_map(|&e| to_u32x4(e)).collect();
  let data_buf = GpuBuffer::from_slice(ctx, &flat, BufferUsages::STORAGE);
  let out_buf = GpuBuffer::<u32>::zeroed(ctx, 4, BufferUsages::STORAGE | BufferUsages::COPY_SRC);

  let params = BatchParams { len: n as u32, _pad: [0; 3] };
  let params_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    label: Some("reduce_params"),
    contents: bytemuck::bytes_of(&params),
    usage: wgpu::BufferUsages::UNIFORM,
  });

  let shader_src = shader_with_gf128(REDUCE_KERNEL_SRC);
  let pipeline = cache.get_or_compile(ctx, "sum_reduce", &shader_src, "sum_reduce");

  let bg_layout = pipeline.get_bind_group_layout(0);
  let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
    label: Some("reduce_bg"),
    layout: &bg_layout,
    entries: &[
      wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
      wgpu::BindGroupEntry { binding: 1, resource: data_buf.inner.as_entire_binding() },
      wgpu::BindGroupEntry { binding: 2, resource: out_buf.inner.as_entire_binding() },
    ],
  });

  // Single workgroup (stride loop handles up to 256 * stride elements)
  Dispatcher::new(ctx).dispatch(pipeline, &[&bind_group], [1, 1, 1]);

  let flat = out_buf.read(ctx).expect("sum_reduce_gpu: read failed");
  from_u32x4([flat[0], flat[1], flat[2], flat[3]])
}

// ─── Eq polynomial expansion ─────────────────────────────────────────────────

const EQ_KERNEL_SRC: &str = include_str!("shaders/eq.wgsl");

/// Compute eq(r, ·) evaluations on GPU.
/// Returns a vector of 2^|r| elements.
pub fn eq_evals_gpu(
  ctx: &GpuContext,
  cache: &mut PipelineCache,
  r: &[GF2_128],
) -> Vec<GF2_128> {
  let n = r.len();
  let total = 1usize << n;

  // Allocate buffer for 2^n elements, initialize first to 1
  let mut init = vec![0u32; total * 4];
  init[0] = 1; // GF128_ONE = (1, 0, 0, 0)
  let evals_buf = GpuBuffer::from_slice(
    ctx,
    &init,
    BufferUsages::STORAGE | BufferUsages::COPY_SRC,
  );

  let shader_src = shader_with_gf128(EQ_KERNEL_SRC);
  let pipeline = cache.get_or_compile(ctx, "eq_expand", &shader_src, "eq_expand");

  let mut cur_len = 1u32;
  for &ri in r {
    let params = EqExpandParams { cur_len, _pad: [0; 3], r_i: to_u32x4(ri) };
    let params_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("eq_params"),
      contents: bytemuck::bytes_of(&params),
      usage: wgpu::BufferUsages::UNIFORM,
    });

    let bg_layout = pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: Some("eq_bg"),
      layout: &bg_layout,
      entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: evals_buf.inner.as_entire_binding() },
      ],
    });

    let wg = cur_len.div_ceil(256);
    Dispatcher::new(ctx).dispatch(pipeline, &[&bind_group], [wg, 1, 1]);

    cur_len *= 2;
  }

  let flat = evals_buf.read(ctx).expect("eq_evals_gpu: read failed");
  flat.chunks_exact(4).take(total).map(|w| from_u32x4([w[0], w[1], w[2], w[3]])).collect()
}

// ─── Uniform structs for new kernels ──────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BatchParams {
  len: u32,
  _pad: [u32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct EqExpandParams {
  cur_len: u32,
  _pad: [u32; 3],
  r_i: [u32; 4],
}
