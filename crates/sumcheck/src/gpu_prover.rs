//! GPU-accelerated sumcheck prover.
//!
//! # Strategy
//!
//! For each round:
//! 1. Launch `sumcheck_round` kernel (single workgroup for half ≤ 2^17;
//!    multi-pass for larger tables — currently single-pass for simplicity).
//! 2. Read back the 3 GF2_128 values `g(0), g(1), g(α)`.
//! 3. Get challenge from transcript.
//! 4. Launch `sumcheck_fold` kernel to fold the table in-place.
//!
//! # Resource reuse
//!
//! Uniform buffers, bind groups, and the staging readback buffer are
//! allocated once before the round loop and updated via `write_buffer`.
//! This eliminates per-round allocation and bind-group creation overhead.

use bytemuck::{Pod, Zeroable};
use field::GF2_128;
use gpu::{GpuBuffer, GpuContext, PipelineCache, shader_with_gf128};
use poly::MlePoly;
use transcript::Transcript;
use wgpu::BufferUsages;

use crate::proof::{RoundPoly, SumcheckProof, alpha};

// ─── Shader sources ──────────────────────────────────────────────────────────

const SUMCHECK_KERNEL_SRC: &str = include_str!("shaders/sumcheck.wgsl");
const ROUND_ENTRY: &str = "sumcheck_round";
const FOLD_ENTRY: &str  = "sumcheck_fold";

// ─── Uniform structs (must match WGSL) ───────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct RoundParams {
    half: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FoldParams {
    half: u32,
    _pad: [u32; 3],
    r: [u32; 4],
}

// ─── GF2_128 ↔ [u32; 4] ──────────────────────────────────────────────────────

pub(crate) fn to_u32x4(v: GF2_128) -> [u32; 4] {
    let lo = v.lo.0;
    let hi = v.hi.0;
    [
        (lo & 0xFFFF_FFFF) as u32,
        (lo >> 32) as u32,
        (hi & 0xFFFF_FFFF) as u32,
        (hi >> 32) as u32,
    ]
}

pub(crate) fn from_u32x4(w: [u32; 4]) -> GF2_128 {
    use field::GF2_64;
    GF2_128 {
        lo: GF2_64((w[0] as u64) | ((w[1] as u64) << 32)),
        hi: GF2_64((w[2] as u64) | ((w[3] as u64) << 32)),
    }
}

// ─── GPU Prover ───────────────────────────────────────────────────────────────

/// Prove `Σ poly(x) = poly.sum()` with GPU-accelerated round polynomial computation.
///
/// The evaluation table is kept on the GPU throughout; only 12 u32s (3 field
/// elements) are transferred back to the CPU per round for the transcript.
pub fn prove_gpu<T: Transcript>(
    poly: &MlePoly,
    transcript: &mut T,
    ctx: &GpuContext,
    cache: &mut PipelineCache,
) -> SumcheckProof {
    let claimed_sum = poly.sum();
    transcript.absorb_field(claimed_sum);

    let shader_src = shader_with_gf128(SUMCHECK_KERNEL_SRC);
    let _a = alpha();

    // ── Pre-allocate GPU buffers (reused every round) ────────────────────

    // Evaluation table
    let flat: Vec<u32> = poly.evals.iter().flat_map(|&e| to_u32x4(e)).collect();
    let table_buf = GpuBuffer::from_slice(ctx, &flat, BufferUsages::STORAGE | BufferUsages::COPY_SRC);

    // Round output: [g0, g1, ga] = 3 × vec4<u32> = 12 u32
    let out_buf = GpuBuffer::<u32>::zeroed(ctx, 12, BufferUsages::STORAGE | BufferUsages::COPY_SRC);

    // Round uniform (written each round)
    let round_uniform = GpuBuffer::<RoundParams>::zeroed(ctx, 1, BufferUsages::UNIFORM);

    // Fold uniform (written each round)
    let fold_uniform = GpuBuffer::<FoldParams>::zeroed(ctx, 1, BufferUsages::UNIFORM);

    // Staging buffer for output readback (reused every round)
    let staging_size = out_buf.size_bytes();
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sc_staging"),
        size: staging_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // ── Create bind groups once ──────────────────────────────────────────

    // ── Pre-compile pipelines (warm the cache) ──────────────────────────
    // Separate scopes so mutable borrows of `cache` don't overlap.
    cache.get_or_compile(ctx, ROUND_ENTRY, &shader_src, ROUND_ENTRY);
    cache.get_or_compile(ctx, FOLD_ENTRY, &shader_src, FOLD_ENTRY);

    // Now take shared references (pipelines are already in the cache).
    let round_pipeline = cache.get(ROUND_ENTRY).unwrap();
    let fold_pipeline = cache.get(FOLD_ENTRY).unwrap();

    let round_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("round_bg"),
        layout: &round_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: round_uniform.inner.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: table_buf.inner.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: out_buf.inner.as_entire_binding() },
        ],
    });

    // (fold_pipeline already obtained above)
    let fold_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fold_bg"),
        layout: &fold_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: fold_uniform.inner.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: table_buf.inner.as_entire_binding() },
        ],
    });

    // ── Round loop ───────────────────────────────────────────────────────

    let mut round_polys = Vec::with_capacity(poly.n_vars as usize);
    let mut current_half = 1u32 << (poly.n_vars - 1);

    for _ in 0..poly.n_vars {
        // 1. Write round params
        round_uniform.write(ctx, &[RoundParams { half: current_half, _pad: [0; 3] }]);

        // 2. Dispatch round + copy to staging in one command buffer
        {
            let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(round_pipeline);
                pass.set_bind_group(0, &round_bg, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&out_buf.inner, 0, &staging, 0, staging_size);
            ctx.queue.submit(std::iter::once(encoder.finish()));
        }

        // 3. Readback via reused staging buffer
        let (g0, g1, ga) = {
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |res| { let _ = tx.send(res); });
            ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
                .expect("poll");
            rx.recv().unwrap().expect("map");
            let view = slice.get_mapped_range();
            let d: &[u32] = bytemuck::cast_slice(&view[..48]);
            let g0 = from_u32x4([d[0], d[1], d[2],  d[3]]);
            let g1 = from_u32x4([d[4], d[5], d[6],  d[7]]);
            let ga = from_u32x4([d[8], d[9], d[10], d[11]]);
            drop(view);
            staging.unmap();
            (g0, g1, ga)
        };

        let rp = RoundPoly([g0, g1, ga]);
        transcript.absorb_field(rp.0[0]);
        transcript.absorb_field(rp.0[1]);
        transcript.absorb_field(rp.0[2]);
        round_polys.push(rp);

        let r = transcript.squeeze_challenge();

        // 4. Write fold params + dispatch fold
        fold_uniform.write(ctx, &[FoldParams {
            half: current_half,
            _pad: [0; 3],
            r: to_u32x4(r),
        }]);

        let wg = current_half.div_ceil(256);
        {
            let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(fold_pipeline);
                pass.set_bind_group(0, &fold_bg, &[]);
                pass.dispatch_workgroups(wg, 1, 1);
            }
            ctx.queue.submit(std::iter::once(encoder.finish()));
        }

        current_half >>= 1;
    }

    // final_eval = table[0]
    let final_data = table_buf.read(ctx).expect("final read");
    let final_eval = from_u32x4([final_data[0], final_data[1], final_data[2], final_data[3]]);

    SumcheckProof { claimed_sum, round_polys, final_eval }
}
