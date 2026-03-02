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
//! The GPU table buffer is sized for the full `2^n_vars` evaluation table.
//! After each fold the active region shrinks by half; we track this with
//! `current_half` and keep the buffer at full size (unused upper half ignored).

use bytemuck::{Pod, Zeroable};
use field::GF2_128;
use gpu::{GpuBuffer, GpuContext, Dispatcher, PipelineCache};
use poly::MlePoly;
use transcript::Transcript;
use wgpu::{util::DeviceExt, BufferUsages};

use crate::proof::{RoundPoly, SumcheckProof, alpha};

// ─── Shader sources ──────────────────────────────────────────────────────────

const SHADER_SRC: &str = include_str!("shaders/sumcheck.wgsl");
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
    let claimed_sum = poly.sum(); // cheap CPU sum; GPU for large polys handled by fold
    transcript.absorb_field(claimed_sum);

    // Upload evaluation table to GPU (packed as u32)
    let flat: Vec<u32> = poly
        .evals
        .iter()
        .flat_map(|&e| to_u32x4(e))
        .collect();
    let table_buf: GpuBuffer<u32> = GpuBuffer::from_slice(
        ctx,
        &flat,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    );

    // Output buffer for round: [g0, g1, ga] = 3 × vec4<u32> = 12 u32
    let out_buf: GpuBuffer<u32> = GpuBuffer::zeroed(ctx, 12, BufferUsages::STORAGE | BufferUsages::COPY_SRC);

    let _a = alpha(); // α = GF2_128::from(2) — computed on GPU side
    let mut round_polys = Vec::with_capacity(poly.n_vars as usize);
    let mut current_half = 1u32 << (poly.n_vars - 1);

    for _ in 0..poly.n_vars {
        // ── Round: compute g(0), g(1), g(α) ─────────────────────────────────
        let round_params = RoundParams { half: current_half, _pad: [0; 3] };
        let round_uniform = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("round_params"),
            contents: bytemuck::bytes_of(&round_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let round_pipeline = cache.get_or_compile(ctx, ROUND_ENTRY, SHADER_SRC, ROUND_ENTRY);
        let bg_layout = round_pipeline.get_bind_group_layout(0);
        let round_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("round_bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: round_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: table_buf.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buf.inner.as_entire_binding(),
                },
            ],
        });

        // Single workgroup: covers up to 256 elements per thread via stride loop
        let dispatcher = Dispatcher::new(ctx);
        dispatcher.dispatch(&round_pipeline, &[&round_bg], [1, 1, 1]);

        // Read g(0), g(1), g(α) from GPU
        let out_data = out_buf.read(ctx).expect("round out read");
        let g0 = from_u32x4([out_data[0], out_data[1], out_data[2],  out_data[3]]);
        let g1 = from_u32x4([out_data[4], out_data[5], out_data[6],  out_data[7]]);
        let ga = from_u32x4([out_data[8], out_data[9], out_data[10], out_data[11]]);

        let rp = RoundPoly([g0, g1, ga]);
        transcript.absorb_field(rp.0[0]);
        transcript.absorb_field(rp.0[1]);
        transcript.absorb_field(rp.0[2]);
        round_polys.push(rp);

        let r = transcript.squeeze_challenge();

        // ── Fold: fix current variable to r ──────────────────────────────────
        let fold_params = FoldParams {
            half: current_half,
            _pad: [0; 3],
            r: to_u32x4(r),
        };
        let fold_uniform = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fold_params"),
            contents: bytemuck::bytes_of(&fold_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let fold_pipeline = cache.get_or_compile(ctx, FOLD_ENTRY, SHADER_SRC, FOLD_ENTRY);
        let fold_bg_layout = fold_pipeline.get_bind_group_layout(0);
        let fold_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fold_bg"),
            layout: &fold_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: fold_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: table_buf.inner.as_entire_binding(),
                },
            ],
        });

        let wg = current_half.div_ceil(256);
        dispatcher.dispatch(&fold_pipeline, &[&fold_bg], [wg, 1, 1]);

        current_half >>= 1;
    }

    // final_eval = table[0]: download first 4 u32s
    let final_data = table_buf.read(ctx).expect("final read");
    let final_eval = from_u32x4([final_data[0], final_data[1], final_data[2], final_data[3]]);

    SumcheckProof { claimed_sum, round_polys, final_eval }
}
