//! GPU-accelerated shard-parallel sumcheck prover.
//!
//! Each workgroup independently computes one shard's sumcheck round,
//! enabling massive parallelism across shards.
//!
//! # Resource reuse
//!
//! Uniform, staging, and challenge buffers are allocated once before the
//! round loop and updated via `queue.write_buffer` each iteration.
//! Bind groups are also created once and reused (buffer handles don't change).

use bytemuck::{Pod, Zeroable};
use field::GF2_128;
use gpu::{GpuBuffer, GpuContext, PipelineCache, shader_with_gf128};
use poly::MlePoly;
use transcript::Transcript;
use wgpu::BufferUsages;

use crate::proof::{RoundPoly, SumcheckProof, alpha};
use crate::gpu_prover::{to_u32x4, from_u32x4};

const SHARD_KERNEL_SRC: &str = include_str!("shaders/shard.wgsl");
const SHARD_ROUND_ENTRY: &str = "shard_sumcheck_round";
const SHARD_FOLD_ENTRY: &str = "shard_fold";

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ShardRoundParams {
    half: u32,
    shard_size: u32,
    n_shards: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ShardFoldParams {
    half: u32,
    shard_size: u32,
    n_shards: u32,
    _pad: u32,
}

/// Prove all shards in parallel on the GPU.
///
/// The full evaluation table is split into `n_shards` contiguous sub-tables,
/// each of size `2^shard_vars`.  A single GPU dispatch handles all shards
/// simultaneously — one workgroup per shard.
///
/// Each shard uses an independently forked transcript for domain separation,
/// producing per-shard challenges that are uploaded to a GPU storage buffer
/// so the fold kernel applies the correct challenge per shard.
pub fn prove_shards_gpu<T: Transcript>(
    poly: &MlePoly,
    transcripts: &mut [T],
    shard_vars: u32,
    ctx: &GpuContext,
    cache: &mut PipelineCache,
) -> Vec<SumcheckProof> {
    let total_vars = poly.n_vars;
    assert!(shard_vars <= total_vars);
    let n_shards = 1u32 << (total_vars - shard_vars);
    let shard_size = 1u32 << shard_vars;
    assert_eq!(transcripts.len(), n_shards as usize);

    let shader_src = shader_with_gf128(SHARD_KERNEL_SRC);
    let _a = alpha();

    // ── Pre-allocate GPU buffers (reused every round) ────────────────────

    // Full evaluation table
    let flat: Vec<u32> = poly.evals.iter().flat_map(|&e| to_u32x4(e)).collect();
    let table_buf = GpuBuffer::from_slice(ctx, &flat, BufferUsages::STORAGE | BufferUsages::COPY_SRC);

    // Round output: 3 GF(2^128) per shard = 12 u32 per shard
    let out_elems = n_shards as usize * 12;
    let out_buf = GpuBuffer::<u32>::zeroed(ctx, out_elems, BufferUsages::STORAGE | BufferUsages::COPY_SRC);

    // Round uniform (written each round via write_buffer)
    let round_uniform = GpuBuffer::<ShardRoundParams>::zeroed(ctx, 1, BufferUsages::UNIFORM);

    // Fold uniform (written each round)
    let fold_uniform = GpuBuffer::<ShardFoldParams>::zeroed(ctx, 1, BufferUsages::UNIFORM);

    // Per-shard challenge buffer (written each round)
    let challenges_buf = GpuBuffer::<u32>::zeroed(
        ctx,
        n_shards as usize * 4,
        BufferUsages::STORAGE,
    );

    // Staging buffer for readback (reused every round)
    let staging_size = out_buf.size_bytes();
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("shard_staging"),
        size: staging_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // ── Pre-compile pipelines (mutable borrow) then take shared refs ───

    cache.get_or_compile(ctx, SHARD_ROUND_ENTRY, &shader_src, SHARD_ROUND_ENTRY);
    cache.get_or_compile(ctx, SHARD_FOLD_ENTRY, &shader_src, SHARD_FOLD_ENTRY);

    let round_pipeline = cache.get(SHARD_ROUND_ENTRY).unwrap();
    let fold_pipeline = cache.get(SHARD_FOLD_ENTRY).unwrap();

    let round_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("shard_round_bg"),
        layout: &round_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: round_uniform.inner.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: table_buf.inner.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: out_buf.inner.as_entire_binding() },
        ],
    });

    let fold_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("shard_fold_bg"),
        layout: &fold_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: fold_uniform.inner.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: table_buf.inner.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: challenges_buf.inner.as_entire_binding() },
        ],
    });

    // ── Compute claimed sums per shard (CPU) ─────────────────────────────

    let claimed_sums: Vec<GF2_128> = (0..n_shards as usize)
        .map(|s| {
            let base = s * shard_size as usize;
            poly.evals[base..base + shard_size as usize]
                .iter()
                .fold(GF2_128::default(), |acc, &e| acc + e)
        })
        .collect();

    for (t, &sum) in transcripts.iter_mut().zip(claimed_sums.iter()) {
        t.absorb_field(sum);
    }

    // ── Round loop ───────────────────────────────────────────────────────

    let mut all_round_polys: Vec<Vec<RoundPoly>> = vec![Vec::new(); n_shards as usize];
    let mut current_half = shard_size >> 1;
    let mut current_shard_size = shard_size;

    for _ in 0..shard_vars {
        // 1. Write round params
        round_uniform.write(ctx, &[ShardRoundParams {
            half: current_half,
            shard_size: current_shard_size,
            n_shards,
            _pad: 0,
        }]);

        // 2. Dispatch round kernel (one workgroup per shard)
        {
            let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(round_pipeline);
                pass.set_bind_group(0, &round_bg, &[]);
                pass.dispatch_workgroups(n_shards, 1, 1);
            }
            // Copy out_buf → staging in the same command buffer
            encoder.copy_buffer_to_buffer(&out_buf.inner, 0, &staging, 0, staging_size);
            ctx.queue.submit(std::iter::once(encoder.finish()));
        }

        // 3. Readback round polys via reused staging buffer
        let out_data = {
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |res| { let _ = tx.send(res); });
            ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
                .expect("poll");
            rx.recv().unwrap().expect("map");
            let view = slice.get_mapped_range();
            let data: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&view[..out_elems * 4]).to_vec();
            drop(view);
            staging.unmap();
            data
        };

        // 4. CPU: per-shard transcript absorb + squeeze → per-shard challenges
        let mut challenges_flat: Vec<u32> = Vec::with_capacity(n_shards as usize * 4);
        for s in 0..n_shards as usize {
            let base = s * 12;
            let g0 = from_u32x4([out_data[base],   out_data[base+1], out_data[base+2],  out_data[base+3]]);
            let g1 = from_u32x4([out_data[base+4], out_data[base+5], out_data[base+6],  out_data[base+7]]);
            let ga = from_u32x4([out_data[base+8], out_data[base+9], out_data[base+10], out_data[base+11]]);

            let rp = RoundPoly([g0, g1, ga]);
            transcripts[s].absorb_field(rp.0[0]);
            transcripts[s].absorb_field(rp.0[1]);
            transcripts[s].absorb_field(rp.0[2]);
            all_round_polys[s].push(rp);

            let r: GF2_128 = transcripts[s].squeeze_challenge();
            challenges_flat.extend_from_slice(&to_u32x4(r));
        }

        // 5. Upload per-shard challenges + fold params, dispatch fold
        challenges_buf.write(ctx, &challenges_flat);
        fold_uniform.write(ctx, &[ShardFoldParams {
            half: current_half,
            shard_size: current_shard_size,
            n_shards,
            _pad: 0,
        }]);

        let total_work = current_half * n_shards;
        let wg = total_work.div_ceil(256);
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
        current_shard_size >>= 1;
    }

    // ── Read final evaluations ───────────────────────────────────────────

    let final_data = table_buf.read(ctx).expect("shard final read");

    (0..n_shards as usize)
        .map(|s| {
            let base = s * (shard_size as usize) * 4;
            let final_eval = from_u32x4([
                final_data[base], final_data[base+1],
                final_data[base+2], final_data[base+3],
            ]);
            SumcheckProof {
                claimed_sum: claimed_sums[s],
                round_polys: all_round_polys[s].clone(),
                final_eval,
            }
        })
        .collect()
}
