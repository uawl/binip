pub mod buffer;
pub mod context;
pub mod error;
pub mod pipeline;
pub mod shaders;

pub use buffer::GpuBuffer;
pub use context::GpuContext;
pub use error::GpuError;
pub use pipeline::{Dispatcher, PipelineCache};
pub use shaders::{GF128_WGSL, shader_with_gf128};

#[cfg(test)]
mod tests {
  use wgpu::BufferUsages;

  use super::*;

  /// GPU 초기화 + 단순 덧셈 컴퓨트 셰이더로 버퍼 왕복 검증.
  ///
  /// 셰이더: 각 원소에 1을 더한다.
  #[test]
  fn test_gpu_roundtrip() {
    let ctx = GpuContext::new().expect("GPU 초기화 실패");
    println!("어댑터: {}", ctx.backend_name());

    // 입력 버퍼 [0, 1, 2, 3, ...]
    let n: usize = 256;
    let input: Vec<u32> = (0..n as u32).collect();
    let in_buf = GpuBuffer::from_slice(&ctx, &input, BufferUsages::COPY_SRC);

    // 출력 버퍼 (0으로 초기화)
    let out_buf = GpuBuffer::<u32>::zeroed(&ctx, n, BufferUsages::COPY_SRC);

    // 셰이더: in[i] + 1 → out[i]
    const WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       in_buf:  array<u32>;
@group(0) @binding(1) var<storage, read_write>  out_buf: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < arrayLength(&in_buf) {
        out_buf[i] = in_buf[i] + 1u;
    }
}
"#;

    let mut cache = PipelineCache::new();
    let pipeline = cache.get_or_compile(&ctx, "add_one", WGSL, "main");

    // 바인드 그룹 수동 생성
    let bg_layout = pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: None,
      layout: &bg_layout,
      entries: &[
        wgpu::BindGroupEntry {
          binding: 0,
          resource: in_buf.inner.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding: 1,
          resource: out_buf.inner.as_entire_binding(),
        },
      ],
    });

    Dispatcher::new(&ctx).dispatch(pipeline, &[&bind_group], [n as u32 / 64, 1, 1]);

    let result = out_buf.read(&ctx).expect("readback 실패");
    for (i, &v) in result.iter().enumerate() {
      assert_eq!(v, i as u32 + 1, "index {i}: expected {}, got {v}", i + 1);
    }
  }
}
