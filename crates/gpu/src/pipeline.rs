use std::collections::HashMap;

use crate::GpuContext;

/// WGSL 컴퓨트 셰이더를 컴파일하고 `ComputePipeline`을 캐시하는 저장소.
///
/// 동일한 `label`로 재호출 시 재컴파일 없이 기존 파이프라인을 반환한다.
///
/// # 예시
/// ```ignore
/// let mut cache = PipelineCache::new();
/// let pipeline = cache.get_or_compile(&ctx, "my_kernel", WGSL_SRC, "main");
/// ```
pub struct PipelineCache {
  pipelines: HashMap<String, wgpu::ComputePipeline>,
}

impl PipelineCache {
  pub fn new() -> Self {
    Self {
      pipelines: HashMap::new(),
    }
  }

  /// `label`에 해당하는 파이프라인이 없으면 `wgsl_src`를 컴파일해 캐시에 삽입하고
  /// 참조를 반환한다.
  ///
  /// - `label`: 캐시 키 및 wgpu 디버그 레이블
  /// - `wgsl_src`: WGSL 셰이더 소스
  /// - `entry_point`: 컴퓨트 진입점 함수 이름 (보통 `"main"`)
  pub fn get_or_compile<'a>(
    &'a mut self,
    ctx: &GpuContext,
    label: &str,
    wgsl_src: &str,
    entry_point: &str,
  ) -> &'a wgpu::ComputePipeline {
    self.pipelines.entry(label.to_owned()).or_insert_with(|| {
      let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
          label: Some(label),
          source: wgpu::ShaderSource::Wgsl(wgsl_src.into()),
        });
      ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
          label: Some(label),
          layout: None, // auto-layout (wgpu가 bind group layout 자동 추론)
          module: &shader,
          entry_point: Some(entry_point),
          compilation_options: wgpu::PipelineCompilationOptions::default(),
          cache: None,
        })
    })
  }

  /// 캐시에서 이미 컴파일된 파이프라인을 불변 참조로 꺼낸다.
  pub fn get(&self, label: &str) -> Option<&wgpu::ComputePipeline> {
    self.pipelines.get(label)
  }

  /// 캐시에서 파이프라인을 제거한다 (셰이더 핫-리로드 등에 사용).
  pub fn invalidate(&mut self, label: &str) {
    self.pipelines.remove(label);
  }
}

impl Default for PipelineCache {
  fn default() -> Self {
    Self::new()
  }
}

/// 컴퓨트 패스 실행을 위한 헬퍼.
///
/// 단일 `dispatch_workgroups` 호출을 커맨드 버퍼로 감싸 제출한다.
pub struct Dispatcher<'a> {
  ctx: &'a GpuContext,
}

impl<'a> Dispatcher<'a> {
  pub fn new(ctx: &'a GpuContext) -> Self {
    Self { ctx }
  }

  /// 파이프라인을 바인드 그룹과 함께 `[x, y, z]` 워크그룹으로 디스패치한다.
  ///
  /// `bind_groups[i]`는 셰이더의 `@group(i)` 에 매핑된다.
  pub fn dispatch(
    &self,
    pipeline: &wgpu::ComputePipeline,
    bind_groups: &[&wgpu::BindGroup],
    workgroups: [u32; 3],
  ) {
    let mut encoder = self
      .ctx
      .device
      .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
      let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
      });
      pass.set_pipeline(pipeline);
      for (i, bg) in bind_groups.iter().enumerate() {
        pass.set_bind_group(i as u32, *bg, &[]);
      }
      pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
    }
    self.ctx.queue.submit(std::iter::once(encoder.finish()));
  }
}
