use crate::error::GpuError;

/// wgpu 디바이스 + 큐를 보유하는 GPU 컨텍스트.  
/// `GpuContext::new()`로 생성하며, 내부적으로 `pollster::block_on`으로
/// 비동기 초기화를 동기화한다.
pub struct GpuContext {
  pub device: wgpu::Device,
  pub queue: wgpu::Queue,
  pub adapter_info: wgpu::AdapterInfo,
}

impl GpuContext {
  /// 고성능 어댑터를 우선 탐색해 디바이스를 초기화한다.
  pub fn new() -> Result<Self, GpuError> {
    pollster::block_on(Self::init_async())
  }

  async fn init_async() -> Result<Self, GpuError> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
      backends: wgpu::Backends::all(),
      ..Default::default()
    });

    let adapter = instance
      .request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
      })
      .await
      .map_err(|_| GpuError::NoAdapter)?;

    let adapter_info = adapter.get_info();

    let (device, queue) = adapter
      .request_device(&wgpu::DeviceDescriptor {
        label: Some("binip"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        memory_hints: wgpu::MemoryHints::default(),
        ..Default::default()
      })
      .await?;

    Ok(Self {
      device,
      queue,
      adapter_info,
    })
  }

  /// 어댑터 이름 + 백엔드 종류를 반환한다.
  pub fn backend_name(&self) -> String {
    format!(
      "{} ({:?})",
      self.adapter_info.name, self.adapter_info.backend
    )
  }
}
