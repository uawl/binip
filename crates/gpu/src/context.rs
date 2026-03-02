use std::mem::ManuallyDrop;

use wgpu::Backends;

use crate::error::GpuError;

/// wgpu 디바이스 + 큐를 보유하는 GPU 컨텍스트.  
/// `GpuContext::new()`로 생성하며, 내부적으로 `pollster::block_on`으로
/// 비동기 초기화를 동기화한다.
///
/// device/queue를 `ManuallyDrop`으로 감싸 프로세스 종료 시 dzn 드라이버의
/// cleanup SIGSEGV를 회피한다. OS가 프로세스와 함께 정리한다.
pub struct GpuContext {
  pub device: ManuallyDrop<wgpu::Device>,
  pub queue: ManuallyDrop<wgpu::Queue>,
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
      flags: wgpu::InstanceFlags::default()
        | wgpu::InstanceFlags::ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER, // For WSL2 dzn
      ..Default::default()
    });

    let adapters = instance.enumerate_adapters(Backends::all()).await;
    // DiscreteGpu > IntegratedGpu > 나머지 순으로 우선 선택
    let adapter = adapters
      .iter()
      .min_by_key(|a| match a.get_info().device_type {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::VirtualGpu => 2,
        wgpu::DeviceType::Other => 3,
        wgpu::DeviceType::Cpu => 4,
      })
      .cloned()
      .ok_or(GpuError::NoAdapter)?;

    let adapter_info = adapter.get_info();

    let (device, queue) = adapter
      .request_device(&wgpu::DeviceDescriptor {
        label: Some("binip"),
        required_features: wgpu::Features::empty(),
        required_limits: adapter.limits(),
        memory_hints: wgpu::MemoryHints::default(),
        ..Default::default()
      })
      .await?;

    Ok(Self {
      device: ManuallyDrop::new(device),
      queue: ManuallyDrop::new(queue),
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

impl Drop for GpuContext {
  fn drop(&mut self) {
    // 진행 중인 GPU 작업 완료 대기
    let _ = self.device.poll(wgpu::PollType::Wait {
      submission_index: None,
      timeout: None,
    });
    // device/queue는 ManuallyDrop이므로 실제 해제하지 않는다.
    // dzn 드라이버의 cleanup SIGSEGV를 회피하기 위함.
  }
}
