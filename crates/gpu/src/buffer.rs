use std::{marker::PhantomData, mem};

use bytemuck::Pod;

use crate::{GpuContext, GpuError};

/// GPU 스토리지 버퍼의 타입-안전 래퍼.
///
/// `T`는 `bytemuck::Pod`를 구현해야 한다 (예: `[u32; 4]`, `f32`).
///
/// # 주요 사용법
/// ```ignore
/// let buf = GpuBuffer::from_slice(&ctx, &data, wgpu::BufferUsages::empty());
/// // ... GPU 커널 실행 ...
/// let result = buf.read(&ctx)?;
/// ```
pub struct GpuBuffer<T: Pod> {
  pub inner: wgpu::Buffer,
  /// 원소(T) 개수
  pub len: usize,
  _phantom: PhantomData<T>,
}

impl<T: Pod> GpuBuffer<T> {
  /// 지정된 usage로 0-초기화 스토리지 버퍼 생성.
  /// `usage`에 `STORAGE`는 자동으로 추가한다.
  pub fn zeroed(ctx: &GpuContext, len: usize, usage: wgpu::BufferUsages) -> Self {
    let byte_size = Self::padded_size(len);
    let inner = ctx.device.create_buffer(&wgpu::BufferDescriptor {
      label: None,
      size: byte_size,
      usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | usage,
      mapped_at_creation: false, // WebGPU spec: always zero-initialised
    });
    Self {
      inner,
      len,
      _phantom: PhantomData,
    }
  }

  /// 슬라이스 데이터로 스토리지 버퍼 생성.
  /// `extra_usage`에 `COPY_SRC` 등 추가 플래그를 지정할 수 있다.
  pub fn from_slice(ctx: &GpuContext, data: &[T], extra_usage: wgpu::BufferUsages) -> Self {
    let byte_size = Self::padded_size(data.len());
    let inner = ctx.device.create_buffer(&wgpu::BufferDescriptor {
      label: None,
      size: byte_size,
      usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra_usage,
      mapped_at_creation: true,
    });
    // 초기화 시 host-ptr 직접 기록 (upload queue 왕복 없음)
    let raw = bytemuck::cast_slice::<T, u8>(data);
    inner
      .slice(..raw.len() as u64)
      .get_mapped_range_mut()
      .copy_from_slice(raw);
    inner.unmap();
    Self {
      inner,
      len: data.len(),
      _phantom: PhantomData,
    }
  }

  /// `queue.write_buffer`로 데이터를 덮어쓴다.
  /// `data.len()` == `self.len`이어야 한다.
  pub fn write(&self, ctx: &GpuContext, data: &[T]) {
    assert_eq!(data.len(), self.len, "GpuBuffer::write: length mismatch");
    ctx
      .queue
      .write_buffer(&self.inner, 0, bytemuck::cast_slice(data));
  }

  /// GPU 버퍼 내용을 CPU Vec<T>로 읽어온다 (staged readback).
  ///
  /// 내부적으로 COPY_DST | MAP_READ staging 버퍼를 생성하고
  /// `device.poll(Maintain::Wait)`로 동기화한 뒤 반환한다.
  pub fn read(&self, ctx: &GpuContext) -> Result<Vec<T>, GpuError> {
    let byte_size = Self::padded_size(self.len);

    // 스테이징 버퍼: CPU로 읽을 수 있는 임시 버퍼
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
      label: Some("staging_readback"),
      size: byte_size,
      usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
      mapped_at_creation: false,
    });

    // self → staging 복사 명령 제출
    let mut encoder = ctx
      .device
      .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(&self.inner, 0, &staging, 0, byte_size);
    ctx.queue.submit(std::iter::once(encoder.finish()));

    // 비동기 맵 요청
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
      let _ = tx.send(res);
    });

    // GPU 완료 대기 (블로킹)
    ctx
      .device
      .poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
      })
      .map_err(|_| GpuError::MapFailed)?;
    rx.recv().unwrap().map_err(|_| GpuError::MapFailed)?;

    let data = {
      let view = slice.get_mapped_range();
      // 실제 원소 수(self.len)만큼만 취한다 (패딩 제외)
      let bytes = &view[..self.len * mem::size_of::<T>()];
      bytemuck::cast_slice::<u8, T>(bytes).to_vec()
    };
    staging.unmap();
    Ok(data)
  }

  /// 버퍼의 바이트 크기 (패딩 포함).
  pub fn size_bytes(&self) -> u64 {
    Self::padded_size(self.len)
  }

  /// wgpu 최소 정렬 요구사항(4 bytes)을 맞춘 바이트 크기.
  fn padded_size(len: usize) -> u64 {
    let bytes = (len * mem::size_of::<T>()).max(4);
    bytes.next_multiple_of(4) as u64
  }
}
