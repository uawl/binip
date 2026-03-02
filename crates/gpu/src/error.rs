use thiserror::Error;

#[derive(Debug, Error)]
pub enum GpuError {
  #[error("no suitable GPU adapter found")]
  NoAdapter,

  #[error("failed to request GPU device: {0}")]
  RequestDevice(#[from] wgpu::RequestDeviceError),

  #[error("buffer map failed")]
  MapFailed,
}
