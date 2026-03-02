//! Shared WGSL shader fragments for GF(2^128) tower arithmetic.
//!
//! Other crates concatenate [`GF128_WGSL`] with their kernel-specific WGSL source
//! before passing to `PipelineCache::get_or_compile`.

/// GF(2^128) tower field arithmetic WGSL source.
///
/// Provides: `clmul32`, `reduce32`, `gf32_mul`, `gf64_mul`, `gf64_add`,
/// `gf128_mul`, `gf128_add`, `gf128_sqr`, `gf128_pow2n`, `gf128_inv`,
/// `GF128_ZERO`, `GF128_ONE`, `GF128_ALPHA`.
pub const GF128_WGSL: &str = include_str!("shaders/gf128.wgsl");

/// Concatenate the shared GF(2^128) arithmetic with a kernel-specific WGSL source.
pub fn shader_with_gf128(kernel_src: &str) -> String {
  format!("{}\n\n{}", GF128_WGSL, kernel_src)
}
