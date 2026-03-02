//! Recursive proof configuration — shard splitting and recursion parameters.

/// Parameters controlling how an MLE is split into shards and how
/// recursive aggregation proceeds.
///
/// # Layout
///
/// A `total_vars`-variable MLE has `2^total_vars` evaluations.
/// It is split into `2^(total_vars - shard_vars)` shards, each containing
/// `2^shard_vars` consecutive evaluations (a sub-MLE).
///
/// At each recursion level, `fan_in` proofs are aggregated into one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecursiveConfig {
  /// Total number of MLE variables (e.g. 20 → 2^20 evaluations).
  pub total_vars: u32,
  /// Number of variables per shard (e.g. 8 → 2^8 = 256 evaluations per shard).
  pub shard_vars: u32,
  /// Number of child proofs aggregated per recursion node (default: 64).
  pub fan_in: u32,
}

impl Default for RecursiveConfig {
  fn default() -> Self {
    Self { total_vars: 20, shard_vars: 10, fan_in: 64 }
  }
}

impl RecursiveConfig {
  /// Build a config for the given total MLE variables.
  ///
  /// Uses `fan_in = 64` and `shard_vars = max(1, total_vars / 2)`.
  pub fn for_n_vars(total_vars: u32) -> Self {
    let shard_vars = (total_vars / 2).max(1).min(total_vars);
    Self { total_vars, shard_vars, fan_in: 64 }
  }
  /// Number of shards: `2^(total_vars - shard_vars)`.
  pub fn n_shards(&self) -> u32 {
    assert!(
      self.total_vars >= self.shard_vars,
      "total_vars ({}) must be >= shard_vars ({})",
      self.total_vars,
      self.shard_vars
    );
    1u32 << (self.total_vars - self.shard_vars)
  }

  /// Recursion depth: `ceil(log_{fan_in}(n_shards))`.
  ///
  /// Returns 0 when `n_shards <= 1`.
  pub fn depth(&self) -> u32 {
    let n = self.n_shards();
    if n <= 1 {
      return 0;
    }
    // ceil(log_k(n)) via repeated division
    let mut remaining = n;
    let mut d = 0u32;
    while remaining > 1 {
      remaining = (remaining + self.fan_in - 1) / self.fan_in;
      d += 1;
    }
    d
  }

  /// Number of recursive proof instances at level `l`.
  ///
  /// Level 0 merges shards, level 1 merges level-0 proofs, etc.
  pub fn instances_at_level(&self, l: u32) -> u32 {
    let mut n = self.n_shards();
    for _ in 0..=l {
      n = (n + self.fan_in - 1) / self.fan_in;
    }
    n
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn default_cfg() -> RecursiveConfig {
    RecursiveConfig { total_vars: 20, shard_vars: 8, fan_in: 8 }
  }

  #[test]
  fn n_shards_basic() {
    assert_eq!(default_cfg().n_shards(), 1 << 12); // 4096
  }

  #[test]
  fn n_shards_equal_vars() {
    let cfg = RecursiveConfig { total_vars: 8, shard_vars: 8, fan_in: 4 };
    assert_eq!(cfg.n_shards(), 1);
  }

  #[test]
  fn depth_default() {
    // 4096 shards, fan_in=8: 4096/8=512 → 512/8=64 → 64/8=8 → 8/8=1 → depth 4
    assert_eq!(default_cfg().depth(), 4);
  }

  #[test]
  fn depth_single_shard() {
    let cfg = RecursiveConfig { total_vars: 8, shard_vars: 8, fan_in: 4 };
    assert_eq!(cfg.depth(), 0);
  }

  #[test]
  fn depth_small() {
    // 4 shards, fan_in=2: 4/2=2 → 2/2=1 → depth 2
    let cfg = RecursiveConfig { total_vars: 4, shard_vars: 2, fan_in: 2 };
    assert_eq!(cfg.n_shards(), 4);
    assert_eq!(cfg.depth(), 2);
  }

  #[test]
  fn instances_at_level_default() {
    let cfg = default_cfg();
    // L0: 4096/8 = 512
    assert_eq!(cfg.instances_at_level(0), 512);
    // L1: 512/8 = 64
    assert_eq!(cfg.instances_at_level(1), 64);
    // L2: 64/8 = 8
    assert_eq!(cfg.instances_at_level(2), 8);
    // L3: 8/8 = 1
    assert_eq!(cfg.instances_at_level(3), 1);
  }
}
