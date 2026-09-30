//! # engine-mlx-spill
//!
//! SSD expert streaming for engine-mlx (MoE models).
//!
//! Uses `mmap` + `madvise(MADV_WILLNEED/DONTNEED)` to stream expert weights
//! from SSD on-demand, keeping only active experts in RAM. The kernel's page
//! cache manages eviction — no userspace LRU needed.
//!
//! Backend-agnostic: this crate only manages expert weight layout and SSD
//! streaming. Compute backend (MLX forward) is provided by the sibling
//! crates in the `engine-mlx-ssd` workspace. The MLX-gated forward/loader
//! path is reserved via the `mlx` feature for future integration.

pub mod config;
pub mod engine_core;
pub mod engine_trait;
pub mod expert_index;
pub mod expert_tracker;
pub mod interface;
pub mod mmap_experts;
pub mod speculative_prefetch;
pub mod types;

pub use config::SpillConfig;
pub use engine_trait::SpillEngineTrait;
pub use expert_index::{ExpertOffset, TensorRange};
pub use expert_tracker::ExpertTracker;
pub use mmap_experts::MmapExperts;
pub use speculative_prefetch::SpeculativePrefetch;