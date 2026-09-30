//! Public trait defining the SSD engine contract.

use std::path::Path;

use anyhow::Result;

use crate::config::SpillConfig;
use crate::mmap_experts::MmapExperts;

/// Trait for creating and accessing mmap-based expert stores.
pub trait ExpertStore: Send + Sync {
    /// Open an expert store from a safetensors file.
    fn open(path: &Path, config: &SpillConfig) -> Result<Self> where Self: Sized;

    /// Prefetch experts into RAM (madvise WILLNEED).
    fn prefetch(&self, layer: usize, experts: &[usize]) -> Result<()>;

    /// Release experts from RAM (madvise DONTNEED).
    fn release(&self, layer: usize, experts: &[usize]) -> Result<()>;

    /// Check if a layer has MoE experts.
    fn is_moe(&self, layer: usize) -> bool;

    /// Number of experts in a layer (0 if not MoE).
    fn expert_count(&self, layer: usize) -> usize;

    /// Hidden dimension size.
    fn hidden_size(&self) -> usize;

    /// Intermediate dimension size per expert.
    fn intermediate_size(&self) -> usize;
}

impl ExpertStore for MmapExperts {
    fn open(path: &Path, config: &SpillConfig) -> Result<Self> {
        Self::open(path, config)
    }

    fn prefetch(&self, layer: usize, experts: &[usize]) -> Result<()> {
        self.prefetch_experts(layer, experts)
    }

    fn release(&self, layer: usize, experts: &[usize]) -> Result<()> {
        self.release_experts(layer, experts)
    }

    fn is_moe(&self, layer: usize) -> bool {
        self.is_moe_layer(layer)
    }

    fn expert_count(&self, layer: usize) -> usize {
        self.index_len(layer)
    }

    fn hidden_size(&self) -> usize {
        self.hidden_size()
    }

    fn intermediate_size(&self) -> usize {
        self.intermediate_size()
    }
}
