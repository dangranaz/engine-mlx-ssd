//! High-level Spill engine trait for consistency with other engines.

use anyhow::Result;
use std::path::Path;

/// Trait defining the Spill streaming engine contract.
pub trait SpillEngineTrait {
    /// Load model from directory.
    fn load(&mut self, model_dir: &Path) -> Result<()>;

    /// Forward pass: process tokens through the SSD streaming graph.
    /// Returns logits from the last position.
    fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>>;

    /// Decode: generate next token given previous token.
    fn step(&mut self, token_id: u32) -> Result<u32>;

    /// Prefill: process prompt tokens and return logits.
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>>;

    /// Reset engine state (KV cache, expert tracker, etc.).
    fn reset(&mut self);

    /// Engine name for logging.
    fn name(&self) -> &str;

    /// Expected tokens per second.
    fn expected_tps(&self) -> f32;
}
