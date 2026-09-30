use std::path::PathBuf;

/// Configuration for the Spill streaming engine.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpillConfig {
    /// Path to the model directory (containing config.json + safetensors).
    pub model_path: PathBuf,

    /// RAM budget in bytes for the expert pool.
    /// Default: 4 GB (leaves room for OS + KV cache on 16 GB machine).
    #[serde(default = "default_ram_budget")]
    pub ram_budget_bytes: usize,

    /// How many layers ahead to prefetch.
    /// Default: 2 (prefetch layer N+2 while computing layer N).
    #[serde(default = "default_prefetch_depth")]
    pub prefetch_depth: usize,

    /// SSD bandwidth in GB/s (for timing estimates).
    /// M1 NVMe: ~2.5, M3/M4: ~3.5-7.0.
    #[serde(default = "default_ssd_bandwidth")]
    pub ssd_bandwidth_gbps: f32,

    /// Path to the LFM 2.5 model for predictive prefetch (Phase 4).
    /// If None, uses previous-token reuse strategy.
    #[serde(default)]
    pub lfm_model_path: Option<PathBuf>,

    /// Whether to pin attention weights in RAM.
    /// Default: true (reduces I/O but uses more RAM).
    #[serde(default = "default_true")]
    pub pin_attention: bool,

    /// Whether to pin shared expert weights in RAM.
    /// Default: true.
    #[serde(default = "default_true")]
    pub pin_shared_expert: bool,

    /// Prefill mode: bulk madvise for all experts, disable speculative prefetch.
    /// Set to true during prefill, false during decode.
    #[serde(default)]
    pub prefill_mode: bool,
}

fn default_ram_budget() -> usize {
    4 * 1024 * 1024 * 1024 // 4 GB
}

fn default_prefetch_depth() -> usize {
    2
}

fn default_ssd_bandwidth() -> f32 {
    2.5
}

fn default_true() -> bool {
    true
}

impl Default for SpillConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            ram_budget_bytes: default_ram_budget(),
            prefetch_depth: default_prefetch_depth(),
            ssd_bandwidth_gbps: default_ssd_bandwidth(),
            lfm_model_path: None,
            pin_attention: true,
            pin_shared_expert: true,
            prefill_mode: false,
        }
    }
}
