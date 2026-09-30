//! Speculative prefetch — triggers madvise during GDN compute.
//!
//! While the GPU is busy with GDN layers (no I/O needed),
//! this module prefetches experts for upcoming MoE layers.

use anyhow::Result;

use crate::expert_tracker::ExpertTracker;
use crate::mmap_experts::MmapExperts;

/// Speculative prefetch controller.
pub struct SpeculativePrefetch {
    tracker: ExpertTracker,
    prefetch_depth: usize,
    hits: u64,
    misses: u64,
}

impl SpeculativePrefetch {
    pub fn new(moe_layer_indices: &[usize], num_experts: usize, prefetch_depth: usize) -> Self {
        Self {
            tracker: ExpertTracker::new(moe_layer_indices, num_experts),
            prefetch_depth,
            hits: 0,
            misses: 0,
        }
    }

    /// Called after router selects experts for a layer.
    /// Records selection and triggers lookahead prefetch.
    pub fn on_expert_selected(
        &mut self,
        store: &MmapExperts,
        layer: usize,
        selected_experts: &[usize],
    ) -> Result<()> {
        // Record for future predictions
        self.tracker.record(layer, selected_experts);

        // Prefetch next MoE layers speculatively
        for depth in 1..=self.prefetch_depth {
            let next_layer = layer + depth * 4; // MoE layers are every 4th (3:1 GDN:attn)
            if !store.is_moe_layer(next_layer) { continue; }

            let predicted = self.tracker.predict(next_layer);
            if !predicted.is_empty() {
                store.prefetch_experts(next_layer, &predicted)?;
                tracing::trace!(
                    target: "nexum::ssd::prefetch",
                    layer = next_layer,
                    experts = ?predicted,
                    "speculative prefetch"
                );
            }
        }
        Ok(())
    }

    /// Called before a decode step starts — prefetch all predicted experts.
    pub fn prefetch_for_step(&self, store: &MmapExperts) -> Result<()> {
        let predictions = self.tracker.predict_all_layers();
        for (layer, experts) in &predictions {
            if !experts.is_empty() {
                store.prefetch_experts(*layer, experts)?;
            }
        }
        Ok(())
    }

    /// Record a cache hit (expert was already in page cache).
    pub fn record_hit(&mut self) { self.hits += 1; }

    /// Record a cache miss (expert required SSD read).
    pub fn record_miss(&mut self) { self.misses += 1; }

    /// Hit rate.
    pub fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total == 0 { return 0.0; }
        self.hits as f32 / total as f32
    }

    /// Update pinned experts (call every ~100 tokens).
    pub fn update_pins(&mut self, store: &MmapExperts, pin_threshold_pct: f32) -> Result<()> {
        self.tracker.update_pins(pin_threshold_pct);
        // Prefetch all pinned experts to ensure they stay in page cache
        for (layer, experts) in self.tracker.pinned_experts() {
            store.prefetch_experts(*layer, experts)?;
        }
        Ok(())
    }

    /// Get prefetch stats for /v1/metrics.
    pub fn stats(&self) -> PrefetchStats {
        PrefetchStats {
            hits: self.hits,
            misses: self.misses,
            hit_rate: self.hit_rate(),
            tracker: self.tracker.stats(),
        }
    }
}

/// Stats for observability.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PrefetchStats {
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f32,
    pub tracker: crate::expert_tracker::TrackerStats,
}
