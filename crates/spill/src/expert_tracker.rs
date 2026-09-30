//! Expert reuse tracking and speculative prefetch.
//!
//! Tracks which experts are activated per layer across tokens.
//! Predicts next-token experts based on reuse patterns.
//! Triggers madvise(WILLNEED) speculatively during GDN layer compute.

use std::collections::HashMap;

/// Per-layer expert activation history.
struct LayerHistory {
    /// Ring buffer of recent expert selections (last N tokens).
    recent: Vec<Vec<usize>>,
    /// Write position in ring buffer.
    pos: usize,
    /// Activation frequency count per expert.
    freq: Vec<u32>,
    /// Total tokens processed.
    total: u64,
}

impl LayerHistory {
    fn new(num_experts: usize, window: usize) -> Self {
        Self {
            recent: vec![Vec::new(); window],
            pos: 0,
            freq: vec![0; num_experts],
            total: 0,
        }
    }

    fn record(&mut self, experts: &[usize]) {
        self.recent[self.pos] = experts.to_vec();
        self.pos = (self.pos + 1) % self.recent.len();
        for &e in experts { self.freq[e] += 1; }
        self.total += 1;
    }

    /// Predict next experts based on last-token reuse (most common pattern).
    fn predict_reuse(&self) -> Vec<usize> {
        // Previous token's experts are 70-90% likely to be reused
        let prev = (self.pos + self.recent.len() - 1) % self.recent.len();
        self.recent[prev].clone()
    }

    /// Get top-K hot experts (most frequently activated).
    fn hot_experts(&self, k: usize) -> Vec<usize> {
        let mut indexed: Vec<(usize, u32)> = self.freq.iter().enumerate()
            .map(|(i, &f)| (i, f)).collect();
        indexed.sort_by(|a, b| b.1.cmp(&a.1));
        indexed.into_iter().take(k).map(|(i, _)| i).collect()
    }
}

/// Expert tracker — manages predictions across all MoE layers.
pub struct ExpertTracker {
    layers: HashMap<usize, LayerHistory>,
    num_experts: usize,
    window: usize,
    /// Experts to pin permanently (hot across all tokens).
    pinned: HashMap<usize, Vec<usize>>,
}

impl ExpertTracker {
    /// Create tracker for a model with given number of experts per MoE layer.
    pub fn new(moe_layer_indices: &[usize], num_experts: usize) -> Self {
        let window = 32; // track last 32 tokens
        let mut layers = HashMap::new();
        for &idx in moe_layer_indices {
            layers.insert(idx, LayerHistory::new(num_experts, window));
        }
        Self { layers, num_experts, window, pinned: HashMap::new() }
    }

    /// Record which experts were activated for a token at a specific layer.
    pub fn record(&mut self, layer: usize, experts: &[usize]) {
        if let Some(hist) = self.layers.get_mut(&layer) {
            hist.record(experts);
        }
    }

    /// Predict which experts will be needed for the next token at given layer.
    /// Returns expert indices sorted by likelihood.
    pub fn predict(&self, layer: usize) -> Vec<usize> {
        let Some(hist) = self.layers.get(&layer) else { return vec![] };

        let mut predicted = hist.predict_reuse();

        // Add hot experts not already in prediction
        let hot = hist.hot_experts(4);
        for e in hot {
            if !predicted.contains(&e) { predicted.push(e); }
        }

        predicted
    }

    /// Predict experts for ALL MoE layers at once (batch prefetch).
    /// Useful before a decode step to prefetch everything needed.
    pub fn predict_all_layers(&self) -> Vec<(usize, Vec<usize>)> {
        self.layers.iter()
            .map(|(&layer, hist)| {
                let mut pred = hist.predict_reuse();
                let hot = hist.hot_experts(4);
                for e in hot {
                    if !pred.contains(&e) { pred.push(e); }
                }
                (layer, pred)
            })
            .collect()
    }

    /// Update pinned experts based on frequency data.
    /// Call periodically (e.g., every 100 tokens).
    pub fn update_pins(&mut self, pin_threshold_pct: f32) {
        self.pinned.clear();
        for (&layer, hist) in &self.layers {
            if hist.total < 10 { continue; }
            let threshold = (hist.total as f32 * pin_threshold_pct) as u32;
            let hot: Vec<usize> = hist.freq.iter().enumerate()
                .filter(|(_, &f)| f > threshold)
                .map(|(i, _)| i)
                .collect();
            if !hot.is_empty() {
                self.pinned.insert(layer, hot);
            }
        }
    }

    /// Get experts that should be pinned (always in RAM).
    pub fn pinned_experts(&self) -> &HashMap<usize, Vec<usize>> {
        &self.pinned
    }

    /// Get stats for metrics.
    pub fn stats(&self) -> TrackerStats {
        let total_predictions: usize = self.layers.values()
            .map(|h| h.recent.iter().filter(|r| !r.is_empty()).count())
            .sum();
        let total_pinned: usize = self.pinned.values().map(|v| v.len()).sum();
        TrackerStats { total_predictions, total_pinned, num_layers: self.layers.len() }
    }
}

/// Stats for observability.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TrackerStats {
    pub total_predictions: usize,
    pub total_pinned: usize,
    pub num_layers: usize,
}
