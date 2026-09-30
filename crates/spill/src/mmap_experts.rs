//! Memory-mapped expert access with madvise-based prefetch.
//!
//! Opens a safetensors file, builds an expert index (byte ranges per expert per layer),
//! and provides madvise(WILLNEED/DONTNEED) for streaming experts from SSD.

use std::collections::HashSet;
use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result};
use memmap2::{Advice, Mmap, MmapOptions};
use safetensors::SafeTensors;

use crate::config::SpillConfig;
use crate::expert_index::{build_expert_offset, ExpertOffset, TensorRange};
#[cfg(feature = "mlx")]
use crate::expert_index::ExpertArrays;

/// Memory-mapped expert access with madvise-based prefetch.
pub struct MmapExperts {
    mmaps: Vec<Mmap>,
    index: Vec<Vec<ExpertOffset>>,
    moe_layers: Vec<usize>,
    hidden_size: usize,
    intermediate_size: usize,
}

impl MmapExperts {
    /// Open safetensors file(s) and build the expert index.
    /// Supports both single-file and multi-shard models.
    pub fn open(path: &Path, _config: &SpillConfig) -> Result<Self> {
        let (model_config, model_dir) = load_model_config(path)?;
        let hidden_size = model_config["hidden_size"].as_u64().unwrap_or(2048) as usize;
        let intermediate_size = model_config["moe_intermediate_size"].as_u64()
            .or_else(|| model_config["intermediate_size"].as_u64())
            .unwrap_or(512) as usize;
        let num_layers = model_config["num_hidden_layers"].as_u64().unwrap_or(64) as usize;
        let num_experts = model_config["num_experts"].as_u64()
            .or_else(|| model_config["num_local_experts"].as_u64())
            .unwrap_or(256) as usize;

        tracing::info!(
            target: "nexum::ssd",
            "Building expert index: {} layers, {} experts/layer, hidden={}, intermediate={}",
            num_layers, num_experts, hidden_size, intermediate_size
        );

        // Determine if sharded or single file
        let index_path = model_dir.join("model.safetensors.index.json");
        let (mmaps, shard_paths) = if index_path.exists() {
            // Multi-shard: open each shard
            let index_str = std::fs::read_to_string(&index_path)?;
            let index_json: serde_json::Value = serde_json::from_str(&index_str)?;
            let weight_map = index_json["weight_map"].as_object().context("weight_map")?;
            let mut shard_names: Vec<String> = weight_map.values()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();
            shard_names.sort();

            let mut mmaps = Vec::new();
            let mut paths = Vec::new();
            for shard in &shard_names {
                let shard_path = model_dir.join(shard);
                let file = File::open(&shard_path)
                    .with_context(|| format!("open shard: {}", shard_path.display()))?;
                let mmap = unsafe { MmapOptions::new().map(&file)? };
                mmaps.push(mmap);
                paths.push(shard_path);
            }
            (mmaps, paths)
        } else {
            // Single file
            let file = File::open(path)
                .with_context(|| format!("open safetensors: {}", path.display()))?;
            let mmap = unsafe { MmapOptions::new().map(&file)? };
            (vec![mmap], vec![path.to_path_buf()])
        };

        // Build expert index across all shards
        // Key insight: a single layer may have tensors split across multiple shards!
        let mut index: Vec<Vec<ExpertOffset>> = vec![Vec::new(); num_layers];
        let mut moe_layers = Vec::new();
        let mut layer_mmap_idx = vec![0usize; num_layers];

        // Parse all shards once
        let parsed_shards: Vec<SafeTensors> = mmaps.iter()
            .map(|m| SafeTensors::deserialize(m).expect("parse safetensors"))
            .collect();

        // Detect expert prefix using first shard that has experts
        let expert_prefix_fn = {
            let mut found_fn: Option<Box<dyn Fn(usize) -> String>> = None;
            for st in &parsed_shards {
                let f = detect_expert_prefix(st, num_layers);
                // Test if this shard has any experts
                let test_name = f(0);
                let gate_test = format!("{test_name}.gate_proj.weight");
                if st.tensor(&gate_test).is_ok() {
                    found_fn = Some(f);
                    break;
                }
            }
            found_fn.unwrap_or_else(|| Box::new(|i| format!("model.layers.{i}.mlp.experts")))
        };

        // For each layer, build expert offsets by searching across all shards
        for layer_idx in 0..num_layers {
            let layer_prefix = expert_prefix_fn(layer_idx);
            let gate_name = format!("{layer_prefix}.gate_proj.weight");

            // Check if any shard has this layer's experts
            let has_experts = parsed_shards.iter().any(|st| st.tensor(&gate_name).is_ok());
            if !has_experts {
                continue;
            }

            moe_layers.push(layer_idx);

            let mut layer_experts = Vec::with_capacity(num_experts);
            for expert_idx in 0..num_experts {
                let offset = build_expert_offset_multi_shard(
                    &parsed_shards, &mmaps, layer_idx, expert_idx, &layer_prefix,
                )?;
                layer_experts.push(offset);
            }
            index[layer_idx] = layer_experts;
        }

        moe_layers.sort();
        moe_layers.dedup();

        tracing::info!(
            target: "nexum::ssd",
            "Expert index built: {} MoE layers, {} experts total, {} shards",
            moe_layers.len(), moe_layers.len() * num_experts, mmaps.len()
        );

        Ok(Self { mmaps, index, moe_layers, hidden_size, intermediate_size })
    }

    /// Prefetch experts for a layer (madvise WILLNEED).
    pub fn prefetch_experts(&self, layer: usize, expert_indices: &[usize]) -> Result<()> {
        let Some(offsets) = self.index.get(layer) else { return Ok(()) };
        if offsets.is_empty() { return Ok(()); }

        for &expert_idx in expert_indices {
            if let Some(offset) = offsets.get(expert_idx) {
                // Prefetch all tensor ranges for this expert
                for range in offset.all_ranges() {
                    let mmap = &self.mmaps[range.mmap_idx];
                    mmap.advise_range(Advice::WillNeed, range.offset, range.len)?;
                }
            }
        }
        Ok(())
    }

    /// Bulk prefetch ALL experts for a layer (prefill mode).
    pub fn bulk_prefetch_layer(&self, layer: usize) -> Result<()> {
        let Some(offsets) = self.index.get(layer) else { return Ok(()) };
        if offsets.is_empty() { return Ok(()); }

        // Prefetch by shard: group ranges by mmap_idx
        for offset in offsets {
            for range in offset.all_ranges() {
                let mmap = &self.mmaps[range.mmap_idx];
                mmap.advise_range(Advice::Sequential, range.offset, range.len)?;
            }
        }
        Ok(())
    }

    /// Release ALL experts for a layer.
    pub fn release_layer(&self, layer: usize) -> Result<()> {
        let Some(offsets) = self.index.get(layer) else { return Ok(()) };
        if offsets.is_empty() { return Ok(()); }

        for offset in offsets {
            for range in offset.all_ranges() {
                let mmap = &self.mmaps[range.mmap_idx];
                unsafe {
                    mmap.unchecked_advise_range(memmap2::UncheckedAdvice::DontNeed, range.offset, range.len)?;
                }
            }
        }
        Ok(())
    }

    /// Release experts from RAM (madvise DONTNEED).
    pub fn release_experts(&self, layer: usize, expert_indices: &[usize]) -> Result<()> {
        let Some(offsets) = self.index.get(layer) else { return Ok(()) };
        if offsets.is_empty() { return Ok(()); }

        for &expert_idx in expert_indices {
            if let Some(offset) = offsets.get(expert_idx) {
                for range in offset.all_ranges() {
                    let mmap = &self.mmaps[range.mmap_idx];
                    unsafe {
                        mmap.unchecked_advise_range(
                            memmap2::UncheckedAdvice::DontNeed,
                            range.offset, range.len,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Create mlx_arrays from an expert's mmap data (zero-copy).
    ///
    /// # Safety
    /// The mmap must outlive the returned ExpertArrays.
    #[cfg(feature = "mlx")]
    pub unsafe fn expert_arrays(&self, layer: usize, expert_idx: usize) -> Result<ExpertArrays> {
        let offsets = self.index.get(layer)
            .and_then(|v| v.get(expert_idx))
            .context("expert_arrays out of range")?;

        use crate::expert_index::mlx_array_from_mmap as from_mmap;

        // Shape for each expert tensor: original shape without the first (num_experts) dim
        let expert_shape = |range: &crate::expert_index::TensorRange| -> Vec<i32> {
            if range.shape.len() > 1 {
                range.shape[1..].iter().map(|&s| s as i32).collect()
            } else {
                vec![1]
            }
        };

        Ok(ExpertArrays {
            gate_w: from_mmap(&self.mmaps[offsets.gate_w.mmap_idx], &offsets.gate_w, &expert_shape(&offsets.gate_w))?,
            gate_s: from_mmap(&self.mmaps[offsets.gate_s.mmap_idx], &offsets.gate_s, &expert_shape(&offsets.gate_s))?,
            gate_b: from_mmap(&self.mmaps[offsets.gate_b.mmap_idx], &offsets.gate_b, &expert_shape(&offsets.gate_b))?,
            up_w: from_mmap(&self.mmaps[offsets.up_w.mmap_idx], &offsets.up_w, &expert_shape(&offsets.up_w))?,
            up_s: from_mmap(&self.mmaps[offsets.up_s.mmap_idx], &offsets.up_s, &expert_shape(&offsets.up_s))?,
            up_b: from_mmap(&self.mmaps[offsets.up_b.mmap_idx], &offsets.up_b, &expert_shape(&offsets.up_b))?,
            down_w: from_mmap(&self.mmaps[offsets.down_w.mmap_idx], &offsets.down_w, &expert_shape(&offsets.down_w))?,
            down_s: from_mmap(&self.mmaps[offsets.down_s.mmap_idx], &offsets.down_s, &expert_shape(&offsets.down_s))?,
            down_b: from_mmap(&self.mmaps[offsets.down_b.mmap_idx], &offsets.down_b, &expert_shape(&offsets.down_b))?,
        })
    }

    /// Prefetch experts for the next layer based on predictions.
    pub fn prefetch_lookahead(&self, current_layer: usize, predicted: &[usize]) -> Result<()> {
        self.prefetch_experts(current_layer + 1, predicted)
    }

    /// Check if a layer is an MoE layer.
    pub fn is_moe_layer(&self, layer: usize) -> bool {
        self.index.get(layer).map_or(false, |v| !v.is_empty())
    }

    /// Get number of MoE layers.
    pub fn num_moe_layers(&self) -> usize {
        self.moe_layers.len()
    }

    /// Get the absolute layer indices that are MoE layers.
    pub fn moe_layer_indices(&self) -> &[usize] {
        &self.moe_layers
    }

    /// Get the mmap for a specific layer (for direct access if needed).
    pub fn mmap(&self) -> &Mmap {
        &self.mmaps[0]
    }

    /// Get hidden size.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Get intermediate size.
    pub fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }

    /// Get number of experts for a specific layer (0 if not MoE).
    pub fn index_len(&self, layer: usize) -> usize {
        self.index.get(layer).map_or(0, |v| v.len())
    }
}


/// Build ExpertOffset for a single expert, searching across multiple shards.
fn build_expert_offset_multi_shard(
    shards: &[SafeTensors],
    mmaps: &[Mmap],
    layer_idx: usize,
    expert_idx: usize,
    layer_prefix: &str,
) -> Result<ExpertOffset> {
    /// Find a tensor across shards, return its range and mmap_idx.
    let find_range = |name: &str| -> Result<TensorRange> {
        let full_name = format!("{layer_prefix}.{name}");
        for (shard_idx, st) in shards.iter().enumerate() {
            if let Ok(tensor) = st.tensor(&full_name) {
                let mmap_base = mmaps[shard_idx].as_ptr() as usize;
                let data = tensor.data();
                let base_offset = (data.as_ptr() as usize) - mmap_base;
                let dtype = format!("{:?}", tensor.dtype());
                let shape = tensor.shape().to_vec();

                let elem_size = match tensor.dtype() {
                    safetensors::Dtype::F32 => 4,
                    safetensors::Dtype::F16 | safetensors::Dtype::BF16 => 2,
                    safetensors::Dtype::U8 => 1,
                    safetensors::Dtype::U32 | safetensors::Dtype::I32 => 4,
                    _ => 4,
                };

                let elems_per_expert: usize = if shape.len() > 1 {
                    shape[1..].iter().product()
                } else {
                    1
                };
                let expert_byte_len = elems_per_expert * elem_size;
                let expert_byte_start = base_offset + expert_idx * expert_byte_len;

                return Ok(TensorRange {
                    offset: expert_byte_start,
                    len: expert_byte_len,
                    dtype,
                    shape,
                    mmap_idx: shard_idx,
                });
            }
        }
        anyhow::bail!("tensor '{}' not found in any shard", full_name)
    };

    // Try "biases" first (Ornith naming), fall back to "bias" (GPT-OSS naming)
    let find_bias = |proj: &str| -> Result<TensorRange> {
        find_range(&format!("{proj}.biases"))
            .or_else(|_| find_range(&format!("{proj}.bias")))
    };

    let gate_w = find_range("gate_proj.weight")?;
    let gate_s = find_range("gate_proj.scales")?;
    let gate_b = find_bias("gate_proj")?;
    let up_w = find_range("up_proj.weight")?;
    let up_s = find_range("up_proj.scales")?;
    let up_b = find_bias("up_proj")?;
    let down_w = find_range("down_proj.weight")?;
    let down_s = find_range("down_proj.scales")?;
    let down_b = find_bias("down_proj")?;

    Ok(ExpertOffset {
        layer: layer_idx, expert: expert_idx,
        gate_w, gate_s, gate_b, up_w, up_s, up_b, down_w, down_s, down_b,
    })
}

fn load_model_config(path: &Path) -> Result<(serde_json::Value, std::path::PathBuf)> {
    // Accept either a directory or a file within the model directory
    let model_dir = if path.is_dir() {
        path.to_path_buf()
    } else {
        path.parent().unwrap_or(Path::new(".")).to_path_buf()
    };
    let config_path = model_dir.join("config.json");
    let text = std::fs::read_to_string(&config_path)
        .with_context(|| format!("read {}", config_path.display()))?;
    let val: serde_json::Value = serde_json::from_str(&text)?;
    Ok((val, model_dir))
}

fn build_index(
    st: &SafeTensors,
    mmap: &[u8],
    num_layers: usize,
    num_experts: usize,
    hidden_size: usize,
    intermediate_size: usize,
) -> Result<(Vec<Vec<ExpertOffset>>, Vec<usize>)> {
    let mut index = Vec::with_capacity(num_layers);
    let mut moe_layers = Vec::new();

    // Auto-detect prefix: try multiple known layouts
    let expert_prefix_fn = detect_expert_prefix(st, num_layers);

    for layer_idx in 0..num_layers {
        let layer_prefix = expert_prefix_fn(layer_idx);
        let gate_name = format!("{layer_prefix}.gate_proj.weight");

        if st.tensor(&gate_name).is_err() {
            index.push(Vec::new());
            continue;
        }

        moe_layers.push(layer_idx);
        let mut layer_experts = Vec::with_capacity(num_experts);

        for expert_idx in 0..num_experts {
            let offset = build_expert_offset(
                st, mmap, layer_idx, expert_idx, &layer_prefix,
                hidden_size, intermediate_size,
            )?;
            layer_experts.push(offset);
        }
        index.push(layer_experts);
    }

    Ok((index, moe_layers))
}

/// Detect expert tensor prefix format by probing known patterns.
fn detect_expert_prefix(st: &SafeTensors, num_layers: usize) -> Box<dyn Fn(usize) -> String> {
    // Pattern 1: Ornith/Qwen3.5-MoE — "language_model.model.layers.{N}.mlp.switch_mlp"
    let ornith_probe = format!("language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight");
    if st.tensor(&ornith_probe).is_ok() {
        return Box::new(move |layer_idx| {
            format!("language_model.model.layers.{layer_idx}.mlp.switch_mlp")
        });
    }

    // Pattern 2: GPT-OSS/Mixtral — "model.layers.{N}.mlp.experts"
    let gptoss_probe = format!("model.layers.0.mlp.experts.gate_proj.weight");
    if st.tensor(&gptoss_probe).is_ok() {
        return Box::new(move |layer_idx| {
            format!("model.layers.{layer_idx}.mlp.experts")
        });
    }

    // Pattern 3: Try to find first MoE layer (might not be layer 0)
    for i in 0..num_layers {
        let ornith = format!("language_model.model.layers.{i}.mlp.switch_mlp.gate_proj.weight");
        if st.tensor(&ornith).is_ok() {
            return Box::new(move |layer_idx| {
                format!("language_model.model.layers.{layer_idx}.mlp.switch_mlp")
            });
        }
        let gptoss = format!("model.layers.{i}.mlp.experts.gate_proj.weight");
        if st.tensor(&gptoss).is_ok() {
            return Box::new(move |layer_idx| {
                format!("model.layers.{layer_idx}.mlp.experts")
            });
        }
    }

    // Fallback: Ornith pattern
    tracing::warn!(target: "nexum::ssd", "Could not detect expert prefix, using Ornith default");
    Box::new(move |layer_idx| {
        format!("language_model.model.layers.{layer_idx}.mlp.switch_mlp")
    })
}
