//! Expert index types and builder.
//!
//! Defines byte-range descriptors for MoE expert weights within a safetensors file,
//! and builds the index by parsing the stacked tensor layout.

use std::ops::Range;

use anyhow::{Context, Result};
use safetensors::SafeTensors;

/// Byte range of a single tensor in the safetensors file.
#[derive(Debug, Clone)]
pub struct TensorRange {
    pub offset: usize,
    pub len: usize,
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Which mmap/shard this tensor belongs to.
    pub mmap_idx: usize,
}

/// All byte ranges for a single MoE expert.
#[derive(Debug, Clone)]
pub struct ExpertOffset {
    pub layer: usize,
    pub expert: usize,
    pub gate_w: TensorRange,
    pub gate_s: TensorRange,
    pub gate_b: TensorRange,
    pub up_w: TensorRange,
    pub up_s: TensorRange,
    pub up_b: TensorRange,
    pub down_w: TensorRange,
    pub down_s: TensorRange,
    pub down_b: TensorRange,
}

impl ExpertOffset {
    /// Contiguous byte range covering all expert tensors (for madvise).
    pub fn contiguous_range(&self) -> Range<usize> {
        let start = self.gate_w.offset;
        let end = self.down_b.offset + self.down_b.len;
        start..end
    }

    /// All individual tensor ranges.
    pub fn all_ranges(&self) -> Vec<&TensorRange> {
        vec![
            &self.gate_w, &self.gate_s, &self.gate_b,
            &self.up_w, &self.up_s, &self.up_b,
            &self.down_w, &self.down_s, &self.down_b,
        ]
    }
}

/// Wrapper for mlx_arrays created on-demand from mmap pointers.
///
/// # Safety
/// The underlying mmap must outlive these arrays.
#[cfg(feature = "mlx")]
pub struct ExpertArrays {
    pub gate_w: engine_mlx_ffi::mlx_array,
    pub gate_s: engine_mlx_ffi::mlx_array,
    pub gate_b: engine_mlx_ffi::mlx_array,
    pub up_w: engine_mlx_ffi::mlx_array,
    pub up_s: engine_mlx_ffi::mlx_array,
    pub up_b: engine_mlx_ffi::mlx_array,
    pub down_w: engine_mlx_ffi::mlx_array,
    pub down_s: engine_mlx_ffi::mlx_array,
    pub down_b: engine_mlx_ffi::mlx_array,
}

/// Build ExpertOffset for a single expert from the stacked tensor layout.
pub(crate) fn build_expert_offset(
    st: &SafeTensors,
    mmap_data: &[u8],
    layer_idx: usize,
    expert_idx: usize,
    layer_prefix: &str,
    _hidden_size: usize,
    _intermediate_size: usize,
) -> Result<ExpertOffset> {
    let mmap_base = mmap_data.as_ptr() as usize;

    /// Compute byte range for one expert slice from a stacked tensor.
    /// Assumes first dimension is num_experts.
    let get_range = |name: &str| -> Result<TensorRange> {
        let full_name = format!("{layer_prefix}.{name}");
        let tensor = st.tensor(&full_name)
            .with_context(|| format!("tensor {full_name} not found"))?;

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

        // Total elements per expert = product of all dims except first (num_experts)
        let elems_per_expert: usize = if shape.len() > 1 {
            shape[1..].iter().product()
        } else {
            1
        };
        let expert_byte_len = elems_per_expert * elem_size;
        let expert_byte_start = base_offset + expert_idx * expert_byte_len;

        Ok(TensorRange { offset: expert_byte_start, len: expert_byte_len, dtype, shape, mmap_idx: 0 })
    };

    // Try "biases" first (Ornith naming), fall back to "bias" (GPT-OSS naming)
    let get_bias = |proj: &str| -> Result<TensorRange> {
        get_range(&format!("{proj}.biases"))
            .or_else(|_| get_range(&format!("{proj}.bias")))
    };

    let gate_w = get_range("gate_proj.weight")?;
    let gate_s = get_range("gate_proj.scales")?;
    let gate_b = get_bias("gate_proj")?;
    let up_w = get_range("up_proj.weight")?;
    let up_s = get_range("up_proj.scales")?;
    let up_b = get_bias("up_proj")?;
    let down_w = get_range("down_proj.weight")?;
    let down_s = get_range("down_proj.scales")?;
    let down_b = get_bias("down_proj")?;

    Ok(ExpertOffset {
        layer: layer_idx, expert: expert_idx,
        gate_w, gate_s, gate_b, up_w, up_s, up_b, down_w, down_s, down_b,
    })
}

/// Create an mlx_array from a byte range in the mmap (zero-copy).
///
/// # Safety
/// The mmap must outlive the returned mlx_array.
#[cfg(feature = "mlx")]
pub(crate) unsafe fn mlx_array_from_mmap(
    mmap: &memmap2::Mmap,
    range: &TensorRange,
    shape: &[i32],
) -> Result<engine_mlx_ffi::mlx_array> {
    use engine_mlx_ffi::{mlx_array_new_data, mlx_dtype};

    let ptr = mmap.as_ptr().add(range.offset) as *const std::ffi::c_void;
    let dtype = match range.dtype.as_str() {
        "F32" | "FLOAT32" => mlx_dtype::MLX_FLOAT32,
        "F16" | "FLOAT16" => mlx_dtype::MLX_FLOAT16,
        "BF16" | "BFloat16" => mlx_dtype::MLX_BFLOAT16,
        "U8" | "UINT8" => mlx_dtype::MLX_UINT8,
        "I32" | "INT32" => mlx_dtype::MLX_INT32,
        "U32" | "UINT32" => mlx_dtype::MLX_UINT32,
        _ => mlx_dtype::MLX_FLOAT32,
    };

    let arr = mlx_array_new_data(ptr, shape.as_ptr(), shape.len() as i32, dtype);
    if arr.ctx.is_null() {
        anyhow::bail!("mlx_array_new_data returned null at offset {}", range.offset);
    }
    Ok(arr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_range_covers_all_tensors() {
        let offset = ExpertOffset {
            layer: 0, expert: 0,
            gate_w: TensorRange { offset: 0, len: 1024, dtype: "F32".into(), shape: vec![], mmap_idx: 0 },
            gate_s: TensorRange { offset: 1024, len: 256, dtype: "F32".into(), shape: vec![], mmap_idx: 0 },
            gate_b: TensorRange { offset: 1280, len: 256, dtype: "F32".into(), shape: vec![], mmap_idx: 0 },
            up_w: TensorRange { offset: 1536, len: 1024, dtype: "F32".into(), shape: vec![], mmap_idx: 0 },
            up_s: TensorRange { offset: 2560, len: 256, dtype: "F32".into(), shape: vec![], mmap_idx: 0 },
            up_b: TensorRange { offset: 2816, len: 256, dtype: "F32".into(), shape: vec![], mmap_idx: 0 },
            down_w: TensorRange { offset: 3072, len: 1024, dtype: "F32".into(), shape: vec![], mmap_idx: 0 },
            down_s: TensorRange { offset: 4096, len: 256, dtype: "F32".into(), shape: vec![], mmap_idx: 0 },
            down_b: TensorRange { offset: 4352, len: 256, dtype: "F32".into(), shape: vec![], mmap_idx: 0 },
        };
        let range = offset.contiguous_range();
        assert_eq!(range.start, 0);
        assert_eq!(range.end, 4608);
    }

    #[test]
    fn all_ranges_returns_nine_tensors() {
        let offset = ExpertOffset {
            layer: 0, expert: 0,
            gate_w: TensorRange { offset: 0, len: 100, dtype: "F32".into(), shape: vec![], mmap_idx: 0 },
            gate_s: TensorRange { offset: 100, len: 50, dtype: "F32".into(), shape: vec![], mmap_idx: 0 },
            gate_b: TensorRange { offset: 150, len: 50, dtype: "F32".into(), shape: vec![], mmap_idx: 0 },
            up_w: TensorRange { offset: 200, len: 100, dtype: "F32".into(), shape: vec![], mmap_idx: 0 },
            up_s: TensorRange { offset: 300, len: 50, dtype: "F32".into(), shape: vec![], mmap_idx: 0 },
            up_b: TensorRange { offset: 350, len: 50, dtype: "F32".into(), shape: vec![], mmap_idx: 0 },
            down_w: TensorRange { offset: 400, len: 100, dtype: "F32".into(), shape: vec![], mmap_idx: 0 },
            down_s: TensorRange { offset: 500, len: 50, dtype: "F32".into(), shape: vec![], mmap_idx: 0 },
            down_b: TensorRange { offset: 550, len: 50, dtype: "F32".into(), shape: vec![], mmap_idx: 0 },
        };
        let ranges = offset.all_ranges();
        assert_eq!(ranges.len(), 9);
        assert_eq!(ranges[0].offset, 0);
        assert_eq!(ranges[8].offset, 550);
    }
}
