//! Generic model types for SSD engine forward pass.
//!
//! These types allow config-driven dispatch without per-model forward code.
//! The loader reads config.json + tensor shapes and builds a Vec<LayerDescriptor>.

#[cfg(feature = "mlx")]
use engine_mlx_ffi::mlx_array;
#[cfg(not(feature = "mlx"))]
#[allow(non_camel_case_types)]
type mlx_array = *mut std::ffi::c_void;

// ─── Quantized Weight ────────────────────────────────────────────────────────

/// A quantized weight with its own quantization parameters.
/// No runtime shape inference needed — the loader determines everything.
#[derive(Clone, Copy)]
pub struct QWeight {
    pub w: mlx_array,
    pub s: mlx_array,
    pub b: mlx_array,
    pub bits: i32,
    pub group_size: i32,
    pub mode: QuantMode,
}

/// Quantization mode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QuantMode {
    /// Standard MLX affine: dequant = scale * packed_value + bias
    Affine,
    /// MXFP4: shared exponent format, U8 scales, no biases.
    /// Requires dequantize before matmul (no fused kernel in MLX C).
    Mxfp4,
    /// Unquantized (plain f16/bf16). Direct matmul.
    None,
}

impl QWeight {
    pub fn affine(w: mlx_array, s: mlx_array, b: mlx_array, bits: i32, group_size: i32) -> Self {
        Self { w, s, b, bits, group_size, mode: QuantMode::Affine }
    }

    pub fn mxfp4(w: mlx_array, s: mlx_array, group_size: i32) -> Self {
        Self { w, s, b: unsafe { std::mem::zeroed() }, bits: 4, group_size, mode: QuantMode::Mxfp4 }
    }

    pub fn unquantized(w: mlx_array) -> Self {
        Self { w, s: unsafe { std::mem::zeroed() }, b: unsafe { std::mem::zeroed() }, bits: 0, group_size: 0, mode: QuantMode::None }
    }
}

// ─── Attention Weights ───────────────────────────────────────────────────────

/// Per-layer attention projection weights.
#[derive(Clone, Copy)]
pub struct AttnWeights {
    pub q: QWeight,
    pub k: QWeight,
    pub v: QWeight,
    pub o: QWeight,
    /// Per-head Q norm weight [head_dim] (optional, BF16). Zero-initialized if absent.
    pub q_norm: mlx_array,
    /// Per-head K norm weight [head_dim] (optional, BF16). Zero-initialized if absent.
    pub k_norm: mlx_array,
}

/// Per-layer Mamba-2 linear attention weights.
#[derive(Clone, Copy)]
pub struct LinearAttnWeights {
    pub in_proj_qkv: QWeight,   // Fused Q+K+V projection
    pub in_proj_z: QWeight,     // Gate Z
    pub in_proj_a: QWeight,     // Decay rate input projection
    pub in_proj_b: QWeight,     // Input-dependent B
    pub conv1d_weight: mlx_array, // Causal 1D conv [d_inner, kernel, 1]
    pub a_log: mlx_array,       // Log decay A [num_heads]
    pub dt_bias: mlx_array,     // Delta time bias [num_heads]
    pub norm_weight: mlx_array, // Group norm weight
    pub out_proj: QWeight,      // Output projection
}

// ─── Layer Types ─────────────────────────────────────────────────────────────

/// The "operator" part of a layer (what processes the sequence).
#[derive(Clone, Copy)]
pub enum OperatorType {
    /// Standard multi-head attention.
    Attention {
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        sliding_window: Option<usize>,
    },
    /// Mamba-2 linear attention (Structured State Space Duality).
    LinearAttn {
        num_heads: i32,
        head_dim: i32,
        d_state: i32,
        d_conv: i32,
        d_inner: i32,
    },
    /// Causal 1D convolution (LFM2 / Liquid style).
    /// No KV cache needed — only a small rolling state of recent hidden vectors.
    Conv {
        kernel_size: i32,
    },
}

/// Per-layer causal conv1d weights (LFM2).
#[derive(Clone, Copy)]
pub struct ConvWeights {
    /// Input projection: hidden → conv_dim
    pub in_proj: QWeight,
    /// Causal conv1d kernel: [conv_dim, kernel_size] (BF16, not quantized)
    pub conv_kernel: mlx_array,
    /// Output projection: conv_dim → hidden
    pub out_proj: QWeight,
}

/// Rolling state for causal conv1d layers (last L hidden states).
pub struct ConvState {
    /// Ring buffer: [kernel_size, hidden_size] — last L input vectors.
    pub buf: mlx_array,
    /// Current write position in the ring buffer.
    pub pos: usize,
}

/// The FFN part of a layer.
pub enum FfnType {
    /// Standard SwiGLU MLP (dense, all params in RAM).
    Dense {
        gate: QWeight,
        up: QWeight,
        down: QWeight,
    },
    /// Mixture of Experts — expert weights on SSD (mmap), router in RAM.
    MoE {
        router: QWeight,
        num_experts: usize,
        top_k: usize,
        /// Expert quantization mode (determines how to execute expert matmul).
        expert_quant: QuantMode,
        expert_bits: i32,
        expert_group_size: i32,
        /// Optional: shared expert (DeepSeek/Ornith style).
        shared_expert: Option<SharedExpert>,
    },
    /// Switch MoE (LFM2) — all experts stacked in single tensors, in RAM.
    /// Format: gate_proj/up_proj/down_proj are [num_experts, out_dim, packed_in].
    /// No mmap needed — small experts fit entirely in memory.
    SwitchMoE {
        router: QWeight,
        /// Expert bias added to router logits before softmax.
        expert_bias: mlx_array,
        /// Stacked expert weights: [num_experts, intermediate, hidden/pack]
        gate_proj_w: mlx_array,
        gate_proj_s: mlx_array,
        gate_proj_b: mlx_array,
        up_proj_w: mlx_array,
        up_proj_s: mlx_array,
        up_proj_b: mlx_array,
        /// Stacked down_proj: [num_experts, hidden, intermediate/pack]
        down_proj_w: mlx_array,
        down_proj_s: mlx_array,
        down_proj_b: mlx_array,
        num_experts: usize,
        top_k: usize,
        expert_bits: i32,
        expert_group_size: i32,
    },
    /// No FFN for this layer (rare).
    None,
}

/// Shared expert weights (always active, blended with MoE output).
pub struct SharedExpert {
    pub gate: QWeight,
    pub up: QWeight,
    pub down: QWeight,
    pub blend_gate: QWeight,
}

// ─── Layer Descriptor ────────────────────────────────────────────────────────

/// Complete descriptor for one transformer layer.
/// The generic forward loop dispatches on this.
pub struct LayerDescriptor {
    pub pre_norm: mlx_array,
    pub post_norm: mlx_array,
    pub operator: OperatorType,
    pub attn: Option<AttnWeights>,
    pub linear_attn: Option<LinearAttnWeights>,
    pub conv: Option<ConvWeights>,
    pub ffn: FfnType,
}

// ─── Model Descriptor ────────────────────────────────────────────────────────

/// Top-level model structure built by the loader.
pub struct ModelDescriptor {
    /// Dequantized embedding table (f16).
    pub embed: mlx_array,
    /// LM head projection.
    pub lm_head: QWeight,
    /// Final layer norm.
    pub final_norm: mlx_array,
    /// Per-layer descriptors.
    pub layers: Vec<LayerDescriptor>,
    /// Model config.
    pub config: ModelConfig,
}

/// Minimal config needed by the forward pass (no loader-specific fields).
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub num_experts_per_tok: usize,
    pub prefill_mode: bool,
}
