//! Engine core types — inlined from nexum-engine-core.
//! Provides Engine/Model traits, Tensor, Shape, EngineError.

use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("config error: {0}")]
    ConfigError(String),
    #[error("load error: {0}")]
    LoadError(String),
    #[error("inference error: {0}")]
    InferenceError(String),
    #[error("weight error: {0}")]
    WeightError(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, EngineError>;

pub trait Engine: Send + Sync {
    fn name(&self) -> &str;
    fn load_model(&self, path: &Path) -> Result<Box<dyn Model>>;
    fn is_available(&self) -> bool;
}

pub trait Model: Send {
    fn name(&self) -> &str;
    fn forward(&mut self, tokens: &[u32]) -> Result<Tensor>;
    fn step(&mut self, token: u32) -> Result<u32>;
    fn reset(&mut self);
    fn offset(&self) -> usize;
    fn vocab_size(&self) -> usize;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType { F32, F16, BF16, I32, U32 }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape(pub Vec<usize>);

impl Shape {
    pub fn total(&self) -> usize { self.0.iter().product() }
}

#[derive(Debug, Clone)]
pub struct Tensor {
    bytes: Vec<u8>,
    shape: Shape,
    dtype: DType,
}

impl Tensor {
    pub fn from_f32(data: &[f32], shape: Shape) -> Self {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self { bytes, shape, dtype: DType::F32 }
    }
    pub fn shape(&self) -> &Shape { &self.shape }
    pub fn as_f32_slice(&self) -> &[f32] {
        let ptr = self.bytes.as_ptr() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, self.shape.total()) }
    }
}
