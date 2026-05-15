//! Inference backend trait + per-backend impls behind features.
//!
//! Backends are gated by Cargo features so cross-compile targets only pull
//! in what they need (libtensorflow on Linux servers, ORT/TFLite/CoreML on
//! mobile and WASM).

use thiserror::Error;

#[derive(Debug, Error)]
pub enum InferError {
    #[error("backend error: {0}")]
    Backend(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Inference of a CHW-flattened batch of pileup images, returning per-row
/// genotype-class probabilities (shape `[batch, NUM_CLASSES]`, row-major).
pub trait InferenceBackend {
    /// `images_nhwc` is a tightly-packed batch of shape `[N, H, W, C]` in
    /// row-major (NHWC) order, dtype f32. Returns a flat
    /// `Vec<f32>` of length `N * num_classes()`.
    fn predict_batch(&self, images_nhwc: &[f32], n: usize) -> Result<Vec<f32>, InferError>;

    fn input_shape(&self) -> [usize; 3];
    fn num_classes(&self) -> usize;

    /// `Some(N)` if the model has a fixed batch dim of size N (callers
    /// must submit exactly N images per `predict_batch` and pad/trim
    /// accordingly). `None` if the batch dim is dynamic.
    fn pinned_batch(&self) -> Option<usize> {
        None
    }
}

#[cfg(feature = "tf")]
pub mod tf;

#[cfg(feature = "ort")]
pub mod ort;
