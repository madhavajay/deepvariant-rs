//! ONNX Runtime backend (the cross-compile-friendly one).
//!
//! Uses the `ort` crate with `load-dynamic` so cargo doesn't bundle the
//! native library; user must have onnxruntime available at runtime
//! (or set `ORT_DYLIB_PATH`). For WASM/iOS/Android we'll swap the
//! loader behind a feature later.

use std::cell::RefCell;
use std::path::Path;

use ndarray::{Array2, Array4};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

use crate::{InferError, InferenceBackend};

pub struct OrtBackend {
    session: RefCell<Session>,
    input_name: String,
    output_name: String,
    input_shape: [usize; 3],
    num_classes: usize,
}

impl OrtBackend {
    pub fn load(onnx_path: impl AsRef<Path>) -> Result<Self, InferError> {
        // ORT's default intra_threads (0 → all physical cores) is a
        // fine baseline on Apple Silicon; forcing 12 + parallel exec
        // mode regressed wall time on full chr20 due to thread
        // oversubscription. Only the env-var overrides remain — leave
        // them unset to keep ORT's defaults.
        let mut builder = Session::builder()
            .map_err(|e| InferError::Backend(format!("ort builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| InferError::Backend(format!("opt level: {e}")))?
            // Fixed-shape input → enable mem pattern (precomputes
            // tensor allocation plan, skips repeated alloc work).
            .with_memory_pattern(true)
            .map_err(|e| InferError::Backend(format!("mem pattern: {e}")))?;
        if let Ok(n) = std::env::var("DV_ORT_INTRA_THREADS").map(|s| s.parse::<usize>()) {
            if let Ok(n) = n {
                builder = builder
                    .with_intra_threads(n)
                    .map_err(|e| InferError::Backend(format!("intra threads: {e}")))?;
            }
        }
        if let Ok(n) = std::env::var("DV_ORT_INTER_THREADS").map(|s| s.parse::<usize>()) {
            if let Ok(n) = n {
                builder = builder
                    .with_inter_threads(n)
                    .map_err(|e| InferError::Backend(format!("inter threads: {e}")))?;
            }
        }

        // Apple platforms: register the CoreML EP. ORT will run nodes
        // it supports on Metal GPU + Neural Engine + CPU and fall back
        // to its CPU EP for unsupported nodes. Set DV_DISABLE_COREML=1
        // to force pure CPU for A/B comparisons.
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            let coreml_off = std::env::var("DV_DISABLE_COREML")
                .ok()
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            if !coreml_off {
                use ort::ep::coreml::{ComputeUnits, ModelFormat, SpecializationStrategy};
                use ort::ep::CoreML;
                let ep = CoreML::default()
                    // ALL = GPU + ANE + CPU; CoreML routes per-op.
                    .with_compute_units(ComputeUnits::All)
                    // NeuralNetwork (default) accepts our tf2onnx
                    // export. MLProgram (newer, faster) fails to parse
                    // the WGS Conv2D with "Required param 'pad' is
                    // missing" — a known tf2onnx → MLProgram quirk.
                    .with_model_format(ModelFormat::NeuralNetwork)
                    // We run inference many times per session; trade
                    // a slower first compile for faster predicts.
                    .with_specialization_strategy(SpecializationStrategy::FastPrediction)
                    .build();
                // NOTE: the WGS ONNX export keeps the batch dim dynamic
                // (`unk__980`). We *do not* call
                // `with_static_input_shapes(true)` — that would tell
                // CoreML to reject any node whose shape depends on the
                // dynamic batch dim, dropping the entire graph back to
                // the CPU EP and producing zero speedup. The default
                // (accept dynamic shapes) lets CoreML run the rest.
                builder = builder
                    .with_execution_providers([ep])
                    .map_err(|e| InferError::Backend(format!("CoreML EP: {e}")))?;
            }
        }

        let session = builder
            .commit_from_file(onnx_path.as_ref())
            .map_err(|e| InferError::Backend(format!("load model: {e}")))?;

        let input = session
            .inputs()
            .first()
            .ok_or_else(|| InferError::Backend("no inputs".into()))?;
        let output = session
            .outputs()
            .first()
            .ok_or_else(|| InferError::Backend("no outputs".into()))?;
        let input_name = input.name().to_string();
        let output_name = output.name().to_string();

        // Hard-code the WGS model shape — this matches example_info.json.
        // (ONNX exports often store dynamic batch as None; we just trust the
        // upstream shape spec rather than parsing TensorElementType here.)
        let input_shape = [100, 221, 7];
        let num_classes = 3;

        Ok(Self {
            session: RefCell::new(session),
            input_name,
            output_name,
            input_shape,
            num_classes,
        })
    }
}

impl InferenceBackend for OrtBackend {
    fn predict_batch(&self, images_nhwc: &[f32], n: usize) -> Result<Vec<f32>, InferError> {
        let [h, w, c] = self.input_shape;
        let expected = n * h * w * c;
        if images_nhwc.len() != expected {
            return Err(InferError::Backend(format!(
                "input length {} != expected {}",
                images_nhwc.len(),
                expected
            )));
        }
        let arr = Array4::from_shape_vec((n, h, w, c), images_nhwc.to_vec())
            .map_err(|e| InferError::Backend(format!("ndarray shape: {e}")))?;
        let input = Tensor::from_array(arr.into_dyn())
            .map_err(|e| InferError::Backend(format!("ort tensor: {e}")))?;
        let mut sess = self.session.borrow_mut();
        let outputs = sess
            .run(ort::inputs![self.input_name.as_str() => input])
            .map_err(|e| InferError::Backend(format!("session.run: {e}")))?;
        let out = outputs
            .get(self.output_name.as_str())
            .ok_or_else(|| InferError::Backend("missing output".into()))?;
        let (_shape, data) = out
            .try_extract_tensor::<f32>()
            .map_err(|e| InferError::Backend(format!("extract: {e}")))?;
        Ok(data.to_vec())
    }

    fn input_shape(&self) -> [usize; 3] {
        self.input_shape
    }

    fn num_classes(&self) -> usize {
        self.num_classes
    }
}

/// ORT backend for the small-model fast path. Same `Session` machinery as
/// `OrtBackend` but the input is a 2-D `[batch, feature_dim]` feature
/// matrix rather than a 4-D NHWC image batch. Output is `[batch, 3]`
/// softmax probabilities (REF, HET, HOM_ALT).
///
/// Doesn't implement `InferenceBackend` because its input contract is
/// different — keeping it as a sibling type keeps both signatures clean.
pub struct SmallModelOrt {
    session: RefCell<Session>,
    input_name: String,
    output_name: String,
    feature_dim: usize,
    num_classes: usize,
}

impl SmallModelOrt {
    pub fn load(onnx_path: impl AsRef<Path>) -> Result<Self, InferError> {
        let session = Session::builder()
            .map_err(|e| InferError::Backend(format!("ort builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| InferError::Backend(format!("opt level: {e}")))?
            .commit_from_file(onnx_path.as_ref())
            .map_err(|e| InferError::Backend(format!("load model: {e}")))?;
        let input = session
            .inputs()
            .first()
            .ok_or_else(|| InferError::Backend("no inputs".into()))?;
        let output = session
            .outputs()
            .first()
            .ok_or_else(|| InferError::Backend("no outputs".into()))?;
        let input_name = input.name().to_string();
        let output_name = output.name().to_string();
        // Trust upstream's WGS small-model spec: 70 features, 3 classes.
        // (Reading shape out of `input.input_type` would let us
        //  auto-detect, but the `ort` 2.0-rc.10 API is awkward and the
        //  numbers are stable across the upstream ports.)
        let feature_dim = 70;
        let num_classes = 3;
        Ok(Self {
            session: RefCell::new(session),
            input_name,
            output_name,
            feature_dim,
            num_classes,
        })
    }

    pub fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    /// Run inference on a flat row-major `[n, feature_dim]` matrix and
    /// return a flat `[n, num_classes]` probability matrix.
    pub fn predict(&self, features: &[f32], n: usize) -> Result<Vec<f32>, InferError> {
        let expected = n * self.feature_dim;
        if features.len() != expected {
            return Err(InferError::Backend(format!(
                "input length {} != expected {}",
                features.len(),
                expected
            )));
        }
        let arr = Array2::from_shape_vec((n, self.feature_dim), features.to_vec())
            .map_err(|e| InferError::Backend(format!("ndarray shape: {e}")))?;
        let input = Tensor::from_array(arr.into_dyn())
            .map_err(|e| InferError::Backend(format!("ort tensor: {e}")))?;
        let mut sess = self.session.borrow_mut();
        let outputs = sess
            .run(ort::inputs![self.input_name.as_str() => input])
            .map_err(|e| InferError::Backend(format!("session.run: {e}")))?;
        let out = outputs
            .get(self.output_name.as_str())
            .ok_or_else(|| InferError::Backend("missing output".into()))?;
        let (_shape, data) = out
            .try_extract_tensor::<f32>()
            .map_err(|e| InferError::Backend(format!("extract: {e}")))?;
        Ok(data.to_vec())
    }
}
