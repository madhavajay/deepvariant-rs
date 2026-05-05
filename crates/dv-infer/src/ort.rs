//! ONNX Runtime backend (the cross-compile-friendly one).
//!
//! Uses the `ort` crate with `load-dynamic` so cargo doesn't bundle the
//! native library; user must have onnxruntime available at runtime
//! (or set `ORT_DYLIB_PATH`). For WASM/iOS/Android we'll swap the
//! loader behind a feature later.

use std::cell::RefCell;
use std::path::Path;

use ndarray::Array4;
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
        let session = Session::builder()
            .map_err(|e| InferError::Backend(format!("ort builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| InferError::Backend(format!("opt level: {e}")))?
            .commit_from_file(onnx_path.as_ref())
            .map_err(|e| InferError::Backend(format!("load model: {e}")))?;

        let input = session
            .inputs
            .first()
            .ok_or_else(|| InferError::Backend("no inputs".into()))?;
        let output = session
            .outputs
            .first()
            .ok_or_else(|| InferError::Backend("no outputs".into()))?;
        let input_name = input.name.clone();
        let output_name = output.name.clone();

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
