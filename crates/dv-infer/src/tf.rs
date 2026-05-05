//! `serving_default` signature of a DeepVariant SavedModel via libtensorflow.
//!
//! Signature (from `model.example_info.json` + saved_model_cli):
//!   input  `input_1`: f32 [N, H, W, C] = [N, 100, 221, 7]
//!   output `classification`: f32 [N, 3]

use std::path::Path;

use tensorflow::{Graph, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};

use crate::{InferError, InferenceBackend};

const SIGNATURE: &str = "serving_default";
const INPUT_KEY: &str = "input_1";
const OUTPUT_KEY: &str = "classification";

pub struct TfBackend {
    bundle: SavedModelBundle,
    graph: Graph,
    input_op_name: String,
    input_op_index: i32,
    output_op_name: String,
    output_op_index: i32,
    input_shape: [usize; 3],
    num_classes: usize,
}

impl TfBackend {
    pub fn load(saved_model_dir: impl AsRef<Path>) -> Result<Self, InferError> {
        let mut graph = Graph::new();
        let bundle = SavedModelBundle::load(
            &SessionOptions::new(),
            ["serve"],
            &mut graph,
            saved_model_dir,
        )
        .map_err(|e| InferError::Backend(format!("SavedModelBundle::load: {e}")))?;

        let sig = bundle
            .meta_graph_def()
            .get_signature(SIGNATURE)
            .map_err(|e| InferError::Backend(format!("missing signature {SIGNATURE}: {e}")))?;

        let input_info = sig
            .get_input(INPUT_KEY)
            .map_err(|e| InferError::Backend(format!("missing input {INPUT_KEY}: {e}")))?;
        let output_info = sig
            .get_output(OUTPUT_KEY)
            .map_err(|e| InferError::Backend(format!("missing output {OUTPUT_KEY}: {e}")))?;

        let input_op_name = input_info.name().name.clone();
        let input_op_index = input_info.name().index;
        let output_op_name = output_info.name().name.clone();
        let output_op_index = output_info.name().index;

        // Validate input shape: [None, H, W, C].
        let input_shape = input_info.shape();
        let input_rank = input_shape
            .dims()
            .ok_or_else(|| InferError::Backend("input rank unknown".into()))?;
        if input_rank != 4 {
            return Err(InferError::Backend(format!(
                "expected rank-4 input, got rank {input_rank}"
            )));
        }
        let h = input_shape[1].ok_or_else(|| InferError::Backend("input H unknown".into()))? as usize;
        let w = input_shape[2].ok_or_else(|| InferError::Backend("input W unknown".into()))? as usize;
        let c = input_shape[3].ok_or_else(|| InferError::Backend("input C unknown".into()))? as usize;

        let output_shape = output_info.shape();
        let output_rank = output_shape
            .dims()
            .ok_or_else(|| InferError::Backend("output rank unknown".into()))?;
        let num_classes = output_shape[output_rank - 1]
            .ok_or_else(|| InferError::Backend("num_classes unknown".into()))?
            as usize;

        Ok(Self {
            bundle,
            graph,
            input_op_name,
            input_op_index,
            output_op_name,
            output_op_index,
            input_shape: [h, w, c],
            num_classes,
        })
    }
}

impl InferenceBackend for TfBackend {
    fn predict_batch(&self, images_nhwc: &[f32], n: usize) -> Result<Vec<f32>, InferError> {
        let [h, w, c] = self.input_shape;
        let expected = n * h * w * c;
        if images_nhwc.len() != expected {
            return Err(InferError::Backend(format!(
                "input length {} != expected {} for batch {n} shape {h}x{w}x{c}",
                images_nhwc.len(),
                expected
            )));
        }

        let input_tensor = Tensor::new(&[n as u64, h as u64, w as u64, c as u64])
            .with_values(images_nhwc)
            .map_err(|e| InferError::Backend(format!("Tensor build: {e}")))?;

        let input_op = self
            .graph
            .operation_by_name_required(&self.input_op_name)
            .map_err(|e| InferError::Backend(format!("input op lookup: {e}")))?;
        let output_op = self
            .graph
            .operation_by_name_required(&self.output_op_name)
            .map_err(|e| InferError::Backend(format!("output op lookup: {e}")))?;

        let mut args = SessionRunArgs::new();
        args.add_feed(&input_op, self.input_op_index, &input_tensor);
        let out_token = args.request_fetch(&output_op, self.output_op_index);

        self.bundle
            .session
            .run(&mut args)
            .map_err(|e| InferError::Backend(format!("session.run: {e}")))?;

        let out: Tensor<f32> = args
            .fetch(out_token)
            .map_err(|e| InferError::Backend(format!("fetch: {e}")))?;

        Ok(out.to_vec())
    }

    fn input_shape(&self) -> [usize; 3] {
        self.input_shape
    }

    fn num_classes(&self) -> usize {
        self.num_classes
    }
}
