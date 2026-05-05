//! ONNX Runtime backend parity check vs the TF backend on the chr20 fixture.
//!
//! Loads the same WGS model both as a SavedModel (TF) and as an ONNX export,
//! runs them on the same chr20 examples, and asserts the probabilities
//! agree within tolerance.

#![cfg(all(feature = "tf", feature = "ort"))]

use std::path::PathBuf;

use prost::Message;

use dv_infer::ort::OrtBackend;
use dv_infer::tf::TfBackend;
use dv_infer::InferenceBackend;
use dv_proto::tf::Example;

const PIXEL_BYTES: usize = 100 * 221 * 7;
const TOLERANCE: f32 = 1e-3;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/quickstart_chr20")
        .join(name)
}

fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models")
}

fn parse_image_f32(payload: &[u8]) -> Vec<f32> {
    let ex = Example::decode(payload).expect("decode tf.Example");
    let features = ex.features.expect("features");
    let f = features.feature.get("image/encoded").expect("image");
    let kind = f.kind.as_ref().expect("kind");
    let bytes = match kind {
        dv_proto::tf::feature::Kind::BytesList(bl) => &bl.value[0],
        _ => panic!(),
    };
    assert_eq!(bytes.len(), PIXEL_BYTES);
    bytes.iter().map(|&b| (b as f32 - 128.0) / 128.0).collect()
}

#[test]
fn ort_matches_tf_on_chr20_examples() {
    // Skip if onnxruntime isn't available at runtime.
    let dylib = model_dir().join("lib/libonnxruntime.so");
    if !dylib.exists() {
        eprintln!("skipping: libonnxruntime not at {}", dylib.display());
        return;
    }
    std::env::set_var("ORT_DYLIB_PATH", dylib);

    let tf_model = TfBackend::load(model_dir().join("wgs")).expect("tf load");
    let ort_model = OrtBackend::load(model_dir().join("wgs.onnx")).expect("ort load");

    let mut reader = dv_io::tfrecord::open_reader(fixture(
        "make_examples.tfrecord-00000-of-00001.gz",
    ))
    .unwrap();
    let mut examples = Vec::new();
    while let Some(rec) = reader.read_record().unwrap() {
        examples.push(parse_image_f32(&rec));
        if examples.len() == 4 {
            break; // 4 is enough for a quick parity check
        }
    }

    let mut batch: Vec<f32> = Vec::with_capacity(examples.len() * PIXEL_BYTES);
    for img in &examples {
        batch.extend_from_slice(img);
    }
    let n = examples.len();
    let tf_out = tf_model.predict_batch(&batch, n).expect("tf predict");
    let ort_out = ort_model.predict_batch(&batch, n).expect("ort predict");
    assert_eq!(tf_out.len(), ort_out.len());

    let mut max_delta: f32 = 0.0;
    for (a, b) in tf_out.iter().zip(ort_out.iter()) {
        let d = (a - b).abs();
        if d > max_delta {
            max_delta = d;
        }
    }
    eprintln!("max prob delta TF vs ORT over {} samples = {max_delta}", n * 3);
    assert!(
        max_delta < TOLERANCE,
        "TF and ORT outputs disagree by {max_delta} > {TOLERANCE}"
    );
}
