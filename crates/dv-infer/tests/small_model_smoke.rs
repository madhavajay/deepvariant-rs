//! Smoke test: load `models/small_wgs.onnx` via the new SmallModelOrt
//! backend, run a tiny all-zeros batch, confirm the output shape is
//! `[batch, 3]` and the rows sum to ~1.0 (softmax invariant).

#![cfg(feature = "ort")]

use std::path::PathBuf;

fn workspace_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

#[test]
fn small_model_loads_and_runs() {
    let dylib = workspace_dir().join("models/lib/libonnxruntime.so");
    let onnx = workspace_dir().join("models/small_wgs.onnx");
    if !dylib.exists() || !onnx.exists() {
        eprintln!("skip: missing assets");
        return;
    }
    std::env::set_var("ORT_DYLIB_PATH", dylib);

    let model = dv_infer::ort::SmallModelOrt::load(&onnx).expect("load");
    assert_eq!(model.feature_dim(), 70);
    assert_eq!(model.num_classes(), 3);

    let n = 4;
    let features = vec![0.0f32; n * model.feature_dim()];
    let probs = model.predict(&features, n).expect("predict");
    assert_eq!(probs.len(), n * model.num_classes());

    for row in 0..n {
        let s: f32 = probs[row * 3..(row + 1) * 3].iter().sum();
        assert!(
            (0.99..=1.01).contains(&s),
            "row {row} probs don't sum to ~1: {s} ({:?})",
            &probs[row * 3..(row + 1) * 3]
        );
    }
    eprintln!(
        "small_wgs.onnx all-zeros prediction (one row): {:?}",
        &probs[..3]
    );
}
