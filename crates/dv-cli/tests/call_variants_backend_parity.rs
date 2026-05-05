//! End-to-end CLI parity: `dv call-variants` produces equivalent
//! CallVariantsOutput records whether you point it at the SavedModel
//! (TF backend) or the ONNX export (ORT backend).
//!
//! The two backends use independent numerical kernels (libtensorflow
//! vs ONNX Runtime), so we don't expect bit-equal floats — we assert
//! that the genotype-probability vectors agree to within 1e-3 and that
//! the implied PASS/REF classification is identical for every record.
//!
//! Skipped unless built with both `tf` and `ort` features.

#![cfg(all(feature = "tf", feature = "ort"))]

use std::path::PathBuf;
use std::process::Command;

use prost::Message;

fn workspace_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn dv_binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_dv"))
}

fn tempdir() -> PathBuf {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let p = std::env::temp_dir().join(format!("dv-backend-parity-{pid}-{nanos}"));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn read_cvos(path: &std::path::Path) -> Vec<dv_proto::dv::CallVariantsOutput> {
    let mut reader = dv_io::tfrecord::open_reader(path).expect("open cvo");
    let mut out = Vec::new();
    while let Some(rec) = reader.read_record().expect("read cvo") {
        out.push(dv_proto::dv::CallVariantsOutput::decode(&rec[..]).expect("decode cvo"));
    }
    out
}

fn run_call_variants(
    binary: &std::path::Path,
    examples: &std::path::Path,
    checkpoint: &std::path::Path,
    output: &std::path::Path,
    cwd: &std::path::Path,
) {
    let status = Command::new(binary)
        .arg("call-variants")
        .arg("--examples").arg(examples)
        .arg("--checkpoint").arg(checkpoint)
        .arg("--output").arg(output)
        .current_dir(cwd)
        .status()
        .expect("spawn dv");
    assert!(status.success(), "dv call-variants failed: {status:?}");
}

#[test]
fn tf_and_ort_agree_on_chr20() {
    let cwd = workspace_dir();
    let examples = cwd.join("testdata/quickstart_chr20/make_examples.tfrecord-00000-of-00001.gz");
    let savedmodel = cwd.join("models/wgs");
    let onnx = cwd.join("models/wgs.onnx");
    let dylib = cwd.join("models/lib/libonnxruntime.so");

    if !examples.exists() {
        eprintln!("skip: missing {}", examples.display());
        return;
    }
    if !savedmodel.exists() || !onnx.exists() {
        eprintln!("skip: missing model assets in {}", cwd.join("models").display());
        return;
    }
    if !dylib.exists() {
        eprintln!("skip: missing onnxruntime dylib at {}", dylib.display());
        return;
    }

    let tmp = tempdir();
    let tf_out = tmp.join("cvo.tf.tfrecord.gz");
    let ort_out = tmp.join("cvo.ort.tfrecord.gz");

    let bin = dv_binary();
    // The `dv` binary auto-routes by checkpoint extension: a directory
    // → TF SavedModel, `.onnx` → ORT.
    run_call_variants(&bin, &examples, &savedmodel, &tf_out, &cwd);
    run_call_variants(&bin, &examples, &onnx, &ort_out, &cwd);

    let tf_cvos = read_cvos(&tf_out);
    let ort_cvos = read_cvos(&ort_out);
    assert_eq!(
        tf_cvos.len(),
        ort_cvos.len(),
        "TF emitted {} records but ORT emitted {}",
        tf_cvos.len(),
        ort_cvos.len()
    );
    assert!(!tf_cvos.is_empty(), "expected at least one CVO");

    let mut max_delta: f64 = 0.0;
    let mut classification_diffs: Vec<String> = Vec::new();
    for (i, (a, b)) in tf_cvos.iter().zip(ort_cvos.iter()).enumerate() {
        // Same variant identity.
        let av = a.variant.as_ref().expect("tf variant");
        let bv = b.variant.as_ref().expect("ort variant");
        assert_eq!(
            (&av.reference_name, av.start, &av.reference_bases, &av.alternate_bases),
            (&bv.reference_name, bv.start, &bv.reference_bases, &bv.alternate_bases),
            "record #{i} variant identity differs"
        );
        // Genotype probabilities close.
        assert_eq!(
            a.genotype_probabilities.len(),
            b.genotype_probabilities.len()
        );
        for (p, q) in a.genotype_probabilities.iter().zip(b.genotype_probabilities.iter()) {
            let d = (p - q).abs();
            if d > max_delta {
                max_delta = d;
            }
        }
        // Both backends should agree on which class wins (= same
        // PASS/RefCall classification post-postprocess).
        let argmax = |v: &[f64]| -> usize {
            v.iter()
                .enumerate()
                .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        };
        let tf_class = argmax(&a.genotype_probabilities);
        let ort_class = argmax(&b.genotype_probabilities);
        if tf_class != ort_class {
            classification_diffs.push(format!(
                "{}:{} {}>{:?}: TF class={} probs={:?}, ORT class={} probs={:?}",
                av.reference_name,
                av.start,
                av.reference_bases,
                av.alternate_bases,
                tf_class,
                a.genotype_probabilities,
                ort_class,
                b.genotype_probabilities,
            ));
        }
    }
    eprintln!(
        "TF/ORT CLI parity: {} records, max prob delta = {max_delta}",
        tf_cvos.len()
    );
    assert!(
        classification_diffs.is_empty(),
        "TF/ORT classification disagreements:\n{}",
        classification_diffs.join("\n")
    );
    assert!(
        max_delta < 1e-3,
        "TF/ORT prob delta {max_delta} > 1e-3 across {} records",
        tf_cvos.len()
    );
}
