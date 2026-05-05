//! End-to-end parity test against upstream `call_variants` on the chr20
//! quickstart fixture.
//!
//! Reads the upstream `make_examples.tfrecord-00000-of-00001.gz` shard,
//! decodes each `tf.Example`, runs inference via the Rust TF backend on
//! the WGS SavedModel, and compares the resulting genotype probabilities
//! against upstream's `call_variants_output-00000-of-00001.tfrecord.gz`.
//!
//! Tolerance: max abs delta < 1e-4 per probability. Floating-point
//! reordering between TF graph runs makes byte-equal infeasible, but
//! same-graph inference should be near-deterministic.

#![cfg(feature = "tf")]

use std::path::PathBuf;

use prost::Message;

use dv_infer::{tf::TfBackend, InferenceBackend};
use dv_proto::dv::CallVariantsOutput;
use dv_proto::tf::Example;

const PIXEL_BYTES: usize = 100 * 221 * 7; // H * W * C, u8
const TOLERANCE: f32 = 1e-4;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/quickstart_chr20")
        .join(name)
}

fn model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/wgs")
}

#[derive(Debug)]
struct ExampleRow {
    image_f32: Vec<f32>,
    variant_bytes: Vec<u8>,
    alt_allele_indices_bytes: Vec<u8>,
}

fn parse_example(payload: &[u8]) -> ExampleRow {
    let ex = Example::decode(payload).expect("decode tf.Example");
    let features = ex.features.expect("tf.Example must have features");

    fn bytes_value<'a>(
        features: &'a std::collections::HashMap<String, dv_proto::tf::Feature>,
        key: &str,
    ) -> &'a [u8] {
        let f = features.get(key).unwrap_or_else(|| panic!("missing key {key}"));
        match f.kind.as_ref().expect("feature.kind") {
            dv_proto::tf::feature::Kind::BytesList(bl) => {
                assert_eq!(bl.value.len(), 1, "expected single bytes value for {key}");
                &bl.value[0]
            }
            other => panic!("expected BytesList for {key}, got {other:?}"),
        }
    }

    let image_raw = bytes_value(&features.feature, "image/encoded");
    assert_eq!(
        image_raw.len(),
        PIXEL_BYTES,
        "image/encoded should be {PIXEL_BYTES} bytes (H*W*C u8)"
    );
    let variant_bytes = bytes_value(&features.feature, "variant/encoded").to_vec();
    let alt_allele_indices_bytes =
        bytes_value(&features.feature, "alt_allele_indices/encoded").to_vec();

    // Preprocess: u8 -> (u8 - 128)/128 in NHWC layout (already NHWC in source).
    let image_f32: Vec<f32> = image_raw
        .iter()
        .map(|&b| (b as f32 - 128.0) / 128.0)
        .collect();

    ExampleRow {
        image_f32,
        variant_bytes,
        alt_allele_indices_bytes,
    }
}

#[test]
fn parity_against_upstream_call_variants() {
    let model = TfBackend::load(model_path()).expect("load WGS SavedModel");
    assert_eq!(model.input_shape(), [100, 221, 7]);
    assert_eq!(model.num_classes(), 3);

    // Read every input example.
    let mut examples_reader =
        dv_io::tfrecord::open_reader(fixture("make_examples.tfrecord-00000-of-00001.gz"))
            .expect("open make_examples");
    let mut rows: Vec<ExampleRow> = Vec::new();
    while let Some(rec) = examples_reader.read_record().unwrap() {
        rows.push(parse_example(&rec));
    }
    eprintln!("parsed {} input examples", rows.len());
    assert_eq!(rows.len(), 24, "fixture has 24 hard-path examples");

    // Read upstream outputs in order.
    let mut upstream_reader = dv_io::tfrecord::open_reader(fixture(
        "call_variants_output-00000-of-00001.tfrecord.gz",
    ))
    .expect("open call_variants_output");
    let mut upstream: Vec<CallVariantsOutput> = Vec::new();
    while let Some(rec) = upstream_reader.read_record().unwrap() {
        upstream.push(CallVariantsOutput::decode(&*rec).expect("decode CVO"));
    }
    assert_eq!(upstream.len(), rows.len(), "row count mismatch");

    // Batch inference (24 fits comfortably in one call).
    let mut batch: Vec<f32> = Vec::with_capacity(rows.len() * PIXEL_BYTES);
    for r in &rows {
        batch.extend_from_slice(&r.image_f32);
    }
    let probs = model
        .predict_batch(&batch, rows.len())
        .expect("inference");
    assert_eq!(probs.len(), rows.len() * 3);

    // Diff against upstream.
    let mut max_delta: f32 = 0.0;
    for (i, (row, up)) in rows.iter().zip(upstream.iter()).enumerate() {
        let ours = &probs[i * 3..(i + 1) * 3];
        let theirs = &up.genotype_probabilities;
        assert_eq!(theirs.len(), 3, "upstream row {i} has {} probs", theirs.len());
        for (k, (&our, &their)) in ours.iter().zip(theirs.iter()).enumerate() {
            let d = (our - their as f32).abs();
            if d > max_delta {
                max_delta = d;
            }
            assert!(
                d <= TOLERANCE,
                "row {i} class {k}: ours={our} upstream={their} delta={d}"
            );
        }

        // alt_allele_indices is unmodified by call_variants — must round-trip byte-equal.
        let up_aai = up
            .alt_allele_indices
            .as_ref()
            .expect("upstream aai present")
            .encode_to_vec();
        assert_eq!(
            up_aai, row.alt_allele_indices_bytes,
            "row {i} alt_allele_indices bytes mismatch"
        );

        // The variant is mutated by upstream call_variants
        // (variantcall_utils.set_model_id → calls[0].info["MID"]="deepvariant"),
        // so a raw byte diff would fail. Instead, decode both, strip the MID
        // info entry, and check structural equality.
        let upstream_variant = up.variant.as_ref().expect("upstream variant present").clone();
        let mut stripped = upstream_variant.clone();
        for call in &mut stripped.calls {
            call.info.remove("MID");
        }
        let our_variant =
            dv_proto::nucleus_v1::Variant::decode(&*row.variant_bytes).expect("decode variant");
        assert_eq!(
            stripped, our_variant,
            "row {i} variant differs (excluding MID annotation)"
        );
    }
    eprintln!(
        "max prob delta across {} rows = {max_delta} (tolerance {TOLERANCE})",
        rows.len()
    );
}
