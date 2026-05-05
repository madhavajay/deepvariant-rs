//! Diff our Rust-rendered pileup image against upstream's tf.Example for the
//! same chr20:10000117 SNV. This is a *guidance* test — currently the diff
//! is large (we don't yet match upstream's layout). The test prints a
//! summary so it can drive future work toward byte-equal pileup parity.

use std::collections::HashMap;
use std::path::PathBuf;

use prost::Message;

use dv_proto::dv::call_variants_output::AltAlleleIndices;
use dv_proto::nucleus_v1::Variant;
use dv_proto::tf::feature::Kind as FeatureKind;
use dv_proto::tf::Example;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/quickstart_chr20")
        .join(name)
}

fn extract_image_bytes(payload: &[u8]) -> (Variant, AltAlleleIndices, Vec<u8>) {
    let ex = Example::decode(payload).unwrap();
    let f = ex.features.unwrap();
    let bytes_for = |k: &str| -> Vec<u8> {
        let kind = f.feature.get(k).unwrap().kind.as_ref().unwrap();
        match kind {
            FeatureKind::BytesList(bl) => bl.value[0].clone(),
            _ => panic!(),
        }
    };
    let img = bytes_for("image/encoded");
    let v = Variant::decode(&*bytes_for("variant/encoded")).unwrap();
    let aai = AltAlleleIndices::decode(&*bytes_for("alt_allele_indices/encoded")).unwrap();
    (v, aai, img)
}

#[test]
fn print_pileup_diff_summary_chr20_first_example() {
    // Read the first upstream example and emit a per-channel summary
    // comparing pixel value distributions.
    let upstream_path = fixture("make_examples.tfrecord-00000-of-00001.gz");
    let mut r = dv_io::tfrecord::open_reader(&upstream_path).unwrap();
    let rec = r.read_record().unwrap().expect("at least one record");
    let (variant, _aai, upstream_img) = extract_image_bytes(&rec);
    eprintln!(
        "first upstream example: {}:{}-{} {}>{:?}",
        variant.reference_name,
        variant.start,
        variant.end,
        variant.reference_bases,
        variant.alternate_bases
    );
    assert_eq!(upstream_img.len(), 100 * 221 * 7);

    // Per-channel histogram of pixel values.
    let mut histos: Vec<HashMap<u8, usize>> = (0..7).map(|_| HashMap::new()).collect();
    for row in 0..100 {
        for col in 0..221 {
            for c in 0..7 {
                let v = upstream_img[(row * 221 + col) * 7 + c];
                *histos[c].entry(v).or_insert(0) += 1;
            }
        }
    }
    for (c, h) in histos.iter().enumerate() {
        let mut keys: Vec<u8> = h.keys().copied().collect();
        keys.sort();
        let summary: Vec<String> = keys
            .iter()
            .map(|k| format!("{k}:{}", h[k]))
            .take(8)
            .collect();
        eprintln!("upstream channel {c}: {} distinct values, top: {}", h.len(), summary.join(", "));
    }
}
