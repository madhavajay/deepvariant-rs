//! End-to-end smoke: parse the dv-make-examples chr20 output and confirm
//! the records decode as valid tf.Example with the expected feature shape.

use std::path::PathBuf;

use dv_core::make_examples::parse_example;

fn out_path() -> PathBuf {
    PathBuf::from("/tmp/dv_rust_examples.tfrecord.gz")
}

#[test]
fn records_have_correct_image_shape() {
    if !out_path().exists() {
        eprintln!("skipping: run `dv make-examples` first to produce {}", out_path().display());
        return;
    }
    let mut r = dv_io::tfrecord::open_reader(out_path()).expect("open tfrecord");
    let mut count = 0usize;
    while let Some(rec) = r.read_record().unwrap() {
        let (variant, _aai, image) = parse_example(&rec).expect("parse example");
        // Each example should have exactly H*W*C = 100*221*7 image bytes.
        assert_eq!(image.len(), 100 * 221 * 7);
        // Variant has a contig.
        assert!(!variant.reference_name.is_empty());
        // Has at least one alt.
        assert!(!variant.alternate_bases.is_empty());
        count += 1;
    }
    eprintln!("decoded {count} examples from chr20 dv-make-examples output");
    assert!(count > 0);
}
