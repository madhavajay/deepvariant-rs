//! Count records in the captured chr20 fixtures so we know what to expect
//! during the parity test.

use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/quickstart_chr20")
        .join(name)
}

fn count(path: &std::path::Path) -> (usize, usize) {
    let mut r = dv_io::tfrecord::open_reader(path).unwrap();
    let mut n = 0usize;
    let mut total = 0usize;
    while let Some(rec) = r.read_record().unwrap() {
        n += 1;
        total += rec.len();
    }
    (n, total)
}

#[test]
fn fixture_record_counts() {
    for name in [
        "make_examples.tfrecord-00000-of-00001.gz",
        "call_variants_output-00000-of-00001.tfrecord.gz",
        "make_examples_call_variant_outputs.tfrecord-00000-of-00001.gz",
        "gvcf.tfrecord-00000-of-00001.gz",
    ] {
        let path = fixture(name);
        let (n, total) = count(&path);
        eprintln!("{name}: {n} records, {total} bytes");
        assert!(n > 0, "{name} should have records");
    }
}
