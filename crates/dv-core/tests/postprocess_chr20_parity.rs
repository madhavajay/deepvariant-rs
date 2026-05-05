//! End-to-end byte-equality parity vs upstream `postprocess_variants` on the
//! chr20 quickstart fixture.

use std::path::PathBuf;

use dv_core::postprocess::{self, PostprocessOptions};
use dv_core::vcf;
use dv_proto::nucleus_v1::ContigInfo;

const FORMAT_KEYS: &[&str] = &["GT", "GQ", "DP", "AD", "VAF", "MID", "PL"];

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/quickstart_chr20")
        .join(name)
}

fn read_vcf_text_gz(path: &std::path::Path) -> String {
    let f = std::fs::File::open(path).expect("open");
    let mut r = flate2::read::GzDecoder::new(std::io::BufReader::new(f));
    let mut s = String::new();
    std::io::Read::read_to_string(&mut r, &mut s).expect("read gz");
    s
}

#[test]
fn postprocess_byte_equal_to_upstream_chr20() {
    let cvos = postprocess::load_cvos(&[
        fixture("call_variants_output-00000-of-00001.tfrecord.gz"),
        fixture("make_examples_call_variant_outputs.tfrecord-00000-of-00001.gz"),
    ])
    .unwrap();
    assert_eq!(cvos.len(), 84);

    let opts = PostprocessOptions::default();
    let variants = postprocess::process_cvos_into_variants(cvos, "NA12878", &opts);
    assert_eq!(variants.len(), 78, "78 PASS-or-filtered variants on chr20 quickstart");

    let mut out = Vec::new();
    let contigs = vec![ContigInfo {
        name: "chr20".into(),
        description: String::new(),
        n_bases: 63_025_520,
        pos_in_fasta: 0,
        extra: std::collections::HashMap::new(),
    }];
    vcf::write_header(&mut out, &contigs, &["NA12878"]).unwrap();
    for v in &variants {
        vcf::write_variant_line(&mut out, v, FORMAT_KEYS).unwrap();
    }
    let ours = String::from_utf8(out).unwrap();
    let theirs = read_vcf_text_gz(&fixture("upstream_output.vcf.gz"));

    if ours != theirs {
        // Print the first 5 differing line pairs to make debugging easy.
        let our_lines: Vec<&str> = ours.lines().collect();
        let their_lines: Vec<&str> = theirs.lines().collect();
        let mut diffs = 0;
        for (i, (a, b)) in our_lines.iter().zip(their_lines.iter()).enumerate() {
            if a != b {
                eprintln!("--- line {i} ---");
                eprintln!("  ours:     {a}");
                eprintln!("  upstream: {b}");
                diffs += 1;
                if diffs >= 5 {
                    break;
                }
            }
        }
        eprintln!(
            "ours: {} lines, upstream: {} lines",
            our_lines.len(),
            their_lines.len()
        );
        panic!("VCF output not byte-equal to upstream");
    }
}
