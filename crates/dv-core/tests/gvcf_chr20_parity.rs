//! End-to-end byte-equality parity vs upstream `postprocess_variants` gVCF
//! on the chr20 quickstart fixture.

use std::path::PathBuf;

use dv_core::{gvcf, postprocess::{self, PostprocessOptions}, vcf};
use dv_proto::nucleus_v1::ContigInfo;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/quickstart_chr20")
        .join(name)
}

fn quickstart_fasta() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../quickstart/input/ucsc.hg19.chr20.unittest.fasta")
}

fn read_gz_text(path: &std::path::Path) -> String {
    let f = std::fs::File::open(path).expect("open");
    let mut r = flate2::read::GzDecoder::new(std::io::BufReader::new(f));
    let mut s = String::new();
    std::io::Read::read_to_string(&mut r, &mut s).expect("read gz");
    s
}

#[test]
fn gvcf_byte_equal_to_upstream_chr20() {
    if !quickstart_fasta().exists() {
        eprintln!("skipping: chr20 quickstart FASTA not at expected path");
        return;
    }

    let cvos = postprocess::load_cvos(&[
        fixture("call_variants_output-00000-of-00001.tfrecord.gz"),
        fixture("make_examples_call_variant_outputs.tfrecord-00000-of-00001.gz"),
    ])
    .unwrap();
    let opts = PostprocessOptions::default();
    let variants = postprocess::process_cvos_into_variants(cvos, "NA12878", &opts);

    let nonvariants = gvcf::load_nonvariants(&[fixture("gvcf.tfrecord-00000-of-00001.gz")]).unwrap();
    let contigs = vec![ContigInfo {
        name: "chr20".into(),
        description: String::new(),
        n_bases: 63_025_520,
        pos_in_fasta: 0,
        extra: std::collections::HashMap::new(),
    }];
    let contig_index_map: std::collections::HashMap<String, u32> =
        contigs.iter().enumerate().map(|(i, c)| (c.name.clone(), i as u32)).collect();

    let reader = noodles::fasta::indexed_reader::Builder::default()
        .build_from_path(quickstart_fasta())
        .expect("open FASTA");
    let cell = std::cell::RefCell::new(reader);
    let mut lookup = move |contig: &str, pos: i64| -> Option<String> {
        let mut r = cell.borrow_mut();
        let region: noodles::core::Region = format!("{}:{}-{}", contig, pos + 1, pos + 1)
            .parse().ok()?;
        r.query(&region).ok().and_then(|rec| {
            std::str::from_utf8(rec.sequence().as_ref())
                .ok()
                .map(|s| s.to_ascii_uppercase())
        })
    };
    let lookup_dyn: &mut dyn FnMut(&str, i64) -> Option<String> = &mut lookup;
    let merged = gvcf::merge_streams(
        variants,
        nonvariants,
        |name| contig_index_map.get(name).copied(),
        Some(&mut *lookup_dyn),
    );

    let mut out = Vec::new();
    vcf::write_header(&mut out, &contigs, &["NA12878"]).unwrap();
    for v in &merged {
        vcf::write_gvcf_line(&mut out, v).unwrap();
    }
    let ours = String::from_utf8(out).unwrap();
    let theirs = read_gz_text(&fixture("upstream_output.g.vcf.gz"));
    if ours != theirs {
        for (i, (a, b)) in ours.lines().zip(theirs.lines()).enumerate() {
            if a != b {
                eprintln!("--- line {i} ---\n  ours:     {a}\n  upstream: {b}");
            }
        }
        panic!("gVCF not byte-equal");
    }
}
