//! Smoke test: read the chr20 quickstart BAM and tally records.

use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/quickstart_chr20")
        .join(name)
}

#[test]
fn read_chr20_quickstart_bam() {
    let (header, mut reader) =
        dv_io::bam::open(fixture("NA12878_S1.chr20.10_10p1mb.bam")).unwrap();
    eprintln!("reference_sequences: {}", header.reference_sequences().len());
    let mut count = 0usize;
    let mut total_len = 0u64;
    use noodles::sam::alignment::Record;
    for rec in reader.records() {
        let r = rec.unwrap();
        count += 1;
        total_len += r.sequence().len() as u64;
    }
    eprintln!("records={count} total_seq_bases={total_len}");
    assert!(count > 1000, "chr20 100kbp slice should have many reads");
}
