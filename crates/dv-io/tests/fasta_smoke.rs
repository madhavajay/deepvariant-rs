//! Smoke test: read a base + a range from the chr20 FASTA.

use std::path::PathBuf;

fn fasta_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../quickstart/input/ucsc.hg19.chr20.unittest.fasta")
}

#[test]
fn fetch_chr20_quickstart_bases() {
    if !fasta_path().exists() {
        eprintln!("skipping: chr20 FASTA not present");
        return;
    }
    let r = dv_io::fasta::open_indexed(fasta_path()).unwrap();
    // chr20 position 10000117 (1-based) was REF=C in the upstream output.
    // 0-based pos = 10000116.
    let b = r.fetch_base("chr20", 10_000_116).unwrap();
    assert_eq!(b, b'C', "chr20 1-based 10000117 should be C, got {}", b as char);

    // Fetch 10 bases starting at 10000115 (0-based).
    let v = r.fetch_range("chr20", 10_000_115, 10_000_125).unwrap();
    assert_eq!(v.len(), 10);
    eprintln!("chr20:10000116-10000125 = {}", String::from_utf8_lossy(&v));
}
