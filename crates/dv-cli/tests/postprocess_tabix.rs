//! Integration test: `dv postprocess-variants` on the chr20 quickstart
//! emits a valid tabix `.tbi` index alongside the `.vcf.gz`, and that
//! index points back to the same set of records.

use std::path::PathBuf;
use std::process::Command;

fn workspace_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn dv_binary() -> PathBuf {
    let target = std::env::var("CARGO_BIN_EXE_dv").map(PathBuf::from);
    target.unwrap_or_else(|_| {
        // Fallback when run outside `cargo test` for this crate.
        let mut p = workspace_dir();
        p.push("target/release/dv");
        if !p.exists() {
            p = workspace_dir();
            p.push("target/debug/dv");
        }
        p
    })
}

#[test]
fn postprocess_emits_tabix_index() {
    let cwd = workspace_dir();
    let cvo = cwd.join("quickstart/output/intermediate_results_dir/call_variants_output-00000-of-00001.tfrecord.gz");
    if !cvo.exists() {
        eprintln!("skip: {} not present", cvo.display());
        return;
    }

    let tmp = tempdir();
    let out_vcf = tmp.join("out.vcf.gz");
    let tbi = tmp.join("out.vcf.gz.tbi");

    let status = Command::new(dv_binary())
        .arg("postprocess-variants")
        .arg("--cvo").arg(&cvo)
        .arg("--output-vcf").arg(&out_vcf)
        .arg("--sample-name").arg("HG002")
        .arg("--contig").arg("chr20:64444167")
        .current_dir(&cwd)
        .status()
        .expect("spawn dv");
    assert!(status.success(), "dv postprocess-variants failed: {status:?}");

    assert!(out_vcf.exists(), "missing {}", out_vcf.display());
    assert!(tbi.exists(), "missing {}", tbi.display());

    // Magic = 0x1F 0x8B (gzip), embedded BGZF "BC" extension at offset 12.
    let bytes = std::fs::read(&tbi).unwrap();
    assert!(bytes.starts_with(&[0x1f, 0x8b]), "tbi must be gzip-framed");
    assert_eq!(&bytes[12..14], b"BC", "tbi must be BGZF (extra subfield)");

    // Roundtrip: noodles must read it and report `chr20`.
    use noodles::csi::BinningIndex;
    let idx = noodles::tabix::fs::read(&tbi).expect("read tbi");
    let header = idx.header().expect("tbi header");
    let names = header.reference_sequence_names();
    assert!(
        names.iter().any(|n| n == "chr20"),
        "expected chr20 in index, got {:?}", names
    );
}

fn tempdir() -> PathBuf {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let p = std::env::temp_dir().join(format!("dv-tbi-{pid}-{nanos}"));
    std::fs::create_dir_all(&p).unwrap();
    p
}
