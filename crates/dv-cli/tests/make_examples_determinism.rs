//! Determinism: parallel and serial `dv make-examples` runs produce
//! byte-identical (decompressed) output.
//!
//! Without this property, downstream byte-equality tests (final VCF,
//! call-variants parity) would fail randomly between thread schedules.
//! Guarantee depends on (a) input-order preservation in the parallel
//! render pass and (b) BTreeMap-encoded tf.Example field order.

#![cfg(feature = "ort")]

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
    PathBuf::from(env!("CARGO_BIN_EXE_dv"))
}

fn tempdir() -> PathBuf {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let p = std::env::temp_dir().join(format!("dv-determinism-{pid}-{nanos}"));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn read_decompressed(path: &std::path::Path) -> Vec<u8> {
    // TFRecord shards are written via flate2's GzEncoder, not BGZF.
    let f = std::fs::File::open(path).expect("open");
    let mut r = flate2::read::GzDecoder::new(f);
    let mut out = Vec::new();
    std::io::copy(&mut r, &mut out).expect("decompress");
    out
}

fn run(rayon_threads: Option<&str>, examples: &std::path::Path, cwd: &std::path::Path, bin: &std::path::Path) {
    let bam = cwd.join("quickstart/input/NA12878_S1.chr20.10_10p1mb.bam");
    let fa = cwd.join("quickstart/input/ucsc.hg19.chr20.unittest.fasta");
    let mut c = Command::new(bin);
    if let Some(t) = rayon_threads {
        c.env("RAYON_NUM_THREADS", t);
    }
    let s = c
        .args(["make-examples"])
        .arg("--reads").arg(bam)
        .arg("--ref-fasta").arg(fa)
        .args(["--region", "chr20:10000000-10100000"])
        .arg("--examples").arg(examples)
        .args(["--sample-name", "NA12878"])
        .current_dir(cwd)
        .status()
        .expect("spawn dv");
    assert!(s.success(), "make-examples failed");
}

#[test]
fn parallel_and_serial_make_examples_match() {
    let cwd = workspace_dir();
    let bam = cwd.join("quickstart/input/NA12878_S1.chr20.10_10p1mb.bam");
    if !bam.exists() {
        eprintln!("skip: missing fixture");
        return;
    }
    let tmp = tempdir();
    let bin = dv_binary();

    let par = tmp.join("par.tfrecord.gz");
    let ser = tmp.join("ser.tfrecord.gz");
    run(None, &par, &cwd, &bin);
    run(Some("1"), &ser, &cwd, &bin);

    let par_bytes = read_decompressed(&par);
    let ser_bytes = read_decompressed(&ser);
    assert_eq!(par_bytes.len(), ser_bytes.len(), "decompressed sizes differ");
    assert!(
        par_bytes == ser_bytes,
        "parallel and serial outputs differ ({} bytes)",
        par_bytes.len()
    );
    eprintln!(
        "parallel/serial determinism: {} bytes byte-identical",
        par_bytes.len()
    );
}
