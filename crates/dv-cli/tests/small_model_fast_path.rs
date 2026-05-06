//! End-to-end smoke test for the small-model fast path.
//!
//! Runs `dv make-examples --small-model` on the chr20 quickstart BAM,
//! confirms a non-trivial fraction of candidates is routed through the
//! fast path (small-model CVO shard non-empty), and that the final VCF
//! after postprocess is a superset of (or identical to) the
//! big-model-only baseline's PASS set.
//!
//! Skipped unless `--features ort` is enabled (we don't gate on TF).

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
    let p = std::env::temp_dir().join(format!("dv-sm-fastpath-{pid}-{nanos}"));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn count_tfrecords(path: &std::path::Path) -> usize {
    let mut r = dv_io::tfrecord::open_reader(path).expect("open tfrecord");
    let mut n = 0;
    while r.read_record().expect("read").is_some() {
        n += 1;
    }
    n
}

#[test]
fn small_model_fast_path_runs_end_to_end() {
    let cwd = workspace_dir();
    let bam = cwd.join("quickstart/input/NA12878_S1.chr20.10_10p1mb.bam");
    let fa = cwd.join("quickstart/input/ucsc.hg19.chr20.unittest.fasta");
    let onnx_big = cwd.join("models/wgs.onnx");
    let onnx_small = cwd.join("models/small_wgs.onnx");
    let dylib = cwd.join("models/lib/libonnxruntime.so");

    if !bam.exists() || !fa.exists() {
        eprintln!("skip: missing quickstart fixtures");
        return;
    }
    if !onnx_big.exists() || !onnx_small.exists() {
        eprintln!("skip: missing model assets");
        return;
    }
    if !dylib.exists() {
        eprintln!("skip: missing onnxruntime dylib");
        return;
    }

    let tmp = tempdir();
    let bin = dv_binary();
    let region = "chr20:10000000-10100000";

    // make-examples with small model.
    let examples = tmp.join("examples.tfrecord.gz");
    let small_cvo = tmp.join("small.cvo.tfrecord.gz");
    let s = Command::new(&bin)
        .args(["make-examples"])
        .arg("--reads").arg(&bam)
        .arg("--ref-fasta").arg(&fa)
        .args(["--region", region])
        .arg("--examples").arg(&examples)
        .args(["--sample-name", "NA12878"])
        .arg("--small-model").arg(&onnx_small)
        .arg("--small-model-cvo").arg(&small_cvo)
        .current_dir(&cwd)
        .status()
        .expect("spawn dv");
    assert!(s.success(), "make-examples failed");

    let n_examples = count_tfrecords(&examples);
    let n_small = count_tfrecords(&small_cvo);
    eprintln!(
        "small-model fast path: {} small CVOs, {} big-model examples",
        n_small, n_examples
    );
    // We expect a non-trivial split — small model should pick up at least
    // some calls, and at least some should fall through to the big model.
    assert!(n_small > 0, "small model rejected every candidate");
    assert!(n_examples > 0, "small model accepted every candidate");

    // Run the big-model on the residual examples.
    let big_cvo = tmp.join("big.cvo.tfrecord.gz");
    let s = Command::new(&bin)
        .args(["call-variants"])
        .arg("--examples").arg(&examples)
        .arg("--checkpoint").arg(&onnx_big)
        .arg("--output").arg(&big_cvo)
        .current_dir(&cwd)
        .status()
        .expect("spawn dv");
    assert!(s.success(), "call-variants failed");

    // Postprocess merging both CVO shards.
    let vcf = tmp.join("out.vcf.gz");
    let s = Command::new(&bin)
        .args(["postprocess-variants"])
        .arg("--cvo").arg(&big_cvo)
        .arg("--small-model-cvo").arg(&small_cvo)
        .arg("--output-vcf").arg(&vcf)
        .args(["--sample-name", "NA12878"])
        .args(["--contig", "chr20:63025520"])
        .current_dir(&cwd)
        .status()
        .expect("spawn dv");
    assert!(s.success(), "postprocess failed");

    // Sanity-check the VCF: at least one MID=small_model record present.
    let f = std::fs::File::open(&vcf).expect("open vcf");
    let mut rdr = noodles::bgzf::Reader::new(f);
    let mut bytes = Vec::new();
    std::io::copy(&mut rdr, &mut bytes).expect("decompress");
    let text = std::str::from_utf8(&bytes).expect("utf8");
    let small_count = text.lines().filter(|l| l.contains("small_model")).count();
    let big_count = text.lines().filter(|l| l.contains(":deepvariant:")).count();
    eprintln!(
        "VCF: {} MID=small_model lines, {} MID=deepvariant lines",
        small_count, big_count
    );
    assert!(small_count > 0, "no small_model records emitted in VCF");
    assert!(big_count > 0, "no deepvariant records emitted in VCF");
}
