//! Parity harness: run the in-browser pipeline's make-examples half
//! (`dv_wasm::pipeline::pipeline_examples`) natively and write the
//! tf.Example records to a TFRecord shard, so native `dv
//! call-variants` + `dv postprocess-variants` can score them and the
//! resulting VCF can be diffed against a native `dv pipeline` run.
//!
//!   cargo run -p dv-wasm --example dump_examples --no-default-features --release \
//!       -- <reads.bam> <ref.fa> <chr:start-end> <out.tfrecord.gz>

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 5 {
        eprintln!("usage: dump_examples <bam> <fa> <chr:start-end> <out.tfrecord.gz>");
        std::process::exit(2);
    }
    let bam = std::fs::read(&a[1]).expect("read bam");
    let fa = std::fs::read(&a[2]).expect("read fa");
    let exs = dv_wasm::pipeline::pipeline_examples(&bam, &fa, &a[3])
        .expect("pipeline_examples");
    let mut w = dv_io::tfrecord::open_writer(&a[4]).expect("open writer");
    for e in &exs {
        w.write_record(&e.example).expect("write record");
    }
    eprintln!("wrote {} examples to {}", exs.len(), a[4]);
}
