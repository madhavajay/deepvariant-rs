//! Diagnostic: dump first few gvcf records.
use std::path::PathBuf;

use prost::Message;

use dv_proto::nucleus_v1::Variant;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/quickstart_chr20")
        .join(name)
}

#[test]
fn dump_gvcf() {
    let mut r = dv_io::tfrecord::open_reader(fixture("gvcf.tfrecord-00000-of-00001.gz")).unwrap();
    let mut count = 0;
    while let Some(rec) = r.read_record().unwrap() {
        if count < 3 {
            let v = Variant::decode(&*rec).unwrap();
            eprintln!(
                "{}:{}-{} ref={} alts={:?} qual={}",
                v.reference_name, v.start, v.end, v.reference_bases, v.alternate_bases, v.quality
            );
            for (j, c) in v.calls.iter().enumerate() {
                eprintln!("  call[{j}] gt={:?} gl={:?}", c.genotype, c.genotype_likelihood);
                let mut keys: Vec<_> = c.info.keys().collect();
                keys.sort();
                for k in keys {
                    eprintln!("    info[{k}]={:?}", c.info[k]);
                }
            }
            let mut iks: Vec<_> = v.info.keys().collect();
            iks.sort();
            for k in iks {
                eprintln!("  variant.info[{k}]={:?}", v.info[k]);
            }
        }
        count += 1;
    }
    eprintln!("total gvcf records: {count}");
}
