//! Diagnostic: dump fields present on the first few CVOs from each fixture.
use std::path::PathBuf;

use dv_core::postprocess;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/quickstart_chr20")
        .join(name)
}

#[test]
fn dump_cvo_fields() {
    for path_name in [
        "call_variants_output-00000-of-00001.tfrecord.gz",
        "make_examples_call_variant_outputs.tfrecord-00000-of-00001.gz",
    ] {
        eprintln!("\n=== {path_name} ===");
        let cvos = postprocess::load_cvos(&[fixture(path_name)]).unwrap();
        for (i, cvo) in cvos.iter().take(2).enumerate() {
            let v = cvo.variant.as_ref().unwrap();
            eprintln!(
                "[{i}] {}:{} {}>{} probs={:?}",
                v.reference_name,
                v.start,
                v.reference_bases,
                v.alternate_bases.join(","),
                cvo.genotype_probabilities,
            );
            for (j, call) in v.calls.iter().enumerate() {
                eprintln!(
                    "    call[{j}] gt={:?} gl={:?}",
                    call.genotype, call.genotype_likelihood
                );
                let mut keys: Vec<_> = call.info.keys().collect();
                keys.sort();
                for k in keys {
                    let lv = &call.info[k];
                    eprintln!("      info[{k}]={:?}", lv);
                }
            }
        }
    }
}
