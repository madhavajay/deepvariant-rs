//! End-to-end smoke test for the phasing pipeline: direct_phasing
//! produces phased variants → phasing_apply mutates the per-call
//! genotype + PS field → vcf::write_variant_line emits a phased VCF
//! line ("0|1" / "1|0" instead of "0/1") with a `PS` value.

use dv_core::direct_phasing::{DirectPhasing, DirectPhasingOptions, PhasedVariant};
use dv_core::phasing_apply::apply_to_variants;
use dv_core::vcf;
use dv_proto::dv::deep_variant_call::SupportingReadsExt;
use dv_proto::dv::deep_variant_call::ReadSupport;
use dv_proto::dv::DeepVariantCall;
use dv_proto::nucleus_v1::{Variant, VariantCall};

fn make_candidate(start: i64, alts: &[(&str, &[&str])]) -> DeepVariantCall {
    let mut variant = Variant::default();
    variant.reference_name = "chr1".into();
    variant.start = start;
    variant.end = start + 1;
    variant.reference_bases = "A".into();
    variant.alternate_bases = alts.iter().map(|(a, _)| (*a).to_string()).collect();
    let mut allele_support_ext = std::collections::BTreeMap::new();
    for (a, reads) in alts {
        let mut sre = SupportingReadsExt::default();
        for r in *reads {
            let mut rs = ReadSupport::default();
            rs.read_name = (*r).to_string();
            sre.read_infos.push(rs);
        }
        allele_support_ext.insert((*a).to_string(), sre);
    }
    DeepVariantCall {
        variant: Some(variant),
        allele_support_ext,
        ..Default::default()
    }
}

fn make_variant_record(start: i64, ref_b: &str, alt: &str) -> Variant {
    let mut v = Variant::default();
    v.reference_name = "chr1".into();
    v.start = start;
    v.end = start + ref_b.len() as i64;
    v.reference_bases = ref_b.into();
    v.alternate_bases = vec![alt.into()];
    let mut call = VariantCall::default();
    call.call_set_name = "HG002".into();
    call.genotype = vec![0, 1]; // unphased het — overridden by phasing
    v.calls = vec![call];
    v
}

#[test]
fn direct_phasing_into_vcf_with_ps_tag() {
    // Three SNV candidates that should all phase together.
    // reads 1,2,3 carry the alt at all three positions (= phase 1)
    // reads 4,5,6 carry the alt at none (= phase 2 / ref).
    let candidates = vec![
        make_candidate(
            100,
            &[
                ("C", &["read1/0", "read2/0", "read3/0"]),
                ("A", &["read4/0", "read5/0", "read6/0"]),
            ],
        ),
        make_candidate(
            110,
            &[
                ("G", &["read1/0", "read2/0", "read3/0"]),
                ("T", &["read4/0", "read5/0", "read6/0"]),
            ],
        ),
        make_candidate(
            120,
            &[
                ("T", &["read1/0", "read2/0", "read3/0"]),
                ("G", &["read4/0", "read5/0", "read6/0"]),
            ],
        ),
    ];
    let reads: Vec<(String, i32)> = (1..=6).map(|i| (format!("read{}", i), 0)).collect();

    // Run direct phasing.
    let mut dp = DirectPhasing::new(DirectPhasingOptions::default());
    let _phases = dp.phase_reads(&candidates, &reads);
    let phased: Vec<PhasedVariant> = dp.phased_variants();
    eprintln!("phased vars: {} entries", phased.len());
    for pv in &phased {
        eprintln!(
            "  pos={} p1={} p2={} first={}",
            pv.position, pv.phase_1_bases, pv.phase_2_bases, pv.is_first_in_block
        );
    }

    // Build matching Variant records (REF/ALT lists in the *same* order
    // as the candidate alts so phase_1_bases / phase_2_bases resolve).
    // direct_phasing iterates alleles in lexicographic order, so the
    // first allele at position 100 will be "A" (alphabetically), not "C".
    // For a SNV with REF=A, the allele "A" *is* the ref, and "C" is alt[0].
    let variants = vec![
        make_variant_record(100, "A", "C"),
        // 110: alts emitted by phasing in lex order are "G", "T".
        // We use REF=G, ALT=T so that "G" resolves to ref (0) and
        // "T" resolves to alt[0] (1). Phase swap → genotype = [1, 0]
        // (T on hap 1, G on hap 2) when phasing reverses them.
        make_variant_record(110, "G", "T"),
        make_variant_record(120, "T", "G"),
    ];

    // Apply phasing.
    let phased_vars = apply_to_variants(&phased, &variants);
    assert_eq!(phased_vars.len(), 3);

    // Every variant should now be phased and have a PS field.
    let mut blocks: std::collections::HashSet<i64> = std::collections::HashSet::new();
    for v in &phased_vars {
        let call = &v.calls[0];
        assert!(call.is_phased, "variant at {} not phased", v.start);
        assert!(call.info.contains_key("PS"));
        // Look up PS value.
        let ps_value = call
            .info
            .get("PS")
            .and_then(|lv| lv.values.first())
            .and_then(|val| match val.kind.as_ref() {
                Some(dv_proto::nucleus_v1::value::Kind::IntValue(n)) => Some(*n as i64),
                _ => None,
            })
            .expect("PS as int");
        blocks.insert(ps_value);
    }
    assert_eq!(blocks.len(), 1, "expected single phasing block; got {:?}", blocks);

    // Render VCF and confirm the genotype field contains a pipe.
    use dv_proto::nucleus_v1::ContigInfo;
    let mut out = Vec::new();
    let contigs = vec![ContigInfo {
        name: "chr1".into(),
        n_bases: 1_000_000,
        ..Default::default()
    }];
    vcf::write_header(&mut out, &contigs, &["HG002"]).unwrap();
    let format_keys: &[&str] = &["GT", "PS"];
    for v in &phased_vars {
        vcf::write_variant_line(&mut out, v, format_keys).unwrap();
    }
    let text = String::from_utf8(out).unwrap();
    eprintln!("--- VCF ---\n{}", text);
    let body: Vec<&str> = text.lines().filter(|l| !l.starts_with('#')).collect();
    assert_eq!(body.len(), 3);
    for line in &body {
        // The GT cell is the second-to-last tab-separated column.
        let cols: Vec<&str> = line.split('\t').collect();
        // FORMAT (GT:PS) at column 8, sample at column 9.
        let sample_col = cols[9];
        assert!(
            sample_col.contains('|'),
            "expected phased GT (with pipe), got {sample_col}"
        );
        let parts: Vec<&str> = sample_col.split(':').collect();
        assert_eq!(parts.len(), 2, "GT:PS expected, got {sample_col}");
        // PS field should be 100 (the start of the block).
        assert_eq!(parts[1], "100", "PS != 100 in {sample_col}");
    }
}
