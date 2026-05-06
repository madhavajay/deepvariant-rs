//! Build tf.Example records for candidate variants.
//!
//! Port-in-progress of `deepvariant/make_examples_native.cc`. The
//! upstream version handles many cases (alt-aligned pileups,
//! multi-sample, methylation, somatic, etc.); this is the SNV-focused
//! foundation. Each example contains:
//!
//!   - `image/encoded`: raw bytes of the H×W×C uint8 pileup image
//!   - `variant/encoded`: serialized Variant proto
//!   - `alt_allele_indices/encoded`: serialized
//!     `CallVariantsOutput.AltAlleleIndices` proto

use std::collections::BTreeMap;

use prost::Message;

use dv_proto::dv::call_variants_output::AltAlleleIndices;
use dv_proto::nucleus_v1::Variant;
use dv_proto::tf::feature::Kind as FeatureKind;
use dv_proto::tf::{BytesList, Example, Feature, Features};

/// Variant-type tag stored in tf.Example. Mirrors upstream's
/// `EncodedVariantType` enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodedVariantType {
    Snp,
    Indel,
    Unknown,
}

/// Classify a variant by its REF/ALT lengths. Mirrors upstream:
///   * REF length 1 AND ≥1 ALT, all ALTs length 1 → SNP
///   * REF length > 1 → INDEL
///   * Any ALT length > 1 → INDEL
///   * Otherwise → UNKNOWN
pub fn encoded_variant_type(variant: &Variant) -> EncodedVariantType {
    if variant.reference_bases.len() == 1 && !variant.alternate_bases.is_empty() {
        if variant.alternate_bases.iter().all(|a| a.len() == 1) {
            return EncodedVariantType::Snp;
        }
    }
    if variant.reference_bases.len() > 1 {
        return EncodedVariantType::Indel;
    }
    if variant.alternate_bases.iter().any(|a| a.len() > 1) {
        return EncodedVariantType::Indel;
    }
    EncodedVariantType::Unknown
}

/// True iff at least one alt is more than one base. Helper for upstream
/// `HasAtLeastOneNonSingleBaseAllele`.
pub fn has_at_least_one_non_single_base_allele(variant: &Variant) -> bool {
    variant.alternate_bases.iter().any(|a| a.len() > 1)
}

/// Multi-allelic mode (mirrors upstream
/// `PileupImageOptions.MultiAllelicMode`). Default is `AddHetAlt` for
/// the WGS pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiAllelicMode {
    /// Emit one example per alt allele.
    NoHetAlt,
    /// Also emit examples for every het pair of alts (i, j) with i<j.
    AddHetAlt,
}

/// Enumerate the alt-allele combinations that get one tf.Example per
/// candidate. For `NoHetAlt` returns `[[alt_0], [alt_1], ...]`. For
/// `AddHetAlt` additionally enumerates `[ref, alt_i]` (REF excluded
/// from the combination — matches upstream) and `[alt_i, alt_j]` for
/// i<j. Order matches upstream's nested loop.
pub fn alt_allele_combinations(variant: &Variant, mode: MultiAllelicMode) -> Vec<Vec<String>> {
    match mode {
        MultiAllelicMode::NoHetAlt => variant
            .alternate_bases
            .iter()
            .map(|a| vec![a.clone()])
            .collect(),
        MultiAllelicMode::AddHetAlt => {
            let mut alts: Vec<&String> = Vec::with_capacity(1 + variant.alternate_bases.len());
            alts.push(&variant.reference_bases);
            for a in &variant.alternate_bases {
                alts.push(a);
            }
            let mut out = Vec::new();
            for i in 0..alts.len() {
                for j in (i + 1)..alts.len() {
                    let mut combo = Vec::new();
                    if i > 0 {
                        combo.push(alts[i].clone());
                    }
                    combo.push(alts[j].clone());
                    out.push(combo);
                }
            }
            out
        }
    }
}

/// Same enumeration but driven by a pre-computed list of allele-index
/// tuples (e.g. from the small-model fast-path `alt_allele_indices`
/// list). Mirrors upstream `AltAlleleCombinationsFromIndices`.
pub fn alt_allele_combinations_from_indices(
    variant: &Variant,
    index_tuples: &[Vec<i32>],
    mode: MultiAllelicMode,
) -> Vec<Vec<String>> {
    match mode {
        MultiAllelicMode::NoHetAlt => index_tuples
            .iter()
            .filter(|t| t.len() == 1)
            .map(|t| vec![variant.alternate_bases[t[0] as usize].clone()])
            .collect(),
        MultiAllelicMode::AddHetAlt => index_tuples
            .iter()
            .map(|t| {
                t.iter()
                    .map(|&i| variant.alternate_bases[i as usize].clone())
                    .collect()
            })
            .collect(),
    }
}

/// Splice an alt-allele bases string into the reference sequence,
/// flanked by `half_width` ref bases on each side, clipped to
/// `[0, contig_n_bases]`. Returns
/// `(haplotype_bases, ref_start, ref_end)` so the caller can re-anchor
/// the realigner. Mirrors upstream `CreateHaplotype` minus the FASTA
/// reader call (caller supplies the prefix/suffix bases).
pub fn create_haplotype(
    variant: &Variant,
    alt: &str,
    half_width: i64,
    contig_n_bases: i64,
    ref_prefix_provider: impl FnOnce(i64, i64) -> String, // (start, end) inclusive-exclusive → bases
    ref_suffix_provider: impl FnOnce(i64, i64) -> String,
) -> (String, i64, i64) {
    let var_start = variant.start;
    let var_end = var_start + variant.reference_bases.len() as i64;
    let ref_start = (var_start - half_width).max(0);
    let ref_end = (var_end + half_width).min(contig_n_bases);
    let prefix = if ref_start < var_start {
        ref_prefix_provider(ref_start, var_start)
    } else {
        String::new()
    };
    let suffix = if ref_end > var_end {
        ref_suffix_provider(var_end, ref_end)
    } else {
        String::new()
    };
    (format!("{}{}{}", prefix, alt, suffix), ref_start, ref_end)
}

/// Encode an alt-combination back to a serialized
/// `CallVariantsOutput.AltAlleleIndices` proto, with the picked alt
/// indices in input order. Mirrors upstream `EncodeAltAlleles`.
pub fn encode_alt_alleles(variant: &Variant, alt_combination: &[String]) -> Vec<u8> {
    let mut alt_indices: BTreeMap<&str, i32> = BTreeMap::new();
    for (i, a) in variant.alternate_bases.iter().enumerate() {
        alt_indices.insert(a.as_str(), i as i32);
    }
    let mut indices: Vec<i32> = Vec::new();
    for a in alt_combination {
        if let Some(idx) = alt_indices.get(a.as_str()) {
            indices.push(*idx);
        }
    }
    AltAlleleIndices { indices }.encode_to_vec()
}

/// Serialize one (variant, alt_allele_indices, pileup_image) triple as a
/// `tf.Example` byte buffer ready to be written to a TFRecord shard.
pub fn build_example(variant: &Variant, alt_indices: &[i32], image: &[u8]) -> Vec<u8> {
    let aai = AltAlleleIndices {
        indices: alt_indices.to_vec(),
    };
    let mut feature_map: BTreeMap<String, Feature> = BTreeMap::new();
    feature_map.insert(
        "image/encoded".to_string(),
        bytes_feature(image.to_vec()),
    );
    feature_map.insert(
        "variant/encoded".to_string(),
        bytes_feature(variant.encode_to_vec()),
    );
    feature_map.insert(
        "alt_allele_indices/encoded".to_string(),
        bytes_feature(aai.encode_to_vec()),
    );
    let example = Example {
        features: Some(Features { feature: feature_map }),
    };
    example.encode_to_vec()
}

fn bytes_feature(bytes: Vec<u8>) -> Feature {
    Feature {
        kind: Some(FeatureKind::BytesList(BytesList { value: vec![bytes] })),
    }
}

/// Decode a tf.Example payload back into its three components, for testing.
pub fn parse_example(payload: &[u8]) -> Result<(Variant, AltAlleleIndices, Vec<u8>), String> {
    let ex = Example::decode(payload).map_err(|e| format!("decode example: {e}"))?;
    let features = ex
        .features
        .ok_or_else(|| "example missing features".to_string())?;
    let bytes_for = |key: &str| -> Result<Vec<u8>, String> {
        let f = features
            .feature
            .get(key)
            .ok_or_else(|| format!("missing feature {key}"))?;
        let kind = f
            .kind
            .as_ref()
            .ok_or_else(|| format!("feature {key} kind missing"))?;
        match kind {
            FeatureKind::BytesList(bl) => bl
                .value
                .first()
                .cloned()
                .ok_or_else(|| format!("feature {key} BytesList empty")),
            other => Err(format!("feature {key} expected bytes, got {other:?}")),
        }
    };
    let image = bytes_for("image/encoded")?;
    let variant_bytes = bytes_for("variant/encoded")?;
    let aai_bytes = bytes_for("alt_allele_indices/encoded")?;
    let variant = Variant::decode(&*variant_bytes).map_err(|e| format!("decode variant: {e}"))?;
    let aai = AltAlleleIndices::decode(&*aai_bytes).map_err(|e| format!("decode aai: {e}"))?;
    Ok((variant, aai, image))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_variant() -> Variant {
        Variant {
            reference_name: "chr20".into(),
            start: 10_000_116,
            end: 10_000_117,
            reference_bases: "C".into(),
            alternate_bases: vec!["T".into()],
            ..Default::default()
        }
    }

    #[test]
    fn round_trip_example() {
        let v = synth_variant();
        let img = vec![42u8; 100 * 221 * 7];
        let bytes = build_example(&v, &[0], &img);
        let (got_v, got_aai, got_img) = parse_example(&bytes).unwrap();
        assert_eq!(got_v, v);
        assert_eq!(got_aai.indices, vec![0]);
        assert_eq!(got_img, img);
    }

    #[test]
    fn round_trip_multi_alt() {
        let mut v = synth_variant();
        v.alternate_bases = vec!["T".into(), "G".into()];
        let img = vec![7u8; 100 * 221 * 7];
        let bytes = build_example(&v, &[0, 1], &img);
        let (got_v, got_aai, _) = parse_example(&bytes).unwrap();
        assert_eq!(got_v.alternate_bases, vec!["T", "G"]);
        assert_eq!(got_aai.indices, vec![0, 1]);
    }

    #[test]
    fn encoded_variant_type_classification() {
        // Single REF base + all single ALT bases → SNP.
        let snp = synth_variant();
        assert_eq!(encoded_variant_type(&snp), EncodedVariantType::Snp);

        // Single REF + multi-base ALT → INDEL (insertion).
        let mut ins = synth_variant();
        ins.alternate_bases = vec!["TG".into()];
        assert_eq!(encoded_variant_type(&ins), EncodedVariantType::Indel);

        // Multi-base REF → INDEL (deletion).
        let mut del = synth_variant();
        del.reference_bases = "CG".into();
        del.alternate_bases = vec!["C".into()];
        assert_eq!(encoded_variant_type(&del), EncodedVariantType::Indel);

        // No alt → UNKNOWN.
        let mut none = synth_variant();
        none.alternate_bases = vec![];
        assert_eq!(encoded_variant_type(&none), EncodedVariantType::Unknown);
    }

    #[test]
    fn has_at_least_one_non_single_base_allele_works() {
        let mut v = synth_variant();
        assert!(!has_at_least_one_non_single_base_allele(&v));
        v.alternate_bases = vec!["T".into(), "GT".into()];
        assert!(has_at_least_one_non_single_base_allele(&v));
    }

    #[test]
    fn alt_allele_combinations_no_het_alt() {
        let mut v = synth_variant();
        v.alternate_bases = vec!["T".into(), "G".into(), "A".into()];
        let combos = alt_allele_combinations(&v, MultiAllelicMode::NoHetAlt);
        assert_eq!(combos, vec![vec!["T".to_string()], vec!["G".to_string()], vec!["A".to_string()]]);
    }

    #[test]
    fn alt_allele_combinations_add_het_alt() {
        let mut v = synth_variant();
        v.reference_bases = "C".into();
        v.alternate_bases = vec!["T".into(), "G".into()];
        let combos = alt_allele_combinations(&v, MultiAllelicMode::AddHetAlt);
        // alts list = [REF=C, T, G]
        // pairs: (0,1) → just T (REF excluded), (0,2) → just G, (1,2) → T,G
        assert_eq!(
            combos,
            vec![
                vec!["T".to_string()],
                vec!["G".to_string()],
                vec!["T".to_string(), "G".to_string()],
            ]
        );
    }

    #[test]
    fn alt_allele_combinations_from_indices_modes() {
        let mut v = synth_variant();
        v.alternate_bases = vec!["T".into(), "G".into(), "A".into()];
        let tuples = vec![vec![0], vec![2], vec![0, 1]];
        // NoHetAlt drops the multi-element tuple.
        let combos = alt_allele_combinations_from_indices(&v, &tuples, MultiAllelicMode::NoHetAlt);
        assert_eq!(combos, vec![vec!["T".to_string()], vec!["A".to_string()]]);
        // AddHetAlt keeps everything.
        let combos = alt_allele_combinations_from_indices(&v, &tuples, MultiAllelicMode::AddHetAlt);
        assert_eq!(
            combos,
            vec![
                vec!["T".to_string()],
                vec!["A".to_string()],
                vec!["T".to_string(), "G".to_string()],
            ]
        );
    }

    #[test]
    fn create_haplotype_centered_window() {
        let v = Variant {
            reference_name: "chr1".into(),
            start: 50,
            end: 51,
            reference_bases: "A".into(),
            alternate_bases: vec!["C".into()],
            ..Default::default()
        };
        let prefix = |s: i64, e: i64| -> String {
            // Synthetic 'L' for left flank, 100 bases.
            "L".repeat((e - s) as usize)
        };
        let suffix = |s: i64, e: i64| -> String { "R".repeat((e - s) as usize) };
        let (hap, rs, re) = create_haplotype(&v, "C", 5, 200, prefix, suffix);
        assert_eq!(rs, 45);
        assert_eq!(re, 56);
        assert_eq!(hap, "LLLLL".to_string() + "C" + "RRRRR");
    }

    #[test]
    fn create_haplotype_clipped_left() {
        // Variant near contig start — left flank truncated.
        let v = Variant {
            reference_name: "chr1".into(),
            start: 2,
            end: 3,
            reference_bases: "A".into(),
            alternate_bases: vec!["C".into()],
            ..Default::default()
        };
        let prefix = |s: i64, e: i64| "L".repeat((e - s) as usize);
        let suffix = |s: i64, e: i64| "R".repeat((e - s) as usize);
        let (hap, rs, re) = create_haplotype(&v, "C", 10, 100, prefix, suffix);
        assert_eq!(rs, 0);
        assert_eq!(re, 13);
        assert_eq!(hap, "LL".to_string() + "C" + &"R".repeat(10));
    }

    #[test]
    fn encode_alt_alleles_round_trip() {
        let mut v = synth_variant();
        v.alternate_bases = vec!["T".into(), "G".into(), "A".into()];
        let combo = vec!["A".to_string(), "T".to_string()];
        let bytes = encode_alt_alleles(&v, &combo);
        let parsed = AltAlleleIndices::decode(&*bytes).unwrap();
        // Order of indices reflects input combination order: A=2 then T=0.
        assert_eq!(parsed.indices, vec![2, 0]);
    }

    #[test]
    fn encode_alt_alleles_unknown_alt_filtered() {
        let v = synth_variant();
        let combo = vec!["X".to_string()]; // not in v.alternate_bases
        let bytes = encode_alt_alleles(&v, &combo);
        let parsed = AltAlleleIndices::decode(&*bytes).unwrap();
        assert!(parsed.indices.is_empty());
    }
}
