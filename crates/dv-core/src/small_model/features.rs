//! Pure feature-engineering functions for the small-model fast path.
//!
//! Each `_get_*` mirrors the same-named method in upstream's
//! `make_small_model_examples.py::FeatureEncoder` and uses integer
//! arithmetic with floor division everywhere upstream does (`100 * a //
//! b`, `sum(...) // n`).

use std::collections::HashMap;

use dv_proto::nucleus_v1::Variant;

use super::ReadAttrs;

// ---------- base features (12) ----------

#[inline]
fn mean_u8<F: Fn(&ReadAttrs) -> u32>(reads: &[ReadAttrs], pick: F, multiplier: u32) -> i32 {
    if reads.is_empty() {
        return 0;
    }
    let sum: u32 = reads.iter().map(&pick).sum();
    ((multiplier * sum) / reads.len() as u32) as i32
}

fn num_reads_supports_ref(ref_reads: &[ReadAttrs]) -> i32 {
    ref_reads.len() as i32
}
fn num_reads_supports_alt(alt_reads: &[ReadAttrs]) -> i32 {
    alt_reads.len() as i32
}
fn alt_indices_depth(ref_reads: &[ReadAttrs], alt_reads: &[ReadAttrs]) -> i32 {
    ref_reads.len() as i32 + alt_reads.len() as i32
}
fn variant_allele_frequency(alt_reads: &[ReadAttrs], total_depth: i32) -> i32 {
    if total_depth <= 0 {
        return 0;
    }
    100 * (alt_reads.len() as i32) / total_depth
}
fn alt_indices_variant_allele_frequency(ref_reads: &[ReadAttrs], alt_reads: &[ReadAttrs]) -> i32 {
    let dp = alt_indices_depth(ref_reads, alt_reads);
    if dp <= 0 {
        return 0;
    }
    100 * (alt_reads.len() as i32) / dp
}

fn ref_mapping_quality(ref_reads: &[ReadAttrs]) -> i32 {
    mean_u8(ref_reads, |r| r.mapping_quality as u32, 1)
}
fn alt_mapping_quality(alt_reads: &[ReadAttrs]) -> i32 {
    mean_u8(alt_reads, |r| r.mapping_quality as u32, 1)
}
fn ref_base_quality(ref_reads: &[ReadAttrs]) -> i32 {
    mean_u8(ref_reads, |r| r.avg_base_quality as u32, 1)
}
fn alt_base_quality(alt_reads: &[ReadAttrs]) -> i32 {
    mean_u8(alt_reads, |r| r.avg_base_quality as u32, 1)
}
fn ref_reverse_strand_ratio(ref_reads: &[ReadAttrs]) -> i32 {
    mean_u8(ref_reads, |r| r.is_reverse_strand as u32, 100)
}
fn alt_reverse_strand_ratio(alt_reads: &[ReadAttrs]) -> i32 {
    mean_u8(alt_reads, |r| r.is_reverse_strand as u32, 100)
}

// ---------- variant features (7) ----------

/// Indices into `variant.alternate_bases` that are NOT in
/// `alt_allele_indices` — i.e. the "exclude" set.
fn excluded_alt_bases<'a>(variant: &'a Variant, alt_allele_indices: &[i32]) -> Vec<&'a String> {
    variant
        .alternate_bases
        .iter()
        .enumerate()
        .filter_map(|(i, alt)| {
            if alt_allele_indices.contains(&(i as i32)) {
                None
            } else {
                Some(alt)
            }
        })
        .collect()
}

/// Mirrors `nucleus.util.variant_utils.is_snp(variant, exclude_alleles)`:
/// after dropping `exclude_alleles` and the gVCF `<*>` symbolic alt, the
/// variant counts as a SNP iff REF is one base and every remaining ALT is
/// one base. Symbolic alts (`<*>`, `<NON_REF>`) are also dropped.
pub fn is_snp(variant: &Variant, alt_allele_indices: &[i32]) -> bool {
    if variant.reference_bases.len() != 1 {
        return false;
    }
    let exclude = excluded_alt_bases(variant, alt_allele_indices);
    let mut found_any = false;
    for (i, alt) in variant.alternate_bases.iter().enumerate() {
        if exclude.iter().any(|s| *s == alt) {
            continue;
        }
        if alt.starts_with('<') {
            continue;
        }
        let _ = i;
        if alt.len() != 1 {
            return false;
        }
        if alt == &variant.reference_bases {
            return false;
        }
        found_any = true;
    }
    found_any
}

fn is_insertion(variant: &Variant, alt_allele_indices: &[i32]) -> bool {
    let exclude = excluded_alt_bases(variant, alt_allele_indices);
    let ref_len = variant.reference_bases.len();
    let mut found_any = false;
    for alt in &variant.alternate_bases {
        if exclude.iter().any(|s| *s == alt) || alt.starts_with('<') {
            continue;
        }
        if alt.len() <= ref_len {
            return false;
        }
        found_any = true;
    }
    found_any
}

fn is_deletion(variant: &Variant, alt_allele_indices: &[i32]) -> bool {
    let exclude = excluded_alt_bases(variant, alt_allele_indices);
    let ref_len = variant.reference_bases.len();
    let mut found_any = false;
    for alt in &variant.alternate_bases {
        if exclude.iter().any(|s| *s == alt) || alt.starts_with('<') {
            continue;
        }
        if alt.len() >= ref_len {
            return false;
        }
        found_any = true;
    }
    found_any
}

fn insertion_length(variant: &Variant, alt_allele_indices: &[i32]) -> i32 {
    let ref_len = variant.reference_bases.len() as i32;
    alt_allele_indices
        .iter()
        .map(|&i| variant.alternate_bases[i as usize].len() as i32 - ref_len)
        .max()
        .unwrap_or(0)
        .max(0)
}

fn deletion_length(variant: &Variant, alt_allele_indices: &[i32]) -> i32 {
    let ref_len = variant.reference_bases.len() as i32;
    alt_allele_indices
        .iter()
        .map(|&i| ref_len - variant.alternate_bases[i as usize].len() as i32)
        .max()
        .unwrap_or(0)
        .max(0)
}

fn is_multiallelic(variant: &Variant) -> bool {
    variant.alternate_bases.len() > 1
}
fn is_multiple_alt_alleles(alt_allele_indices: &[i32]) -> bool {
    alt_allele_indices.len() > 1
}

// ---------- main entry point ----------

pub fn compute_with_window(
    variant: &Variant,
    alt_allele_indices: &[i32],
    ref_reads: &[ReadAttrs],
    alt_reads: &[ReadAttrs],
    total_depth: i32,
    vaf_at_position: &HashMap<i64, i32>,
    window_size: usize,
) -> Vec<f32> {
    let mut out: Vec<f32> = Vec::with_capacity(19 + window_size);

    // 12 base features (insertion-order in upstream's enum).
    out.push(num_reads_supports_ref(ref_reads) as f32);
    out.push(num_reads_supports_alt(alt_reads) as f32);
    out.push(alt_indices_depth(ref_reads, alt_reads) as f32);
    out.push(total_depth as f32);
    out.push(variant_allele_frequency(alt_reads, total_depth) as f32);
    out.push(alt_indices_variant_allele_frequency(ref_reads, alt_reads) as f32);
    out.push(ref_mapping_quality(ref_reads) as f32);
    out.push(alt_mapping_quality(alt_reads) as f32);
    out.push(ref_base_quality(ref_reads) as f32);
    out.push(alt_base_quality(alt_reads) as f32);
    out.push(ref_reverse_strand_ratio(ref_reads) as f32);
    out.push(alt_reverse_strand_ratio(alt_reads) as f32);

    // 7 variant features.
    out.push(is_snp(variant, alt_allele_indices) as i32 as f32);
    out.push(is_insertion(variant, alt_allele_indices) as i32 as f32);
    out.push(is_deletion(variant, alt_allele_indices) as i32 as f32);
    out.push(insertion_length(variant, alt_allele_indices) as f32);
    out.push(deletion_length(variant, alt_allele_indices) as f32);
    out.push(is_multiallelic(variant) as i32 as f32);
    out.push(is_multiple_alt_alleles(alt_allele_indices) as i32 as f32);

    // window_size VAF context features at offsets [-half, half].
    if window_size > 0 {
        let half = (window_size / 2) as i64;
        for offset in -half..=half {
            let pos = variant.start + offset;
            out.push(vaf_at_position.get(&pos).copied().unwrap_or(0) as f32);
        }
    }

    debug_assert_eq!(out.len(), 19 + window_size);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use dv_proto::nucleus_v1::Variant;

    fn snp_variant() -> Variant {
        Variant {
            reference_name: "chr1".into(),
            start: 100,
            end: 101,
            reference_bases: "A".into(),
            alternate_bases: vec!["C".into()],
            ..Default::default()
        }
    }

    fn ins_variant() -> Variant {
        Variant {
            reference_name: "chr1".into(),
            start: 100,
            end: 101,
            reference_bases: "A".into(),
            alternate_bases: vec!["AGT".into()],
            ..Default::default()
        }
    }

    fn del_variant() -> Variant {
        Variant {
            reference_name: "chr1".into(),
            start: 100,
            end: 104,
            reference_bases: "ACGT".into(),
            alternate_bases: vec!["A".into()],
            ..Default::default()
        }
    }

    fn multi_variant() -> Variant {
        Variant {
            reference_name: "chr1".into(),
            start: 100,
            end: 101,
            reference_bases: "A".into(),
            alternate_bases: vec!["C".into(), "AT".into()],
            ..Default::default()
        }
    }

    fn read(mq: u8, bq: u8, rev: bool) -> ReadAttrs {
        ReadAttrs {
            mapping_quality: mq,
            avg_base_quality: bq,
            is_reverse_strand: rev,
        }
    }

    #[test]
    fn variant_classification() {
        assert!(is_snp(&snp_variant(), &[0]));
        assert!(!is_snp(&ins_variant(), &[0]));
        assert!(!is_snp(&del_variant(), &[0]));
        assert!(is_insertion(&ins_variant(), &[0]));
        assert!(!is_insertion(&snp_variant(), &[0]));
        assert!(is_deletion(&del_variant(), &[0]));
        assert!(!is_deletion(&ins_variant(), &[0]));
        assert!(is_multiallelic(&multi_variant()));

        // Multi-allelic with one SNP and one INS:
        //   alt_allele_indices=(0,) → look only at "C" → IS_SNP=true
        //   alt_allele_indices=(1,) → look only at "AT" → IS_INSERTION=true
        //   alt_allele_indices=(0,1) → both → mixed → none of snp/ins/del
        let m = multi_variant();
        assert!(is_snp(&m, &[0]));
        assert!(is_insertion(&m, &[1]));
        assert!(!is_snp(&m, &[0, 1]));
        assert!(!is_insertion(&m, &[0, 1]));
    }

    #[test]
    fn indel_lengths() {
        assert_eq!(insertion_length(&ins_variant(), &[0]), 2); // 3-1
        assert_eq!(deletion_length(&del_variant(), &[0]), 3); // 4-1
        assert_eq!(insertion_length(&snp_variant(), &[0]), 0);
    }

    #[test]
    fn base_features_basic() {
        let v = snp_variant();
        let refs = vec![read(60, 30, false), read(60, 30, true), read(40, 35, false)];
        let alts = vec![read(60, 30, false), read(60, 25, true)];
        let total_depth = 5; // = ref + alt for biallelic
        let vaf = HashMap::new();
        let f = compute_with_window(&v, &[0], &refs, &alts, total_depth, &vaf, 0);
        // Vector should be 19 (no VAF context).
        assert_eq!(f.len(), 19);
        // Index 0: num_reads_supports_ref = 3
        assert_eq!(f[0], 3.0);
        // Index 1: num_reads_supports_alt = 2
        assert_eq!(f[1], 2.0);
        // Index 2: alt_indices_depth = 5
        assert_eq!(f[2], 5.0);
        // Index 3: total_depth = 5
        assert_eq!(f[3], 5.0);
        // Index 4: VAF = 100*2/5 = 40
        assert_eq!(f[4], 40.0);
        // Index 5: alt_indices_VAF = 100*2/5 = 40
        assert_eq!(f[5], 40.0);
        // Index 6: ref_MAPQ = (60+60+40)/3 = 53
        assert_eq!(f[6], 53.0);
        // Index 7: alt_MAPQ = (60+60)/2 = 60
        assert_eq!(f[7], 60.0);
        // Index 8: ref_BQ = (30+30+35)/3 = 31  (integer floor)
        assert_eq!(f[8], 31.0);
        // Index 9: alt_BQ = (30+25)/2 = 27
        assert_eq!(f[9], 27.0);
        // Index 10: ref_strand_ratio = 100*1/3 = 33
        assert_eq!(f[10], 33.0);
        // Index 11: alt_strand_ratio = 100*1/2 = 50
        assert_eq!(f[11], 50.0);
        // Index 12: is_snp = 1
        assert_eq!(f[12], 1.0);
        // Index 13-15: is_ins/is_del = 0, lengths = 0
        assert_eq!(f[13], 0.0);
        assert_eq!(f[14], 0.0);
        assert_eq!(f[15], 0.0);
        assert_eq!(f[16], 0.0);
        // Index 17: is_multiallelic = 0
        assert_eq!(f[17], 0.0);
        // Index 18: is_multiple_alt_alleles = 0
        assert_eq!(f[18], 0.0);
    }

    #[test]
    fn full_70_feature_vector_for_snp() {
        let v = snp_variant();
        let refs: Vec<ReadAttrs> = (0..10).map(|_| read(60, 30, false)).collect();
        let alts: Vec<ReadAttrs> = (0..10).map(|_| read(60, 30, true)).collect();
        let mut vaf = HashMap::new();
        // Pad VAF window so we can detect order.
        for offset in -25i64..=25 {
            vaf.insert(v.start + offset, (offset.abs() as i32) % 100);
        }
        let f = compute_with_window(&v, &[0], &refs, &alts, 20, &vaf, 51);
        assert_eq!(f.len(), 70);
        // VAF context starts at index 19; offset=-25 should be 25
        assert_eq!(f[19], 25.0);
        // Center (offset=0) is at index 19+25 = 44, value 0
        assert_eq!(f[44], 0.0);
        // Last (offset=+25) is at 69, value 25
        assert_eq!(f[69], 25.0);
    }

    #[test]
    fn empty_reads_safe() {
        let v = snp_variant();
        let f = compute_with_window(&v, &[0], &[], &[], 0, &HashMap::new(), 0);
        for x in &f[0..12] {
            assert_eq!(*x, 0.0);
        }
    }
}
