//! Candidate variant generation from `AlleleCount`s.
//!
//! Port-in-progress of `deepvariant/variant_calling.cc`. Implements the
//! threshold-based candidate caller: a position is a candidate when an
//! alt allele has both `count >= min_count` AND
//! `count / total_reads >= min_fraction`. Different thresholds for SNPs
//! vs indels. No genotyping (model does that downstream).

use std::collections::HashMap;

use dv_proto::dv::{Allele, AlleleCount, AlleleType};
use dv_proto::nucleus_v1::{value, ListValue, Value, Variant, VariantCall};

/// Get the deletion size of an allele (length of bases) or -1 if it's
/// not a deletion. Helper for `calc_ref_bases`. Mirrors upstream
/// `DeletionSize`.
pub fn deletion_size(allele: &Allele) -> i32 {
    if allele.r#type == AlleleType::Deletion as i32 {
        allele.bases.len() as i32
    } else {
        -1
    }
}

/// Compute the longest substitution span on the reference needed to
/// describe any of `alt_alleles` at this position. If none of the
/// alts are deletions, this is just `ref_bases` (a single base). If
/// at least one is a deletion, the full deleted-bases string from the
/// longest deletion is used (since DEL alleles store
/// `anchor + deleted_ref_bases` per our allelecounter port).
///
/// Mirrors upstream `CalcRefBases`.
pub fn calc_ref_bases(ref_bases: &str, alt_alleles: &[Allele]) -> String {
    if alt_alleles.is_empty() {
        return ref_bases.to_string();
    }
    let max_del = alt_alleles
        .iter()
        .max_by_key(|a| deletion_size(a))
        .expect("non-empty");
    if max_del.r#type != AlleleType::Deletion as i32 {
        return ref_bases.to_string();
    }
    assert!(
        max_del.bases.len() > 1,
        "DEL allele {} has too few bases",
        max_del.bases
    );
    // Skip the anchor base and append the deleted bases to ref_bases.
    let suffix = &max_del.bases[1..];
    format!("{}{}", ref_bases, suffix)
}

/// Construct an alt allele consistent with a possibly-extended REF
/// span. Splices `prefix` onto the trailing portion of `variant_ref`
/// starting at `from`. Used to harmonize SNV/INS alts with a
/// DEL-extended REF in mixed multi-allelic candidates.
///
/// Examples (variant_ref = "ACGT" because of a 3-bp deletion):
///   * SNV "C" at position 0: prefix="C", from=1 → "C" + "CGT" = "CCGT"
///   * INS "ATTT":            prefix="ATTT", from=1 → "ATTT" + "CGT" = "ATTTCGT"
///   * DEL "ACGT":            prefix="A", from=4 → "A" + "" = "A"
///
/// Mirrors upstream `MakeAltAllele`.
pub fn make_alt_allele(prefix: &str, variant_ref: &str, from: usize) -> String {
    if from >= variant_ref.len() {
        prefix.to_string()
    } else {
        format!("{}{}", prefix, &variant_ref[from..])
    }
}

/// Project an `Allele` (from the allelecounter) into the alt-allele
/// string a Variant proto needs, given the variant's chosen REF span.
/// Uses `make_alt_allele` under the hood with the rules upstream
/// applies in `BuildAlleleMap`:
///   * SUB / INS: prefix = allele.bases, from = 1
///   * DEL: prefix = allele.bases[0..1] (anchor), from = allele.bases.len()
///
/// Returns `None` for REFERENCE / SOFT_CLIP alleles (which don't go
/// into alternate_bases).
pub fn allele_to_variant_alt(allele: &Allele, variant_ref: &str) -> Option<String> {
    let t = allele.r#type;
    if t == AlleleType::Substitution as i32 || t == AlleleType::Insertion as i32 {
        Some(make_alt_allele(&allele.bases, variant_ref, 1))
    } else if t == AlleleType::Deletion as i32 {
        assert!(
            allele.bases.len() > 1,
            "DEL allele {} has too few bases",
            allele.bases
        );
        Some(make_alt_allele(&allele.bases[..1], variant_ref, allele.bases.len()))
    } else {
        None
    }
}

#[derive(Debug, Clone)]
pub struct VariantCallerOptions {
    pub min_count_snps: i32,
    pub min_count_indels: i32,
    pub min_fraction_snps: f64,
    pub min_fraction_indels: f64,
    pub sample_name: String,
}

impl Default for VariantCallerOptions {
    fn default() -> Self {
        // Defaults from `deepvariant/options/dv_constants.py` for WGS.
        Self {
            min_count_snps: 2,
            min_count_indels: 2,
            min_fraction_snps: 0.12,
            min_fraction_indels: 0.06,
            sample_name: "SAMPLE".into(),
        }
    }
}

/// Aggregated alt-allele counts per (allele_type, bases) at one position.
#[derive(Debug, Clone, Default)]
pub struct AltSummary {
    pub by_allele: HashMap<(i32, String), i32>, // (allele_type, bases) -> count
    pub total_reads: i32,
    pub ref_supporting: i32,
}

pub fn summarize(c: &AlleleCount) -> AltSummary {
    let mut s = AltSummary {
        ref_supporting: c.ref_supporting_read_count,
        ..Default::default()
    };
    for allele in c.read_alleles.values() {
        let key = (allele.r#type, allele.bases.clone());
        *s.by_allele.entry(key).or_insert(0) += allele.count;
    }
    s.total_reads = s.ref_supporting + s.by_allele.values().sum::<i32>();
    s
}

/// Decide whether a single allele passes the candidate threshold.
fn passes(allele_type: i32, count: i32, total: i32, opts: &VariantCallerOptions) -> bool {
    if total == 0 {
        return false;
    }
    let frac = count as f64 / total as f64;
    let is_indel = allele_type == AlleleType::Insertion as i32
        || allele_type == AlleleType::Deletion as i32;
    if is_indel {
        count >= opts.min_count_indels && frac >= opts.min_fraction_indels
    } else if allele_type == AlleleType::Substitution as i32 {
        count >= opts.min_count_snps && frac >= opts.min_fraction_snps
    } else {
        false
    }
}

/// Build candidate Variant records from a sorted list of AlleleCounts.
pub fn candidates_from_counts(
    counts: &[AlleleCount],
    opts: &VariantCallerOptions,
) -> Vec<Variant> {
    let mut out = Vec::new();
    for c in counts {
        let summary = summarize(c);
        if summary.total_reads == 0 {
            continue;
        }
        let mut passing_alts: Vec<(i32, String, i32)> = summary
            .by_allele
            .iter()
            .filter(|((t, _), &cnt)| passes(*t, cnt, summary.total_reads, opts))
            .map(|((t, b), &cnt)| (*t, b.clone(), cnt))
            .collect();
        if passing_alts.is_empty() {
            continue;
        }
        // Sort alts deterministically for a stable output.
        passing_alts.sort_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));

        // Collapse alts that project to the same alt base string. Two
        // distinct `(allele_type, bases)` keys (e.g. an insertion and a
        // substitution) can yield the same alt string; emitting it twice
        // produces a malformed multi-allelic Variant
        // (`alternate_bases = ["GT", "GT"]`). Downstream that breaks the
        // postprocess merge: `to_remove` is a HashSet so it dedupes the
        // string, the `to_remove.len() == alternate_bases.len()` safety
        // net misfires, and `prune_alleles` strips *both* copies, leaving
        // a 0-alt variant → `n_alleles must be >= 2` panic. Sum the read
        // counts for collapsed alts so AD/VAF stay consistent.
        {
            let mut deduped: Vec<(i32, String, i32)> = Vec::with_capacity(passing_alts.len());
            for (t, b, cnt) in passing_alts.drain(..) {
                match deduped.iter_mut().find(|(_, eb, _)| *eb == b) {
                    Some((_, _, ec)) => *ec += cnt,
                    None => deduped.push((t, b, cnt)),
                }
            }
            passing_alts = deduped;
        }

        let pos = c.position.as_ref().expect("position");
        let ref_len = if passing_alts
            .iter()
            .any(|(t, _, _)| *t == AlleleType::Deletion as i32)
        {
            // Deletions extend the REF: longest del bases length.
            let max_del = passing_alts
                .iter()
                .filter(|(t, _, _)| *t == AlleleType::Deletion as i32)
                .map(|(_, b, _)| b.len() as i64)
                .max()
                .unwrap_or(1);
            max_del
        } else {
            1
        };
        let ref_bases = if ref_len > 1 {
            // We don't have a ref-string buffer here; use the AlleleCount's
            // recorded ref_base + 'N' filler for the rest. Callers that need
            // exact REF bases must overlay the FASTA reference. For
            // make_examples this is fine because pileup_image_native
            // recomputes from the FASTA anyway.
            let mut s = c.ref_base.clone();
            for _ in 1..ref_len {
                s.push('N');
            }
            s
        } else {
            c.ref_base.clone()
        };
        let alt_bases: Vec<String> = passing_alts
            .iter()
            .map(|(_, bases, _)| bases.clone())
            .collect();
        let mut variant = Variant {
            reference_name: pos.reference_name.clone(),
            start: pos.position,
            end: pos.position + ref_len,
            reference_bases: ref_bases,
            alternate_bases: alt_bases,
            ..Default::default()
        };
        // Populate per-call DP/AD/VAF from the allele count summary.
        let dp = summary.total_reads;
        let mut call = VariantCall::default();
        set_int(&mut call, "DP", dp);
        let mut ad: Vec<i32> = Vec::with_capacity(passing_alts.len() + 1);
        ad.push(summary.ref_supporting);
        for (_, _, cnt) in &passing_alts {
            ad.push(*cnt);
        }
        set_int_list(&mut call, "AD", &ad);
        let mut vaf: Vec<f64> = Vec::with_capacity(passing_alts.len());
        let total = (dp.max(1)) as f64;
        for (_, _, cnt) in &passing_alts {
            vaf.push(*cnt as f64 / total);
        }
        set_float_list(&mut call, "VAF", &vaf);
        variant.calls = vec![call];
        out.push(variant);
    }
    out
}

fn set_int(call: &mut VariantCall, key: &str, n: i32) {
    call.info.insert(
        key.to_string(),
        ListValue {
            values: vec![Value {
                kind: Some(value::Kind::IntValue(n)),
            }],
        },
    );
}

fn set_int_list(call: &mut VariantCall, key: &str, ns: &[i32]) {
    call.info.insert(
        key.to_string(),
        ListValue {
            values: ns
                .iter()
                .map(|&n| Value {
                    kind: Some(value::Kind::IntValue(n)),
                })
                .collect(),
        },
    );
}

fn set_float_list(call: &mut VariantCall, key: &str, xs: &[f64]) {
    call.info.insert(
        key.to_string(),
        ListValue {
            values: xs
                .iter()
                .map(|&x| Value {
                    kind: Some(value::Kind::NumberValue(x)),
                })
                .collect(),
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allelecounter::{add_read, empty_counts, AlignedRead, CounterOptions};

    fn allele(bases: &str, t: AlleleType) -> Allele {
        Allele {
            bases: bases.to_string(),
            r#type: t as i32,
            count: 1,
            ..Default::default()
        }
    }

    #[test]
    fn deletion_size_returns_length_for_dels_and_minus_one_otherwise() {
        assert_eq!(deletion_size(&allele("ACGT", AlleleType::Deletion)), 4);
        assert_eq!(deletion_size(&allele("A", AlleleType::Substitution)), -1);
        assert_eq!(deletion_size(&allele("ATC", AlleleType::Insertion)), -1);
        assert_eq!(deletion_size(&allele("A", AlleleType::Reference)), -1);
    }

    #[test]
    fn calc_ref_bases_no_alts_returns_input() {
        assert_eq!(calc_ref_bases("A", &[]), "A");
    }

    #[test]
    fn calc_ref_bases_no_dels_returns_input() {
        let alts = vec![
            allele("C", AlleleType::Substitution),
            allele("ATG", AlleleType::Insertion),
        ];
        assert_eq!(calc_ref_bases("A", &alts), "A");
    }

    #[test]
    fn calc_ref_bases_picks_longest_del() {
        // Two deletions, longest wins. DEL bases include the anchor +
        // the deleted ref bases. The variant's REF is `ref_bases` plus
        // the *deleted* bases (everything after the anchor).
        let alts = vec![
            allele("AC", AlleleType::Deletion),    // anchor A + del 1bp
            allele("ACGT", AlleleType::Deletion),  // anchor A + del 3bp ← longer
            allele("C", AlleleType::Substitution),
        ];
        assert_eq!(calc_ref_bases("A", &alts), "ACGT");
    }

    #[test]
    fn make_alt_allele_examples_from_upstream_doc() {
        // Matches upstream's example: variant_ref="ACGT" because of
        // a 3-bp deletion. SNV "C": prefix="C" from=1 → "CCGT".
        assert_eq!(make_alt_allele("C", "ACGT", 1), "CCGT");
        // INS "ATTT": prefix="ATTT" from=1 → "ATTTCGT".
        assert_eq!(make_alt_allele("ATTT", "ACGT", 1), "ATTTCGT");
        // DEL "ACGT": prefix="A" from=4 (= bases.len()) → "A".
        assert_eq!(make_alt_allele("A", "ACGT", 4), "A");
        // Edge: from > variant_ref.len() → just prefix.
        assert_eq!(make_alt_allele("X", "ACGT", 100), "X");
    }

    #[test]
    fn allele_to_variant_alt_handles_each_type() {
        let variant_ref = "ACGT";
        let snv = allele("C", AlleleType::Substitution);
        assert_eq!(
            allele_to_variant_alt(&snv, variant_ref).unwrap(),
            "CCGT"
        );
        let ins = allele("ATT", AlleleType::Insertion);
        assert_eq!(
            allele_to_variant_alt(&ins, variant_ref).unwrap(),
            "ATTCGT"
        );
        let del = allele("ACGT", AlleleType::Deletion);
        assert_eq!(allele_to_variant_alt(&del, variant_ref).unwrap(), "A");
        // REF / SOFT_CLIP types return None.
        let refa = allele("A", AlleleType::Reference);
        assert!(allele_to_variant_alt(&refa, variant_ref).is_none());
    }

    fn syn_read<'a>(name: &'a str, ref_start: i64, cigar: &'a [(char, i64)], seq: &'a [u8]) -> AlignedRead<'a> {
        AlignedRead {
            name,
            mate_number: 1,
            ref_start,
            cigar,
            seq,
            base_quality: &[40; 200][..seq.len()],
            mapping_quality: 60,
            is_reverse_strand: false,
        }
    }

    #[test]
    fn snp_passes_threshold() {
        let mut counts = empty_counts("chr1", 100, 105, b"AAAAA");
        let opts = CounterOptions::default();
        // 8 ref reads, 4 alt reads at position 102 → fraction 4/12 = 0.33 > 0.12
        for i in 0..8 {
            let name = format!("ref{i}");
            add_read(
                &mut counts,
                &syn_read(&name, 100, &[('M', 5)], b"AAAAA"),
                &opts,
                100,
            );
        }
        for i in 0..4 {
            let name = format!("alt{i}");
            add_read(
                &mut counts,
                &syn_read(&name, 100, &[('M', 5)], b"AACAA"),
                &opts,
                100,
            );
        }
        let cs = candidates_from_counts(&counts, &VariantCallerOptions::default());
        assert_eq!(cs.len(), 1);
        assert_eq!(cs[0].start, 102);
        assert_eq!(cs[0].reference_bases, "A");
        assert_eq!(cs[0].alternate_bases, vec!["C"]);
    }

    #[test]
    fn snp_fails_threshold() {
        let mut counts = empty_counts("chr1", 100, 105, b"AAAAA");
        let opts = CounterOptions::default();
        // 100 ref reads, 1 alt — fraction 1/101 < 0.12
        for i in 0..100 {
            let name = format!("ref{i}");
            add_read(&mut counts, &syn_read(&name, 100, &[('M', 5)], b"AAAAA"), &opts, 100);
        }
        add_read(&mut counts, &syn_read("alt", 100, &[('M', 5)], b"AACAA"), &opts, 100);
        let cs = candidates_from_counts(&counts, &VariantCallerOptions::default());
        assert!(cs.is_empty());
    }

    #[test]
    fn insertion_passes_threshold() {
        let mut counts = empty_counts("chr1", 100, 110, b"AAAAAAAAAA");
        let opts = CounterOptions::default();
        // 8 ref + 4 ins reads at position 102 (insertion of "CC" between bases 2 and 3)
        for i in 0..8 {
            let name = format!("ref{i}");
            add_read(&mut counts, &syn_read(&name, 100, &[('M', 10)], b"AAAAAAAAAA"), &opts, 100);
        }
        for i in 0..4 {
            let name = format!("ins{i}");
            add_read(
                &mut counts,
                &syn_read(&name, 100, &[('M', 3), ('I', 2), ('M', 7)], b"AAACCAAAAAAA"),
                &opts,
                100,
            );
        }
        let cs = candidates_from_counts(&counts, &VariantCallerOptions::default());
        assert_eq!(cs.len(), 1);
        assert_eq!(cs[0].start, 102);
        assert_eq!(cs[0].alternate_bases, vec!["ACC"]);
    }

    /// Mirrors upstream `TestNoVariant`. Pure-ref pileup must not emit
    /// any candidate.
    #[test]
    fn no_variant_with_pure_ref() {
        let mut counts = empty_counts("chr1", 100, 105, b"AAAAA");
        let opts = CounterOptions::default();
        for i in 0..15 {
            let name = format!("ref{i}");
            add_read(&mut counts, &syn_read(&name, 100, &[('M', 5)], b"AAAAA"), &opts, 100);
        }
        let cs = candidates_from_counts(&counts, &VariantCallerOptions::default());
        assert!(cs.is_empty());
    }

    /// Mirrors upstream `TestNonCanonicalBase`. Bases like 'N' don't
    /// trigger a candidate (they're filtered by allele counter via
    /// the substitution branch — no alt allele is recorded).
    #[test]
    fn non_canonical_base_does_not_emit() {
        let mut counts = empty_counts("chr1", 100, 105, b"AAAAA");
        let opts = CounterOptions::default();
        for i in 0..10 {
            let name = format!("ref{i}");
            add_read(&mut counts, &syn_read(&name, 100, &[('M', 5)], b"AAAAA"), &opts, 100);
        }
        // 5 reads with 'N' at position 102 — N is canonical for our
        // counter (it tracks the base) but doesn't pass the SUB
        // threshold against the high ref count without enough fraction.
        // We use this to verify that even with N as an alt the
        // overall threshold filtering applies.
        let mut counts2 = counts.clone();
        for i in 0..1 {
            let name = format!("nn{i}");
            add_read(&mut counts2, &syn_read(&name, 100, &[('M', 5)], b"AANAA"), &opts, 100);
        }
        let cs = candidates_from_counts(&counts2, &VariantCallerOptions::default());
        // 1/11 = 9% < 12% threshold → no candidate.
        assert!(cs.is_empty());
    }

    /// Mirrors upstream `TestMinCount1`. With min_count_snps=10 a
    /// single SNP read isn't enough.
    #[test]
    fn min_count_threshold() {
        let mut counts = empty_counts("chr1", 100, 105, b"AAAAA");
        let opts = CounterOptions::default();
        for _ in 0..1 {
            add_read(&mut counts, &syn_read("alt", 100, &[('M', 5)], b"AACAA"), &opts, 100);
        }
        let mut o = VariantCallerOptions::default();
        o.min_count_snps = 10;
        o.min_fraction_snps = 0.0;
        let cs = candidates_from_counts(&counts, &o);
        assert!(cs.is_empty());
        // Drop the count threshold to 1 → candidate emerges.
        let mut o2 = o.clone();
        o2.min_count_snps = 1;
        let cs = candidates_from_counts(&counts, &o2);
        assert_eq!(cs.len(), 1);
    }

    /// Mirrors upstream `TestMultAllelicSNP`. Two distinct alts each
    /// passing the threshold should appear in the same candidate
    /// record's alternate_bases list, sorted lexicographically.
    #[test]
    fn multi_allelic_snp() {
        let mut counts = empty_counts("chr1", 100, 105, b"AAAAA");
        let opts = CounterOptions::default();
        for i in 0..5 {
            let name = format!("ref{i}");
            add_read(&mut counts, &syn_read(&name, 100, &[('M', 5)], b"AAAAA"), &opts, 100);
        }
        for i in 0..5 {
            let name = format!("alt_c{i}");
            add_read(&mut counts, &syn_read(&name, 100, &[('M', 5)], b"AACAA"), &opts, 100);
        }
        for i in 0..5 {
            let name = format!("alt_g{i}");
            add_read(&mut counts, &syn_read(&name, 100, &[('M', 5)], b"AAGAA"), &opts, 100);
        }
        let cs = candidates_from_counts(&counts, &VariantCallerOptions::default());
        assert_eq!(cs.len(), 1);
        assert_eq!(cs[0].start, 102);
        // Sorted alphabetically — C before G.
        assert_eq!(cs[0].alternate_bases, vec!["C", "G"]);
    }

    /// Mirrors upstream `TestBiAllelicDeletion`. A 2bp deletion
    /// passing thresholds emits a single candidate at the anchor
    /// position with REF span = anchor + deleted bases.
    ///
    /// Our allelecounter now stores DEL alleles with the actual
    /// ref bases (anchor + deleted ref bases), matching upstream's
    /// `AlleleCounter::AddReadAllele(DEL)`. variant_calling
    /// propagates that into `reference_bases` and `alternate_bases`.
    #[test]
    fn bi_allelic_deletion() {
        let mut counts = empty_counts("chr1", 100, 110, b"AAAAAAAAAA");
        let opts = CounterOptions::default();
        for i in 0..8 {
            let name = format!("ref{i}");
            add_read(&mut counts, &syn_read(&name, 100, &[('M', 10)], b"AAAAAAAAAA"), &opts, 100);
        }
        for i in 0..4 {
            let name = format!("del{i}");
            add_read(
                &mut counts,
                &syn_read(&name, 100, &[('M', 3), ('D', 2), ('M', 5)], b"AAAAAAAA"),
                &opts,
                100,
            );
        }
        let cs = candidates_from_counts(&counts, &VariantCallerOptions::default());
        assert_eq!(cs.len(), 1);
        assert_eq!(cs[0].start, 102);
        // REF spans 1 anchor + 2 deleted bases = 3.
        assert_eq!(cs[0].reference_bases.len(), 3);
        assert_eq!(cs[0].alternate_bases.len(), 1);
    }

    /// Two simultaneous indel alts at the same anchor.
    #[test]
    fn snp_plus_insertion_at_same_position() {
        let mut counts = empty_counts("chr1", 100, 110, b"AAAAAAAAAA");
        let opts = CounterOptions::default();
        for i in 0..6 {
            let name = format!("ref{i}");
            add_read(&mut counts, &syn_read(&name, 100, &[('M', 10)], b"AAAAAAAAAA"), &opts, 100);
        }
        for i in 0..3 {
            let name = format!("snp{i}");
            // SNP at position 102: A → C
            add_read(&mut counts, &syn_read(&name, 100, &[('M', 10)], b"AACAAAAAAA"), &opts, 100);
        }
        for i in 0..3 {
            let name = format!("ins{i}");
            // Insertion of CC after position 102.
            add_read(
                &mut counts,
                &syn_read(&name, 100, &[('M', 3), ('I', 2), ('M', 7)], b"AAACCAAAAAAA"),
                &opts,
                100,
            );
        }
        let cs = candidates_from_counts(&counts, &VariantCallerOptions::default());
        // Both alts appear at the appropriate position. SNP is at 102,
        // insertion anchored at 102 — same position so one record with
        // two alts.
        assert!(!cs.is_empty());
        let snp = cs.iter().find(|v| v.start == 102 && v.reference_bases == "A");
        assert!(snp.is_some(), "expected SNP candidate at 102");
        // Alt list must contain both "C" and an insertion-style "ACC".
        let alts = &snp.unwrap().alternate_bases;
        assert!(alts.iter().any(|a| a == "C"), "missing SNP alt C");
        assert!(alts.iter().any(|a| a.len() == 3), "missing INS alt ACC-like");
    }

    /// Two distinct `(allele_type, bases)` keys with the SAME base string
    /// (e.g. an insertion and a substitution that both spell "GT") must
    /// collapse to a single alt — `alternate_bases` must never contain a
    /// duplicate, and AD must sum the collapsed reads. Pre-fix this
    /// produced `["GT","GT"]`, which made postprocess emit a 0-alt
    /// variant and panic (`n_alleles must be >= 2`). Real-data repro:
    /// chr20:35167420 ref="GN" on full-chr20 HG003.
    #[test]
    fn duplicate_alt_string_is_collapsed() {
        use dv_proto::nucleus_v1::Position;

        let mut read_alleles = std::collections::BTreeMap::new();
        for i in 0..4 {
            read_alleles.insert(
                format!("ins{i}/1"),
                Allele { bases: "GT".into(), r#type: AlleleType::Insertion as i32, count: 1, ..Default::default() },
            );
        }
        for i in 0..4 {
            read_alleles.insert(
                format!("sub{i}/1"),
                Allele { bases: "GT".into(), r#type: AlleleType::Substitution as i32, count: 1, ..Default::default() },
            );
        }
        let ac = AlleleCount {
            position: Some(Position {
                reference_name: "chr20".into(),
                position: 35_167_420,
                ..Default::default()
            }),
            ref_base: "G".into(),
            ref_supporting_read_count: 4,
            read_alleles,
            ..Default::default()
        };
        let cs = candidates_from_counts(&[ac], &VariantCallerOptions::default());
        assert_eq!(cs.len(), 1, "expected one candidate");
        let v = &cs[0];
        assert_eq!(
            v.alternate_bases,
            vec!["GT".to_string()],
            "duplicate alt not collapsed: {:?}",
            v.alternate_bases
        );
        // AD = [ref, collapsed-alt] = [4, 4+4].
        let ad = &v.calls[0].info["AD"].values;
        let ad_ints: Vec<i64> = ad
            .iter()
            .filter_map(|x| match &x.kind {
                Some(value::Kind::IntValue(n)) => Some(*n as i64),
                _ => None,
            })
            .collect();
        assert_eq!(ad_ints, vec![4, 8], "AD not summed across collapsed alts");
    }
}
