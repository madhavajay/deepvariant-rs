//! Candidate variant generation from `AlleleCount`s.
//!
//! Port-in-progress of `deepvariant/variant_calling.cc`. Implements the
//! threshold-based candidate caller: a position is a candidate when an
//! alt allele has both `count >= min_count` AND
//! `count / total_reads >= min_fraction`. Different thresholds for SNPs
//! vs indels. No genotyping (model does that downstream).

use std::collections::HashMap;

use dv_proto::dv::{AlleleCount, AlleleType};
use dv_proto::nucleus_v1::{value, ListValue, Value, Variant, VariantCall};

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
}
