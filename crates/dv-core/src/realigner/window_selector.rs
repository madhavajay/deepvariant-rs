//! Port of `deepvariant/realigner/window_selector.cc`.
//!
//! Walks per-position AlleleCounts and returns a per-position support
//! score that downstream code uses to pick out windows for de-novo
//! haplotype assembly.
//!
//! Algorithm:
//!   - For each position's AlleleCount, select alt alleles whose
//!     `count >= min_allele_support`.
//!   - For each alt allele, increment counts in a span around the
//!     anchor position whose width depends on allele type:
//!       * SUBSTITUTION: span is 1bp, exactly at anchor
//!       * INSERTION / SOFT_CLIP: span is `2 * len(bases) - 1`,
//!         centered such that the span is
//!         `[anchor + 1 - (len-1), anchor + len)`
//!       * DELETION: span is `[anchor + 1, anchor + len)`
//!   - Optionally apply a strict 8% VAF threshold to short insertions
//!     (length <= 2).

use crate::allelecounter::summarize_alleles;
use dv_proto::dv::{AlleleCount, AlleleType};

#[derive(Debug, Clone, Copy)]
pub struct WindowSelectorOptions {
    pub min_allele_support: i32,
    pub enable_strict_insertion_filter: bool,
}

impl Default for WindowSelectorOptions {
    fn default() -> Self {
        Self {
            min_allele_support: 2,
            enable_strict_insertion_filter: true,
        }
    }
}

fn allele_passes(
    allele_type: i32,
    count: i32,
    bases: &str,
    total: i32,
    opts: &WindowSelectorOptions,
) -> bool {
    if allele_type == AlleleType::Reference as i32
        || allele_type == AlleleType::Unspecified as i32
    {
        return false;
    }
    if count < opts.min_allele_support {
        return false;
    }
    if opts.enable_strict_insertion_filter
        && allele_type == AlleleType::Insertion as i32
        && bases.len() <= 2
    {
        let total = total.max(1) as f32;
        if (count as f32) / total < 0.08 {
            return false;
        }
    }
    true
}

fn update_counts(amount: i32, start: i64, end: i64, scores: &mut [i32]) {
    let lo = start.max(0) as usize;
    let hi = end.min(scores.len() as i64) as usize;
    for i in lo..hi {
        scores[i] += amount;
    }
}

/// Compute per-position window-selector candidate scores.
/// Returns a vector of length `counts.len()`. The i-th value is the sum of
/// alt-allele read support contributions affecting position i.
pub fn variant_reads_candidates(
    counts: &[AlleleCount],
    opts: &WindowSelectorOptions,
) -> Vec<i32> {
    let mut scores = vec![0i32; counts.len()];
    for (i, c) in counts.iter().enumerate() {
        let total = c.ref_supporting_read_count
            + c.ref_nonconfident_read_count
            + c.read_alleles.len() as i32;
        let summary = summarize_alleles(c);
        for allele in &summary {
            if !allele_passes(allele.r#type, allele.count, &allele.bases, total, opts) {
                continue;
            }
            let pos = i as i64;
            let len = allele.bases.len() as i64;
            let (start, end) = if allele.r#type == AlleleType::Substitution as i32 {
                (pos, pos + 1)
            } else if allele.r#type == AlleleType::Insertion as i32
                || allele.r#type == AlleleType::SoftClip as i32
            {
                (pos + 1 - (len - 1), pos + len)
            } else if allele.r#type == AlleleType::Deletion as i32 {
                (pos + 1, pos + len)
            } else {
                continue;
            };
            update_counts(allele.count, start, end, &mut scores);
        }
    }
    scores
}

/// Convert per-position scores into a list of `[start, end)` windows by
/// merging consecutive positions that score above `min_score`.
pub fn windows_from_scores(scores: &[i32], min_score: i32) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    let mut start: Option<usize> = None;
    for (i, &s) in scores.iter().enumerate() {
        if s >= min_score {
            start.get_or_insert(i);
        } else if let Some(s0) = start.take() {
            out.push((s0, i));
        }
    }
    if let Some(s0) = start {
        out.push((s0, scores.len()));
    }
    out
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
    fn snv_scores_position_only() {
        let mut counts = empty_counts("c", 100, 105, b"AAAAA");
        let opts = CounterOptions::default();
        for i in 0..3 {
            let name = format!("alt{i}");
            add_read(&mut counts, &syn_read(&name, 100, &[('M', 5)], b"AACAA"), &opts, 100);
        }
        let scores = variant_reads_candidates(&counts, &WindowSelectorOptions::default());
        assert_eq!(scores, vec![0, 0, 3, 0, 0]);
    }

    #[test]
    fn deletion_scores_span() {
        // Read 3M3D3M anchored at offset 100. Deletion is anchored at offset 2
        // and spans 3 ref bases. Window selector should score positions 3..5.
        let mut counts = empty_counts("c", 100, 109, b"AAAAAAAAA");
        let opts = CounterOptions::default();
        for i in 0..3 {
            let name = format!("d{i}");
            add_read(
                &mut counts,
                &syn_read(&name, 100, &[('M', 3), ('D', 3), ('M', 3)], b"AAAAAA"),
                &opts,
                100,
            );
        }
        let scores = variant_reads_candidates(&counts, &WindowSelectorOptions::default());
        // Deletion length=4 bytes (anchor + 3 deleted). start = anchor + 1 = 3,
        // end = anchor + 4 = 6. Positions [3, 6) get +3.
        assert_eq!(&scores[0..3], &[0, 0, 0]);
        assert_eq!(&scores[3..6], &[3, 3, 3]);
        assert_eq!(&scores[6..9], &[0, 0, 0]);
    }

    #[test]
    fn insertion_scores_span() {
        // 3M3I3M: insertion anchored at position 2, length-3 INS bases.
        let mut counts = empty_counts("c", 100, 106, b"AAAAAA");
        let opts = CounterOptions::default();
        for i in 0..3 {
            let name = format!("i{i}");
            add_read(
                &mut counts,
                &syn_read(&name, 100, &[('M', 3), ('I', 3), ('M', 3)], b"AAACCCAAA"),
                &opts,
                100,
            );
        }
        let scores = variant_reads_candidates(&counts, &WindowSelectorOptions::default());
        // INS allele bases = anchor + inserted = 4 bytes long (len=4).
        // start = anchor + 1 - (4-1) = 0, end = anchor + 4 = 6.
        // Positions [0, 6) get +3.
        for &s in &scores {
            assert_eq!(s, 3);
        }
    }

    #[test]
    fn min_allele_support_filters() {
        let mut counts = empty_counts("c", 100, 105, b"AAAAA");
        let opts = CounterOptions::default();
        // Just 1 alt read — below default min_allele_support = 2.
        add_read(&mut counts, &syn_read("alt", 100, &[('M', 5)], b"AACAA"), &opts, 100);
        let scores = variant_reads_candidates(&counts, &WindowSelectorOptions::default());
        assert_eq!(scores, vec![0; 5]);
    }

    #[test]
    fn windows_merging() {
        let scores = vec![0, 0, 3, 5, 0, 1, 4, 0, 2];
        let ws = windows_from_scores(&scores, 2);
        assert_eq!(ws, vec![(2, 4), (6, 7), (8, 9)]);
    }
}
