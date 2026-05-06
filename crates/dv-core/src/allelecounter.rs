//! Per-position allele tallying from aligned reads.
//!
//! Port-in-progress of `deepvariant/allelecounter.cc` — covers the core
//! CIGAR walk and per-position allele aggregation for SNV/INS/DEL alleles.
//! Upstream's full implementation (~1000 LOC) additionally handles
//! methylation, low-quality flagging, sample-level aggregation, and DeepTrio
//! multi-sample fusion. Those are deferred.
//!
//! The output is a `Vec<AlleleCount>` (one per position in the requested
//! range), matching the upstream proto layout — so downstream code that
//! consumes AlleleCount can be ported against this.

use dv_proto::dv::{Allele, AlleleCount, AlleleType};
use dv_proto::nucleus_v1::Position;

/// A read summarized as a CIGAR-walked alignment (the bare minimum
/// AlleleCounter needs from a noodles BAM record).
#[derive(Debug, Clone)]
pub struct AlignedRead<'a> {
    /// Read fragment name (used as the key in `read_alleles`).
    pub name: &'a str,
    pub mate_number: i32,
    /// 0-based position on the reference where the alignment starts.
    pub ref_start: i64,
    /// CIGAR ops as `(op_char, length)` pairs.
    pub cigar: &'a [(char, i64)],
    /// Per-base sequence (uppercase, length must match CIGAR's read-advancing total).
    pub seq: &'a [u8],
    /// Per-base quality (length matches `seq`).
    pub base_quality: &'a [u8],
    pub mapping_quality: u8,
    pub is_reverse_strand: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct CounterOptions {
    pub min_base_quality: u8,
    pub min_mapping_quality: u8,
}

impl Default for CounterOptions {
    fn default() -> Self {
        Self {
            min_base_quality: 10,
            min_mapping_quality: 10,
        }
    }
}

/// Initialize an `AlleleCount` at every position in `[start, end)` of `contig`,
/// pre-populated with reference bases from `ref_bases` (length == end - start).
pub fn empty_counts(
    contig: &str,
    start: i64,
    end: i64,
    ref_bases: &[u8],
) -> Vec<AlleleCount> {
    assert_eq!(
        ref_bases.len(),
        (end - start) as usize,
        "ref_bases length must match [start, end)"
    );
    (start..end)
        .enumerate()
        .map(|(i, pos)| AlleleCount {
            position: Some(Position {
                reference_name: contig.to_string(),
                position: pos,
                reverse_strand: false,
            }),
            ref_base: (ref_bases[i] as char).to_ascii_uppercase().to_string(),
            ..Default::default()
        })
        .collect()
}

/// Add one read's allele evidence to the per-position counts.
pub fn add_read(counts: &mut [AlleleCount], read: &AlignedRead, opts: &CounterOptions, region_start: i64) {
    if read.mapping_quality < opts.min_mapping_quality {
        return;
    }
    let read_id = format!("{}/{}", read.name, read.mate_number);

    let mut ref_pos = read.ref_start;
    let mut read_pos = 0usize;

    // Pending insertion bases keyed to the *previous* ref position.
    // We accumulate insertion bases while consuming an `I` op.
    let mut idx_iter = 0usize;
    let cigar = read.cigar;
    while idx_iter < cigar.len() {
        let (op, len) = cigar[idx_iter];
        let len_usize = len as usize;
        match op {
            'M' | '=' | 'X' => {
                for k in 0..len_usize {
                    let rp = ref_pos + k as i64;
                    let pos_idx = rp - region_start;
                    if pos_idx >= 0 && pos_idx < counts.len() as i64 {
                        let read_b = read.seq[read_pos + k].to_ascii_uppercase();
                        let bq = read.base_quality[read_pos + k];
                        let count = &mut counts[pos_idx as usize];
                        let ref_b = count.ref_base.as_bytes()[0].to_ascii_uppercase();
                        if read_b == ref_b {
                            if bq >= opts.min_base_quality {
                                count.ref_supporting_read_count += 1;
                            } else {
                                count.ref_nonconfident_read_count += 1;
                            }
                        } else if bq >= opts.min_base_quality {
                            count.read_alleles.insert(
                                read_id.clone(),
                                Allele {
                                    bases: (read_b as char).to_string(),
                                    r#type: AlleleType::Substitution as i32,
                                    count: 1,
                                    is_low_quality: false,
                                    avg_base_quality: bq as i32,
                                    mapping_quality: read.mapping_quality as i32,
                                    is_reverse_strand: read.is_reverse_strand,
                                    ..Default::default()
                                },
                            );
                        }
                    }
                }
                ref_pos += len;
                read_pos += len_usize;
            }
            'I' => {
                // Insertion is keyed to the previous ref position
                // (the base before the insertion).
                let anchor_ref = ref_pos - 1;
                let pos_idx = anchor_ref - region_start;
                if pos_idx >= 0 && pos_idx < counts.len() as i64 {
                    let count = &mut counts[pos_idx as usize];
                    let ref_b = count.ref_base.as_bytes()[0].to_ascii_uppercase();
                    let mut bases = String::with_capacity(len_usize + 1);
                    bases.push(ref_b as char);
                    let mut min_bq = u8::MAX;
                    for k in 0..len_usize {
                        let b = read.seq[read_pos + k].to_ascii_uppercase();
                        bases.push(b as char);
                        let bq = read.base_quality[read_pos + k];
                        if bq < min_bq {
                            min_bq = bq;
                        }
                    }
                    if min_bq >= opts.min_base_quality {
                        count.read_alleles.insert(
                            read_id.clone(),
                            Allele {
                                bases,
                                r#type: AlleleType::Insertion as i32,
                                count: 1,
                                is_low_quality: false,
                                avg_base_quality: min_bq as i32,
                                mapping_quality: read.mapping_quality as i32,
                                is_reverse_strand: read.is_reverse_strand,
                                ..Default::default()
                            },
                        );
                    }
                }
                read_pos += len_usize;
            }
            'D' => {
                // Deletion keyed to the previous ref position. Bases =
                // anchor ref base + the deleted ref bases (matching
                // upstream's `AlleleCounter::AddReadAllele(DEL)` which
                // splices in the actual reference). When the deleted
                // span runs off the end of the AlleleCount slice
                // (e.g. read trails past the region) the missing
                // positions fall back to 'N' so the bases length still
                // reflects the deletion size.
                let anchor_ref = ref_pos - 1;
                let pos_idx = anchor_ref - region_start;
                if pos_idx >= 0 && pos_idx < counts.len() as i64 {
                    let ref_b = counts[pos_idx as usize]
                        .ref_base
                        .as_bytes()[0]
                        .to_ascii_uppercase();
                    let mut bases = String::with_capacity(len_usize + 1);
                    bases.push(ref_b as char);
                    for k in 0..len_usize {
                        let del_idx = pos_idx + 1 + k as i64;
                        if del_idx >= 0 && (del_idx as usize) < counts.len() {
                            let del_b = counts[del_idx as usize]
                                .ref_base
                                .as_bytes()[0]
                                .to_ascii_uppercase();
                            bases.push(del_b as char);
                        } else {
                            bases.push('N');
                        }
                    }
                    let count = &mut counts[pos_idx as usize];
                    count.read_alleles.insert(
                        read_id.clone(),
                        Allele {
                            bases,
                            r#type: AlleleType::Deletion as i32,
                            count: 1,
                            is_low_quality: false,
                            avg_base_quality: 0,
                            mapping_quality: read.mapping_quality as i32,
                            is_reverse_strand: read.is_reverse_strand,
                            ..Default::default()
                        },
                    );
                }
                ref_pos += len;
            }
            'N' => {
                // Skipped reference (e.g., RNA-seq splice). No allele evidence;
                // just advance ref.
                ref_pos += len;
            }
            'S' => {
                // Soft-clipped read bases; advance read but not ref.
                read_pos += len_usize;
            }
            'H' | 'P' => {
                // Hard clip / pad: nothing to do.
            }
            _ => {
                // Unknown op — bail early to avoid silent miscounts.
                return;
            }
        }
        idx_iter += 1;
    }
}

/// Aggregate per-position alleles into a sorted summary list.
pub fn summarize_alleles(count: &AlleleCount) -> Vec<Allele> {
    let mut by_key: std::collections::HashMap<(i32, String), (i32, u8, u8, bool)> =
        std::collections::HashMap::new();
    for allele in count.read_alleles.values() {
        let key = (allele.r#type, allele.bases.clone());
        let entry = by_key.entry(key).or_insert((
            0,
            0,
            0,
            allele.is_reverse_strand,
        ));
        entry.0 += 1;
    }
    let mut out: Vec<Allele> = by_key
        .into_iter()
        .map(|((t, bases), (count, _bq, _mq, rev))| Allele {
            bases,
            r#type: t,
            count,
            is_low_quality: false,
            avg_base_quality: 0,
            mapping_quality: 0,
            is_reverse_strand: rev,
            ..Default::default()
        })
        .collect();
    if count.ref_supporting_read_count > 0 {
        out.push(Allele {
            bases: count.ref_base.clone(),
            r#type: AlleleType::Reference as i32,
            count: count.ref_supporting_read_count,
            ..Default::default()
        });
    }
    out.sort_by(|a, b| (a.r#type, a.bases.clone()).cmp(&(b.r#type, b.bases.clone())));
    out
}

/// Total number of reads contributing to this position (ref + non-ref).
pub fn total_count(c: &AlleleCount) -> i32 {
    c.ref_supporting_read_count + c.ref_nonconfident_read_count + c.read_alleles.len() as i32
}

/// Sum of read counts grouped by (allele_type, bases) — analog of
/// upstream `SumAlleleCounts` (single AlleleCount). Optionally
/// excludes low-quality alleles. Mirrors upstream's REF-injection
/// behavior: when ref_supporting_read_count > 0 and `track_ref_reads`
/// is false, a synthetic REFERENCE allele is appended with that count.
pub fn sum_allele_counts(count: &AlleleCount, include_low_quality: bool) -> Vec<Allele> {
    use std::collections::BTreeMap;
    let mut sums: BTreeMap<(i32, String), i32> = BTreeMap::new();
    for allele in count.read_alleles.values() {
        if !include_low_quality && allele.is_low_quality {
            continue;
        }
        *sums
            .entry((allele.r#type, allele.bases.clone()))
            .or_insert(0) += 1;
    }
    let mut out: Vec<Allele> = sums
        .into_iter()
        .map(|((t, bases), c)| Allele {
            bases,
            r#type: t,
            count: c,
            ..Default::default()
        })
        .collect();
    if count.ref_supporting_read_count > 0 && !count.track_ref_reads {
        out.push(Allele {
            bases: count.ref_base.clone(),
            r#type: AlleleType::Reference as i32,
            count: count.ref_supporting_read_count,
            ..Default::default()
        });
    }
    out
}

/// Span variant of `sum_allele_counts` — sums across multiple
/// AlleleCount slots (e.g. when a candidate spans more than one ref
/// base). Mirrors upstream's `SumAlleleCounts(span<AlleleCount>)`.
pub fn sum_allele_counts_span(counts: &[AlleleCount], include_low_quality: bool) -> Vec<Allele> {
    use std::collections::BTreeMap;
    let mut sums: BTreeMap<(i32, String), i32> = BTreeMap::new();
    for ac in counts {
        for allele in ac.read_alleles.values() {
            if !include_low_quality && allele.is_low_quality {
                continue;
            }
            *sums
                .entry((allele.r#type, allele.bases.clone()))
                .or_insert(0) += 1;
        }
    }
    let mut out: Vec<Allele> = sums
        .into_iter()
        .map(|((t, bases), c)| Allele {
            bases,
            r#type: t,
            count: c,
            ..Default::default()
        })
        .collect();
    let total_ref: i32 = counts.iter().map(|c| c.ref_supporting_read_count).sum();
    if total_ref > 0
        && !counts.is_empty()
        && !counts[0].track_ref_reads
    {
        out.push(Allele {
            bases: counts[0].ref_base.clone(),
            r#type: AlleleType::Reference as i32,
            count: total_ref,
            ..Default::default()
        });
    }
    out
}

/// Total number of reads at a single position counting both ref and
/// non-ref support. Excludes REFERENCE-typed alleles in `read_alleles`
/// (which can appear when `track_ref_reads` is set) to avoid double
/// counting them with `ref_supporting_read_count`. Mirrors upstream
/// `TotalAlleleCounts`.
pub fn total_allele_counts(count: &AlleleCount, include_low_quality: bool) -> i32 {
    let mut total = count
        .read_alleles
        .values()
        .filter(|a| {
            (!a.is_low_quality || include_low_quality) && a.r#type != AlleleType::Reference as i32
        })
        .count() as i32;
    total += count.ref_supporting_read_count;
    total
}

/// Span variant of `total_allele_counts`.
pub fn total_allele_counts_span(counts: &[AlleleCount], include_low_quality: bool) -> i32 {
    counts
        .iter()
        .map(|c| total_allele_counts(c, include_low_quality))
        .sum()
}

/// True iff every base in `seq[offset..offset+len]` is one of the
/// canonical bases A/C/G/T/N. Mirrors upstream
/// `nucleus::IsCanonicalBase` applied to the slice. Note: upstream's
/// canonical set includes N — non-canonical bases are e.g. R/Y/W/S
/// (IUPAC ambiguity codes that are rare in modern reads).
fn is_canonical_slice(seq: &[u8], offset: usize, len: usize) -> bool {
    seq[offset..offset + len]
        .iter()
        .all(|&b| matches!(b, b'A' | b'C' | b'G' | b'T' | b'N' | b'a' | b'c' | b'g' | b't' | b'n'))
}

/// Decide whether the bases at `[offset, offset+len)` of `read` can
/// contribute to allele counting. Returns
/// `(usable, is_low_quality)`. Mirrors upstream `CanBasesBeUsed`:
///   * Any non-canonical base (e.g. IUPAC ambiguity) → unusable.
///   * `keep_legacy_behavior=true` AND any per-base BQ below threshold
///     → unusable (matches the v1 behavior).
///   * `keep_legacy_behavior=false` AND average BQ across the span
///     below threshold → usable but flagged low-quality.
pub fn can_bases_be_used(
    seq: &[u8],
    quality: &[u8],
    offset: usize,
    len: usize,
    min_base_quality: u8,
    keep_legacy_behavior: bool,
) -> (bool, bool) {
    if offset + len > seq.len() || offset + len > quality.len() {
        return (false, false);
    }
    let mut indel_bq: u32 = 0;
    for i in 0..len {
        indel_bq += quality[offset + i] as u32;
        if quality[offset + i] < min_base_quality && keep_legacy_behavior {
            return (false, false);
        }
    }
    if !is_canonical_slice(seq, offset, len) {
        return (false, false);
    }
    let mut is_low_quality = false;
    if !keep_legacy_behavior {
        if (indel_bq as i64) < (min_base_quality as i64) * (len as i64) {
            is_low_quality = true;
        }
    }
    (true, is_low_quality)
}

/// Average base quality across the span. For DEL operations there's
/// no in-read coverage to average, so upstream returns the BQ of the
/// base immediately before the deletion (or 0 at offset 0). Mirrors
/// upstream `GetAvgBaseQuality`.
pub fn get_avg_base_quality(
    quality: &[u8],
    cigar_op: char,
    offset: usize,
    len: usize,
) -> i32 {
    if cigar_op == 'D' {
        let idx = offset.saturating_sub(1);
        return quality.get(idx).copied().unwrap_or(0) as i32;
    }
    if len == 0 {
        return 0;
    }
    let sum: u32 = (0..len)
        .filter_map(|i| quality.get(offset + i).copied())
        .map(|q| q as u32)
        .sum();
    (sum / len.max(1) as u32) as i32
}

/// Locate the AlleleCount at `pos` in a sorted slice via binary
/// search. Returns `Some(index)` on hit, `None` otherwise. Mirrors
/// upstream `AlleleIndex`.
pub fn allele_index(counts: &[AlleleCount], pos: i64) -> Option<usize> {
    let idx = counts.partition_point(|c| {
        c.position
            .as_ref()
            .map(|p| p.position < pos)
            .unwrap_or(false)
    });
    if idx >= counts.len() {
        return None;
    }
    let p = counts[idx].position.as_ref().map(|p| p.position).unwrap_or(-1);
    if p == pos {
        Some(idx)
    } else {
        None
    }
}

/// Convert a 5mC modification byte (0..=255) to a probability in
/// [0.0, 1.0]. The encoding matches `BaseModifications` in the BAM
/// MM/ML tag spec — a value of 0 indicates either no modification
/// data or a strong unmodified call.
#[inline]
pub fn methylation_probability(level: u8) -> f64 {
    level as f64 / 255.0
}

/// Decide whether a given base position is "called methylated" given
/// the per-base 5mC level. Mirrors upstream `IsMethylated`. When
/// methylation calling is disabled or no MM/ML data is available, the
/// answer is always false.
pub fn is_methylated(
    level_at_offset: Option<u8>,
    enable_methylation_calling: bool,
    threshold: f64,
) -> bool {
    if !enable_methylation_calling {
        return false;
    }
    let lvl = match level_at_offset {
        Some(0) => return false,
        Some(l) => l,
        None => return false,
    };
    methylation_probability(lvl) > threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    fn syn_read<'a>(
        name: &'a str,
        ref_start: i64,
        cigar: &'a [(char, i64)],
        seq: &'a [u8],
    ) -> AlignedRead<'a> {
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
    fn ref_only_reads() {
        // Region [100, 110), all-ref bases AAAAAAAAAA.
        let ref_bases = b"AAAAAAAAAA";
        let mut counts = empty_counts("chr1", 100, 110, ref_bases);
        let opts = CounterOptions::default();
        let r1 = syn_read("r1", 100, &[('M', 10)], b"AAAAAAAAAA");
        let r2 = syn_read("r2", 100, &[('M', 10)], b"AAAAAAAAAA");
        add_read(&mut counts, &r1, &opts, 100);
        add_read(&mut counts, &r2, &opts, 100);
        for c in &counts {
            assert_eq!(c.ref_supporting_read_count, 2);
            assert!(c.read_alleles.is_empty());
            assert_eq!(total_count(c), 2);
        }
    }

    #[test]
    fn snv_at_position_5() {
        let ref_bases = b"AAAAAAAAAA";
        let mut counts = empty_counts("chr1", 100, 110, ref_bases);
        let opts = CounterOptions::default();
        let r = syn_read("r1", 100, &[('M', 10)], b"AAAAACAAAA"); // SNV at offset 5
        add_read(&mut counts, &r, &opts, 100);
        for (i, c) in counts.iter().enumerate() {
            if i == 5 {
                assert_eq!(c.ref_supporting_read_count, 0);
                assert_eq!(c.read_alleles.len(), 1);
                let a = c.read_alleles.values().next().unwrap();
                assert_eq!(a.bases, "C");
                assert_eq!(a.r#type, AlleleType::Substitution as i32);
            } else {
                assert_eq!(c.ref_supporting_read_count, 1);
            }
        }
    }

    #[test]
    fn insertion_at_position_3() {
        // Read aligns 3M3I3M: 3 ref-matched, 3 inserted, 3 ref-matched.
        // Insertion is anchored to ref position before INS, i.e. ref base #2 (offset 2).
        let ref_bases = b"AAAAAA";
        let mut counts = empty_counts("chr1", 100, 106, ref_bases);
        let opts = CounterOptions::default();
        let r = syn_read("r1", 100, &[('M', 3), ('I', 3), ('M', 3)], b"AAACCCAAA");
        add_read(&mut counts, &r, &opts, 100);
        // First 3 positions: ref count 1
        for i in 0..3 {
            assert_eq!(counts[i].ref_supporting_read_count, 1);
        }
        // Position 2 (anchor of insertion) should also have an insertion allele
        let alleles = &counts[2].read_alleles;
        assert_eq!(alleles.len(), 1);
        let a = alleles.values().next().unwrap();
        assert_eq!(a.r#type, AlleleType::Insertion as i32);
        assert_eq!(a.bases, "ACCC"); // anchor + inserted CCC
        // Last 3 positions: ref count 1
        for i in 3..6 {
            assert_eq!(counts[i].ref_supporting_read_count, 1);
        }
    }

    /// Mirrors upstream `TestDeletionSize2`. Verifies the DEL allele's
    /// `bases` field stores actual reference bases (anchor + deleted
    /// bases) rather than 'N' filler.
    #[test]
    fn deletion_size_2_uses_actual_ref_bases() {
        // Ref bases TGCCT — 5 bases.
        // Read: 1M 2D 2M = match T, delete CC, match GT.
        let ref_bases = b"TGCCT";
        let mut counts = empty_counts("chr1", 100, 105, ref_bases);
        // Reset bases to actual ref bases (test ref only had T's
        // earlier; here we pass the real sequence).
        for (i, c) in counts.iter_mut().enumerate() {
            c.ref_base = (ref_bases[i] as char).to_string();
        }
        let opts = CounterOptions::default();
        // Read: ref starts at 100, 1M then 2D then 2M.
        // Read seq is "TGT" (matches positions 0, 3, 4 of ref).
        // But upstream's MakeRead expects the read seq to match the
        // pre-deletion bases — we use "TGT" same as upstream test.
        let read = AlignedRead {
            name: "r1",
            mate_number: 1,
            ref_start: 100,
            cigar: &[('M', 1), ('D', 2), ('M', 2)],
            seq: b"TGT",
            base_quality: &[40u8; 3],
            mapping_quality: 60,
            is_reverse_strand: false,
        };
        add_read(&mut counts, &read, &opts, 100);

        // Position 0 is the anchor — should have a DEL allele.
        let alleles = &counts[0].read_alleles;
        assert_eq!(alleles.len(), 1);
        let a = alleles.values().next().unwrap();
        assert_eq!(a.r#type, AlleleType::Deletion as i32);
        // Bases = anchor "T" + deleted "GC" = "TGC" — actual ref bases,
        // NOT "TNN".
        assert_eq!(a.bases, "TGC");
    }

    #[test]
    fn deletion_of_3_after_position_2() {
        // Read 3M3D3M: ref AAA + skip 3 ref + AAA. Read seq is 6 long.
        let ref_bases = b"AAAAAAAAA";
        let mut counts = empty_counts("chr1", 100, 109, ref_bases);
        let opts = CounterOptions::default();
        let r = syn_read("r1", 100, &[('M', 3), ('D', 3), ('M', 3)], b"AAAAAA");
        add_read(&mut counts, &r, &opts, 100);
        // First 3 ref positions: ref count 1, plus one DEL allele on the 3rd
        for i in 0..3 {
            assert_eq!(counts[i].ref_supporting_read_count, 1);
        }
        // The DEL allele anchors at position 2 (offset 2).
        let alleles = &counts[2].read_alleles;
        assert_eq!(alleles.len(), 1);
        let a = alleles.values().next().unwrap();
        assert_eq!(a.r#type, AlleleType::Deletion as i32);
        assert_eq!(a.bases.len(), 4); // anchor + 3 deleted bases
        // Positions 3-5 are the deleted positions: no read evidence here
        for i in 3..6 {
            assert_eq!(counts[i].ref_supporting_read_count, 0);
            assert!(counts[i].read_alleles.is_empty());
        }
        // Positions 6-8 have ref again
        for i in 6..9 {
            assert_eq!(counts[i].ref_supporting_read_count, 1);
        }
    }

    #[test]
    fn soft_clip_skipped() {
        // 2S5M: first 2 read bases soft-clipped, then 5 matches.
        let ref_bases = b"AAAAA";
        let mut counts = empty_counts("chr1", 100, 105, ref_bases);
        let opts = CounterOptions::default();
        let r = syn_read("r1", 100, &[('S', 2), ('M', 5)], b"NNAAAAA");
        add_read(&mut counts, &r, &opts, 100);
        for c in &counts {
            assert_eq!(c.ref_supporting_read_count, 1);
        }
    }

    #[test]
    fn low_mapq_skipped() {
        let ref_bases = b"AAAA";
        let mut counts = empty_counts("chr1", 100, 104, ref_bases);
        let opts = CounterOptions {
            min_mapping_quality: 30,
            ..Default::default()
        };
        let mut r = syn_read("r1", 100, &[('M', 4)], b"AAAA");
        r.mapping_quality = 20;
        add_read(&mut counts, &r, &opts, 100);
        for c in &counts {
            assert_eq!(c.ref_supporting_read_count, 0);
        }
    }

    #[test]
    fn low_baseq_to_nonconfident() {
        let ref_bases = b"AAAA";
        let mut counts = empty_counts("chr1", 100, 104, ref_bases);
        let opts = CounterOptions {
            min_base_quality: 30,
            min_mapping_quality: 0,
        };
        let mut r = syn_read("r1", 100, &[('M', 4)], b"AAAA");
        let bq: &[u8] = &[20, 20, 20, 20];
        r.base_quality = bq;
        add_read(&mut counts, &r, &opts, 100);
        for c in &counts {
            assert_eq!(c.ref_supporting_read_count, 0);
            assert_eq!(c.ref_nonconfident_read_count, 1);
        }
    }

    /// Build a synthetic AlleleCount populated with N reads supporting
    /// each of `alts`, plus `ref_count` REF reads. Helper for the
    /// summary helpers below.
    fn synth_count(
        ref_base: &str,
        ref_count: i32,
        alts: &[(&str, i32, AlleleType, bool)], // (bases, count, type, is_low_quality)
    ) -> AlleleCount {
        let mut c = AlleleCount {
            position: Some(Position {
                reference_name: "chr1".into(),
                position: 100,
                reverse_strand: false,
            }),
            ref_base: ref_base.into(),
            ref_supporting_read_count: ref_count,
            ..Default::default()
        };
        let mut rid = 0usize;
        for (bases, n, t, lq) in alts {
            for _ in 0..*n {
                c.read_alleles.insert(
                    format!("read{}/1", rid),
                    Allele {
                        bases: (*bases).into(),
                        r#type: *t as i32,
                        count: 1,
                        is_low_quality: *lq,
                        ..Default::default()
                    },
                );
                rid += 1;
            }
        }
        c
    }

    #[test]
    fn sum_allele_counts_groups_by_type_and_bases() {
        let c = synth_count(
            "A",
            5,
            &[
                ("C", 3, AlleleType::Substitution, false),
                ("C", 1, AlleleType::Substitution, true),
                ("G", 2, AlleleType::Substitution, false),
            ],
        );
        let with_lq = sum_allele_counts(&c, true);
        // 3 unique alts (C non-lq + C lq aggregated, G) + REF synth = 3+REF
        // Actually: ("C", SUB) appears 4 times (3 + 1). G 2 times. REF 5.
        let total: i32 = with_lq.iter().map(|a| a.count).sum();
        assert_eq!(total, 4 + 2 + 5);

        let no_lq = sum_allele_counts(&c, false);
        // dropping the LQ "C" leaves 3 C + 2 G + 5 REF.
        let total: i32 = no_lq.iter().map(|a| a.count).sum();
        assert_eq!(total, 3 + 2 + 5);
    }

    #[test]
    fn sum_allele_counts_span_aggregates_across_positions() {
        let mut c1 = synth_count(
            "A",
            5,
            &[("C", 2, AlleleType::Substitution, false)],
        );
        let c2 = synth_count(
            "A",
            3,
            &[("C", 1, AlleleType::Substitution, false)],
        );
        // ensure positions differ
        c1.position.as_mut().unwrap().position = 100;
        let mut c2 = c2;
        c2.position.as_mut().unwrap().position = 101;
        let res = sum_allele_counts_span(&[c1, c2], true);
        // C is summed = 3, REF = 5+3 = 8
        let c_count: i32 = res.iter().filter(|a| a.bases == "C").map(|a| a.count).sum();
        let ref_count: i32 = res
            .iter()
            .filter(|a| a.r#type == AlleleType::Reference as i32)
            .map(|a| a.count)
            .sum();
        assert_eq!(c_count, 3);
        assert_eq!(ref_count, 8);
    }

    #[test]
    fn total_allele_counts_excludes_ref_typed_alts() {
        // track_ref_reads off → REFERENCE in read_alleles is unusual,
        // but the total should still include alt+ref_supporting only.
        let c = synth_count(
            "A",
            4,
            &[
                ("C", 2, AlleleType::Substitution, false),
                ("CC", 1, AlleleType::Insertion, false),
                // a REFERENCE-typed entry (synthetic; should be skipped):
                ("A", 3, AlleleType::Reference, false),
            ],
        );
        // Expected: 2 (C SUB) + 1 (CC INS) + 4 (ref_supporting) = 7
        assert_eq!(total_allele_counts(&c, true), 7);
    }

    #[test]
    fn total_allele_counts_lq_filter() {
        let c = synth_count(
            "A",
            0,
            &[
                ("C", 3, AlleleType::Substitution, true),
                ("G", 2, AlleleType::Substitution, false),
            ],
        );
        assert_eq!(total_allele_counts(&c, true), 5);
        assert_eq!(total_allele_counts(&c, false), 2); // drops LQ
    }

    #[test]
    fn can_bases_be_used_canonical_and_quality_logic() {
        let seq = b"ACGTNX"; // X is non-canonical
        let q = vec![30u8, 30, 30, 30, 30, 30];
        // All canonical, BQ 30, threshold 20 → usable, not low-quality.
        let (ok, lq) = can_bases_be_used(seq, &q, 0, 5, 20, false);
        assert!(ok && !lq);
        // Hits non-canonical X at offset 5.
        let (ok, _) = can_bases_be_used(seq, &q, 5, 1, 20, false);
        assert!(!ok);
        // Legacy mode: any low-BQ kills it.
        let q2 = vec![10u8, 30, 30, 30, 30, 30];
        let (ok, _) = can_bases_be_used(seq, &q2, 0, 4, 20, true);
        assert!(!ok);
        // Modern mode: averaged below threshold → flagged but usable.
        let q3 = vec![5u8, 5, 5, 30, 30, 30];
        let (ok, lq) = can_bases_be_used(seq, &q3, 0, 4, 20, false);
        assert!(ok && lq);
        // Out of bounds → not usable.
        let (ok, _) = can_bases_be_used(seq, &q, 0, 100, 20, false);
        assert!(!ok);
    }

    #[test]
    fn get_avg_base_quality_match_and_del() {
        let q = vec![30u8, 40, 50, 60];
        // Match: 4-base average of 30/40/50/60 = 45.
        assert_eq!(get_avg_base_quality(&q, 'M', 0, 4), 45);
        // Insertion 2 bases at offset 1: (40+50)/2 = 45.
        assert_eq!(get_avg_base_quality(&q, 'I', 1, 2), 45);
        // Deletion at offset 2 → BQ at offset 1 = 40.
        assert_eq!(get_avg_base_quality(&q, 'D', 2, 3), 40);
        // Deletion at offset 0 → BQ at offset 0 = 30.
        assert_eq!(get_avg_base_quality(&q, 'D', 0, 3), 30);
    }

    #[test]
    fn allele_index_binary_search() {
        let counts = empty_counts("chr1", 100, 110, b"AAAAAAAAAA");
        assert_eq!(allele_index(&counts, 100), Some(0));
        assert_eq!(allele_index(&counts, 109), Some(9));
        assert_eq!(allele_index(&counts, 105), Some(5));
        assert_eq!(allele_index(&counts, 110), None);
        assert_eq!(allele_index(&counts, 99), None);
    }

    #[test]
    fn methylation_helpers() {
        assert!((methylation_probability(128) - 128.0 / 255.0).abs() < 1e-6);
        // Disabled → always false.
        assert!(!is_methylated(Some(128), false, 0.5));
        // 128/255 ≈ 0.5019 > 0.5 → true.
        assert!(is_methylated(Some(128), true, 0.5));
        // Above 0.51 → false.
        assert!(!is_methylated(Some(128), true, 0.51));
        // Level==0 is treated as no data.
        assert!(!is_methylated(Some(0), true, 0.0));
        assert!(!is_methylated(None, true, 0.0));
    }
}
