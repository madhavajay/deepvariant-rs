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
                // Deletion keyed to the previous ref position; bases =
                // ref_base + deleted ref bases.
                let anchor_ref = ref_pos - 1;
                let pos_idx = anchor_ref - region_start;
                if pos_idx >= 0 && pos_idx < counts.len() as i64 {
                    let count = &mut counts[pos_idx as usize];
                    let ref_b = count.ref_base.as_bytes()[0].to_ascii_uppercase();
                    // We don't have direct access to the deleted ref bases here
                    // (would need ref_bases buffer); store anchor only and set
                    // bases length via deletion length signal.
                    let mut bases = String::with_capacity(len_usize + 1);
                    bases.push(ref_b as char);
                    // Insert deleted-base sentinel marker: upstream stores the
                    // actual ref bases here, but for tally purposes the length
                    // suffices. We mark them with 'N' so callers can identify
                    // the deletion length without a ref lookup.
                    for _ in 0..len_usize {
                        bases.push('N');
                    }
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
}
