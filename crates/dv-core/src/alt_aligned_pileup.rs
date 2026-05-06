//! Alt-aligned pileup helpers — port of
//! `deepvariant/alt_aligned_pileup_lib.cc` (~317 LOC).
//!
//! Used by the `--alt_aligned_pileup={diff,base}_channels` modes,
//! which re-render the pileup with reads realigned to each alt
//! haplotype and stack two extra channels per alt:
//!   * #9, #10  (DIFF_CHANNELS_ALTERNATE_ALLELE_{1,2})
//!   * #20, #21 (BASE_CHANNELS_ALTERNATE_ALLELE_{1,2})
//!
//! This module ports the pieces that don't need a full FASTA reader:
//!   * `trim_cigar` — pure-function CIGAR trimmer
//!   * `trim_read`  — slice a read down to a sub-range
//!   * `calculate_alignment_region` — clip the pileup window to the
//!     contig bounds
//!   * `cigar_ref_length`, `cigar_read_length` — small helpers
//!
//! `RealignReadsToHaplotype` (the FastPassAligner-backed entry point)
//! is intentionally left as a thin wrapper over our existing
//! `realigner::fast_pass_aligner` and lives at the call site (see
//! `dv-cli` integration); the trimming primitives below are what the
//! upstream test suite parameterizes most heavily, so coverage parity
//! is concentrated here.

/// CIGAR ops that consume the reference (advance ref_pos).
#[inline]
fn ref_advancing(op: char) -> bool {
    matches!(op, 'M' | '=' | 'X' | 'D' | 'N')
}

/// CIGAR ops that consume the read (advance read_pos).
#[inline]
fn read_advancing(op: char) -> bool {
    matches!(op, 'M' | '=' | 'X' | 'I' | 'S')
}

/// Total reference span of a CIGAR string (counts M/=/X/D/N ops).
pub fn cigar_ref_length(cigar: &[(char, i64)]) -> i64 {
    cigar
        .iter()
        .filter(|(op, _)| ref_advancing(*op))
        .map(|(_, len)| *len)
        .sum()
}

/// Total read span of a CIGAR string (counts M/=/X/I/S ops).
pub fn cigar_read_length(cigar: &[(char, i64)]) -> i64 {
    cigar
        .iter()
        .filter(|(op, _)| read_advancing(*op))
        .map(|(_, len)| *len)
        .sum()
}

/// Output of `trim_cigar`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrimmedCigar {
    pub cigar: Vec<(char, i64)>,
    /// Read-relative position where the trimmed alignment starts.
    pub read_start: i64,
    /// Read length after trimming.
    pub new_read_length: i64,
}

/// Trim a CIGAR string against a sub-range of the reference. The
/// function "consumes" `ref_start` reference bases off the front and
/// keeps the next `ref_length` reference bases. Insertions that fall
/// inside the consumed prefix are discarded; insertions inside the kept
/// window are preserved.
///
/// Mirrors upstream `TrimCigar` exactly.
pub fn trim_cigar(cigar: &[(char, i64)], ref_start: i64, ref_length: i64) -> TrimmedCigar {
    let mut trim_remaining = ref_start;
    let mut ref_to_cover_remaining = ref_length;
    let mut read_start = 0i64;
    let mut new_read_length = 0i64;
    let mut new_cigar: Vec<(char, i64)> = Vec::new();

    for (op, op_len) in cigar.iter().copied() {
        let mut c_len = op_len;
        let advances_ref = ref_advancing(op);
        let advances_read = read_advancing(op);
        let mut ref_step: i64 = if advances_ref { c_len } else { 0 };

        // Phase 1: consume the leading trim window.
        if trim_remaining > 0 {
            if ref_step <= trim_remaining {
                trim_remaining -= ref_step;
                if advances_read {
                    read_start += c_len;
                }
                continue;
            } else {
                ref_step -= trim_remaining;
                if advances_read {
                    read_start += trim_remaining;
                }
                c_len = ref_step;
                trim_remaining = 0;
            }
        }

        // Phase 2: cover the kept ref window.
        if trim_remaining == 0 {
            if ref_step <= ref_to_cover_remaining {
                new_cigar.push((op, c_len));
                ref_to_cover_remaining -= ref_step;
                if advances_read {
                    new_read_length += c_len;
                }
            } else {
                c_len = ref_to_cover_remaining;
                new_cigar.push((op, c_len));
                if advances_read {
                    new_read_length += c_len;
                }
                ref_to_cover_remaining = 0;
                break;
            }
        }
    }

    TrimmedCigar {
        cigar: new_cigar,
        read_start,
        new_read_length,
    }
}

/// Genomic region used by `trim_read` and `calculate_alignment_region`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Range {
    pub reference_name: String,
    pub start: i64,
    pub end: i64,
}

/// Read inputs/outputs to `trim_read`. We carry just enough to faithfully
/// reproduce the trim — full BAM/proto round-tripping isn't needed by
/// the alt-aligned pileup builder which feeds these straight back into
/// the renderer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrimmableRead {
    pub fragment_name: String,
    pub read_number: i32,
    pub ref_name: String,
    pub ref_start: i64,
    pub cigar: Vec<(char, i64)>,
    pub aligned_sequence: Vec<u8>,
    pub aligned_quality: Vec<u8>,
    pub mapping_quality: u8,
}

/// Slice a read down to its overlap with `region`. Returns a new
/// `TrimmableRead` with adjusted ref_start (clipped to the region's
/// start when the original began earlier) and trimmed
/// sequence/quality/cigar.
///
/// Panics if the resulting `ref_length` would be ≤ 0 — caller is
/// expected to filter.
pub fn trim_read(read: &TrimmableRead, region: &Range) -> TrimmableRead {
    let read_start = read.ref_start;
    let trim_left = (region.start - read_start).max(0);
    let ref_length = region.end - region.start.max(read_start);
    assert!(ref_length > 0, "trim_read: ref_length must be > 0");

    let trimmed = trim_cigar(&read.cigar, trim_left, ref_length);
    let TrimmedCigar {
        cigar: new_cigar,
        read_start: read_trim,
        new_read_length,
    } = trimmed;
    let read_trim = read_trim as usize;
    let new_read_length = new_read_length as usize;

    assert!(read_trim + new_read_length <= read.aligned_sequence.len());
    assert!(read_trim + new_read_length <= read.aligned_quality.len());

    let new_seq = read.aligned_sequence[read_trim..read_trim + new_read_length].to_vec();
    let new_qual = read.aligned_quality[read_trim..read_trim + new_read_length].to_vec();

    let new_ref_start = if trim_left != 0 {
        region.start
    } else {
        read.ref_start
    };

    TrimmableRead {
        fragment_name: read.fragment_name.clone(),
        read_number: read.read_number,
        ref_name: read.ref_name.clone(),
        ref_start: new_ref_start,
        cigar: new_cigar,
        aligned_sequence: new_seq,
        aligned_quality: new_qual,
        mapping_quality: read.mapping_quality,
    }
}

/// Trim a vector of reads to the region; drop any whose remaining
/// CIGAR-on-reference length is shorter than `min_overlap`. Records
/// each kept read's *original* alignment start position so the caller
/// can map back if needed (mirrors upstream's
/// `original_alignment_positions` out-vector).
pub fn trim_reads(
    reads: &[&TrimmableRead],
    region: &Range,
    min_overlap: i64,
) -> (Vec<TrimmableRead>, Vec<i64>) {
    let mut out = Vec::new();
    let mut original_positions = Vec::new();
    for r in reads {
        // Skip reads with no positive overlap.
        let overlap_start = region.start.max(r.ref_start);
        if region.end <= overlap_start {
            continue;
        }
        let trimmed = trim_read(r, region);
        if cigar_ref_length(&trimmed.cigar) < min_overlap || trimmed.aligned_sequence.is_empty() {
            continue;
        }
        original_positions.push(r.ref_start);
        out.push(trimmed);
    }
    (out, original_positions)
}

/// Compute a `Range` of the same width as the pileup image, centered
/// on the variant, clipped to the contig's `[0, n_bases]` bounds.
pub fn calculate_alignment_region(
    variant_ref_name: &str,
    variant_start: i64,
    variant_ref_bases_len: i64,
    half_width: i64,
    contig_n_bases: i64,
) -> Range {
    let ref_end = variant_start + variant_ref_bases_len;
    Range {
        reference_name: variant_ref_name.to_string(),
        start: (variant_start - half_width).max(0),
        end: (ref_end + half_width).min(contig_n_bases),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cigar(parts: &[(char, i64)]) -> Vec<(char, i64)> {
        parts.to_vec()
    }

    /// Mirrors upstream `TrimCigarTests` parameterized cases.
    #[test]
    fn trim_cigar_with_ins() {
        let c = make_cigar(&[('M', 20), ('I', 5), ('M', 10)]);
        let t = trim_cigar(&c, 10, 20);
        assert_eq!(t.cigar, make_cigar(&[('M', 10), ('I', 5), ('M', 10)]));
        assert_eq!(t.read_start, 10);
        assert_eq!(t.new_read_length, 25);
    }

    #[test]
    fn trim_cigar_with_del() {
        let c = make_cigar(&[('M', 20), ('D', 5), ('M', 10)]);
        let t = trim_cigar(&c, 10, 20);
        assert_eq!(t.cigar, make_cigar(&[('M', 10), ('D', 5), ('M', 5)]));
        assert_eq!(t.read_start, 10);
        assert_eq!(t.new_read_length, 15);
    }

    #[test]
    fn trim_cigar_ref_start_inside_ins() {
        let c = make_cigar(&[('M', 20), ('I', 5), ('M', 20)]);
        let t = trim_cigar(&c, 22, 10);
        assert_eq!(t.cigar, make_cigar(&[('M', 10)]));
        assert_eq!(t.read_start, 27);
        assert_eq!(t.new_read_length, 10);
    }

    #[test]
    fn trim_cigar_ref_start_inside_del() {
        let c = make_cigar(&[('M', 20), ('D', 5), ('M', 20)]);
        let t = trim_cigar(&c, 22, 10);
        assert_eq!(t.cigar, make_cigar(&[('D', 3), ('M', 7)]));
        assert_eq!(t.read_start, 20);
        assert_eq!(t.new_read_length, 7);
    }

    #[test]
    fn trim_cigar_ref_start_past_read_end() {
        let c = make_cigar(&[('M', 20), ('I', 5), ('M', 10)]);
        let t = trim_cigar(&c, 50, 20);
        assert!(t.cigar.is_empty());
        // upstream returns read_start=35 (= 20M + 5I + 10M ≥ 35)
        assert_eq!(t.read_start, 35);
        assert_eq!(t.new_read_length, 0);
    }

    #[test]
    fn trim_cigar_ref_length_past_read_end() {
        let c = make_cigar(&[('M', 20), ('I', 5), ('M', 10)]);
        let t = trim_cigar(&c, 10, 40);
        assert_eq!(t.cigar, make_cigar(&[('M', 10), ('I', 5), ('M', 10)]));
        assert_eq!(t.read_start, 10);
        assert_eq!(t.new_read_length, 25);
    }

    fn make_read(
        ref_start: i64,
        bases: &[u8],
        cigar: &[(char, i64)],
        quality: &[u8],
    ) -> TrimmableRead {
        TrimmableRead {
            fragment_name: "test_read".into(),
            read_number: 0,
            ref_name: "chr1".into(),
            ref_start,
            cigar: cigar.to_vec(),
            aligned_sequence: bases.to_vec(),
            aligned_quality: quality.to_vec(),
            mapping_quality: 60,
        }
    }

    /// Mirrors upstream `TrimReadTests` first parameterized case (22M
    /// trimmed to 5M).
    #[test]
    fn trim_read_22m_inner_window() {
        let r = make_read(
            10,
            b"ACGTACGTAAAAAAGTGTGATC",
            &[('M', 22)],
            &(1..=22u8).collect::<Vec<_>>(),
        );
        let region = Range {
            reference_name: "chr1".into(),
            start: 15,
            end: 20,
        };
        let t = trim_read(&r, &region);
        assert_eq!(t.aligned_sequence, b"CGTAA".to_vec());
        assert_eq!(t.cigar, vec![('M', 5)]);
        assert_eq!(t.aligned_quality, vec![6, 7, 8, 9, 10]);
        assert_eq!(t.ref_start, 15);
    }

    #[test]
    fn trim_read_with_inner_insertion() {
        let r = make_read(
            10,
            b"ACGTACGTAAAAAAGTGTGATC",
            &[('M', 2), ('I', 3), ('M', 17)],
            &(1..=22u8).collect::<Vec<_>>(),
        );
        let region = Range {
            reference_name: "chr1".into(),
            start: 15,
            end: 20,
        };
        let t = trim_read(&r, &region);
        assert_eq!(t.aligned_sequence, b"AAAAA".to_vec());
        assert_eq!(t.cigar, vec![('M', 5)]);
        assert_eq!(t.aligned_quality, vec![9, 10, 11, 12, 13]);
        assert_eq!(t.ref_start, 15);
    }

    #[test]
    fn trim_read_with_inner_deletion() {
        let r = make_read(
            10,
            b"ACGTACGTAAAAAAGTGTGATC",
            &[('M', 2), ('D', 3), ('M', 20)],
            &(1..=22u8).collect::<Vec<_>>(),
        );
        let region = Range {
            reference_name: "chr1".into(),
            start: 15,
            end: 20,
        };
        let t = trim_read(&r, &region);
        assert_eq!(t.aligned_sequence, b"GTACG".to_vec());
        assert_eq!(t.cigar, vec![('M', 5)]);
        assert_eq!(t.aligned_quality, vec![3, 4, 5, 6, 7]);
        assert_eq!(t.ref_start, 15);
    }

    #[test]
    fn trim_read_region_starts_before_read() {
        let r = make_read(
            10,
            b"ACGTACGTAAAAAAGTGTGATC",
            &[('M', 22)],
            &(1..=22u8).collect::<Vec<_>>(),
        );
        let region = Range {
            reference_name: "chr1".into(),
            start: 8,
            end: 13,
        };
        let t = trim_read(&r, &region);
        assert_eq!(t.aligned_sequence, b"ACG".to_vec());
        assert_eq!(t.cigar, vec![('M', 3)]);
        assert_eq!(t.aligned_quality, vec![1, 2, 3]);
        assert_eq!(t.ref_start, 10); // unchanged, since trim_left==0
    }

    #[test]
    fn trim_read_full_overlap_keeps_everything() {
        let q = (1..=22u8).collect::<Vec<_>>();
        let r = make_read(10, b"ACGTACGTAAAAAAGTGTGATC", &[('M', 22)], &q);
        let region = Range {
            reference_name: "chr1".into(),
            start: 10,
            end: 32,
        };
        let t = trim_read(&r, &region);
        assert_eq!(t.aligned_sequence, b"ACGTACGTAAAAAAGTGTGATC".to_vec());
        assert_eq!(t.cigar, vec![('M', 22)]);
        assert_eq!(t.aligned_quality, q);
        assert_eq!(t.ref_start, 10);
    }

    /// Mirrors upstream `CalculateAlignmentRegionTests`.
    #[test]
    fn calculate_alignment_region_centered() {
        let r = calculate_alignment_region("chr1", 11, 1, 10, 43);
        assert_eq!(r, Range { reference_name: "chr1".into(), start: 1, end: 22 });
    }

    #[test]
    fn calculate_alignment_region_left_clip() {
        let r = calculate_alignment_region("chr1", 5, 1, 10, 43);
        assert_eq!(r, Range { reference_name: "chr1".into(), start: 0, end: 16 });
    }

    #[test]
    fn calculate_alignment_region_right_clip() {
        let r = calculate_alignment_region("chr1", 40, 1, 10, 43);
        assert_eq!(r, Range { reference_name: "chr1".into(), start: 30, end: 43 });
    }

    #[test]
    fn calculate_alignment_region_full_clip() {
        let r = calculate_alignment_region("chr1", 20, 1, 100, 43);
        assert_eq!(r, Range { reference_name: "chr1".into(), start: 0, end: 43 });
    }

    #[test]
    fn calculate_alignment_region_clip_to_contig_end() {
        let r = calculate_alignment_region("chr1", 40, 1, 20, 43);
        assert_eq!(r, Range { reference_name: "chr1".into(), start: 20, end: 43 });
    }

    #[test]
    fn cigar_lengths_helpers() {
        let c = vec![('M', 5), ('I', 3), ('D', 2), ('M', 4), ('S', 6)];
        assert_eq!(cigar_ref_length(&c), 5 + 2 + 4); // M+D+M
        assert_eq!(cigar_read_length(&c), 5 + 3 + 4 + 6); // M+I+M+S
    }

    #[test]
    fn trim_reads_drops_short_overlaps() {
        let q = (1..=22u8).collect::<Vec<_>>();
        let r1 = make_read(10, b"ACGTACGTAAAAAAGTGTGATC", &[('M', 22)], &q);
        let r2 = make_read(10, b"ACG", &[('M', 3)], &[1, 2, 3]);
        let region = Range {
            reference_name: "chr1".into(),
            start: 0,
            end: 14,
        };
        // r1 keeps 4 ref bases (M=4) → below min_overlap=15 → dropped.
        // r2 keeps 3 (M=3) → also dropped.
        let (kept, _) = trim_reads(&[&r1, &r2], &region, 15);
        assert!(kept.is_empty());
        let (kept, _) = trim_reads(&[&r1, &r2], &region, 3);
        // Both kept now.
        assert_eq!(kept.len(), 2);
    }
}
