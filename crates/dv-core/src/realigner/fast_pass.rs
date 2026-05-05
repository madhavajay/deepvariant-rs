//! Simplified port of `deepvariant/realigner/fast_pass_aligner.cc`.
//!
//! Upstream is ~1000 LOC and uses a k-mer index to skip the SSW step
//! for reads that match a haplotype perfectly (or with ≤3 mismatches).
//! This port skips that optimization and runs SSW on every read against
//! every haplotype — slower (O(R * H * L²)) but produces the same
//! ranking and final alignments.
//!
//! Use case: given a set of candidate haplotypes from the de-Bruijn
//! graph and a set of reads in a window, score each haplotype by the
//! sum of its read alignment scores and rank them. The top-K
//! haplotypes are then aligned to the reference and reads are
//! "realigned" through the haplotype→ref mapping.

use crate::realigner::ssw::{align, Alignment, ScoreParams};

/// Per-read alignment to a haplotype.
#[derive(Debug, Clone)]
pub struct ReadAlignmentToHaplotype {
    pub read_index: usize,
    /// `None` if the read could not be aligned (score == 0).
    pub alignment: Option<Alignment>,
}

/// All reads' alignments to one haplotype.
#[derive(Debug, Clone)]
pub struct HaplotypeReadsAlignment {
    pub haplotype_index: usize,
    pub haplotype_score: i32,
    pub read_alignments: Vec<ReadAlignmentToHaplotype>,
    /// Alignment of this haplotype against the reference.
    pub haplotype_to_ref: Option<Alignment>,
    pub is_reference: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct FastPassOptions {
    pub score: ScoreParams,
}

impl Default for FastPassOptions {
    fn default() -> Self {
        Self {
            score: ScoreParams::default(),
        }
    }
}

/// Align every read against every haplotype, summing per-read scores
/// into a haplotype score. Also align each haplotype to the reference.
///
/// `reads` and `haplotypes` are slices of byte sequences. `reference`
/// must be one of the haplotypes (or the closest one) for the
/// `is_reference` flag to be set correctly — we mark a haplotype as
/// reference if its bytes equal `reference`.
pub fn align_reads_to_haplotypes(
    reads: &[&[u8]],
    haplotypes: &[Vec<u8>],
    reference: &[u8],
    opts: &FastPassOptions,
) -> Vec<HaplotypeReadsAlignment> {
    let mut out = Vec::with_capacity(haplotypes.len());
    for (h_idx, hap) in haplotypes.iter().enumerate() {
        let mut total = 0i32;
        let mut per_read = Vec::with_capacity(reads.len());
        for (r_idx, read) in reads.iter().enumerate() {
            let aln = align(read, hap, opts.score);
            if let Some(a) = aln.as_ref() {
                total += a.score;
            }
            per_read.push(ReadAlignmentToHaplotype {
                read_index: r_idx,
                alignment: aln,
            });
        }
        let hap_to_ref = align(hap, reference, opts.score);
        out.push(HaplotypeReadsAlignment {
            haplotype_index: h_idx,
            haplotype_score: total,
            read_alignments: per_read,
            haplotype_to_ref: hap_to_ref,
            is_reference: hap.as_slice() == reference,
        });
    }
    // Sort by score descending so callers can grab the top K.
    out.sort_by(|a, b| b.haplotype_score.cmp(&a.haplotype_score));
    out
}

/// Map a 0-based position within a haplotype back to a 0-based position on
/// the reference, given the haplotype→reference alignment.
/// Returns `None` if the haplotype position falls outside the alignment.
pub fn haplotype_pos_to_ref_pos(
    hap_pos: usize,
    hap_to_ref: &Alignment,
    ref_pos_of_alignment_start: usize,
) -> Option<usize> {
    if hap_pos < hap_to_ref.query_begin || hap_pos >= hap_to_ref.query_end {
        return None;
    }
    let mut q_consumed = 0usize;
    let mut r_consumed = 0usize;
    let target_offset = hap_pos - hap_to_ref.query_begin;
    for (op, len) in &hap_to_ref.cigar {
        match *op {
            'M' => {
                if q_consumed + len > target_offset {
                    let inner = target_offset - q_consumed;
                    return Some(ref_pos_of_alignment_start + hap_to_ref.ref_begin + r_consumed + inner);
                }
                q_consumed += len;
                r_consumed += len;
            }
            'I' => {
                if q_consumed + len > target_offset {
                    // Falls inside an insertion: clamp to the prior ref pos.
                    return Some(ref_pos_of_alignment_start + hap_to_ref.ref_begin + r_consumed);
                }
                q_consumed += len;
            }
            'D' => {
                r_consumed += len;
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_haplotype_with_more_supporting_reads() {
        let reference: &[u8] = b"AAACCCTGGGTTT";
        let alt: &[u8] = b"AAACCCGGGGTTT";
        let haplotypes: Vec<Vec<u8>> = vec![reference.to_vec(), alt.to_vec()];

        // 3 reads support the alt, 1 supports the ref.
        let reads: Vec<&[u8]> = vec![alt, alt, alt, reference];

        let aligned = align_reads_to_haplotypes(&reads, &haplotypes, reference, &FastPassOptions::default());
        // Sort puts higher-scoring haplotype first.
        assert!(aligned[0].haplotype_score >= aligned[1].haplotype_score);
        // The alt haplotype should win because more reads support it.
        let top_idx = aligned[0].haplotype_index;
        assert_eq!(haplotypes[top_idx].as_slice(), alt);
    }

    #[test]
    fn ref_haplotype_marked_correctly() {
        let reference: &[u8] = b"ACGTACGTACGT";
        let alt: &[u8] = b"ACGTACCTACGT";
        let haplotypes = vec![reference.to_vec(), alt.to_vec()];
        let reads: Vec<&[u8]> = vec![reference];
        let aligned = align_reads_to_haplotypes(&reads, &haplotypes, reference, &FastPassOptions::default());
        let ref_aln = aligned.iter().find(|a| a.is_reference).expect("ref haplotype");
        assert_eq!(haplotypes[ref_aln.haplotype_index].as_slice(), reference);
    }

    #[test]
    fn read_alignment_skipped_when_no_overlap() {
        // Reads are entirely different from haplotype → SSW returns None.
        let haplotype = b"AAAAAAAAAAAA".to_vec();
        let read: &[u8] = b"GGGGGGGGGGGG";
        let aligned = align_reads_to_haplotypes(
            &[read],
            &[haplotype.clone()],
            &haplotype,
            &FastPassOptions::default(),
        );
        let ra = &aligned[0].read_alignments[0];
        assert!(ra.alignment.is_none());
        assert_eq!(aligned[0].haplotype_score, 0);
    }

    #[test]
    fn haplotype_pos_round_trip_for_perfect_match() {
        // ref == haplotype: hap_pos i should map to ref_pos i.
        let reference: &[u8] = b"AAACCCGGGTTTAGCT";
        let haplotypes = vec![reference.to_vec()];
        let aligned = align_reads_to_haplotypes(
            &[reference],
            &haplotypes,
            reference,
            &FastPassOptions::default(),
        );
        let hap_to_ref = aligned[0].haplotype_to_ref.as_ref().unwrap();
        for i in hap_to_ref.query_begin..hap_to_ref.query_end {
            assert_eq!(haplotype_pos_to_ref_pos(i, hap_to_ref, 0), Some(i));
        }
    }
}
