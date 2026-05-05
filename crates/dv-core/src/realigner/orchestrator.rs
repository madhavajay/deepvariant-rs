//! Realigner orchestrator: glues window_selector → de Bruijn assembly →
//! fast_pass alignment together.
//!
//! Upstream's realigner does substantially more bookkeeping (write
//! realigned reads back into a BAM-like stream, propagate the
//! haplotype's CIGAR into the read's CIGAR via the
//! `hap_to_ref_positions_map`, etc.). This is a foundation that
//! covers the algorithmic surface needed for candidate discovery.

use crate::realigner::debruijn::{DeBruijnGraph, DeBruijnOptions, ReadInput};
use crate::realigner::fast_pass::{
    align_reads_to_haplotypes, FastPassOptions, HaplotypeReadsAlignment,
};
use crate::realigner::ssw::{align as ssw_align, ScoreParams};
use crate::realigner::window_selector::WindowSelectorOptions;
use dv_proto::nucleus_v1::Variant;

#[derive(Debug, Clone)]
pub struct RealignerOptions {
    pub debruijn: DeBruijnOptions,
    pub window: WindowSelectorOptions,
    pub fast_pass: FastPassOptions,
    /// Top-K haplotypes to keep after fast_pass ranking.
    pub max_haplotypes: usize,
    /// Padding bases around each window for assembly context.
    pub window_padding: i64,
}

impl Default for RealignerOptions {
    fn default() -> Self {
        Self {
            debruijn: DeBruijnOptions::default(),
            window: WindowSelectorOptions::default(),
            fast_pass: FastPassOptions::default(),
            max_haplotypes: 64,
            window_padding: 30,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WindowResult {
    /// 0-based half-open ref range covered by the assembly window.
    pub start: i64,
    pub end: i64,
    /// Candidate haplotype byte sequences (sorted lexicographically).
    pub haplotypes: Vec<Vec<u8>>,
    /// Per-haplotype read alignments and scores.
    pub alignments: Vec<HaplotypeReadsAlignment>,
}

/// Sweep the realigner over a sequence of windows and merge any newly
/// discovered variants into the candidate list.
///
/// `windows` is `[(start, end), ...]` — typically the output of
/// `window_selector::windows_from_scores` translated from local indices
/// to global ref coordinates.
///
/// `existing_keys` is a set of `(start, ref, sorted_alts)` triples
/// already in the candidate set; new realigner variants matching any of
/// these are skipped to avoid duplicates.
///
/// Returns the new variants discovered. Caller is responsible for
/// merging these with the existing candidate list and re-running
/// downstream stages (pileup rendering, etc.).
pub fn discover_variants_from_realigner(
    contig: &str,
    windows: &[(i64, i64)],
    fetch_ref: impl Fn(i64, i64) -> Option<Vec<u8>>,
    fetch_reads: impl Fn(i64, i64) -> Vec<crate::realigner::debruijn::ReadInput<'static>>,
    existing_keys: &std::collections::HashSet<(i64, String, Vec<String>)>,
    opts: &RealignerOptions,
) -> Vec<Variant> {
    let mut new_variants = Vec::new();
    for &(start, end) in windows {
        let ref_bases = match fetch_ref(start, end) {
            Some(b) => b,
            None => continue,
        };
        let reads = fetch_reads(start, end);
        let result = match realign_window(start, end, &ref_bases, &reads, opts) {
            Some(r) => r,
            None => continue,
        };
        for hap in &result.haplotypes {
            if hap.as_slice() == ref_bases {
                continue;
            }
            for v in variants_from_haplotype(contig, start, &ref_bases, hap) {
                let mut alts = v.alternate_bases.clone();
                alts.sort();
                let key = (v.start, v.reference_bases.clone(), alts);
                if existing_keys.contains(&key) {
                    continue;
                }
                new_variants.push(v);
            }
        }
    }
    // Deduplicate within new_variants too.
    let mut seen: std::collections::HashSet<(i64, String, Vec<String>)> =
        std::collections::HashSet::new();
    new_variants.retain(|v| {
        let mut alts = v.alternate_bases.clone();
        alts.sort();
        seen.insert((v.start, v.reference_bases.clone(), alts))
    });
    new_variants
}

/// Walk the haplotype→reference alignment and emit a `Variant` for each
/// I/D/X CIGAR run in the alignment.
///
/// `contig` and `ref_window_start` are used to set the reference name and
/// 0-based variant.start. `ref_bases` covers `[ref_window_start,
/// ref_window_start + ref_bases.len())`.
pub fn variants_from_haplotype(
    contig: &str,
    ref_window_start: i64,
    ref_bases: &[u8],
    haplotype: &[u8],
) -> Vec<Variant> {
    let aln = match ssw_align(haplotype, ref_bases, ScoreParams::default()) {
        Some(a) => a,
        None => return Vec::new(),
    };
    let mut variants = Vec::new();
    let mut ref_pos = aln.ref_begin as i64;
    let mut hap_pos = aln.query_begin as i64;
    for (op, len) in &aln.cigar {
        let len_i = *len as i64;
        match *op {
            'M' => {
                // Walk and emit SNVs at any mismatched bases.
                for i in 0..len_i {
                    let r = ref_bases[(ref_pos + i) as usize];
                    let h = haplotype[(hap_pos + i) as usize];
                    if r != h {
                        let global_start = ref_window_start + ref_pos + i;
                        variants.push(Variant {
                            reference_name: contig.to_string(),
                            start: global_start,
                            end: global_start + 1,
                            reference_bases: (r as char).to_string(),
                            alternate_bases: vec![(h as char).to_string()],
                            ..Default::default()
                        });
                    }
                }
                ref_pos += len_i;
                hap_pos += len_i;
            }
            'I' => {
                // Insertion in haplotype relative to ref. Emit at anchor pos.
                let anchor_pos = (ref_pos - 1).max(0);
                let anchor = ref_bases.get(anchor_pos as usize).copied().unwrap_or(b'N');
                let mut bases = vec![anchor];
                bases.extend_from_slice(
                    &haplotype[hap_pos as usize..(hap_pos + len_i) as usize],
                );
                let global_start = ref_window_start + anchor_pos;
                variants.push(Variant {
                    reference_name: contig.to_string(),
                    start: global_start,
                    end: global_start + 1,
                    reference_bases: (anchor as char).to_string(),
                    alternate_bases: vec![String::from_utf8_lossy(&bases).to_string()],
                    ..Default::default()
                });
                hap_pos += len_i;
            }
            'D' => {
                // Deletion in haplotype relative to ref. Emit del with
                // anchor + deleted bases as REF, anchor as ALT.
                let anchor_pos = (ref_pos - 1).max(0);
                let anchor = ref_bases.get(anchor_pos as usize).copied().unwrap_or(b'N');
                let mut ref_str = vec![anchor];
                ref_str.extend_from_slice(
                    &ref_bases[ref_pos as usize..(ref_pos + len_i) as usize],
                );
                let global_start = ref_window_start + anchor_pos;
                variants.push(Variant {
                    reference_name: contig.to_string(),
                    start: global_start,
                    end: global_start + len_i + 1,
                    reference_bases: String::from_utf8_lossy(&ref_str).to_string(),
                    alternate_bases: vec![(anchor as char).to_string()],
                    ..Default::default()
                });
                ref_pos += len_i;
            }
            _ => {}
        }
    }
    variants
}

/// Run the realigner over a single window.
/// `ref_bases` covers `[start, end)` exactly. Reads' aligned_sequence
/// must be uppercase ACGT; quality must match in length.
pub fn realign_window(
    start: i64,
    end: i64,
    ref_bases: &[u8],
    reads: &[ReadInput<'_>],
    opts: &RealignerOptions,
) -> Option<WindowResult> {
    debug_assert_eq!(
        ref_bases.len() as i64,
        end - start,
        "ref_bases must cover [start,end)"
    );
    let graph = DeBruijnGraph::build(ref_bases, reads, &opts.debruijn)?;
    let mut haplotypes = graph.candidate_haplotypes();
    if haplotypes.is_empty() {
        return None;
    }
    haplotypes.truncate(opts.max_haplotypes);

    // Convert reads into &[u8] slices for fast_pass.
    let read_slices: Vec<&[u8]> = reads.iter().map(|r| r.aligned_sequence).collect();
    let alignments =
        align_reads_to_haplotypes(&read_slices, &haplotypes, ref_bases, &opts.fast_pass);

    Some(WindowResult {
        start,
        end,
        haplotypes,
        alignments,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read<'a>(seq: &'a [u8], qual: &'a [u8]) -> ReadInput<'a> {
        ReadInput {
            aligned_sequence: seq,
            aligned_quality: qual,
            mapping_quality: 60,
        }
    }

    #[test]
    fn discovers_alt_haplotype_in_window() {
        let reference = b"AAACCCTGGGTTT";
        let alt = b"AAACCCGGGGTTT";
        let qual = vec![40u8; alt.len()];
        let reads_input: Vec<ReadInput<'_>> = (0..10)
            .map(|_| read(alt.as_slice(), &qual))
            .collect();

        let mut opts = RealignerOptions::default();
        opts.debruijn.min_k = 4;
        opts.debruijn.max_k = 8;
        opts.debruijn.min_edge_weight = 1;
        opts.debruijn.min_base_quality = 0;
        opts.debruijn.min_mapq = 0;

        let res = realign_window(100, 113, reference, &reads_input, &opts).expect("window");
        // Both ref and alt should be among candidate haplotypes.
        assert!(res.haplotypes.iter().any(|h| h.as_slice() == reference));
        assert!(res.haplotypes.iter().any(|h| h.as_slice() == alt));
        // The alt haplotype should outscore ref (more reads support it).
        assert!(res.alignments[0].haplotype_score >= res.alignments[1].haplotype_score);
        let top = &res.alignments[0];
        assert_eq!(res.haplotypes[top.haplotype_index].as_slice(), alt);
    }

    #[test]
    fn variants_from_haplotype_snv() {
        // hap differs from ref by a single SNV at offset 5.
        let reference = b"AAACCCTTTGGGAAA";
        let hap = b"AAACCATTTGGGAAA"; // C→A at offset 5 of ref
        let vs = variants_from_haplotype("chr1", 100, reference, hap);
        assert_eq!(vs.len(), 1);
        assert_eq!(vs[0].start, 105);
        assert_eq!(vs[0].reference_bases, "C");
        assert_eq!(vs[0].alternate_bases, vec!["A"]);
    }

    #[test]
    fn variants_from_haplotype_insertion() {
        let reference = b"AAACCCTTTGGGAAA";
        let hap = b"AAACCCXXTTTGGGAAA"; // 2bp insert after pos 5
        let vs = variants_from_haplotype("chr1", 100, reference, hap);
        assert_eq!(vs.len(), 1);
        // INS anchored at pos 105 (offset 5)
        assert_eq!(vs[0].start, 105);
        assert_eq!(vs[0].reference_bases, "C");
        assert_eq!(vs[0].alternate_bases, vec!["CXX"]);
    }

    #[test]
    fn variants_from_haplotype_deletion() {
        let reference = b"AAACCCTTTGGGAAA";
        let hap = b"AAACCCGGGAAA"; // 3bp deletion of TTT
        let vs = variants_from_haplotype("chr1", 100, reference, hap);
        assert_eq!(vs.len(), 1);
        assert_eq!(vs[0].start, 105);
        assert_eq!(vs[0].reference_bases.len(), 4); // anchor + 3 deleted
        assert_eq!(vs[0].alternate_bases.len(), 1);
        assert_eq!(vs[0].alternate_bases[0].len(), 1);
    }

    #[test]
    fn variants_from_identical_haplotype_is_empty() {
        let reference = b"AAACCCTTTGGGAAA";
        let vs = variants_from_haplotype("chr1", 100, reference, reference);
        assert!(vs.is_empty());
    }

    #[test]
    fn no_alt_evidence_returns_only_reference() {
        let reference = b"AAACCCGGGTTTACGT";
        let qual = vec![40u8; reference.len()];
        let reads_input: Vec<ReadInput<'_>> = (0..5)
            .map(|_| read(reference.as_slice(), &qual))
            .collect();
        let mut opts = RealignerOptions::default();
        opts.debruijn.min_k = 4;
        opts.debruijn.max_k = 10;
        opts.debruijn.min_edge_weight = 1;
        opts.debruijn.min_base_quality = 0;
        opts.debruijn.min_mapq = 0;

        let res = realign_window(0, reference.len() as i64, reference, &reads_input, &opts)
            .expect("window");
        assert_eq!(res.haplotypes.len(), 1);
        assert_eq!(res.haplotypes[0].as_slice(), reference);
    }
}
