//! End-to-end smoke: feed the chr20 quickstart BAM into the AlleleCounter
//! over the same 10kb region used by the upstream pipeline, and assert
//! that the output looks plausible (per-position read coverage in the
//! expected range; non-empty alt evidence somewhere in the region).

use std::path::PathBuf;

use dv_core::allelecounter::{add_read, empty_counts, total_count, AlignedRead, CounterOptions};
use dv_io::fasta;
use noodles::sam::alignment::record::cigar::op::Kind as CigarKind;
use noodles::sam::alignment::record::QualityScores;
use noodles::sam::alignment::Record;

const REGION_START: i64 = 9_999_999; // 0-based; matches upstream quickstart 1-based 10000000-10010000
const REGION_END: i64 = 10_010_000;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/quickstart_chr20")
        .join(name)
}

fn quickstart_fasta() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../quickstart/input/ucsc.hg19.chr20.unittest.fasta")
}

fn cigar_op_to_char(op: CigarKind) -> char {
    use CigarKind::*;
    match op {
        Match => 'M',
        Insertion => 'I',
        Deletion => 'D',
        Skip => 'N',
        SoftClip => 'S',
        HardClip => 'H',
        Pad => 'P',
        SequenceMatch => '=',
        SequenceMismatch => 'X',
    }
}

#[test]
fn chr20_quickstart_allele_counts_plausible() {
    if !quickstart_fasta().exists() {
        eprintln!("skipping: chr20 quickstart FASTA missing");
        return;
    }

    let fa = fasta::open_indexed(quickstart_fasta()).unwrap();
    let ref_bases = fa.fetch_range("chr20", REGION_START, REGION_END).unwrap();
    assert_eq!(ref_bases.len(), (REGION_END - REGION_START) as usize);

    let mut counts = empty_counts("chr20", REGION_START, REGION_END, &ref_bases);
    let opts = CounterOptions::default();

    let (header, mut reader) =
        dv_io::bam::open(fixture("NA12878_S1.chr20.10_10p1mb.bam")).unwrap();
    let mut consumed_reads = 0usize;
    for rec in reader.records() {
        let r = rec.unwrap();
        // Only chr20 in this slice, so we don't filter by reference name.
        let Some(start) = r.alignment_start() else {
            continue;
        };
        let start_0based = usize::from(start.unwrap()) as i64 - 1;
        // Skip reads that don't overlap our region.
        let read_len_on_ref: i64 = r
            .cigar()
            .iter()
            .filter_map(|op| {
                let op = op.ok()?;
                let len = op.len() as i64;
                Some(match op.kind() {
                    CigarKind::Match
                    | CigarKind::SequenceMatch
                    | CigarKind::SequenceMismatch
                    | CigarKind::Deletion
                    | CigarKind::Skip => len,
                    _ => 0,
                })
            })
            .sum();
        if start_0based + read_len_on_ref < REGION_START || start_0based >= REGION_END {
            continue;
        }

        let cigar_owned: Vec<(char, i64)> = r
            .cigar()
            .iter()
            .filter_map(|op| {
                let op = op.ok()?;
                Some((cigar_op_to_char(op.kind()), op.len() as i64))
            })
            .collect();
        let seq: Vec<u8> = r.sequence().iter().map(|b| b.to_ascii_uppercase()).collect();
        let bq: Vec<u8> = r
            .quality_scores()
            .iter()
            .map(|q| q.unwrap_or(0))
            .collect();
        let name = r
            .name()
            .map(|n| std::str::from_utf8(n.as_ref()).unwrap_or("").to_string())
            .unwrap_or_default();
        let mate_no = if r.flags().is_first_segment() { 1 } else { 2 };
        let mq = r.mapping_quality().map(|q| q.get()).unwrap_or(255);
        let is_rev = r.flags().is_reverse_complemented();

        let read = AlignedRead {
            name: &name,
            mate_number: mate_no,
            ref_start: start_0based,
            cigar: &cigar_owned,
            seq: &seq,
            base_quality: &bq,
            mapping_quality: mq,
            is_reverse_strand: is_rev,
        };
        add_read(&mut counts, &read, &opts, REGION_START);
        consumed_reads += 1;
    }
    eprintln!("processed {consumed_reads} reads spanning the chr20 region");

    // Spot-check coverage: at least the middle of the region should have some
    // depth (chr20 NA12878 quickstart is well-covered).
    let middle = counts.len() / 2;
    let mid_total = total_count(&counts[middle]);
    eprintln!("center-of-region total reads = {mid_total}");
    assert!(mid_total > 5, "expected nontrivial coverage at region center, got {mid_total}");

    // At chr20 1-based 10000117 (REF=C ALT=T was a real call), there should
    // be SUB allele evidence.
    let snv_idx = (10_000_116 - REGION_START) as usize;
    let snv_count = &counts[snv_idx];
    let alt_evidence = snv_count.read_alleles.len();
    eprintln!(
        "chr20:10000117 ref-supporting={}, alt-evidence reads={}",
        snv_count.ref_supporting_read_count, alt_evidence
    );
    assert!(alt_evidence > 0, "expected SUB evidence at known SNV site");

    // suppress unused header warning
    let _ = header;
}
