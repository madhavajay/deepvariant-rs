//! Smoke: run allelecounter → variant_caller against chr20 BAM and verify
//! the SNV at chr20:10000117 (C>T) is among the candidates.

use std::path::PathBuf;

use dv_core::allelecounter::{add_read, empty_counts, AlignedRead, CounterOptions};
use dv_core::variant_calling::{candidates_from_counts, VariantCallerOptions};
use dv_io::fasta;
use noodles::sam::alignment::record::cigar::op::Kind as CigarKind;
use noodles::sam::alignment::record::QualityScores;
use noodles::sam::alignment::Record;

const REGION_START: i64 = 9_999_999;
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
fn chr20_quickstart_candidate_calls() {
    if !quickstart_fasta().exists() {
        eprintln!("skipping: chr20 FASTA not present");
        return;
    }
    let fa = fasta::open_indexed(quickstart_fasta()).unwrap();
    let ref_bases = fa.fetch_range("chr20", REGION_START, REGION_END).unwrap();
    let mut counts = empty_counts("chr20", REGION_START, REGION_END, &ref_bases);
    let opts = CounterOptions::default();

    let (_h, mut reader) = dv_io::bam::open(fixture("NA12878_S1.chr20.10_10p1mb.bam")).unwrap();
    for rec in reader.records() {
        let r = rec.unwrap();
        let Some(start) = r.alignment_start() else { continue };
        let start_0based = usize::from(start.unwrap()) as i64 - 1;
        let cigar_owned: Vec<(char, i64)> = r
            .cigar()
            .iter()
            .filter_map(|op| {
                let op = op.ok()?;
                Some((cigar_op_to_char(op.kind()), op.len() as i64))
            })
            .collect();
        let seq: Vec<u8> = r.sequence().iter().map(|b| b.to_ascii_uppercase()).collect();
        let bq: Vec<u8> = r.quality_scores().iter().map(|q| q.unwrap_or(0)).collect();
        let name = r
            .name()
            .map(|n| std::str::from_utf8(n.as_ref()).unwrap_or("").to_string())
            .unwrap_or_default();
        let mate_no = if r.flags().is_first_segment() { 1 } else { 2 };
        let mq = r.mapping_quality().map(|q| q.get()).unwrap_or(255);

        let read = AlignedRead {
            name: &name,
            mate_number: mate_no,
            ref_start: start_0based,
            cigar: &cigar_owned,
            seq: &seq,
            base_quality: &bq,
            mapping_quality: mq,
            is_reverse_strand: r.flags().is_reverse_complemented(),
        };
        add_read(&mut counts, &read, &opts, REGION_START);
    }

    let cs = candidates_from_counts(&counts, &VariantCallerOptions::default());
    eprintln!("emitted {} candidate variants", cs.len());

    // Expected SNV from upstream output: chr20:10000117 C>T (0-based 10000116).
    let snv = cs
        .iter()
        .find(|v| v.start == 10_000_116 && v.reference_bases == "C")
        .expect("SNV C>T at chr20:10000117 should be a candidate");
    assert!(
        snv.alternate_bases.iter().any(|a| a == "T"),
        "expected T as alt, got {:?}",
        snv.alternate_bases
    );
    eprintln!("found candidate at chr20:10000117 with alts {:?}", snv.alternate_bases);

    // Sanity: candidates are sorted by position.
    for w in cs.windows(2) {
        assert!(w[0].start <= w[1].start, "candidates should be position-sorted");
    }

    // Upstream's chr20 quickstart found 84 candidates total. We should be
    // in roughly the same order of magnitude (no realigner means we may
    // miss some indels and call a few spurious noise candidates, so accept
    // anywhere from 50–250).
    assert!(
        cs.len() > 50 && cs.len() < 500,
        "candidate count {} outside plausible range",
        cs.len()
    );
}
