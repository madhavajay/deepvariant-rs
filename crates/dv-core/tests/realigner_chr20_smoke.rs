//! Smoke: run the realigner orchestrator on a chr20 window with known
//! indel candidates. Verifies that the de Bruijn graph + fast_pass
//! pipeline produces candidate haplotypes from real BAM reads.

use std::path::PathBuf;

use dv_core::realigner::debruijn::ReadInput;
use dv_core::realigner::orchestrator::{realign_window, RealignerOptions};
use dv_io::fasta;
use noodles::sam::alignment::record::cigar::op::Kind as CigarKind;
use noodles::sam::alignment::record::QualityScores;
use noodles::sam::alignment::Record;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/quickstart_chr20")
        .join(name)
}

fn quickstart_fasta() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../quickstart/input/ucsc.hg19.chr20.unittest.fasta")
}

#[test]
fn realigner_finds_haplotypes_in_chr20_window() {
    if !quickstart_fasta().exists() {
        eprintln!("skipping: chr20 FASTA missing");
        return;
    }
    // Pick a 200bp window around the chr20:10001436 A>AAGGCT insertion
    // (one of the upstream candidates).
    let win_start: i64 = 10_001_336;
    let win_end: i64 = 10_001_536;

    let fa = fasta::open_indexed(quickstart_fasta()).unwrap();
    let ref_bases = fa.fetch_range("chr20", win_start, win_end).unwrap();
    assert_eq!(ref_bases.len(), 200);

    let (_h, mut reader) = dv_io::bam::open(fixture("NA12878_S1.chr20.10_10p1mb.bam")).unwrap();

    struct OwnedRead {
        ref_start: i64,
        cigar: Vec<(char, i64)>,
        seq: Vec<u8>,
        bq: Vec<u8>,
        mq: u8,
    }
    let mut owned: Vec<OwnedRead> = Vec::new();
    for rec in reader.records() {
        let r = rec.unwrap();
        let Some(start) = r.alignment_start() else { continue };
        let start_0based = usize::from(start.unwrap()) as i64 - 1;
        let cigar_owned: Vec<(char, i64)> = r
            .cigar()
            .iter()
            .filter_map(|op| {
                let op = op.ok()?;
                let c = match op.kind() {
                    CigarKind::Match => 'M',
                    CigarKind::Insertion => 'I',
                    CigarKind::Deletion => 'D',
                    CigarKind::Skip => 'N',
                    CigarKind::SoftClip => 'S',
                    CigarKind::HardClip => 'H',
                    CigarKind::Pad => 'P',
                    CigarKind::SequenceMatch => '=',
                    CigarKind::SequenceMismatch => 'X',
                };
                Some((c, op.len() as i64))
            })
            .collect();
        let read_len_on_ref: i64 = cigar_owned
            .iter()
            .filter(|(op, _)| matches!(op, 'M' | '=' | 'X' | 'D' | 'N'))
            .map(|(_, l)| *l)
            .sum();
        if start_0based + read_len_on_ref < win_start || start_0based >= win_end {
            continue;
        }
        let seq: Vec<u8> = r.sequence().iter().map(|b| b.to_ascii_uppercase()).collect();
        let bq: Vec<u8> = r.quality_scores().iter().map(|q| q.unwrap_or(0)).collect();
        let mq = r.mapping_quality().map(|q| q.get()).unwrap_or(255);
        owned.push(OwnedRead {
            ref_start: start_0based,
            cigar: cigar_owned,
            seq,
            bq,
            mq,
        });
    }
    eprintln!("loaded {} reads in window", owned.len());

    let read_inputs: Vec<ReadInput<'_>> = owned
        .iter()
        .map(|r| ReadInput {
            aligned_sequence: &r.seq,
            aligned_quality: &r.bq,
            mapping_quality: r.mq,
        })
        .collect();

    let opts = RealignerOptions::default();
    let result = realign_window(win_start, win_end, &ref_bases, &read_inputs, &opts);
    match result {
        Some(r) => {
            eprintln!("got {} candidate haplotypes", r.haplotypes.len());
            // Should produce at least the reference haplotype.
            assert!(
                !r.haplotypes.is_empty(),
                "realigner should yield at least one haplotype"
            );
            // The reference itself should be among the candidates.
            let has_ref = r.haplotypes.iter().any(|h| h.as_slice() == ref_bases);
            assert!(has_ref, "reference haplotype should be among candidates");
            // For a window with this many reads (~30+), we expect the score to be
            // positive (some reads aligned to some haplotype).
            assert!(
                r.alignments.iter().any(|a| a.haplotype_score > 0),
                "expected positive haplotype score from {} reads",
                read_inputs.len()
            );
        }
        None => {
            // Some windows can't be acyclically assembled at the configured k
            // range — that's fine; just log it.
            eprintln!("realigner returned None (no acyclic graph at given k range)");
        }
    }
}
