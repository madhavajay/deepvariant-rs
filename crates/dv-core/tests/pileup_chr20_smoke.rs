//! End-to-end smoke: render a pileup image for the SNV at chr20:10000117
//! using the chr20 quickstart BAM + FASTA. Verifies dimensions and that the
//! variant column shows alt evidence in the read rows.

use std::path::PathBuf;

use dv_core::pileup_image::{
    channels::ChannelKind,
    layout::{render, PileupRead, VariantContext},
    options::PileupOptions,
};
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
fn render_pileup_image_at_known_snv() {
    if !quickstart_fasta().exists() {
        eprintln!("skipping: chr20 FASTA missing");
        return;
    }
    // Variant: chr20:10000117 C>T (1-based) → 0-based pos = 10000116.
    let variant_pos: i64 = 10_000_116;
    let width = 221usize;
    let height = 100usize;
    let center = width / 2;
    let win_start = variant_pos - center as i64;
    let win_end = win_start + width as i64;

    let fa = fasta::open_indexed(quickstart_fasta()).unwrap();
    let ref_bases = fa.fetch_range("chr20", win_start, win_end).unwrap();
    assert_eq!(ref_bases.len(), width);
    assert_eq!(ref_bases[center], b'C', "REF at variant center should be C");

    // Pull reads overlapping the window.
    let (_h, mut reader) = dv_io::bam::open(fixture("NA12878_S1.chr20.10_10p1mb.bam")).unwrap();

    struct OwnedRead {
        ref_start: i64,
        cigar: Vec<(char, i64)>,
        seq: Vec<u8>,
        bq: Vec<u8>,
        mq: u8,
        is_rev: bool,
        frag: i32,
        supports_variant: bool,
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
                Some((cigar_op_to_char(op.kind()), op.len() as i64))
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
        let is_rev = r.flags().is_reverse_complemented();
        let frag = r.template_length() as i32;

        // Determine if this read supports the alt T at variant_pos.
        // Walk the CIGAR until we land on variant_pos and read the base.
        let mut ref_pos = start_0based;
        let mut read_pos = 0usize;
        let mut supports = false;
        for &(op, len) in &cigar_owned {
            let len_us = len as usize;
            match op {
                'M' | '=' | 'X' => {
                    if ref_pos <= variant_pos && variant_pos < ref_pos + len {
                        let off = (variant_pos - ref_pos) as usize;
                        if read_pos + off < seq.len() {
                            supports = seq[read_pos + off] == b'T';
                        }
                        break;
                    }
                    ref_pos += len;
                    read_pos += len_us;
                }
                'I' | 'S' => read_pos += len_us,
                'D' | 'N' => ref_pos += len,
                _ => {}
            }
        }
        owned.push(OwnedRead {
            ref_start: start_0based,
            cigar: cigar_owned,
            seq,
            bq,
            mq,
            is_rev,
            frag,
            supports_variant: supports,
        });
    }
    // Sort: reads that cover the variant center first (so they get painted
    // when we cap at 75 read rows). This mirrors how upstream sorts reads
    // for a candidate-centered pileup image.
    owned.sort_by_key(|r| {
        let read_len_on_ref: i64 = r
            .cigar
            .iter()
            .filter(|(op, _)| matches!(op, 'M' | '=' | 'X' | 'D' | 'N'))
            .map(|(_, l)| *l)
            .sum();
        let read_end = r.ref_start + read_len_on_ref;
        let covers_center = r.ref_start <= variant_pos && variant_pos < read_end;
        // Sort key: covers-center reads first, then by ref_start.
        (!covers_center, r.ref_start)
    });

    let pileup_reads: Vec<PileupRead<'_>> = owned
        .iter()
        .map(|r| PileupRead {
            ref_start: r.ref_start,
            cigar: &r.cigar,
            seq: &r.seq,
            base_quality: &r.bq,
            mapping_quality: r.mq,
            is_reverse_strand: r.is_rev,
            fragment_length: r.frag,
            supports_variant: r.supports_variant,
            hp_tag: 0,
            fragment_name: "smoke",
            read_number: 0,
        })
        .collect();

    let opts = PileupOptions::default();
    let kinds = [
        ChannelKind::ReadBase,
        ChannelKind::BaseQuality,
        ChannelKind::MappingQuality,
        ChannelKind::Strand,
        ChannelKind::ReadSupportsVariant,
        ChannelKind::BaseDiffersFromRef,
        ChannelKind::InsertSize,
    ];
    let img = render(
        win_start,
        width,
        height,
        5,
        &ref_bases,
        &pileup_reads,
        &kinds,
        &opts,
        Some(VariantContext { variant_pos, min_base_quality_at_call: 10 }),
        42,
    );
    assert_eq!(img.len(), height * width * kinds.len());

    // At the center column on a read row, the read_base channel should show
    // either the C (ref) or T (alt) color — never zero (since the variant has
    // ~55x coverage we should get plenty of read rows filled).
    let mut nonzero = 0usize;
    for row in 5..height {
        let pixel = img[(row * width + center) * kinds.len() + 0]; // read_base channel
        if pixel != 0 {
            nonzero += 1;
        }
    }
    eprintln!("variant column has {nonzero} non-zero read rows of {}", height - 5);
    assert!(nonzero >= 20, "expected ≥20 read rows at variant col, got {nonzero}");
}
