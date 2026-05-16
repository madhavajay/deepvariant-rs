//! Full make-examples + postprocess pipeline, in-memory, single
//! threaded, for the browser. Faithful port of `dv-cli`'s
//! `pipeline_cmd` orchestration but reading BAM from a byte buffer
//! (noodles over a `Cursor`, no `std::fs`, no CRAM C deps) so it
//! builds for `wasm32`.
//!
//! Flow (browser):
//!   bam_bytes + fasta_bytes + region
//!     → [Rust/wasm] parse BAM, allele count, candidates, realigner
//!       (≤8 haplotypes/window), pileup render  → tf.Example bytes
//!     → [JS] onnxruntime-web (WebGPU) per example → 3-class probs
//!     → [Rust/wasm] postprocess (group_cvos dup-alt fix) → VCF text

use std::collections::HashMap;
use std::io::Cursor;

use dv_core::allelecounter::{add_read, empty_counts, AlignedRead, CounterOptions};
use dv_core::make_examples::build_example;
use dv_core::pileup_image::{
    channels::ChannelKind,
    layout::{render, PileupRead, VariantContext},
    options::PileupOptions,
};
use dv_core::postprocess::{self, PostprocessOptions};
use dv_core::realigner::{
    debruijn::{DeBruijnGraph, DeBruijnOptions, ReadInput},
    orchestrator::variants_from_haplotype,
    window_selector::{variant_reads_candidates, windows_from_scores, WindowSelectorOptions},
};
use dv_core::variant_calling::{candidates_from_counts, VariantCallerOptions};
use dv_proto::dv::{call_variants_output::AltAlleleIndices, CallVariantsOutput};
use dv_proto::nucleus_v1::{ContigInfo, Variant};

#[cfg(feature = "wasm-bindgen")]
use wasm_bindgen::prelude::*;

/// Mirrors `dv-cli`'s MAX_HAPLOTYPES_PER_WINDOW (C++ fork's
/// `_MAX_HAPLOTYPES = 8`).
const MAX_HAPLOTYPES_PER_WINDOW: usize = 8;

#[derive(Clone)]
struct OwnedRead {
    ref_start: i64,
    ref_end: i64,
    cigar: Vec<(char, i64)>,
    seq: Vec<u8>,
    bq: Vec<u8>,
    mq: u8,
    is_rev: bool,
    frag: i32,
    name: String,
    mate: i32,
    hp: u8,
}

/// 0-based half-open region, matching `dv-cli`'s internal convention
/// (a `--region chr:A-B` literal is 1-based inclusive; start-1, end).
struct Region {
    reference_name: String,
    start: i64,
    end: i64,
}

fn parse_region(s: &str) -> Result<Region, String> {
    let (rn, rng) = s
        .split_once(':')
        .ok_or_else(|| format!("bad region {s:?} (want chr:start-end)"))?;
    let (a, b) = rng
        .split_once('-')
        .ok_or_else(|| format!("bad region {s:?} (want chr:start-end)"))?;
    let a: i64 = a.replace(',', "").parse().map_err(|_| "bad region start")?;
    let b: i64 = b.replace(',', "").parse().map_err(|_| "bad region end")?;
    Ok(Region {
        reference_name: rn.to_string(),
        start: a - 1,
        end: b,
    })
}

/// Minimal FASTA: `>name [desc]\n` then sequence lines. Uppercased.
fn parse_fasta(bytes: &[u8]) -> HashMap<String, Vec<u8>> {
    let mut map: HashMap<String, Vec<u8>> = HashMap::new();
    let mut cur: Option<String> = None;
    for line in bytes.split(|&c| c == b'\n') {
        if line.first() == Some(&b'>') {
            let name = line[1..]
                .split(|&c| c == b' ' || c == b'\t' || c == b'\r')
                .next()
                .unwrap_or(&[]);
            let name = String::from_utf8_lossy(name).into_owned();
            map.entry(name.clone()).or_default();
            cur = Some(name);
        } else if let Some(c) = &cur {
            let v = map.get_mut(c).unwrap();
            for &b in line {
                if b != b'\r' {
                    v.push(b.to_ascii_uppercase());
                }
            }
        }
    }
    map
}

/// Reference bases for `[start,end)` on `contig`, N-padded if the
/// range runs past the parsed sequence (matches dv-io's behaviour
/// closely enough for rendering).
fn fetch_range(
    fasta: &HashMap<String, Vec<u8>>,
    contig: &str,
    start: i64,
    end: i64,
) -> Option<Vec<u8>> {
    if start < 0 || end <= start {
        return None;
    }
    let seq = fasta.get(contig)?;
    let mut out = Vec::with_capacity((end - start) as usize);
    for p in start..end {
        out.push(*seq.get(p as usize).unwrap_or(&b'N'));
    }
    Some(out)
}

fn cigar_op_to_char(op: noodles::sam::alignment::record::cigar::op::Kind) -> char {
    use noodles::sam::alignment::record::cigar::op::Kind as K;
    match op {
        K::Match => 'M',
        K::Insertion => 'I',
        K::Deletion => 'D',
        K::Skip => 'N',
        K::SoftClip => 'S',
        K::HardClip => 'H',
        K::Pad => 'P',
        K::SequenceMatch => '=',
        K::SequenceMismatch => 'X',
    }
}

/// Parse BAM bytes → reads overlapping `[region.start, region.end)`.
/// Same record filtering as `dv-cli::process_shard`.
fn parse_bam(bam: &[u8], region: &Region) -> Result<Vec<OwnedRead>, String> {
    use noodles::sam::alignment::record::QualityScores;
    use noodles::sam::alignment::Record;

    let mut reader = noodles::bam::io::Reader::new(Cursor::new(bam));
    let header = reader.read_header().map_err(|e| format!("bam header: {e}"))?;
    let mut reads = Vec::new();

    for result in reader.records() {
        let r = result.map_err(|e| format!("bam record: {e}"))?;
        // Resolve the record's reference name; skip other contigs.
        let rname = match r.reference_sequence(&header) {
            Some(Ok((name, _))) => String::from_utf8_lossy(name.as_ref()).into_owned(),
            _ => continue,
        };
        if rname != region.reference_name {
            continue;
        }
        let Some(Ok(start)) = r.alignment_start() else {
            continue;
        };
        let start_0based = usize::from(start) as i64 - 1;
        let flags = r.flags();
        if flags.is_secondary()
            || flags.is_supplementary()
            || flags.is_unmapped()
            || flags.is_duplicate()
            || flags.is_qc_fail()
        {
            continue;
        }
        let mq_check = r.mapping_quality().map(|q| q.get()).unwrap_or(255);
        if mq_check < 10 {
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
        let read_len_on_ref: i64 = cigar_owned
            .iter()
            .filter(|(op, _)| matches!(op, 'M' | '=' | 'X' | 'D' | 'N'))
            .map(|(_, l)| *l)
            .sum();
        if start_0based + read_len_on_ref < region.start || start_0based >= region.end {
            continue;
        }
        let seq: Vec<u8> = r.sequence().iter().map(|b| b.to_ascii_uppercase()).collect();
        let bq: Vec<u8> = r.quality_scores().iter().map(|q| q.unwrap_or(0)).collect();
        let is_rev = flags.is_reverse_complemented();
        let frag = r.template_length();
        let name = r
            .name()
            .map(|n| String::from_utf8_lossy(n.as_ref()).into_owned())
            .unwrap_or_default();
        let mate = if flags.is_first_segment() { 1 } else { 2 };
        let hp = {
            use noodles::sam::alignment::record::data::field::Tag;
            match r
                .data()
                .get(&Tag::new(b'H', b'P'))
                .and_then(|res| res.ok())
                .and_then(|v| v.as_int())
            {
                Some(1) => 1u8,
                Some(2) => 2u8,
                _ => 0u8,
            }
        };
        reads.push(OwnedRead {
            ref_start: start_0based,
            ref_end: start_0based + read_len_on_ref,
            cigar: cigar_owned,
            seq,
            bq,
            mq: mq_check,
            is_rev,
            frag,
            name,
            mate,
            hp,
        });
    }
    reads.sort_by_key(|r| r.ref_start);
    Ok(reads)
}

/// One rendered example, ready for inference.
pub struct RenderedExample {
    /// tf.Example proto bytes (variant + alt_allele_indices + image),
    /// identical to a make-examples shard record.
    pub example: Vec<u8>,
    pub variant: Variant,
    pub alt_indices: Vec<i32>,
}

/// Native-side core: BAM bytes + FASTA bytes + region → examples.
/// Pure compute (no I/O); reused by both the wasm surface and tests.
pub fn pipeline_examples(
    bam: &[u8],
    fasta: &[u8],
    region_str: &str,
) -> Result<Vec<RenderedExample>, String> {
    let region = parse_region(region_str)?;
    let fasta = parse_fasta(fasta);
    let ref_bases = fetch_range(&fasta, &region.reference_name, region.start, region.end)
        .ok_or_else(|| {
            format!(
                "reference {}:{}-{} not in FASTA",
                region.reference_name, region.start, region.end
            )
        })?;

    let owned = parse_bam(bam, &region)?;
    let read_starts: Vec<i64> = owned.iter().map(|r| r.ref_start).collect();
    let max_read_span = owned
        .iter()
        .map(|r| r.ref_end - r.ref_start)
        .max()
        .unwrap_or(0);

    // Allele counts over the region.
    let mut counts = empty_counts(
        &region.reference_name,
        region.start,
        region.end,
        &ref_bases,
    );
    let counter_opts = CounterOptions::default();
    for r in &owned {
        let aligned = AlignedRead {
            name: &r.name,
            mate_number: r.mate,
            ref_start: r.ref_start,
            cigar: &r.cigar,
            seq: &r.seq,
            base_quality: &r.bq,
            mapping_quality: r.mq,
            is_reverse_strand: r.is_rev,
        };
        add_read(&mut counts, &aligned, &counter_opts, region.start);
    }

    let mut cands = candidates_from_counts(&counts, &VariantCallerOptions::default());

    // Realigner-driven candidate expansion (single-threaded port of
    // pipeline_cmd's block; ≤8 haplotypes/window).
    {
        let scores = variant_reads_candidates(&counts, &WindowSelectorOptions::default());
        let raw_windows = windows_from_scores(&scores, 3);
        let mut padded: Vec<(i64, i64)> = raw_windows
            .iter()
            .map(|(s, e)| {
                (
                    region.start + *s as i64 - 50,
                    region.start + *e as i64 + 50,
                )
            })
            .collect();
        padded.sort_by_key(|w| w.0);
        let mut merged: Vec<(i64, i64)> = Vec::new();
        for w in padded {
            match merged.last_mut() {
                Some(last) if w.0 <= last.1 => last.1 = last.1.max(w.1),
                _ => merged.push(w),
            }
        }
        let dbg_opts = DeBruijnOptions::default();
        let mut new_cands: Vec<Variant> = Vec::new();
        for (ws, we) in &merged {
            let Some(win_ref) =
                fetch_range(&fasta, &region.reference_name, *ws, *we)
                    .filter(|b| b.len() == (we - ws) as usize)
            else {
                continue;
            };
            let lo = read_starts.partition_point(|&s| s + max_read_span <= *ws);
            let hi = read_starts.partition_point(|&s| s < *we);
            let win_reads: Vec<&OwnedRead> =
                owned[lo..hi].iter().filter(|r| r.ref_end > *ws).collect();
            let read_inputs: Vec<ReadInput<'_>> = win_reads
                .iter()
                .map(|r| ReadInput {
                    aligned_sequence: &r.seq,
                    aligned_quality: &r.bq,
                    mapping_quality: r.mq,
                })
                .collect();
            let Some(graph) = DeBruijnGraph::build(&win_ref, &read_inputs, &dbg_opts) else {
                continue;
            };
            for hap in graph
                .candidate_haplotypes()
                .into_iter()
                .take(MAX_HAPLOTYPES_PER_WINDOW)
            {
                if hap.as_slice() == win_ref.as_slice() {
                    continue;
                }
                for nv in
                    variants_from_haplotype(&region.reference_name, *ws, &win_ref, &hap)
                {
                    new_cands.push(nv);
                }
            }
        }
        let mut existing: std::collections::HashSet<(i64, String, Vec<String>)> =
            std::collections::HashSet::new();
        for v in &cands {
            let mut alts = v.alternate_bases.clone();
            alts.sort();
            existing.insert((v.start, v.reference_bases.clone(), alts));
        }
        for nv in new_cands {
            let mut alts = nv.alternate_bases.clone();
            alts.sort();
            let key = (nv.start, nv.reference_bases.clone(), alts);
            if existing.insert(key) {
                cands.push(nv);
            }
        }
        cands.sort_by(|a, b| {
            (a.reference_name.as_str(), a.start, a.end).cmp(&(
                b.reference_name.as_str(),
                b.start,
                b.end,
            ))
        });
    }

    // Pass 1 + 2: render each candidate's pileup image.
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
    let width = opts.width;
    let height = opts.height;
    let center = (width / 2) as i64;

    let mut out = Vec::new();
    for v in &cands {
        let win_start = v.start - center;
        let win_end = win_start + width as i64;
        let Some(img_ref) =
            fetch_range(&fasta, &v.reference_name, win_start, win_end)
                .filter(|b| b.len() == width)
        else {
            continue;
        };
        let variant_pos = v.start;

        const READ_OVERLAP_BUFFER_BP: i64 = 5;
        let query_start = v.start - READ_OVERLAP_BUFFER_BP;
        let query_end = v.end + READ_OVERLAP_BUFFER_BP;
        let lo = read_starts.partition_point(|&s| s + max_read_span <= query_start);
        let hi = read_starts.partition_point(|&s| s < query_end);
        let window_reads: Vec<&OwnedRead> =
            owned[lo..hi].iter().filter(|r| r.ref_end > query_start).collect();

        let alt_set: std::collections::HashSet<u8> = v
            .alternate_bases
            .iter()
            .filter_map(|a| a.as_bytes().first().copied())
            .collect();
        let supports = |r: &OwnedRead| -> bool {
            let mut ref_pos = r.ref_start;
            let mut read_pos = 0usize;
            for &(op, len) in &r.cigar {
                let len_us = len as usize;
                match op {
                    'M' | '=' | 'X' => {
                        if ref_pos <= variant_pos && variant_pos < ref_pos + len {
                            let off = (variant_pos - ref_pos) as usize;
                            if read_pos + off < r.seq.len() {
                                return alt_set.contains(&r.seq[read_pos + off]);
                            }
                            return false;
                        }
                        ref_pos += len;
                        read_pos += len_us;
                    }
                    'I' | 'S' => read_pos += len_us,
                    'D' | 'N' => ref_pos += len,
                    _ => {}
                }
            }
            false
        };
        let pileup_reads: Vec<PileupRead<'_>> = window_reads
            .iter()
            .map(|r| PileupRead {
                ref_start: r.ref_start,
                cigar: &r.cigar,
                seq: &r.seq,
                base_quality: &r.bq,
                mapping_quality: r.mq,
                is_reverse_strand: r.is_rev,
                fragment_length: r.frag,
                supports_variant: supports(r),
                hp_tag: r.hp,
                fragment_name: &r.name,
                read_number: r.mate - 1,
            })
            .collect();
        let ctx = Some(VariantContext {
            variant_pos,
            min_base_quality_at_call: 10,
        });
        let img = render(
            win_start,
            width,
            height,
            5,
            &img_ref,
            &pileup_reads,
            &kinds,
            &opts,
            ctx,
            42,
        );
        // Match native pass2's `variant_with_call`: ensure a call
        // exists so downstream `set_model_id` can stamp MID (realigner
        // -origin variants otherwise have no call → MID missing).
        let mut vc = v.clone();
        if vc.calls.is_empty() {
            vc.calls.push(Default::default());
        }
        // Native pipeline_cmd emits ONE example per alt allele
        // (`alt_indices = vec![i]`), all sharing this variant's image
        // (the pileup/supports use the full alt set, so the render is
        // identical across alts). postprocess groups the per-alt CVOs
        // and `merge_predictions_multi` rebuilds the multi-allelic
        // call + full PL. Emitting a single all-alts example instead
        // breaks that merge (truncated PL, wrong alt pruning).
        for i in 0..v.alternate_bases.len() {
            let alt_indices = vec![i as i32];
            let example = build_example(&vc, &alt_indices, &img);
            out.push(RenderedExample {
                example,
                variant: vc.clone(),
                alt_indices,
            });
        }
    }
    Ok(out)
}

/// Stitch (example, predictions) pairs into a VCF. `probs_flat` is
/// `examples.len() * 3` row-major f64. Reuses dv-core postprocess
/// (group_cvos dup-alt fix, merge, genotype, filter).
pub fn examples_to_vcf(
    examples: &[(Variant, Vec<i32>)],
    probs_flat: &[f64],
    contigs: &[(String, i64)],
    sample: &str,
) -> Result<String, String> {
    if probs_flat.len() != examples.len() * 3 {
        return Err(format!(
            "probs len {} != examples {} * 3",
            probs_flat.len(),
            examples.len()
        ));
    }
    let cvos: Vec<CallVariantsOutput> = examples
        .iter()
        .enumerate()
        .map(|(i, (v, aai))| CallVariantsOutput {
            variant: Some(v.clone()),
            alt_allele_indices: Some(AltAlleleIndices {
                indices: aai.clone(),
            }),
            genotype_probabilities: probs_flat[i * 3..i * 3 + 3].to_vec(),
            debug_info: None,
        })
        .collect();
    let variants =
        postprocess::process_cvos_into_variants(cvos, sample, &PostprocessOptions::default());

    let contig_infos: Vec<ContigInfo> = contigs
        .iter()
        .map(|(name, len)| ContigInfo {
            name: name.clone(),
            n_bases: *len,
            ..Default::default()
        })
        .collect();
    const FORMAT_KEYS: &[&str] = &["GT", "GQ", "DP", "AD", "VAF", "MID", "PL"];
    let mut buf: Vec<u8> = Vec::new();
    dv_core::vcf::write_header(&mut buf, &contig_infos, &[sample])
        .map_err(|e| format!("vcf header: {e}"))?;
    for v in &variants {
        dv_core::vcf::write_variant_line(&mut buf, v, FORMAT_KEYS)
            .map_err(|e| format!("vcf line: {e}"))?;
    }
    String::from_utf8(buf).map_err(|e| format!("vcf utf8: {e}"))
}

// ---- wasm-bindgen surface ----

/// Opaque list of rendered examples (avoids returning Vec<Vec<u8>>
/// across the wasm boundary). JS iterates `len()`/`example(i)`,
/// runs ORT, then calls `examples_to_vcf_js`.
#[cfg(feature = "wasm-bindgen")]
#[wasm_bindgen]
pub struct Pipeline {
    examples: Vec<RenderedExample>,
}

#[cfg(feature = "wasm-bindgen")]
#[wasm_bindgen]
impl Pipeline {
    /// Run the make-examples half on an uploaded BAM + reference.
    #[wasm_bindgen(constructor)]
    pub fn new(bam: &[u8], fasta: &[u8], region: &str) -> Result<Pipeline, JsError> {
        let examples = pipeline_examples(bam, fasta, region).map_err(|e| JsError::new(&e))?;
        Ok(Pipeline { examples })
    }

    pub fn len(&self) -> usize {
        self.examples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// tf.Example bytes for example `i` (feed to `example_to_model_input`).
    pub fn example(&self, i: usize) -> Vec<u8> {
        self.examples[i].example.clone()
    }

    /// Build the final VCF text from flat `[n*3]` f32 predictions
    /// (row-major, matching `example(i)` order).
    pub fn to_vcf(
        &self,
        probs_flat: &[f32],
        contigs_json: &str,
        sample: &str,
    ) -> Result<String, JsError> {
        let pairs: Vec<(Variant, Vec<i32>)> = self
            .examples
            .iter()
            .map(|e| (e.variant.clone(), e.alt_indices.clone()))
            .collect();
        let probs: Vec<f64> = probs_flat.iter().map(|&p| p as f64).collect();
        // contigs_json: [["chr20",64444167], ...]
        let contigs: Vec<(String, i64)> = parse_contigs_json(contigs_json)
            .map_err(|e| JsError::new(&e))?;
        examples_to_vcf(&pairs, &probs, &contigs, sample).map_err(|e| JsError::new(&e))
    }
}

#[cfg(feature = "wasm-bindgen")]
fn parse_contigs_json(s: &str) -> Result<Vec<(String, i64)>, String> {
    // Tiny hand parser for [["name",len],...] to avoid a serde dep.
    let mut out = Vec::new();
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'"' {
            let j = s[i + 1..].find('"').ok_or("bad contigs json")? + i + 1;
            let name = s[i + 1..j].to_string();
            let rest = &s[j + 1..];
            let comma = rest.find(',').ok_or("bad contigs json")?;
            let after = &rest[comma + 1..];
            let end = after
                .find(|c: char| c == ']')
                .ok_or("bad contigs json")?;
            let len: i64 = after[..end]
                .trim()
                .parse()
                .map_err(|_| "bad contig length")?;
            out.push((name, len));
            i = j + 1 + comma + 1 + end;
        } else {
            i += 1;
        }
    }
    Ok(out)
}
