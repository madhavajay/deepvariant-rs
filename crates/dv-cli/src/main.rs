use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use prost::Message;

use dv_core::postprocess::{self, PostprocessOptions};
use dv_core::vcf;
use dv_infer::InferenceBackend;
use dv_io::tfrecord;
use dv_proto::dv::{call_variants_output::AltAlleleIndices, CallVariantsOutput};
use dv_proto::nucleus_v1::{ContigInfo, ListValue, Value, Variant};
use dv_proto::tf::{feature::Kind as FeatureKind, Example};

const PIXEL_BYTES: usize = 100 * 221 * 7;
const MODEL_ID: &str = "deepvariant";

#[derive(Parser)]
#[command(name = "dv", version, about = "DeepVariant in Rust")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run the model on TFRecord pileup-image examples.
    CallVariants {
        /// Path to a make_examples TFRecord shard (`*.tfrecord(.gz)`).
        #[arg(long)]
        examples: PathBuf,
        /// Path to the SavedModel directory (e.g. `models/wgs/`).
        #[arg(long)]
        checkpoint: PathBuf,
        /// Path to write the CallVariantsOutput TFRecord shard.
        #[arg(long)]
        output: PathBuf,
        /// Inference batch size.
        #[arg(long, default_value_t = 32)]
        batch_size: usize,
    },
    /// Build pileup-image examples from BAM/FASTA (SNV-only foundation).
    MakeExamples {
        /// Input BAM file.
        #[arg(long)]
        reads: PathBuf,
        /// Reference FASTA (with `.fai` index).
        #[arg(long)]
        ref_fasta: PathBuf,
        /// Region literal like `chr20:10,000,000-10,010,000` (1-based inclusive).
        #[arg(long)]
        region: String,
        /// Output TFRecord shard (`.tfrecord.gz` recommended).
        #[arg(long)]
        examples: PathBuf,
        /// Sample name for the output column.
        #[arg(long, default_value = "SAMPLE")]
        sample_name: String,
        /// Optional ONNX small-model checkpoint. When set, biallelic SNV
        /// candidates are first run through the small model; only those
        /// failing the GQ threshold fall through to the big-model image
        /// pipeline.
        #[arg(long)]
        small_model: Option<PathBuf>,
        /// CVO output for small-model accepts. Required when
        /// `--small-model` is set.
        #[arg(long)]
        small_model_cvo: Option<PathBuf>,
    },
    /// Stitch CallVariantsOutput shards into a VCF (and optional gVCF).
    PostprocessVariants {
        /// CallVariantsOutput TFRecord shard from the large model.
        #[arg(long)]
        cvo: PathBuf,
        /// Optional CallVariantsOutput TFRecord shard from the small-model
        /// fast path.
        #[arg(long)]
        small_model_cvo: Option<PathBuf>,
        /// Optional gVCF non-variant block TFRecord shard (from
        /// `make_examples --gvcf`).
        #[arg(long)]
        nonvariant_site_tfrecord: Option<PathBuf>,
        /// Output VCF (.vcf or .vcf.gz).
        #[arg(long)]
        output_vcf: PathBuf,
        /// Output gVCF (.g.vcf or .g.vcf.gz). Requires `--nonvariant-site-tfrecord`.
        #[arg(long)]
        output_gvcf: Option<PathBuf>,
        /// Sample name for the output column.
        #[arg(long, default_value = "SAMPLE")]
        sample_name: String,
        /// Reference contigs (name:length pairs). For chr20 quickstart use
        /// `--contig chr20:63025520`.
        #[arg(long = "contig")]
        contigs: Vec<String>,
        /// Reference FASTA — required for gVCF if non-variant blocks may be
        /// right-truncated by overlapping variants.
        #[arg(long)]
        ref_fasta: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::CallVariants {
            examples,
            checkpoint,
            output,
            batch_size,
        } => call_variants(&examples, &checkpoint, &output, batch_size),
        Cmd::MakeExamples {
            reads,
            ref_fasta,
            region,
            examples,
            sample_name,
            small_model,
            small_model_cvo,
        } => make_examples_cmd(
            &reads,
            &ref_fasta,
            &region,
            &examples,
            &sample_name,
            small_model.as_deref(),
            small_model_cvo.as_deref(),
        ),
        Cmd::PostprocessVariants {
            cvo,
            small_model_cvo,
            nonvariant_site_tfrecord,
            output_vcf,
            output_gvcf,
            sample_name,
            contigs,
            ref_fasta,
        } => postprocess_variants(
            &cvo,
            small_model_cvo.as_deref(),
            nonvariant_site_tfrecord.as_deref(),
            &output_vcf,
            output_gvcf.as_deref(),
            &sample_name,
            &contigs,
            ref_fasta.as_deref(),
        ),
    }
}

const POSTPROCESS_FORMAT_KEYS: &[&str] =
    &["GT", "GQ", "DP", "AD", "VAF", "MID", "PL"];

fn parse_contigs(specs: &[String]) -> Result<Vec<ContigInfo>> {
    let mut out = Vec::with_capacity(specs.len());
    for (i, spec) in specs.iter().enumerate() {
        let (name, len) = spec.split_once(':').context("--contig must be NAME:LENGTH")?;
        out.push(ContigInfo {
            name: name.to_string(),
            description: String::new(),
            n_bases: len.parse().context("contig length")?,
            pos_in_fasta: i as i32,
            extra: std::collections::HashMap::new(),
        });
    }
    Ok(out)
}

/// One read held by the make-examples flow as plain owned data, so it
/// can be re-borrowed across multiple per-candidate iterations without
/// re-parsing the BAM.
struct OwnedRead {
    ref_start: i64,
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

fn make_examples_cmd(
    reads_path: &std::path::Path,
    ref_path: &std::path::Path,
    region_literal: &str,
    examples_path: &std::path::Path,
    sample_name: &str,
    small_model_path: Option<&std::path::Path>,
    small_model_cvo_path: Option<&std::path::Path>,
) -> Result<()> {
    use dv_core::allelecounter::{add_read, empty_counts, AlignedRead, CounterOptions};
    use dv_core::make_examples::build_example;
    use dv_core::nucleus::ranges;
    use dv_core::pileup_image::{
        channels::ChannelKind,
        layout::{render, PileupRead, VariantContext},
        options::PileupOptions,
    };
    use dv_core::variant_calling::{candidates_from_counts, VariantCallerOptions};
    use noodles::sam::alignment::record::cigar::op::Kind as CigarKind;
    use noodles::sam::alignment::record::QualityScores;
    #[allow(unused_imports)]
    use noodles::sam::alignment::Record;

    let region = ranges::parse_literal(region_literal).map_err(|e| anyhow::anyhow!(e))?;
    tracing::info!(?reads_path, ?ref_path, ?examples_path, region=?region, "make_examples");

    let fa = dv_io::fasta::open_indexed(ref_path).context("open FASTA")?;
    let ref_bases = fa
        .fetch_range(&region.reference_name, region.start, region.end)
        .ok_or_else(|| anyhow::anyhow!("FASTA region missing"))?;
    let ref_len = ref_bases.len() as i64;
    anyhow::ensure!(ref_len == region.end - region.start);

    let mut counts = empty_counts(&region.reference_name, region.start, region.end, &ref_bases);
    let counter_opts = CounterOptions::default();

    fn cigar_op_to_char(op: CigarKind) -> char {
        match op {
            CigarKind::Match => 'M',
            CigarKind::Insertion => 'I',
            CigarKind::Deletion => 'D',
            CigarKind::Skip => 'N',
            CigarKind::SoftClip => 'S',
            CigarKind::HardClip => 'H',
            CigarKind::Pad => 'P',
            CigarKind::SequenceMatch => '=',
            CigarKind::SequenceMismatch => 'X',
        }
    }


    let (_h, mut reader) = dv_io::bam::open(reads_path).context("open BAM")?;
    let mut owned: Vec<OwnedRead> = Vec::new();
    for rec in reader.records() {
        let r = rec?;
        let Some(start) = r.alignment_start() else { continue };
        let start_0based = usize::from(start.unwrap()) as i64 - 1;
        // Filter reads matching upstream's read_requirements default:
        // skip secondary/supplementary, unmapped, duplicates, QC-fail, and
        // low mapping quality (< 10).
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
        let mq = r.mapping_quality().map(|q| q.get()).unwrap_or(255);
        let is_rev = r.flags().is_reverse_complemented();
        let frag = r.template_length() as i32;
        let name = r
            .name()
            .map(|n| std::str::from_utf8(n.as_ref()).unwrap_or("").to_string())
            .unwrap_or_default();
        let mate = if r.flags().is_first_segment() { 1 } else { 2 };
        // HP aux tag: 1 → haplotype 1, 2 → haplotype 2, anything else → 0.
        let hp_tag = {
            use noodles::sam::alignment::record::data::field::Tag;
            let data = r.data();
            let raw = data
                .get(&Tag::new(b'H', b'P'))
                .and_then(|res| res.ok())
                .and_then(|v| v.as_int());
            match raw {
                Some(1) => 1u8,
                Some(2) => 2u8,
                _ => 0u8,
            }
        };
        owned.push(OwnedRead {
            ref_start: start_0based,
            cigar: cigar_owned,
            seq,
            bq,
            mq,
            is_rev,
            frag,
            name,
            mate,
            hp: hp_tag,
        });
    }
    tracing::info!(reads_loaded = owned.len(), "loaded reads");

    // Run allele counter.
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

    // Run candidate caller.
    let mut cands = candidates_from_counts(&counts, &VariantCallerOptions::default());
    tracing::info!(initial_candidates = cands.len(), "candidate variants");

    // Realigner-driven candidate expansion: walk window_selector hot spots,
    // assemble a de Bruijn graph from local reads, and add any haplotype-vs-ref
    // variants we don't already have. Recovers indel candidates that don't
    // appear in any single read's CIGAR but emerge from local reassembly.
    {
        use dv_core::realigner::{
            debruijn::{DeBruijnGraph, DeBruijnOptions, ReadInput},
            orchestrator::variants_from_haplotype,
            window_selector::{variant_reads_candidates, windows_from_scores, WindowSelectorOptions},
        };
        let scores = variant_reads_candidates(&counts, &WindowSelectorOptions::default());
        // Threshold: any position with score >= 3 supports of evidence.
        let raw_windows = windows_from_scores(&scores, 3);
        // Pad each window by ~50bp on each side and merge overlapping ones.
        let mut padded: Vec<(i64, i64)> = raw_windows
            .iter()
            .map(|(s, e)| (region.start + *s as i64 - 50, region.start + *e as i64 + 50))
            .collect();
        padded.sort_by_key(|w| w.0);
        let mut merged: Vec<(i64, i64)> = Vec::new();
        for w in padded {
            match merged.last_mut() {
                Some(last) if w.0 <= last.1 => last.1 = last.1.max(w.1),
                _ => merged.push(w),
            }
        }

        let mut existing: std::collections::HashSet<(i64, String, Vec<String>)> =
            std::collections::HashSet::new();
        for v in &cands {
            let mut alts = v.alternate_bases.clone();
            alts.sort();
            existing.insert((v.start, v.reference_bases.clone(), alts));
        }

        let dbg_opts = DeBruijnOptions::default();
        let mut added = 0usize;
        for (ws, we) in &merged {
            let win_ref = match fa.fetch_range(&region.reference_name, *ws, *we) {
                Some(b) if b.len() == (we - ws) as usize => b,
                _ => continue,
            };
            // Find reads overlapping this window.
            let win_reads: Vec<&OwnedRead> = owned
                .iter()
                .filter(|r| {
                    let read_len_on_ref: i64 = r
                        .cigar
                        .iter()
                        .filter(|(op, _)| matches!(op, 'M' | '=' | 'X' | 'D' | 'N'))
                        .map(|(_, l)| *l)
                        .sum();
                    let read_end = r.ref_start + read_len_on_ref;
                    r.ref_start < *we && read_end > *ws
                })
                .collect();
            let read_inputs: Vec<ReadInput<'_>> = win_reads
                .iter()
                .map(|r| ReadInput {
                    aligned_sequence: &r.seq,
                    aligned_quality: &r.bq,
                    mapping_quality: r.mq,
                })
                .collect();
            let graph = match DeBruijnGraph::build(&win_ref, &read_inputs, &dbg_opts) {
                Some(g) => g,
                None => continue,
            };
            for hap in graph.candidate_haplotypes() {
                if hap.as_slice() == win_ref {
                    continue;
                }
                for nv in variants_from_haplotype(&region.reference_name, *ws, &win_ref, &hap) {
                    let mut alts = nv.alternate_bases.clone();
                    alts.sort();
                    let key = (nv.start, nv.reference_bases.clone(), alts);
                    if existing.insert(key) {
                        cands.push(nv);
                        added += 1;
                    }
                }
            }
        }
        // Keep candidates sorted by genomic coord for deterministic output.
        cands.sort_by(|a, b| {
            (a.reference_name.as_str(), a.start, a.end).cmp(&(
                b.reference_name.as_str(),
                b.start,
                b.end,
            ))
        });
        tracing::info!(
            realigner_added = added,
            total_candidates = cands.len(),
            assembly_windows = merged.len(),
            "realigner candidate expansion"
        );
    }

    // ---- small-model fast path setup ----
    // Pre-compute VAF (alt fraction × 100) per position from the allele
    // counter output. The small model's 51-element context window reads
    // from this map.
    let mut vaf_at_position: std::collections::HashMap<i64, i32> =
        std::collections::HashMap::with_capacity(counts.len());
    for c in &counts {
        let pos = c.position.as_ref().expect("position").position;
        let alt_count: i32 = c.read_alleles.values().map(|a| a.count).sum();
        let total = c.ref_supporting_read_count + alt_count;
        if total > 0 {
            vaf_at_position.insert(pos, 100 * alt_count / total);
        }
    }
    let small_model_loaded = if let Some(p) = small_model_path {
        try_set_ort_dylib_path();
        Some(dv_infer::ort::SmallModelOrt::load(p).context("load small model")?)
    } else {
        None
    };
    let mut small_model_writer = match (small_model_loaded.is_some(), small_model_cvo_path) {
        (true, Some(p)) => Some(dv_io::tfrecord::open_writer(p).context("open small CVO")?),
        (true, None) => anyhow::bail!("--small-model also requires --small-model-cvo"),
        _ => None,
    };
    let mut small_model_emitted = 0usize;

    // Render examples.
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

    let mut writer = dv_io::tfrecord::open_writer(examples_path).context("open TFRecord")?;
    let mut emitted = 0usize;
    for v in &cands {
        let win_start = v.start - center;
        let win_end = win_start + width as i64;
        let img_ref = match fa.fetch_range(&v.reference_name, win_start, win_end) {
            Some(b) if b.len() == width => b,
            _ => continue,
        };
        // Upstream's `Query(region)` for the pileup uses a tight region
        // around the variant: `[variant.start - read_overlap_buffer_bp,
        // variant.end + read_overlap_buffer_bp)` where buffer defaults to 5.
        // (See `make_examples_native.cc:643-648`.)
        const READ_OVERLAP_BUFFER_BP: i64 = 5;
        let query_start = v.start - READ_OVERLAP_BUFFER_BP;
        let query_end = v.end + READ_OVERLAP_BUFFER_BP;
        let window_reads: Vec<&OwnedRead> = owned
            .iter()
            .filter(|r| {
                let read_len_on_ref: i64 = r
                    .cigar
                    .iter()
                    .filter(|(op, _)| matches!(op, 'M' | '=' | 'X' | 'D' | 'N'))
                    .map(|(_, l)| *l)
                    .sum();
                let read_end = r.ref_start + read_len_on_ref;
                r.ref_start < query_end && read_end > query_start
            })
            .collect();
        let variant_pos = v.start;
        // For each read, check whether it supports any of the alt alleles
        // by reading its base at the variant position.
        let alt_set: std::collections::HashSet<u8> = v
            .alternate_bases
            .iter()
            .filter_map(|a| a.as_bytes().first().copied())
            .collect();
        let read_supports = |r: &OwnedRead| -> bool {
            let mut ref_pos = r.ref_start;
            let mut read_pos = 0usize;
            for &(op, len) in &r.cigar {
                let len_us = len as usize;
                match op {
                    'M' | '=' | 'X' => {
                        if ref_pos <= variant_pos && variant_pos < ref_pos + len {
                            let off = (variant_pos - ref_pos) as usize;
                            if read_pos + off < r.seq.len() {
                                let b = r.seq[read_pos + off];
                                return alt_set.contains(&b);
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
        // Diagnostic: dump supports classification for chr20:10001019.
        if v.start == 10_001_018 && std::env::var("DV_DEBUG_SUPPORTS").is_ok() {
            for r in &window_reads {
                let s = read_supports(r);
                eprintln!(
                    "  supports={} ref_start={} name={} mate={}",
                    s, r.ref_start, r.name, r.mate
                );
            }
        }
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
                supports_variant: read_supports(r),
                hp_tag: r.hp,
                fragment_name: &r.name,
                read_number: r.mate - 1,
            })
            .collect();
        let ctx = Some(VariantContext {
            variant_pos: variant_pos,
            min_base_quality_at_call: 10,
        });
        let img = render(
            win_start,
            width,
            height,
            5, // upstream default reference_band_height
            &img_ref,
            &pileup_reads,
            &kinds,
            &opts,
            ctx,
            42, // upstream default random_seed
        );

        // For each alt-allele combo, try the small-model fast path first.
        // If it accepts, emit a CVO with MID=small_model and skip the
        // image render for that combo. Otherwise fall through to the
        // big-model image example.
        for (i, _) in v.alternate_bases.iter().enumerate() {
            let alt_indices = vec![i as i32];

            // Try small model on biallelic SNVs only — for indels and
            // multiallelics our read-support classification (1-base
            // match) isn't faithful enough to feed the model.
            let try_small = small_model_loaded.is_some()
                && v.alternate_bases.len() == 1
                && dv_core::small_model::features::is_snp(v, &alt_indices);

            if try_small {
                let alt_byte = v.alternate_bases[i].as_bytes()[0];
                let (refs, alts, total_depth) =
                    classify_reads_at(&owned, variant_pos, &v.reference_bases, alt_byte);
                let feat = dv_core::small_model::compute(
                    v,
                    &alt_indices,
                    &refs,
                    &alts,
                    total_depth,
                    &vaf_at_position,
                );
                let model = small_model_loaded.as_ref().unwrap();
                let probs = model.predict(&feat, 1)?;
                if dv_core::small_model::passes_threshold(v, &alt_indices, &probs) {
                    let mut variant_with_call = v.clone();
                    if variant_with_call.calls.is_empty() {
                        variant_with_call.calls.push(Default::default());
                    }
                    variant_with_call.calls[0].call_set_name = sample_name.to_string();
                    set_model_id(&mut variant_with_call, "small_model");
                    let cvo = CallVariantsOutput {
                        variant: Some(variant_with_call),
                        alt_allele_indices: Some(AltAlleleIndices {
                            indices: alt_indices.clone(),
                        }),
                        genotype_probabilities: probs.iter().map(|&p| p as f64).collect(),
                        debug_info: None,
                    };
                    let writer = small_model_writer.as_mut().unwrap();
                    writer.write_record(&cvo.encode_to_vec())?;
                    small_model_emitted += 1;
                    continue;
                }
            }

            let mut variant_with_call = v.clone();
            // Preserve the DP/AD/VAF info populated by variant_calling; just
            // set the sample name. If no call exists yet, add one.
            if variant_with_call.calls.is_empty() {
                variant_with_call.calls.push(Default::default());
            }
            variant_with_call.calls[0].call_set_name = sample_name.to_string();
            let bytes = build_example(&variant_with_call, &alt_indices, &img);
            writer.write_record(&bytes)?;
            emitted += 1;
        }
    }
    writer.flush()?;
    if let Some(mut w) = small_model_writer {
        w.flush()?;
    }
    tracing::info!(
        examples = emitted,
        small_model_cvos = small_model_emitted,
        "wrote examples"
    );
    Ok(())
}

/// Classify reads overlapping `variant_pos` into ref-supporting,
/// alt-supporting (where the read base equals `alt_byte`), and report
/// the total number of reads with coverage at this position. Walks each
/// read's CIGAR to find its base at `variant_pos`. Skips clipped/skipped
/// regions.
///
/// Restricted to single-base alts — caller guarantees this is a
/// biallelic SNV.
fn classify_reads_at(
    owned: &[OwnedRead],
    variant_pos: i64,
    ref_bases: &str,
    alt_byte: u8,
) -> (Vec<dv_core::small_model::ReadAttrs>, Vec<dv_core::small_model::ReadAttrs>, i32) {
    let ref_byte = ref_bases.as_bytes()[0].to_ascii_uppercase();
    let alt_byte = alt_byte.to_ascii_uppercase();
    let mut refs: Vec<dv_core::small_model::ReadAttrs> = Vec::new();
    let mut alts: Vec<dv_core::small_model::ReadAttrs> = Vec::new();
    let mut total = 0i32;

    'reads: for r in owned {
        let mut ref_pos = r.ref_start;
        let mut read_pos = 0usize;
        for &(op, len) in &r.cigar {
            let len_us = len as usize;
            match op {
                'M' | '=' | 'X' => {
                    if ref_pos <= variant_pos && variant_pos < ref_pos + len {
                        let off = (variant_pos - ref_pos) as usize;
                        if read_pos + off < r.seq.len() {
                            let b = r.seq[read_pos + off].to_ascii_uppercase();
                            let bq = r.bq.get(read_pos + off).copied().unwrap_or(0);
                            let attrs = dv_core::small_model::ReadAttrs {
                                mapping_quality: r.mq,
                                avg_base_quality: bq,
                                is_reverse_strand: r.is_rev,
                            };
                            total += 1;
                            if b == ref_byte {
                                refs.push(attrs);
                            } else if b == alt_byte {
                                alts.push(attrs);
                            }
                            // else: supports some other allele; counts
                            // toward total_depth but not ref/alt.
                        }
                        continue 'reads;
                    }
                    ref_pos += len;
                    read_pos += len_us;
                }
                'I' | 'S' => read_pos += len_us,
                'D' | 'N' => ref_pos += len,
                _ => {}
            }
        }
    }

    (refs, alts, total)
}

fn open_text_writer(path: &std::path::Path) -> Result<Box<dyn std::io::Write>> {
    let f = std::fs::File::create(path)
        .with_context(|| format!("create {}", path.display()))?;
    let buf = std::io::BufWriter::new(f);
    Ok(if path.extension().and_then(|e| e.to_str()) == Some("gz") {
        // Use BGZF (block gzip) so the output can be tabix-indexed.
        Box::new(noodles::bgzf::Writer::new(buf))
    } else {
        Box::new(buf)
    })
}

/// Build a tabix `.tbi` index alongside `path` if it ends in `.gz`.
/// No-op for plain `.vcf` outputs; index path is `<path>.tbi`.
fn maybe_emit_tbi(path: &std::path::Path) -> Result<()> {
    if path.extension().and_then(|e| e.to_str()) != Some("gz") {
        return Ok(());
    }
    let index = noodles::vcf::fs::index(path)
        .with_context(|| format!("build tabix index for {}", path.display()))?;
    let tbi_path = {
        let mut s = path.as_os_str().to_owned();
        s.push(".tbi");
        std::path::PathBuf::from(s)
    };
    let f = std::fs::File::create(&tbi_path)
        .with_context(|| format!("create {}", tbi_path.display()))?;
    let mut writer = noodles::tabix::io::Writer::new(std::io::BufWriter::new(f));
    writer
        .write_index(&index)
        .with_context(|| format!("write {}", tbi_path.display()))?;
    tracing::info!(path = %tbi_path.display(), "wrote tabix index");
    Ok(())
}

fn postprocess_variants(
    cvo: &std::path::Path,
    small_model_cvo: Option<&std::path::Path>,
    nonvariant_site_tfrecord: Option<&std::path::Path>,
    output_vcf: &std::path::Path,
    output_gvcf: Option<&std::path::Path>,
    sample_name: &str,
    contig_specs: &[String],
    ref_fasta: Option<&std::path::Path>,
) -> Result<()> {
    tracing::info!(?cvo, ?output_vcf, ?output_gvcf, sample_name, "postprocess_variants");
    let contigs = parse_contigs(contig_specs)?;

    let mut paths: Vec<&std::path::Path> = vec![cvo];
    if let Some(p) = small_model_cvo {
        paths.push(p);
    }
    let cvos = postprocess::load_cvos(&paths)?;
    tracing::info!(loaded = cvos.len(), "loaded CVOs");

    let opts = PostprocessOptions::default();
    let variants = postprocess::process_cvos_into_variants(cvos, sample_name, &opts);
    tracing::info!(produced = variants.len(), "produced variants");

    // VCF
    let mut vcf_writer = open_text_writer(output_vcf)?;
    vcf::write_header(&mut vcf_writer, &contigs, &[sample_name])?;
    let mut emitted_vcf = 0usize;
    for v in &variants {
        // Upstream emits all FILTER values (PASS, RefCall, LowQual, NoCall).
        // The `--only_keep_pass` flag is per-call and defaults to false.
        vcf::write_variant_line(&mut vcf_writer, v, POSTPROCESS_FORMAT_KEYS)?;
        emitted_vcf += 1;
    }
    vcf_writer.flush()?;
    drop(vcf_writer);
    tracing::info!(emitted_vcf, "wrote VCF");
    maybe_emit_tbi(output_vcf)?;

    // gVCF (optional)
    if let (Some(gvcf_path), Some(nv_path)) = (output_gvcf, nonvariant_site_tfrecord) {
        emit_gvcf(
            variants,
            nv_path,
            ref_fasta,
            gvcf_path,
            &contigs,
            sample_name,
        )?;
    }

    Ok(())
}

fn emit_gvcf(
    variants: Vec<dv_proto::nucleus_v1::Variant>,
    nv_path: &std::path::Path,
    ref_fasta: Option<&std::path::Path>,
    gvcf_path: &std::path::Path,
    contigs: &[ContigInfo],
    sample_name: &str,
) -> Result<()> {
    let nonvariants = dv_core::gvcf::load_nonvariants(&[nv_path])?;
    tracing::info!(loaded_nonvariants = nonvariants.len(), "loaded gvcf blocks");
    let contig_index_map: std::collections::HashMap<String, u32> = contigs
        .iter()
        .enumerate()
        .map(|(i, c)| (c.name.clone(), i as u32))
        .collect();

    let merged = if let Some(p) = ref_fasta {
        let reader = noodles::fasta::indexed_reader::Builder::default()
            .build_from_path(p)
            .with_context(|| format!("open FASTA {}", p.display()))?;
        let cell = std::cell::RefCell::new(reader);
        let mut lookup = move |contig: &str, pos: i64| -> Option<String> {
            let mut r = cell.borrow_mut();
            let region: noodles::core::Region = format!("{}:{}-{}", contig, pos + 1, pos + 1)
                .parse()
                .ok()?;
            r.query(&region).ok().and_then(|rec| {
                std::str::from_utf8(rec.sequence().as_ref())
                    .ok()
                    .map(|s| s.to_ascii_uppercase())
            })
        };
        let lookup_dyn: &mut dyn FnMut(&str, i64) -> Option<String> = &mut lookup;
        dv_core::gvcf::merge_streams(
            variants,
            nonvariants,
            |name| contig_index_map.get(name).copied(),
            Some(&mut *lookup_dyn),
        )
    } else {
        dv_core::gvcf::merge_streams(
            variants,
            nonvariants,
            |name| contig_index_map.get(name).copied(),
            None,
        )
    };

    let mut g_writer = open_text_writer(gvcf_path)?;
    vcf::write_header(&mut g_writer, contigs, &[sample_name])?;
    let mut emitted_gvcf = 0usize;
    for v in &merged {
        vcf::write_gvcf_line(&mut g_writer, v)?;
        emitted_gvcf += 1;
    }
    g_writer.flush()?;
    drop(g_writer);
    tracing::info!(emitted_gvcf, "wrote gVCF");
    maybe_emit_tbi(gvcf_path)?;
    Ok(())
}

struct ExampleRow {
    image_f32: Vec<f32>,
    variant: Variant,
    alt_allele_indices: AltAlleleIndices,
}

fn parse_example(payload: &[u8]) -> Result<ExampleRow> {
    let ex = Example::decode(payload).context("decode tf.Example")?;
    let features = ex.features.context("tf.Example missing features")?;

    let bytes_for = |key: &str| -> Result<Vec<u8>> {
        let f = features
            .feature
            .get(key)
            .with_context(|| format!("missing feature {key}"))?;
        match f.kind.as_ref().context("feature missing kind")? {
            FeatureKind::BytesList(bl) => {
                anyhow::ensure!(bl.value.len() == 1, "feature {key} has {} bytes values", bl.value.len());
                Ok(bl.value[0].clone())
            }
            other => anyhow::bail!("feature {key} expected BytesList, got {other:?}"),
        }
    };

    let image_raw = bytes_for("image/encoded")?;
    anyhow::ensure!(
        image_raw.len() == PIXEL_BYTES,
        "image/encoded is {} bytes, expected {PIXEL_BYTES}",
        image_raw.len()
    );
    let variant = Variant::decode(&*bytes_for("variant/encoded")?)?;
    let alt_allele_indices = AltAlleleIndices::decode(&*bytes_for("alt_allele_indices/encoded")?)?;

    let image_f32 = image_raw
        .iter()
        .map(|&b| (b as f32 - 128.0) / 128.0)
        .collect();

    Ok(ExampleRow {
        image_f32,
        variant,
        alt_allele_indices,
    })
}

/// Mirror `variantcall_utils.set_model_id` — set `info["MID"] = ListValue([model_id])`
/// on the first call of the variant. Matches upstream call_variants behavior.
fn set_model_id(variant: &mut Variant, model_id: &str) {
    if let Some(call) = variant.calls.first_mut() {
        call.info.insert(
            "MID".to_string(),
            ListValue {
                values: vec![Value {
                    kind: Some(dv_proto::nucleus_v1::value::Kind::StringValue(
                        model_id.to_string(),
                    )),
                }],
            },
        );
    }
}

/// Pick an inference backend based on the model path's shape.
///   - `*.onnx`              → ORT (priority backend, no libtensorflow needed)
///   - directory or `*/saved_model.pb` → TF SavedModel
///
/// Returns a clear error if the requested backend isn't compiled in (e.g.
/// `--no-default-features --features tf` and the user passed an `.onnx`
/// path).
fn load_backend(path: &std::path::Path) -> Result<Box<dyn InferenceBackend>> {
    let is_onnx = path.extension().and_then(|e| e.to_str()) == Some("onnx");
    let is_dir = path.is_dir();

    if is_onnx {
        #[cfg(feature = "ort")]
        {
            try_set_ort_dylib_path();
            tracing::info!(?path, backend = "ort", "loading model");
            let m = dv_infer::ort::OrtBackend::load(path).context("load ONNX model")?;
            return Ok(Box::new(m));
        }
        #[cfg(not(feature = "ort"))]
        anyhow::bail!(
            ".onnx path given but dv-cli was built without the `ort` feature; \
             rebuild with `--features ort` (it's the default)"
        );
    }

    if is_dir {
        #[cfg(feature = "tf")]
        {
            tracing::info!(?path, backend = "tf", "loading model");
            let m = dv_infer::tf::TfBackend::load(path).context("load SavedModel")?;
            return Ok(Box::new(m));
        }
        #[cfg(not(feature = "tf"))]
        anyhow::bail!(
            "SavedModel directory given but dv-cli was built without the `tf` \
             feature; rebuild with `--features tf` or pass an `.onnx` path"
        );
    }

    anyhow::bail!(
        "checkpoint path {} is neither an `.onnx` file nor a SavedModel directory",
        path.display()
    )
}

/// If the user hasn't already set ORT_DYLIB_PATH, try to find a bundled
/// `libonnxruntime.so` next to the workspace `models/` directory or
/// alongside the binary.
#[cfg(feature = "ort")]
fn try_set_ort_dylib_path() {
    if std::env::var_os("ORT_DYLIB_PATH").is_some() {
        return;
    }
    let candidates: Vec<std::path::PathBuf> = {
        let mut v = Vec::new();
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                v.push(dir.join("libonnxruntime.so"));
                if let Some(p) = dir.parent() {
                    v.push(p.join("models/lib/libonnxruntime.so"));
                }
                if let Some(p) = dir.parent().and_then(|p| p.parent()) {
                    v.push(p.join("models/lib/libonnxruntime.so"));
                }
            }
        }
        if let Ok(cwd) = std::env::current_dir() {
            v.push(cwd.join("models/lib/libonnxruntime.so"));
        }
        v
    };
    for c in candidates {
        if c.exists() {
            tracing::info!(path = %c.display(), "auto-set ORT_DYLIB_PATH");
            std::env::set_var("ORT_DYLIB_PATH", c);
            return;
        }
    }
}

fn call_variants(
    examples: &std::path::Path,
    checkpoint: &std::path::Path,
    output: &std::path::Path,
    batch_size: usize,
) -> Result<()> {
    tracing::info!(?examples, ?checkpoint, ?output, batch_size, "call_variants");
    anyhow::ensure!(batch_size > 0, "batch_size must be > 0");

    let model = load_backend(checkpoint)?;
    let [h, w, c] = model.input_shape();
    anyhow::ensure!(
        h * w * c == PIXEL_BYTES,
        "model input {h}x{w}x{c} != fixture pixel count {PIXEL_BYTES}"
    );

    let mut reader = tfrecord::open_reader(examples).context("open examples")?;
    let mut writer = tfrecord::open_writer(output).context("open output")?;
    let mut total = 0usize;

    let mut batch_rows: Vec<ExampleRow> = Vec::with_capacity(batch_size);
    let mut flat_buf: Vec<f32> = Vec::with_capacity(batch_size * PIXEL_BYTES);

    let flush = |rows: &mut Vec<ExampleRow>,
                     buf: &mut Vec<f32>,
                     writer: &mut tfrecord::Writer<Box<dyn std::io::Write>>,
                     total: &mut usize|
     -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let n = rows.len();
        let probs = model.predict_batch(buf, n)?;
        anyhow::ensure!(probs.len() == n * 3);
        for (i, row) in rows.drain(..).enumerate() {
            let mut variant = row.variant;
            set_model_id(&mut variant, MODEL_ID);
            let cvo = CallVariantsOutput {
                variant: Some(variant),
                alt_allele_indices: Some(row.alt_allele_indices),
                genotype_probabilities: probs[i * 3..(i + 1) * 3]
                    .iter()
                    .map(|&p| p as f64)
                    .collect(),
                debug_info: None,
            };
            let bytes = cvo.encode_to_vec();
            writer.write_record(&bytes)?;
            *total += 1;
        }
        buf.clear();
        Ok(())
    };

    while let Some(rec) = reader.read_record()? {
        let row = parse_example(&rec)?;
        flat_buf.extend_from_slice(&row.image_f32);
        batch_rows.push(row);
        if batch_rows.len() >= batch_size {
            flush(&mut batch_rows, &mut flat_buf, &mut writer, &mut total)?;
        }
    }
    flush(&mut batch_rows, &mut flat_buf, &mut writer, &mut total)?;
    writer.flush()?;
    tracing::info!(total, "wrote CallVariantsOutput records");
    Ok(())
}
