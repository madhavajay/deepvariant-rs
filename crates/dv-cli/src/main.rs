use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

mod serve;
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

/// Cap on assembled haplotypes carried out of each de-Bruijn window
/// before the (expensive) SSW pass in `variants_from_haplotype`.
/// Mirrors the C++ Apple-Silicon fork's `_MAX_HAPLOTYPES = 8`
/// (`realigner.py`): ~95.7% of assembled windows already produce ≤8
/// haplotypes; the long tail (up to `max_num_paths`=256) drives
/// disproportionate Smith-Waterman cost in complex/repeat regions.
/// `candidate_haplotypes()` returns them lexicographically sorted, so
/// this keeps the first 8 — the fork measured −14.7% on make_examples
/// with no accuracy loss (INDEL accuracy slightly improved).
const MAX_HAPLOTYPES_PER_WINDOW: usize = 8;

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
    /// Run make-examples and call-variants concurrently in-process.
    /// The pileup-image rendering pass streams (variant, alt_indices,
    /// image) tuples through an internal channel into the inference
    /// worker, eliminating the disk roundtrip and overlapping
    /// rendering with inference.
    Pipeline {
        /// Input BAM file.
        #[arg(long)]
        reads: PathBuf,
        /// Reference FASTA (with `.fai` index).
        #[arg(long)]
        ref_fasta: PathBuf,
        /// Region literal like `chr20:10,000,000-10,010,000` (1-based inclusive).
        #[arg(long)]
        region: String,
        /// Path to write the CallVariantsOutput TFRecord shard.
        #[arg(long)]
        output: PathBuf,
        /// Path to the SavedModel directory or `.onnx` file.
        #[arg(long)]
        checkpoint: PathBuf,
        /// Sample name for the output column.
        #[arg(long, default_value = "SAMPLE")]
        sample_name: String,
        /// Inference batch size.
        #[arg(long, default_value_t = 128)]
        batch_size: usize,
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
        /// Optional input BAM. When present, runs direct phasing on the
        /// variant calls and adds `0|1`/`1|0` GT separators + `PS`
        /// (phase-set) info to records that fall in a phasing block.
        #[arg(long)]
        phase_reads: Option<PathBuf>,
    },
    /// Serve a drag-and-drop web UI that runs `dv pipeline` on an
    /// uploaded BAM/CRAM and streams stage progress to the browser.
    Serve {
        /// Reference FASTA (with `.fai`). Required for CRAM decode.
        #[arg(long)]
        ref_fasta: PathBuf,
        /// SavedModel directory or `.onnx` checkpoint.
        #[arg(long)]
        checkpoint: PathBuf,
        /// TCP port to listen on.
        #[arg(long, default_value_t = 8080)]
        port: u16,
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
        Cmd::Pipeline {
            reads,
            ref_fasta,
            region,
            output,
            checkpoint,
            sample_name,
            batch_size,
        } => pipeline_cmd(
            &reads,
            &ref_fasta,
            &region,
            &output,
            &checkpoint,
            &sample_name,
            batch_size,
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
            phase_reads,
        } => postprocess_variants(
            &cvo,
            small_model_cvo.as_deref(),
            nonvariant_site_tfrecord.as_deref(),
            &output_vcf,
            output_gvcf.as_deref(),
            &sample_name,
            &contigs,
            ref_fasta.as_deref(),
            phase_reads.as_deref(),
        ),
        Cmd::Serve {
            ref_fasta,
            checkpoint,
            port,
        } => serve::serve_cmd(&ref_fasta, &checkpoint, port),
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
            extra: std::collections::BTreeMap::new(),
        });
    }
    Ok(out)
}

/// One read held by the make-examples flow as plain owned data, so it
/// can be re-borrowed across multiple per-candidate iterations without
/// re-parsing the BAM.
struct OwnedRead {
    ref_start: i64,
    /// `ref_start + sum(M/=/X/D/N CIGAR ops)`. Cached at BAM-load time
    /// because the overlap filters in realigner / pass2_render call
    /// this for every read on every query — recomputing it from the
    /// CIGAR on each call dominates make_examples on big regions.
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

    let mut stage_t = std::time::Instant::now();
    let lap = |label: &str, t: &mut std::time::Instant| {
        let dt = t.elapsed();
        tracing::info!(stage = label, ms = dt.as_millis() as u64, "stage");
        *t = std::time::Instant::now();
    };

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


    // Format-agnostic reader: BAM by default, CRAM when the path
    // ends in `.cram`. CRAM is reference-compressed so we pass the
    // FASTA path through.
    let (_h, mut reader) =
        dv_io::reads::open(reads_path, Some(ref_path)).context("open alignment input")?;
    let mut owned: Vec<OwnedRead> = Vec::new();
    reader.for_each_record_in_region(&_h, &region.reference_name, region.start, region.end, |r| {
        let Some(start) = r.alignment_start() else { return Ok(()) };
        let start_0based = usize::from(start.unwrap()) as i64 - 1;
        // Filter reads matching upstream's read_requirements default:
        // skip secondary/supplementary, unmapped, duplicates, QC-fail, and
        // low mapping quality (< 10). The `Record` trait returns
        // Result for fallible accessors; a parse error reading any
        // field means we should skip the record rather than crash.
        let flags = match r.flags() {
            Ok(f) => f,
            Err(_) => return Ok(()),
        };
        if flags.is_secondary()
            || flags.is_supplementary()
            || flags.is_unmapped()
            || flags.is_duplicate()
            || flags.is_qc_fail()
        {
            return Ok(());
        }
        let mq_check = r
            .mapping_quality()
            .and_then(|q| q.ok())
            .map(|q| q.get())
            .unwrap_or(255);
        if mq_check < 10 {
            return Ok(());
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
            return Ok(());
        }
        let seq: Vec<u8> = r.sequence().iter().map(|b| b.to_ascii_uppercase()).collect();
        let bq: Vec<u8> = r.quality_scores().iter().map(|q| q.unwrap_or(0)).collect();
        let mq = mq_check;
        let is_rev = flags.is_reverse_complemented();
        let frag = r.template_length().unwrap_or(0) as i32;
        let name = r
            .name()
            .map(|n| std::str::from_utf8(n.as_ref()).unwrap_or("").to_string())
            .unwrap_or_default();
        let mate = if flags.is_first_segment() { 1 } else { 2 };
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
            ref_end: start_0based + read_len_on_ref,
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
        Ok(())
    })?;
    tracing::info!(reads_loaded = owned.len(), "loaded reads");
    lap("bam_read", &mut stage_t);

    // Sort by ref_start so we can binary-search a per-query slice
    // instead of scanning all reads. BAM is mostly already sorted but
    // reads with the same start can come in any order; sort_by_key is
    // stable and cheap on already-sorted data.
    owned.sort_by_key(|r| r.ref_start);
    let read_starts: Vec<i64> = owned.iter().map(|r| r.ref_start).collect();
    // Largest "footprint" of any read so a query [qs, qe] only needs to
    // look at reads with `ref_start >= qs - max_read_span`. Reads here
    // are <300bp typically; this caps the lower-bound search distance.
    let max_read_span: i64 = owned.iter().map(|r| r.ref_end - r.ref_start).max().unwrap_or(0);
    lap("bam_sort", &mut stage_t);

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

    lap("allele_counter", &mut stage_t);

    // Run candidate caller.
    let mut cands = candidates_from_counts(&counts, &VariantCallerOptions::default());
    tracing::info!(initial_candidates = cands.len(), "candidate variants");
    lap("candidate_caller", &mut stage_t);

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

        let dbg_opts = DeBruijnOptions::default();

        // Pre-fetch FASTA windows serially (the Indexed FASTA reader
        // wraps a RefCell and is !Sync — can't be shared across rayon
        // threads). Each fetch is cheap; the heavy work is the DBG
        // build below.
        let win_refs: Vec<Option<Vec<u8>>> = merged
            .iter()
            .map(|(ws, we)| {
                fa.fetch_range(&region.reference_name, *ws, *we)
                    .filter(|b| b.len() == (we - ws) as usize)
            })
            .collect();

        // Parallel: build a DBG per window, expand each candidate
        // haplotype into variants. Each window's work is independent;
        // dedupe against the existing candidate set happens serially
        // after the par_iter.
        use rayon::prelude::*;
        let new_cands: Vec<Variant> = merged
            .par_iter()
            .zip(win_refs.par_iter())
            .filter_map(|((ws, we), win_ref_opt)| {
                let win_ref = win_ref_opt.as_ref()?;
                // Binary-search the candidate slice of reads, then
                // filter by the precomputed ref_end. This avoids the
                // O(reads) linear scan and the O(cigar) walk per read.
                let lo = read_starts.partition_point(|&s| s + max_read_span <= *ws);
                let hi = read_starts.partition_point(|&s| s < *we);
                let win_reads: Vec<&OwnedRead> = owned[lo..hi]
                    .iter()
                    .filter(|r| r.ref_end > *ws)
                    .collect();
                let read_inputs: Vec<ReadInput<'_>> = win_reads
                    .iter()
                    .map(|r| ReadInput {
                        aligned_sequence: &r.seq,
                        aligned_quality: &r.bq,
                        mapping_quality: r.mq,
                    })
                    .collect();
                let graph = DeBruijnGraph::build(win_ref, &read_inputs, &dbg_opts)?;
                let mut out: Vec<Variant> = Vec::new();
                for hap in graph
                    .candidate_haplotypes()
                    .into_iter()
                    .take(MAX_HAPLOTYPES_PER_WINDOW)
                {
                    if hap.as_slice() == win_ref {
                        continue;
                    }
                    for nv in
                        variants_from_haplotype(&region.reference_name, *ws, win_ref, &hap)
                    {
                        out.push(nv);
                    }
                }
                Some(out)
            })
            .flatten()
            .collect();

        // Serial dedupe + merge into shared candidate list.
        let mut existing: std::collections::HashSet<(i64, String, Vec<String>)> =
            std::collections::HashSet::new();
        for v in &cands {
            let mut alts = v.alternate_bases.clone();
            alts.sort();
            existing.insert((v.start, v.reference_bases.clone(), alts));
        }
        let mut added = 0usize;
        for nv in new_cands {
            let mut alts = nv.alternate_bases.clone();
            alts.sort();
            let key = (nv.start, nv.reference_bases.clone(), alts);
            if existing.insert(key) {
                cands.push(nv);
                added += 1;
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
    lap("realigner", &mut stage_t);

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

    let emitted;

    // ---- Pass 1 (serial): small-model decisions + FASTA pre-fetch ----
    //
    // For each candidate we decide whether each alt-allele combo takes
    // the small-model fast path. Any that doesn't gets queued in
    // `pending_renders` along with its FASTA window. Rendering then runs
    // in parallel (Pass 2) and writes happen serially afterward (Pass 3).
    struct PendingRender {
        variant: Variant,
        alt_indices: Vec<i32>,
        img_ref: Vec<u8>,
        win_start: i64,
        variant_pos: i64,
    }
    let mut pending_renders: Vec<PendingRender> = Vec::new();
    for v in &cands {
        let win_start = v.start - center;
        let win_end = win_start + width as i64;
        let img_ref = match fa.fetch_range(&v.reference_name, win_start, win_end) {
            Some(b) if b.len() == width => b,
            _ => continue,
        };
        let variant_pos = v.start;

        // For each alt-allele combo, try the small-model fast path
        // first. If it accepts, emit a CVO with MID=small_model now.
        // Otherwise queue the (variant, alt_indices) for parallel image
        // rendering in Pass 2.
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
            pending_renders.push(PendingRender {
                variant: v.clone(),
                alt_indices,
                img_ref: img_ref.clone(),
                win_start,
                variant_pos,
            });
        }
    }

    lap("pass1_small_model_decisions", &mut stage_t);

    // ---- Pass 2 (parallel): render images ----
    //
    // Each entry in `pending_renders` is independent — `owned` is shared
    // by-reference and otherwise everything the closure needs is owned
    // by the entry. rayon handles work-stealing across cores; on chr20
    // quickstart this drops the render wall time roughly proportional to
    // CPU count.
    use rayon::prelude::*;
    let rendered: Vec<(usize, Vec<u8>)> = pending_renders
        .par_iter()
        .enumerate()
        .map(|(idx, p)| {
            const READ_OVERLAP_BUFFER_BP: i64 = 5;
            let query_start = p.variant.start - READ_OVERLAP_BUFFER_BP;
            let query_end = p.variant.end + READ_OVERLAP_BUFFER_BP;
            // See realigner above for rationale. Binary-search the
            // candidate slice, then filter by precomputed ref_end.
            let lo = read_starts.partition_point(|&s| s + max_read_span <= query_start);
            let hi = read_starts.partition_point(|&s| s < query_end);
            let window_reads: Vec<&OwnedRead> = owned[lo..hi]
                .iter()
                .filter(|r| r.ref_end > query_start)
                .collect();
            let alt_set: std::collections::HashSet<u8> = p
                .variant
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
                            if ref_pos <= p.variant_pos && p.variant_pos < ref_pos + len {
                                let off = (p.variant_pos - ref_pos) as usize;
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
                variant_pos: p.variant_pos,
                min_base_quality_at_call: 10,
            });
            let img = render(
                p.win_start,
                width,
                height,
                5,
                &p.img_ref,
                &pileup_reads,
                &kinds,
                &opts,
                ctx,
                42,
            );
            let mut variant_with_call = p.variant.clone();
            if variant_with_call.calls.is_empty() {
                variant_with_call.calls.push(Default::default());
            }
            variant_with_call.calls[0].call_set_name = sample_name.to_string();
            let bytes = build_example(&variant_with_call, &p.alt_indices, &img);
            (idx, bytes)
        })
        .collect();

    lap("pass2_render_parallel", &mut stage_t);

    // ---- Pass 3 (parallel-gzip): write in canonical (input) order ----
    //
    // par_iter preserves index order via `enumerate()`, but we explicitly
    // sort by index here to keep the output deterministic regardless of
    // rayon's scheduling. Then chunked-parallel-gzip the whole batch in
    // one shot — the per-record streaming Writer was a single-thread
    // bottleneck (~5 min on full chr20).
    let mut rendered = rendered;
    rendered.sort_by_key(|(idx, _)| *idx);
    let payloads: Vec<Vec<u8>> = rendered.into_iter().map(|(_, bytes)| bytes).collect();
    emitted = payloads.len();
    dv_io::tfrecord::write_records_gz_parallel(examples_path, &payloads)
        .context("write examples TFRecord")?;
    if let Some(mut w) = small_model_writer {
        w.flush()?;
    }
    lap("pass3_write", &mut stage_t);
    tracing::info!(
        examples = emitted,
        small_model_cvos = small_model_emitted,
        "wrote examples"
    );
    Ok(())
}

/// Read BAM records overlapping `[shard_start, shard_end)` and run
/// the allele counter for that sub-region. Returns
/// `(owned, counts)`:
///
///   - `owned`: reads with `ref_start ∈ [shard_start, shard_end)`. A
///     read straddling shard boundaries is emitted exactly once, by
///     the shard whose start range contains its `ref_start`.
///   - `counts`: per-position allele counts for `[shard_start, shard_end)`.
///     Reads spanning shards still contribute to counts inside this
///     shard's range — `add_read` ignores positions outside the
///     supplied counts slice.
fn process_shard(
    reads_path: &std::path::Path,
    ref_path: &std::path::Path,
    contig: &str,
    shard_start: i64,
    shard_end: i64,
) -> Result<(Vec<OwnedRead>, Vec<dv_proto::dv::AlleleCount>)> {
    use dv_core::allelecounter::{add_read, empty_counts, AlignedRead, CounterOptions};
    use noodles::sam::alignment::record::cigar::op::Kind as CigarKind;
    use noodles::sam::alignment::record::QualityScores;
    #[allow(unused_imports)]
    use noodles::sam::alignment::Record;

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

    let fa = dv_io::fasta::open_indexed(ref_path).context("open FASTA")?;
    let ref_bases = fa
        .fetch_range(contig, shard_start, shard_end)
        .ok_or_else(|| anyhow::anyhow!("FASTA shard {contig}:{shard_start}-{shard_end} missing"))?;
    let mut counts = empty_counts(contig, shard_start, shard_end, &ref_bases);
    let counter_opts = CounterOptions::default();

    let (h, mut reader) =
        dv_io::reads::open(reads_path, Some(ref_path)).context("open alignment input")?;
    let mut all_reads: Vec<OwnedRead> = Vec::new();
    reader.for_each_record_in_region(&h, contig, shard_start, shard_end, |r| {
        let Some(start) = r.alignment_start() else { return Ok(()) };
        let start_0based = usize::from(start.unwrap()) as i64 - 1;
        let flags = match r.flags() {
            Ok(f) => f,
            Err(_) => return Ok(()),
        };
        if flags.is_secondary()
            || flags.is_supplementary()
            || flags.is_unmapped()
            || flags.is_duplicate()
            || flags.is_qc_fail()
        {
            return Ok(());
        }
        let mq_check = r
            .mapping_quality()
            .and_then(|q| q.ok())
            .map(|q| q.get())
            .unwrap_or(255);
        if mq_check < 10 {
            return Ok(());
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
        if start_0based + read_len_on_ref < shard_start || start_0based >= shard_end {
            return Ok(());
        }
        let seq: Vec<u8> = r.sequence().iter().map(|b| b.to_ascii_uppercase()).collect();
        let bq: Vec<u8> = r.quality_scores().iter().map(|q| q.unwrap_or(0)).collect();
        let mq = mq_check;
        let is_rev = flags.is_reverse_complemented();
        let frag = r.template_length().unwrap_or(0) as i32;
        let name = r
            .name()
            .map(|n| std::str::from_utf8(n.as_ref()).unwrap_or("").to_string())
            .unwrap_or_default();
        let mate = if flags.is_first_segment() { 1 } else { 2 };
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
        all_reads.push(OwnedRead {
            ref_start: start_0based,
            ref_end: start_0based + read_len_on_ref,
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
        Ok(())
    })?;

    // Reads straddling boundaries still contribute to in-shard
    // positions — `add_read` clamps writes to the counts slice.
    for r in &all_reads {
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
        add_read(&mut counts, &aligned, &counter_opts, shard_start);
    }

    // Owned partition: only reads whose start is in this shard's range.
    all_reads.retain(|r| r.ref_start >= shard_start && r.ref_start < shard_end);
    Ok((all_reads, counts))
}

/// Run make-examples and call-variants concurrently in-process.
///
/// Mirrors `make_examples_cmd` up to the candidate set + pending
/// renders, then replaces Pass 2 (collect-and-write) with a streaming
/// pipeline:
///
///   - rayon par_iter renders pileup-images and pushes
///     `(variant, alt_indices, image_u8)` onto a bounded channel.
///   - A separate inference worker thread loads the ORT/CoreML model,
///     drains the channel, batches, calls `predict_batch`, and writes
///     `CallVariantsOutput` records.
///
/// Net effect: the disk roundtrip is gone (no intermediate
/// `examples.tfrecord.gz`), and rendering overlaps with inference
/// instead of running fully sequentially. On macOS+CoreML this is the
/// equivalent of the C++ fork's `fast_pipeline` shared-memory IPC, but
/// all in-process.
fn pipeline_cmd(
    reads_path: &std::path::Path,
    ref_path: &std::path::Path,
    region_literal: &str,
    output_cvo_path: &std::path::Path,
    checkpoint_path: &std::path::Path,
    sample_name: &str,
    batch_size: usize,
) -> Result<()> {
    use dv_core::nucleus::ranges;
    use dv_core::pileup_image::{
        channels::ChannelKind,
        layout::{render, PileupRead, VariantContext},
        options::PileupOptions,
    };
    use dv_core::variant_calling::{candidates_from_counts, VariantCallerOptions};

    let region = ranges::parse_literal(region_literal).map_err(|e| anyhow::anyhow!(e))?;
    tracing::info!(
        ?reads_path,
        ?ref_path,
        ?output_cvo_path,
        ?checkpoint_path,
        region = ?region,
        batch_size,
        "pipeline"
    );
    anyhow::ensure!(batch_size > 0, "batch_size must be > 0");

    let mut stage_t = std::time::Instant::now();
    let lap = |label: &str, t: &mut std::time::Instant| {
        let dt = t.elapsed();
        tracing::info!(stage = label, ms = dt.as_millis() as u64, "stage");
        *t = std::time::Instant::now();
    };

    // ---- Region-shard pre-pass2 (BAM read + allele counter) ----
    //
    // Splits the input region into N=perf_cores sub-regions. Each
    // shard does its own BAM indexed query + allele counter in
    // parallel. Reads straddling shard boundaries contribute to
    // counts on both sides but are emitted to `owned` exactly once
    // (by the shard whose start range contains their `ref_start`),
    // so the merged owned list is dedupe-free.
    let n_shards = std::env::var("DV_SHARDS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
        })
        .max(1);
    let region_width = region.end - region.start;
    let shard_width = ((region_width + n_shards as i64 - 1) / n_shards as i64).max(1);
    let shard_ranges: Vec<(i64, i64)> = (0..n_shards as i64)
        .map(|i| {
            let s = region.start + i * shard_width;
            let e = (s + shard_width).min(region.end);
            (s, e)
        })
        .filter(|(s, e)| s < e)
        .collect();
    tracing::info!(shards = shard_ranges.len(), shard_width, "shard pre-pass2");

    use rayon::prelude::*;
    let shard_results: Result<Vec<(Vec<OwnedRead>, Vec<dv_proto::dv::AlleleCount>)>> =
        shard_ranges
            .par_iter()
            .map(|&(s, e)| {
                process_shard(reads_path, ref_path, &region.reference_name, s, e)
            })
            .collect();
    let shard_outputs = shard_results?;
    let mut owned: Vec<OwnedRead> = Vec::new();
    let mut counts: Vec<dv_proto::dv::AlleleCount> = Vec::new();
    for (mut shard_owned, mut shard_counts) in shard_outputs {
        owned.append(&mut shard_owned);
        counts.append(&mut shard_counts);
    }
    tracing::info!(reads_loaded = owned.len(), "loaded reads");
    lap("shard_pre_pass2", &mut stage_t);

    // Owned reads from shards are roughly sorted (each shard is in
    // genomic order and shards run in genomic order), but
    // intra-shard ordering is preserved from BAM order which is
    // sorted by ref_start anyway. Final stable sort to be safe.
    owned.sort_by_key(|r| r.ref_start);
    let read_starts: Vec<i64> = owned.iter().map(|r| r.ref_start).collect();
    let max_read_span: i64 = owned.iter().map(|r| r.ref_end - r.ref_start).max().unwrap_or(0);
    lap("bam_sort", &mut stage_t);

    // FASTA handle for the realigner / pass1 / pass2 fetches below.
    // Each shard opened its own; we open a fresh one here for the
    // remaining (single-thread) FASTA work.
    let fa = dv_io::fasta::open_indexed(ref_path).context("open FASTA")?;

    let mut cands = candidates_from_counts(&counts, &VariantCallerOptions::default());
    tracing::info!(initial_candidates = cands.len(), "candidate variants");
    lap("candidate_caller", &mut stage_t);

    // Realigner-driven candidate expansion (mirrors make_examples_cmd).
    {
        use dv_core::realigner::{
            debruijn::{DeBruijnGraph, DeBruijnOptions, ReadInput},
            orchestrator::variants_from_haplotype,
            window_selector::{variant_reads_candidates, windows_from_scores, WindowSelectorOptions},
        };
        let scores = variant_reads_candidates(&counts, &WindowSelectorOptions::default());
        let raw_windows = windows_from_scores(&scores, 3);
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
        let dbg_opts = DeBruijnOptions::default();
        let win_refs: Vec<Option<Vec<u8>>> = merged
            .iter()
            .map(|(ws, we)| {
                fa.fetch_range(&region.reference_name, *ws, *we)
                    .filter(|b| b.len() == (we - ws) as usize)
            })
            .collect();
        use rayon::prelude::*;
        let new_cands: Vec<Variant> = merged
            .par_iter()
            .zip(win_refs.par_iter())
            .filter_map(|((ws, we), win_ref_opt)| {
                let win_ref = win_ref_opt.as_ref()?;
                let lo = read_starts.partition_point(|&s| s + max_read_span <= *ws);
                let hi = read_starts.partition_point(|&s| s < *we);
                let win_reads: Vec<&OwnedRead> = owned[lo..hi]
                    .iter()
                    .filter(|r| r.ref_end > *ws)
                    .collect();
                let read_inputs: Vec<ReadInput<'_>> = win_reads
                    .iter()
                    .map(|r| ReadInput {
                        aligned_sequence: &r.seq,
                        aligned_quality: &r.bq,
                        mapping_quality: r.mq,
                    })
                    .collect();
                let graph = DeBruijnGraph::build(win_ref, &read_inputs, &dbg_opts)?;
                let mut out: Vec<Variant> = Vec::new();
                for hap in graph
                    .candidate_haplotypes()
                    .into_iter()
                    .take(MAX_HAPLOTYPES_PER_WINDOW)
                {
                    if hap.as_slice() == win_ref {
                        continue;
                    }
                    for nv in
                        variants_from_haplotype(&region.reference_name, *ws, win_ref, &hap)
                    {
                        out.push(nv);
                    }
                }
                Some(out)
            })
            .flatten()
            .collect();
        let mut existing: std::collections::HashSet<(i64, String, Vec<String>)> =
            std::collections::HashSet::new();
        for v in &cands {
            let mut alts = v.alternate_bases.clone();
            alts.sort();
            existing.insert((v.start, v.reference_bases.clone(), alts));
        }
        let mut added = 0usize;
        for nv in new_cands {
            let mut alts = nv.alternate_bases.clone();
            alts.sort();
            let key = (nv.start, nv.reference_bases.clone(), alts);
            if existing.insert(key) {
                cands.push(nv);
                added += 1;
            }
        }
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
    lap("realigner", &mut stage_t);

    // Build the PendingRender list (mirrors make_examples_cmd Pass 1
    // but skips the small-model fast path — we don't expose
    // --small-model in the pipeline subcommand for now).
    struct PendingRender {
        variant: Variant,
        alt_indices: Vec<i32>,
        img_ref: Vec<u8>,
        win_start: i64,
        variant_pos: i64,
    }
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
    let mut pending_renders: Vec<PendingRender> = Vec::new();
    for v in &cands {
        let win_start = v.start - center;
        let win_end = win_start + width as i64;
        let img_ref = match fa.fetch_range(&v.reference_name, win_start, win_end) {
            Some(b) if b.len() == width => b,
            _ => continue,
        };
        for (i, _) in v.alternate_bases.iter().enumerate() {
            let alt_indices = vec![i as i32];
            pending_renders.push(PendingRender {
                variant: v.clone(),
                alt_indices,
                img_ref: img_ref.clone(),
                win_start,
                variant_pos: v.start,
            });
        }
    }
    lap("pass1_prepare_renders", &mut stage_t);

    // ---- Spawn inference worker ----
    let (tx, rx) = std::sync::mpsc::sync_channel::<(Variant, Vec<i32>, Vec<u8>)>(256);
    let checkpoint = checkpoint_path.to_path_buf();
    let output = output_cvo_path.to_path_buf();
    let bs = batch_size;
    let cv_handle: std::thread::JoinHandle<Result<usize>> = std::thread::spawn(move || {
        try_set_ort_dylib_path();
        let model = load_backend(&checkpoint).context("load model")?;
        let [h, w, c] = model.input_shape();
        anyhow::ensure!(
            h * w * c == PIXEL_BYTES,
            "model input {h}x{w}x{c} != fixture pixel count {PIXEL_BYTES}"
        );
        let mut writer = dv_io::tfrecord::open_writer(&output).context("open CVO output")?;
        let mut total = 0usize;
        // If the ONNX has a fixed batch dim (e.g. 128 from
        // `normalize_onnx_pads.py`), use that. The user-specified
        // batch_size is overridden in that case so we always submit
        // shape-conforming batches to CoreML / ORT — partial batches
        // would otherwise fall back to the CPU EP and tank perf.
        let bs = model.pinned_batch().unwrap_or(bs);
        let mut batch_meta: Vec<(Variant, Vec<i32>)> = Vec::with_capacity(bs);
        let mut flat_buf: Vec<f32> = Vec::with_capacity(bs * PIXEL_BYTES);
        let flush =
            |bm: &mut Vec<(Variant, Vec<i32>)>,
             fb: &mut Vec<f32>,
             writer: &mut dv_io::tfrecord::Writer<Box<dyn std::io::Write>>,
             total: &mut usize|
             -> Result<()> {
                if bm.is_empty() {
                    return Ok(());
                }
                let actual_n = bm.len();
                // Pad the input buffer up to `bs` zero images if the
                // last batch is partial. Output for the padded slots
                // is discarded.
                let need = bs * PIXEL_BYTES;
                if fb.len() < need {
                    fb.resize(need, 0.0);
                }
                let probs = model.predict_batch(fb, bs)?;
                anyhow::ensure!(probs.len() == bs * 3);
                for (i, (mut variant, alt_indices)) in bm.drain(..).enumerate() {
                    set_model_id(&mut variant, MODEL_ID);
                    let cvo = CallVariantsOutput {
                        variant: Some(variant),
                        alt_allele_indices: Some(AltAlleleIndices {
                            indices: alt_indices,
                        }),
                        genotype_probabilities: probs[i * 3..(i + 1) * 3]
                            .iter()
                            .map(|&p| p as f64)
                            .collect(),
                        debug_info: None,
                    };
                    writer.write_record(&cvo.encode_to_vec())?;
                    *total += 1;
                }
                let _ = actual_n;
                fb.clear();
                Ok(())
            };
        while let Ok((variant, alt_indices, img_u8)) = rx.recv() {
            anyhow::ensure!(
                img_u8.len() == PIXEL_BYTES,
                "image bytes {} != PIXEL_BYTES {PIXEL_BYTES}",
                img_u8.len()
            );
            for &b in &img_u8 {
                flat_buf.push((b as f32 - 128.0) / 128.0);
            }
            batch_meta.push((variant, alt_indices));
            if batch_meta.len() >= bs {
                flush(&mut batch_meta, &mut flat_buf, &mut writer, &mut total)?;
            }
        }
        flush(&mut batch_meta, &mut flat_buf, &mut writer, &mut total)?;
        writer.flush()?;
        Ok(total)
    });

    // ---- Pass 2 (parallel render → channel) ----
    pending_renders.par_iter().for_each(|p| {
        const READ_OVERLAP_BUFFER_BP: i64 = 5;
        let query_start = p.variant.start - READ_OVERLAP_BUFFER_BP;
        let query_end = p.variant.end + READ_OVERLAP_BUFFER_BP;
        let lo = read_starts.partition_point(|&s| s + max_read_span <= query_start);
        let hi = read_starts.partition_point(|&s| s < query_end);
        let window_reads: Vec<&OwnedRead> = owned[lo..hi]
            .iter()
            .filter(|r| r.ref_end > query_start)
            .collect();
        let alt_set: std::collections::HashSet<u8> = p
            .variant
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
                        if ref_pos <= p.variant_pos && p.variant_pos < ref_pos + len {
                            let off = (p.variant_pos - ref_pos) as usize;
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
            variant_pos: p.variant_pos,
            min_base_quality_at_call: 10,
        });
        let img = render(
            p.win_start,
            width,
            height,
            5,
            &p.img_ref,
            &pileup_reads,
            &kinds,
            &opts,
            ctx,
            42,
        );
        let mut variant_with_call = p.variant.clone();
        if variant_with_call.calls.is_empty() {
            variant_with_call.calls.push(Default::default());
        }
        variant_with_call.calls[0].call_set_name = sample_name.to_string();
        // Channel send only fails if the receiver hung up; that means
        // the worker died and we'll surface its error after join.
        let _ = tx.send((variant_with_call, p.alt_indices.clone(), img));
    });
    drop(tx);
    lap("pass2_render_streamed", &mut stage_t);

    // ---- Wait for inference worker ----
    let total = cv_handle
        .join()
        .map_err(|_| anyhow::anyhow!("inference worker panicked"))??;
    lap("inference_drain", &mut stage_t);

    tracing::info!(emitted = total, "pipeline complete");
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

/// Run direct phasing on a list of called Variant records using the
/// supplied BAM as the read source. Returns a new Vec with phased
/// genotypes + PS tags applied to records that fall in a phasing block.
///
/// For each het SNV, walks reads overlapping the variant position to
/// build a `DeepVariantCall.allele_support_ext` (alt → list of read
/// keys) and a `ref_support_ext` (ref-supporting reads). Indel and
/// homozygous variants are not phased — direct_phasing's
/// `candidate_filter` skips them anyway.
fn phase_called_variants(
    variants: &[Variant],
    bam_path: &std::path::Path,
) -> Result<Vec<Variant>> {
    use dv_core::direct_phasing::{DirectPhasing, DirectPhasingOptions};
    use dv_core::phasing_apply::apply_to_variants;
    use dv_proto::dv::deep_variant_call::{ReadSupport, SupportingReadsExt};
    use dv_proto::dv::DeepVariantCall;
    use noodles::sam::alignment::record::cigar::op::Kind as CigarKind;
    use noodles::sam::alignment::record::QualityScores;

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

    /// Sub-set of fields we need to do per-position read classification.
    struct OwnedRead {
        ref_start: i64,
        cigar: Vec<(char, i64)>,
        seq: Vec<u8>,
        name: String,
        mate: i32,
    }

    let (_h, mut reader) =
        dv_io::bam::open(bam_path).context("open BAM for phasing")?;
    let mut owned: Vec<OwnedRead> = Vec::new();
    for rec in reader.records() {
        let r = rec?;
        let Some(start) = r.alignment_start() else { continue };
        let start_0based = usize::from(start.unwrap()) as i64 - 1;
        let flags = r.flags();
        if flags.is_secondary()
            || flags.is_supplementary()
            || flags.is_unmapped()
            || flags.is_duplicate()
            || flags.is_qc_fail()
        {
            continue;
        }
        if r.mapping_quality().map(|q| q.get()).unwrap_or(255) < 10 {
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
        let seq: Vec<u8> = r
            .sequence()
            .iter()
            .map(|b| b.to_ascii_uppercase())
            .collect();
        let _q: Vec<u8> = r.quality_scores().iter().map(|q| q.unwrap_or(0)).collect();
        let name = r
            .name()
            .map(|n| std::str::from_utf8(n.as_ref()).unwrap_or("").to_string())
            .unwrap_or_default();
        let mate = if flags.is_first_segment() { 1 } else { 2 };
        owned.push(OwnedRead {
            ref_start: start_0based,
            cigar: cigar_owned,
            seq,
            name,
            mate,
        });
    }
    tracing::info!(reads_loaded = owned.len(), "phasing: loaded reads");

    // Walk each candidate's overlapping reads, classify by base at the
    // variant position, build a DeepVariantCall.
    fn read_base_at(r: &OwnedRead, pos: i64) -> Option<u8> {
        let mut ref_pos = r.ref_start;
        let mut read_pos = 0usize;
        for &(op, len) in &r.cigar {
            let len_us = len as usize;
            match op {
                'M' | '=' | 'X' => {
                    if ref_pos <= pos && pos < ref_pos + len {
                        let off = (pos - ref_pos) as usize;
                        return r.seq.get(read_pos + off).copied();
                    }
                    ref_pos += len;
                    read_pos += len_us;
                }
                'I' | 'S' => read_pos += len_us,
                'D' | 'N' => ref_pos += len,
                _ => {}
            }
        }
        None
    }

    // Build a DeepVariantCall for every variant, keeping only those that
    // direct_phasing's CandidateFilter would accept (≥2 called alleles
    // or ≥3 REF reads, no INDELs).
    let mut candidates: Vec<DeepVariantCall> = Vec::new();
    let mut all_read_keys: std::collections::BTreeSet<(String, i32)> =
        std::collections::BTreeSet::new();
    for v in variants {
        // Only het biallelic SNVs are useful — others fall through.
        if v.reference_bases.len() != 1 {
            continue;
        }
        if v.alternate_bases.is_empty() {
            continue;
        }
        if v.alternate_bases.iter().any(|a| a.len() != 1) {
            continue;
        }
        let ref_b = v.reference_bases.as_bytes()[0];
        let mut ref_reads = Vec::new();
        let mut alt_reads: std::collections::HashMap<String, Vec<ReadSupport>> =
            std::collections::HashMap::new();
        for alt in &v.alternate_bases {
            alt_reads.insert(alt.clone(), Vec::new());
        }
        for r in &owned {
            let read_overlaps = {
                let read_len_on_ref: i64 = r
                    .cigar
                    .iter()
                    .filter(|(op, _)| matches!(op, 'M' | '=' | 'X' | 'D' | 'N'))
                    .map(|(_, l)| *l)
                    .sum();
                r.ref_start <= v.start && v.start < r.ref_start + read_len_on_ref
            };
            if !read_overlaps {
                continue;
            }
            let Some(b) = read_base_at(r, v.start) else { continue };
            let b = b.to_ascii_uppercase();
            let key = format!("{}/{}", r.name, r.mate - 1);
            all_read_keys.insert((r.name.clone(), r.mate - 1));
            if b == ref_b {
                let mut rs = ReadSupport::default();
                rs.read_name = key;
                ref_reads.push(rs);
            } else {
                for alt in &v.alternate_bases {
                    if alt.as_bytes()[0].to_ascii_uppercase() == b {
                        let mut rs = ReadSupport::default();
                        rs.read_name = key.clone();
                        alt_reads.get_mut(alt).unwrap().push(rs);
                        break;
                    }
                }
            }
        }
        let mut allele_support_ext = std::collections::BTreeMap::new();
        for (alt, reads) in alt_reads {
            allele_support_ext.insert(alt, SupportingReadsExt { read_infos: reads });
        }
        let dv_call = DeepVariantCall {
            variant: Some(v.clone()),
            allele_support_ext,
            ref_support_ext: Some(SupportingReadsExt { read_infos: ref_reads }),
            ..Default::default()
        };
        candidates.push(dv_call);
    }
    candidates.sort_by_key(|c| c.variant.as_ref().map(|v| v.start).unwrap_or(0));
    tracing::info!(phasing_candidates = candidates.len(), "running direct_phasing");

    let read_list: Vec<(String, i32)> = all_read_keys.into_iter().collect();
    let mut dp = DirectPhasing::new(DirectPhasingOptions::default());
    let _phases = dp.phase_reads(&candidates, &read_list);
    let phased = dp.phased_variants();
    tracing::info!(phased_variants = phased.len(), "direct_phasing complete");

    let out = apply_to_variants(&phased, variants);
    Ok(out)
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
    phase_reads_path: Option<&std::path::Path>,
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
    let mut variants = postprocess::process_cvos_into_variants(cvos, sample_name, &opts);
    tracing::info!(produced = variants.len(), "produced variants");

    // Optional: run direct phasing over the called variants using the
    // input BAM, then apply the resulting `0|1` GTs and PS tags.
    if let Some(bam_path) = phase_reads_path {
        let phased = phase_called_variants(&variants, bam_path)?;
        variants = phased;
    }

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

    // Pinned-batch override: if the model has a fixed N (e.g. 128
    // from `normalize_onnx_pads.py`), use that for back-pressure.
    let batch_size = model.pinned_batch().unwrap_or(batch_size);
    let mut batch_rows: Vec<ExampleRow> = Vec::with_capacity(batch_size);
    let mut flat_buf: Vec<f32> = Vec::with_capacity(batch_size * PIXEL_BYTES);

    // Stage time accumulators (across all batches).
    let mut t_decode_us: u64 = 0;
    let mut t_predict_us: u64 = 0;
    let mut t_write_us: u64 = 0;

    let flush = |rows: &mut Vec<ExampleRow>,
                     buf: &mut Vec<f32>,
                     writer: &mut tfrecord::Writer<Box<dyn std::io::Write>>,
                     total: &mut usize,
                     t_predict_us: &mut u64,
                     t_write_us: &mut u64|
     -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let actual_n = rows.len();
        // Pad partial batches up to the fixed batch size (no-op when
        // the model accepts dynamic batches and the row buffer
        // happens to be full).
        let need = batch_size * PIXEL_BYTES;
        if buf.len() < need {
            buf.resize(need, 0.0);
        }
        let n = batch_size;
        let t = std::time::Instant::now();
        let probs = model.predict_batch(buf, n)?;
        *t_predict_us += t.elapsed().as_micros() as u64;
        anyhow::ensure!(probs.len() == n * 3);
        let _ = actual_n;
        let t = std::time::Instant::now();
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
        *t_write_us += t.elapsed().as_micros() as u64;
        buf.clear();
        Ok(())
    };

    while let Some(rec) = reader.read_record()? {
        let t = std::time::Instant::now();
        let row = parse_example(&rec)?;
        t_decode_us += t.elapsed().as_micros() as u64;
        flat_buf.extend_from_slice(&row.image_f32);
        batch_rows.push(row);
        if batch_rows.len() >= batch_size {
            flush(
                &mut batch_rows,
                &mut flat_buf,
                &mut writer,
                &mut total,
                &mut t_predict_us,
                &mut t_write_us,
            )?;
        }
    }
    flush(
        &mut batch_rows,
        &mut flat_buf,
        &mut writer,
        &mut total,
        &mut t_predict_us,
        &mut t_write_us,
    )?;
    writer.flush()?;
    tracing::info!(
        total,
        decode_ms = t_decode_us / 1000,
        predict_ms = t_predict_us / 1000,
        write_ms = t_write_us / 1000,
        "wrote CallVariantsOutput records"
    );
    Ok(())
}
