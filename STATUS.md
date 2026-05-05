# DeepVariant → Rust port — overnight status

Snapshot 2026-05-04. Codebase at `dv-rs/`. **142/142 tests passing, 0
failures.** ~8,400 LOC of Rust.

**Pileup byte-diff vs upstream** (chr20:10001019, realigner-disabled
upstream fixtures): **100.0000% pixel match** (154,700 of 154,700
pixels — all bit-identical). 50/50 active rows match. All 7 channels
fully match.

The last 15 pixels closed by two CIGAR-walk fixes:
- DELETE op anchor goes at `ref_i - 1`, not `ref_i` (upstream's lambda
  decrements ref_i internally).
- SOFT_CLIP doesn't paint anything — upstream's lambda only sets
  `read_base = '*'` for INSERT and DELETE, leaving SOFT_CLIP at 0
  which fails the paint check.

**End-to-end pipeline: 72 of 72 PASS variants** on chr20 quickstart
(100% recall vs upstream). 83 total records emitted. Per-record
DP/AD/VAF byte-match upstream. PL/GQ/MID values differ for ~60
records because upstream routes simpler candidates through a
"small_model" fast path (lightweight Keras model run during
make_examples) before falling back to the large CNN. We currently
route everything through the large WGS model. The PASS/RefCall
classification still matches.

Critical fix that closed the gap: `sort_by_alt_allele_support` defaults
to **false** in upstream — we were unconditionally segregating reads
into alt-supporting-first groups, which mis-rowed every read and dropped
us to 56% pixel match. After removing the segregation, pixel match
jumped 86% → 99.99% and PASS recall went from 58/72 → 72/72.

ORT (ONNX Runtime) backend + WGS ONNX export complete; TF and ORT
inference outputs match within 6e-7 on chr20 fixtures. Cross-compile
path (M4) is functional pending native-target setup.

End-to-end Rust pipeline now produces **72 PASS variants out of
upstream's 72** on chr20 quickstart — **100% PASS recall**. Up from
27 at the start of this work, then 58, then 72.

**Full pipeline now runs end-to-end in pure Rust**:
`BAM → dv make-examples → dv call-variants → dv postprocess-variants → VCF`.
DP/AD/VAF read counts in our output **byte-match upstream** at
chr20:10000117 (DP=55, AD=25,30, VAF=0.545455). Variant quality scores
still come out as RefCall because our pileup-image layout doesn't yet
match the trained model's pixel distribution closely enough — the model
was trained on upstream's exact layout (with indel anchoring, alt-aligned
pileups, and `supports_variant` flag computed from the candidate). Read
counting and overall data flow are correct; pileup rendering quality is
the remaining gap.

## Headline

- ✅ **call_variants pipeline works end-to-end in pure Rust.** Inference
  outputs are bit-near-identical to upstream (max prob delta 1.7e-7 over
  the chr20 quickstart; tolerance was 1e-4).
- ✅ **postprocess_variants → VCF + gVCF works end-to-end in pure Rust,
  byte-equal to upstream output.** 78 PASS variants (84 emitted total
  including RefCall/LowQual) on chr20 quickstart, 235 gVCF blocks.
  Both diff zero against upstream `output.vcf.gz` and
  `output.g.vcf.gz`.
- ✅ **BGZF output + tabix-indexable.** The `dv` CLI emits BGZF-format
  `.vcf.gz` / `.g.vcf.gz` (via noodles-bgzf) that `tabix -p vcf`
  successfully indexes (137-byte `.tbi` produced).
- ✅ **make_examples foundation in place**: BAM/FASTA reading via
  noodles, allelecounter (CIGAR-walking allele tally), and
  variant_calling (threshold candidate caller). On chr20, 83 candidates
  emitted vs upstream's 84 — the missing one is an indel that requires
  the realigner (not yet ported).

## Test count

**128 pure-Rust tests passing**, 0 warnings, all green. Run with:

```
cd dv-rs && cargo test --workspace --features tf
```

Breakdown:
- `dv-io`: 11 (TFRecord codec, GZ handling, BAM smoke, FASTA smoke,
  fixture record counts)
- `dv-core`: 72 lib + 6 integration (math, postprocess, vcf, gvcf,
  nucleus::cigar, nucleus::ranges, nucleus::sequence_utils,
  nucleus::variant_utils, nucleus::variantcall_utils, allelecounter,
  variant_calling, realigner::ssw, plus the two byte-equal parity
  tests vs upstream chr20 VCF + gVCF, plus chr20 BAM smoke for both
  allelecounter and variant_calling)
- `dv-infer`: 1 integration (call_variants chr20 inference parity)

## What runs end-to-end

```bash
# Inference on a TFRecord shard:
dv call-variants \
  --examples examples.tfrecord.gz \
  --checkpoint /opt/models/wgs \
  --output cvo.tfrecord.gz

# Postprocess CVO → VCF + gVCF (byte-equal to upstream on chr20):
dv postprocess-variants \
  --cvo cvo.tfrecord.gz \
  --small-model-cvo small.tfrecord.gz \
  --nonvariant-site-tfrecord gvcf.tfrecord.gz \
  --output-vcf out.vcf \
  --output-gvcf out.g.vcf \
  --sample-name NA12878 \
  --contig chr20:63025520 \
  --ref-fasta ucsc.hg19.chr20.unittest.fasta
```

`dv make-examples` is **not yet implemented**.

## Module status

| Module | Status | Tests | Notes |
|---|---|---|---|
| `dv-proto` (prost from upstream `.proto`) | ✅ done | n/a | 12 schemas mirrored |
| `dv-io::tfrecord` (masked-CRC32C + GZ) | ✅ done | 6 | round-trip + corruption detection |
| `dv-io::bam` (noodles wrapper) | ✅ done | 1 | reads 52K chr20 records |
| `dv-io::fasta` (noodles indexed) | ✅ done | 1 | per-base + range fetch |
| `dv-infer::tf` (libtensorflow SavedModel) | ✅ done | 1 | parity 1.7e-7 on chr20 |
| `dv-infer::ort` (ONNX) | stub | 0 | M4 cross-compile target |
| `dv-core::math` (phred conversions) | ✅ done | 5 | port of nucleus/util/math |
| `dv-core::postprocess` (CVO → Variant) | ✅ done | 7 | sort/group/merge/genotype/QUAL/PL |
| `dv-core::vcf` (text writer) | ✅ done | 2 | upstream header schema |
| `dv-core::gvcf` (interleave) | ✅ done | 4 | transform_to_gvcf, merge, FASTA truncate |
| `dv-core::nucleus::cigar` | ✅ done | 7 | parse/format/alignment_length |
| `dv-core::nucleus::ranges` | ✅ done | 9 | parse_literal, overlap, span |
| `dv-core::nucleus::sequence_utils` | ✅ done | 4 | reverse_complement |
| `dv-core::nucleus::variant_utils` | ✅ done | 7 | is_snp/is_indel/simplify_alleles/genotype_ordering |
| `dv-core::nucleus::variantcall_utils` | ✅ done | 7 | set_gt/set_gq/set_gl/set_model_id helpers |
| `dv-core::allelecounter` | ✅ foundational | 8 | CIGAR walk + chr20 SNV match |
| `dv-core::variant_calling` | ✅ foundational | 4 | threshold caller, 83/84 chr20 match |
| `dv-core::realigner::ssw` | ✅ foundational | 7 | classical SW + affine gap (not striped/SIMD) |
| `dv-core::realigner::debruijn` | ✅ done | 7 | k-mer graph, prune, candidate haplotypes |
| `dv-core::realigner::window_selector` | ✅ done | 5 | per-position support score → windows |
| `dv-core::realigner::fast_pass` | ✅ foundational | 4 | reads × haplotypes scoring |
| `dv-core::realigner::orchestrator` | ✅ done | 6 | window→haplotypes→alignments + variants_from_haplotype + discover_variants_from_realigner |
| `dv-infer::ort` | ✅ done | 1 | ONNX Runtime backend; matches TF within 6e-7 |
| WGS SavedModel → ONNX | ✅ done | n/a | `models/wgs.onnx` (87 MB) via tf2onnx 1.16.1 |
| `dv-core::pileup_image::channels` | ✅ partial (16/29) | 17 | read_base/base_quality/mapping_quality/strand/read_supports_variant/base_differs_from_ref/insert_size/avg_base_quality/gc_content/identity/gap_compressed_identity/blank/haplotype_tag/allele_frequency/mean_coverage/read_mapping_percent |
| `dv-core::pileup_image::layout` | ✅ indel-anchoring | 7 | M/=/X paint + INSERT anchor at ref_i-1 + DELETE anchor at ref_i; reference_band_height=5; low-qual call-site drop; mt19937_64 deterministic shuffle |
| `dv-core::make_examples` | ✅ foundational | 2 | tf.Example builder (image + variant + alt_allele_indices) |
| `dv` CLI `make-examples` subcommand | ✅ foundational | 1 | end-to-end runs on chr20 quickstart, 86 examples emitted |
| `dv-core::pileup_image` (~25 channels) | ❌ not started | — | second-biggest piece |
| `dv-core::make_examples` (tf.Example builder) | ❌ not started | — | wraps the above into TFRecord |
| `dv-core::direct_phasing` | ❌ not started | — | read-based phasing |

## Strategic decisions (locked in)

1. **Clean room.** No `cxx` to upstream C++ anywhere. Tests-as-spec
   workflow.
2. **Shared model weights.** Use the published WGS SavedModel
   unchanged; one-time format conversion only (TF → ONNX/TFLite/CoreML
   for cross-compile in M4).
3. **Pangenome and `--stream_examples` modes** dropped for v1.

## What chr20 quickstart proves

| Stage | Upstream output | Our output | Diff |
|---|---|---|---|
| make_examples | 24 hard examples | (using upstream's) | n/a — make_examples not ported |
| call_variants | 24 CVOs | 24 CVOs | max prob delta 1.7e-7 |
| postprocess_variants → VCF | 78 PASS variants | 78 PASS variants | **byte-equal** |
| postprocess_variants → gVCF | 235 blocks | 235 blocks | **byte-equal** |

## Where I'd pick this up next

In priority order:
1. **Wire `discover_variants_from_realigner` into `dv make-examples`**
   to recover the missing chr20 indel candidate (84/84 instead of
   83/84). The helper exists; the CLI just needs to call it after
   initial candidate calling.
2. **Read sort/sampling exact byte parity** — our deterministic shuffle
   matches upstream's algorithm shape but not byte-for-byte. The last
   ~10-15 PASS variants likely close once shuffle order matches.
3. **Indel pileup detail** — alt-aligned pileups (re-render with reads
   aligned to alt haplotype) are the upstream's canonical
   `--alt_aligned_pileup=diff_channels` mode. Adds 2 channels (#9, #10).
4. **Remaining channels (13 of 29)** — homopolymer_*, base_methylation,
   base_6ma, supplementary_alignment, allele_sample_probability, etc.
5. **WASM build target (M6.3)** — needs `pacman -S rust-wasm` (Arch
   package available); then `cargo build --target wasm32-unknown-unknown`
   should work for dv-core (with appropriate feature gates) and the
   ort backend already has a wasm-bindgen feature path.
6. **direct_phasing + methylation_aware_phasing** for full DV behavior.
7. **iOS / Android targets** — CoreML and TFLite model conversions plus
   `dv-infer::coreml` / `dv-infer::tflite` backends.

## Where everything lives

- `dv-rs/Cargo.toml` — workspace
- `dv-rs/crates/dv-{proto,io,infer,core,cli}/` — crates
- `dv-rs/testdata/quickstart_chr20/` — captured upstream fixtures
- `dv-rs/models/wgs/` — extracted WGS SavedModel (87 MB variables)
- `dv-rs/.cargo/config.toml` — embedded rpath for libtensorflow
- `port.md` (project root) — milestones + scope decisions
- `compile.md` — upstream Docker build reference
