# DeepVariant → Rust port — TODO

This file lists the work that's still outstanding, grouped by impact.
A summary of what's already landed is at the top of each section
(checked items). Recent end-to-end performance work — methodology,
per-stage timings, and the C++ comparison — lives in `benchmark.md`.

## Performance (May 2026) — shipped this push

Headline: HG003 chr20 on M2 Max, end-to-end (BAM in, CVO out):

| Configuration | wall |
|---|---|
| Rust port — original baseline | >40 min (never finished) |
| Rust port — `dv pipeline`, NeuralNetwork CoreML, sharded pre-pass2 | **2 m 55 s** |
| Rust port — projected `dv pipeline` + MLProgram batch=128 pinned | est. ~1 m 30 s (not locked in on full chr20 yet) |
| C++ Metal+CoreML+fast-pipeline fork (same machine) | 3 m 06 s |

Smoke (1 Mbp slice, 2879 examples), inference only:

| Backend | predict_ms |
|---|---|
| ORT-CPU baseline | 20,200 |
| CoreML NeuralNetwork (dynamic batch) | 4,118 |
| CoreML MLProgram (dynamic batch — partial-batch fallback regression) | 21,500 |
| **CoreML MLProgram, batch=128 pinned + zero-padded last batch** | **1,275** |

- [x] **Indexed BAM `.bai` query in `dv-io::reads`** — auto-detects
  sibling `.bai`/`.csi`, exposes `for_each_record_in_region`.
  bam_read on the chr20 1 Mbp quick bench: 14 s → 0.4 s.
- [x] **CRAM `.crai` indexed query.** Sibling `.crai` triggers
  `noodles::cram::IndexedReader`. Same auto-detection pattern as
  the BAM path.
- [x] **Sorted-read overlap index** in `dv-cli` make-examples.
  Caches `ref_end` once at BAM-load time + binary-searches a
  ref-start-ordered slice per query, killing the O(reads × queries)
  inner filter in the realigner / pass2 render. Realigner on full
  chr20: 36 min → 42 s. Pass-2 render on full chr20: would-be ~45 min
  → 7 s.
- [x] **Parallel-chunked gzip writer** in `dv-io::tfrecord`
  (`write_records_gz_parallel`). Splits the record stream across
  rayon workers; concatenated gzip streams are valid for any reader.
  pass3_write on full chr20: 297 s → 11 s.
- [x] **CoreML execution provider for ORT on macOS / iOS** — gated
  `#[cfg(any(target_os = "macos", target_os = "ios"))]` in
  `dv-infer::ort`. `MLComputeUnits::All` (GPU + ANE + CPU),
  `SpecializationStrategy::FastPrediction`. Disable via
  `DV_DISABLE_COREML=1` for A/B against pure ORT-CPU.
- [x] **CoreML compiled-model cache** at
  `~/.cache/deepvariant-rs/coreml` (override `DV_COREML_CACHE`).
  First load pays ~20 s compile, subsequent loads skip it.
- [x] **`dv pipeline` subcommand** — in-process ME→CV via a bounded
  `std::sync::mpsc::sync_channel`; rayon par_iter pass-2 streams
  `(variant, alt_indices, image_u8)` tuples into an inference worker
  thread that loads the model and writes CVOs. The Rust analogue of
  the C++ fork's `fast_pipeline` shared-memory bridge, all in a
  single process. Eliminates the intermediate
  `examples.tfrecord.gz` roundtrip and overlaps rendering with
  inference.
- [x] **Region-shard pre-pass2 in `dv pipeline`.** Splits the input
  region into N=perf-cores sub-regions; each shard does its own
  indexed BAM query + allele counter in parallel. Reads spanning
  shard boundaries still contribute to in-shard allele counts but
  are emitted to the merged owned list exactly once (the start-range
  shard owns them). chr20 pre-pass2 wall: 38 s → 7 s.
- [x] **Top-level `./benchmark.sh`** + `scripts/benchmark_rust.sh`,
  with `--quick` (1 Mbp) and `--full` (full chr20) presets. Mirrors
  the C++ fork's `benchmark.md` reproducibility recipe but for the
  Rust port.
- [x] **ONNX MLProgram normalization** (`scripts/normalize_onnx_pads.py`).
  Adds explicit `pads=[0,...]` to Conv / MaxPool / etc. and pins the
  batch dim so CoreML MLProgram (the newer, faster format) accepts
  our tf2onnx export. `OrtBackend` detects the pinned dim via
  `pinned_batch()` and the `dv pipeline` + `dv call-variants`
  workers pad partial batches with zero images and trim the output;
  smoke test predict 4.1 s (NeuralNetwork) → **1.3 s** (MLProgram,
  batch=128 pinned, GPU+ANE on every batch).
- [x] **Default to MLProgram in `dv-infer::ort`** with env-var
  fallback `DV_COREML_FORMAT=NeuralNetwork`. `benchmark.sh` now
  runs the normalisation script after `tf2onnx` so the produced
  `models/wgs/model.onnx` is MLProgram-safe by default.

### Outstanding — finishing the perf pass

Ordered by what unblocks the headline WGS number first.

- [x] **[P0 — blocker] Postprocess panic on full chr20.** Fixed.
  Root cause: at chr20:35167420 the candidate caller emitted a
  malformed variant with a **duplicate alt** (`alternate_bases =
  ["GT","GT"]`) — two distinct `(allele_type, bases)` allele-counter
  keys projecting to the same alt string. Downstream, `to_remove`
  in `get_alt_alleles_to_remove` is a `HashSet` so it dedupes the
  string; the `to_remove.len() == alternate_bases.len()` safety net
  then misfires (1 ≠ 2) and `prune_alleles` strips *both* copies,
  leaving a 0-alt variant → `n_alleles must be >= 2`.
  Fixes (both landed):
  1. **Source:** `variant_calling::candidates_from_counts` now
     collapses alts that project to the same base string, summing
     their read counts so AD/VAF stay consistent. A candidate can
     no longer carry a duplicate alt.
  2. **Defensive:** `postprocess::group_cvos` now keys on
     `(ref_name, start, end, sorted_alts)` instead of just the
     range, so two *different* candidates at the same range (e.g.
     an allele-counter SNV/INS and a realigner-assembled deletion)
     are processed as separate VCF records rather than merged with
     a mismatched canonical alt list.
  Regression tests:
  `variant_calling::tests::duplicate_alt_string_is_collapsed` and
  `postprocess::tests::group_cvos_splits_distinct_alt_sets_at_same_range`.
  Verified end-to-end: full-chr20 `dv pipeline` (208,882 CVOs) →
  `dv postprocess-variants` exits 0, emits 205,198 VCF records, the
  chr20:35167421 locus now produces two valid records
  (`G→GT` insertion + `GT→G` deletion), zero duplicate-alt records
  across the whole VCF.

- [x] **[P1 — measurement] Lock in MLProgram + batch-pinning on
  full chr20.** Measured (clean, uncontended, M2 Max,
  `models/wgs/model.onnx` batch=128 + explicit pads, 208,882 CVOs):
  `pass2_render_streamed` 124.8 s → **106.2 s**, pipeline wall
  2 m 55 s → **~2 m 34 s**, postprocess **1.4 s**, end-to-end
  **~2 m 35 s** (~17 % faster than the C++ fork's 3 m 06 s).
  The earlier "pass2 → ~50 s, e2e ~1 m 30 s" projection was wrong:
  the 3× the inference smoke bench showed does NOT translate,
  because `pass2_render_streamed` overlaps CPU rendering with
  GPU/ANE inference and is **render-bound** at full-chr20 scale —
  faster inference only trims the inference-was-long-pole slice.
  The new long pole is the renderer (106 s). `benchmark.md`
  updated with measured numbers + the honest note.

- [ ] **[P1 — headline number] Run the full WGS end-to-end on this
  M2 Max.** Per-chromosome dispatch (24 `dv pipeline` invocations,
  concat CVOs through `dv postprocess-variants`). Postprocess panic
  unblocked, but **now blocked on data**: the local benchmark set
  is chr20-only
  (`HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam`).
  A real WGS run needs the whole-genome HG003 BAM (~100 GB+),
  not yet downloaded. WGS numbers in `benchmark.md` remain a
  linear projection (~2 h 04 m), not measured.

- [ ] **[P1 — new long pole] Speed up `pass2_render_streamed`
  (106 s).** Now the dominant stage. Candidates: SIMD pileup
  encode, fewer per-pixel allocations in the channel encoders,
  coarser-grained rayon batching to cut task overhead. The model
  backend is no longer the bottleneck here.

- [x] **[P2 — perf, portable] Haplotype cap (≤8 candidates per
  DBG).** Done. Ported the C++ fork's `_MAX_HAPLOTYPES = 8`
  (`realigner.py:783`): the two uncapped `graph.candidate_haplotypes()`
  loops in the dv-cli hot path (`pipeline` + `make-examples`) now
  `.take(MAX_HAPLOTYPES_PER_WINDOW)` after the lexicographic sort.
  The library `realign_window` path already capped (its
  `RealignerOptions.max_haplotypes` default stays 64; only the
  dv-cli inline DBG loop was uncapped). Measured clean full-chr20:
  realigner **35.9 s → 24.6 s (−31.5 %, −11.3 s)**; candidates
  208,882 → 207,938 (−944, −0.45 %, the noisy long tail);
  postprocess clean, 204,254 VCF records, zero dup-alt. Matches
  the fork's "95.7 % of windows already ≤8, no accuracy loss,
  INDEL slightly improved."

- [x] **[P2 — infra] Stable benchmark snapshot per commit.** Done.
  `scripts/benchmark_rust.sh` now captures `git rev-parse --short
  HEAD` (with a `-dirty` suffix when the tree is modified) and, in
  addition to the rolling `benchmark_results.json`, writes an
  immutable `$BENCH_DIR/rust_runs_<sha>.json` per commit. `git_sha`
  and the actual `region` are recorded in the snapshot metadata so
  the perf journey can be graphed over time / regression-checked in
  CI. Clean commits get `rust_runs_<sha>.json`; dirty trees get
  `rust_runs_<sha>-dirty.json` so a committed baseline is never
  clobbered by an ad-hoc run.

## P0 — closes the last accuracy gap on chr20 quickstart

- [x] **Small-model fast path.** Ported `make_small_model_examples.py`
  feature engineering + keras→ONNX export + ORT inference + threshold
  decision. `dv make-examples --small-model` routes biallelic SNVs
  through the small model first; on chr20 quickstart 196 of 364
  candidates take the fast path. End-to-end test in
  `crates/dv-cli/tests/small_model_fast_path.rs`.

- [x] **Wire realigner into `dv make-examples` candidate flow.**
  Now called inline (window selector → de Bruijn assembly →
  `variants_from_haplotype`). Recovers indels that don't appear in any
  single read's CIGAR. Multi-allelic indels in long repeats still
  missing from byte-diff sweep — needs a haplotype-vs-haplotype
  realignment pass (or larger k for the assembler).

## P1 — pileup parity edge cases (currently 100% on chr20:10001019)

- [x] **Alt-aligned pileups (library code).** `dv_core::alt_aligned_pileup`
  ports `trim_cigar`, `trim_read`, `trim_reads`,
  `calculate_alignment_region`, `cigar_ref/read_length` from
  `alt_aligned_pileup_lib.cc`. 18 unit tests cover the upstream
  parameterized test cases. Wiring this re-render path through to the
  pileup builder so channels #9/#10/#20/#21 actually populate is the
  remaining piece (requires plumbing FastPassAligner output back into
  the per-candidate render loop).

- [x] **Run pileup byte-diff on more variants.** Sweep harness now runs
  all 32 norealign upstream examples; current state is 100% on the SNV
  floor (chr20:10001019), 99.06% overall pixel match across 25
  compared records, with 7 multi-allelic / repeat-region indels still
  missing from our candidate set (deeper realigner work).

- [x] **Indel-anchor edge cases.** Six unit tests in
  `pileup_image::layout::tests` cover: leading H+S CIGAR combo, long
  insertions (50I), long deletions (50D), INSERT at contig start (no
  anchor at `-1`), DELETE at contig start, and INSERT as first read op
  when `ref_start > 0` (anchor IS painted, not skipped). The existing
  `ref_i > 0` and `read_i > 0 && ref_i > 0` short-circuits hold for
  every case.

## P2 — channel coverage (16 of 29 channels ported)

These channels aren't in the WGS default set so they don't affect chr20
parity, but they're needed for WES/PacBio/ONT model variants:

Encoder logic for the WGS-extended channels is now ported as pure
functions in `pileup_image::channels`. Wiring them into the pileup
builder (BAM aux-tag plumbing, alt-aligned re-rendering) is what's
left:

- [x] CH_HAPLOTYPE_TAG (#7) — encoder done; HP-tag now parsed from BAM
      aux fields in `dv make-examples` and threaded through to the
      pileup builder.
- [x] CH_DIFF_CHANNELS_ALTERNATE_ALLELE_{1,2} (#9, #10) — library-level
      support landed. `ChannelKind::DiffChannelsAlternateAllele{1,2}` plus
      `from_proto_index(9|10)` plumb the routing; the per-pixel encoder
      reuses `BaseDiffersFromRef` against an alt-haplotype reference.
      `alt_aligned_pileup::build_alt_haplotype_ref` splices the alt allele
      into the image window; `realign_to_alt_haplotype` uses the SSW
      aligner (`realigner::ssw`) to produce the realigned CIGAR + ref_start.
      Tests at `crates/dv-core/src/alt_aligned_pileup.rs` and integration
      tests in `pileup_image::layout::tests`. CLI wiring (an
      `--alt-aligned-pileup` flag) is still off by default — the WGS model
      doesn't use these channels and there's no upstream alt-aligned
      fixture to byte-diff against; we'll wire it when a consuming model
      lands.
- [x] CH_BASE_CHANNELS_ALTERNATE_ALLELE_{1,2} (#20, #21) — same machinery,
      `alt_aligned_underlying()` routes these through the `ReadBase`
      encoder. Same wiring caveat.
- [x] CH_BASE_METHYLATION (#23), CH_BASE_6MA (#24) — encoder done; needs
      MM/ML BAM aux-tag parsing
- [x] CH_HOMOPOLYMER_WEIGHTED (#17), CH_IS_HOMOPOLYMER (#16) — done & wireable
- [x] CH_HOMOPOLYMER_INSERTION_QUALITY (#28),
      CH_HOMOPOLYMER_DELETION_QUALITY (#29) — encoder done; needs `tp`
      Ultima-tag parsing
- [x] CH_SUPPLEMENTARY_ALIGNMENT (#26) — encoder done; needs SAM flag wiring
- [x] CH_ALLELE_SAMPLE_PROBABILITY (#27) — encoder done; needs allele-support
      counts threaded from candidate to pileup row
- [x] CH_READ_SUPPORTS_VARIANT_FUZZY (#25) — encoder done. Classifier
      (`fuzzy_support::classify_read_support`) ported as a self-contained
      helper that takes plain Rust types (variant_alts, allele_support,
      alt_phases, read_key, read_hp). 12 unit tests; needs HP/ALT_PS
      plumbing through the candidate path to wire end-to-end.

## P3 — broader coverage / robustness

- [ ] **Generalization tests.** Run our pipeline on chr1, chr22, X
  segments and confirm parity. Run on a different model (WES) and
  confirm. Add to CI.

- [x] **`direct_phasing.cc` port** (1006 LOC). DP-based read phasing
  graph with backtracking and broken-path handling. 25 unit tests
  cover the upstream `*_test.cc` cases. `dv_core::direct_phasing`.

- [x] **`methylation_aware_phasing.cc` port** (483 LOC). Wilcoxon
  rank-sum test (with Abramowitz–Stegun erf), informative-site
  filter, per-read voting, iterative phasing loop. 11 unit tests.
  `dv_core::methylation_aware_phasing`.

- [x] **Multi-shard / multi-region parallelism.** rayon-parallel
  per-candidate image rendering in `dv make-examples` (Pass 1
  decisions / Pass 2 parallel render / Pass 3 sequential write). New
  `tests/make_examples_determinism.rs` confirms parallel and serial
  shards are byte-identical (BTreeMap-encoded tf.Example). Upstream
  GNU-parallel-shard-by-region remains as a separate orchestration
  layer; can be added at the dv-cli level if needed.

- [x] **CRAM support.** `noodles-cram` enabled in `dv-io`'s feature
  gate; `dv-io::reads::open` dispatches to BAM or CRAM by file extension
  with a shared `for_each_record(&dyn Record, ...)` callback API. CRAM
  decompression goes through a `noodles::fasta::Repository` with an
  `IndexedReader` adapter for sequence reconstruction. `dv make-examples
  --reads file.cram --ref-fasta ref.fasta` works end-to-end on the
  small test fixture. CRAM's C-only deps (bzip2-sys, lzma-sys) are
  isolated to the `dv-io` crate's local feature so `dv-core` and
  `dv-wasm` still build cleanly for `wasm32-unknown-unknown`.

- [x] **CRAM robustness — noodles-cram panic on real WGS CRAMs.**
  Fixed. `noodles-cram 0.78.0` `.expect("invalid reference base")`'d
  at `record/sequence/iter.rs:118` whenever a read carried a
  Substitution feature over a non-ACGTN reference base — IUPAC
  ambiguity codes in GRCh38 decoy/HLA, or N in gaps (its
  `Base::try_from` only accepts A/C/G/T/N). samtools/htslib reads
  the same file fine. Fixed by vendoring noodles-cram 0.78.0 to
  `third_party/noodles-cram/` and applying the **exact upstream
  0.93.0 change** — `Base::try_from(ref).unwrap_or(Base::N)` — via
  `[patch.crates-io]` (one functional line; avoids a wide
  meta-crate bump since 0.93 also changed internal APIs).
  Validated: `dv pipeline` directly on the NA06985 chr22 CRAM now
  exits 0 and the resulting VCF is **byte-identical** to the
  samtools-transcode→BAM path (134,610 records, `diff` exit 0).
  CRAM-direct chr22 wall 141 s vs BAM 113 s — CRAM trades ~28 s of
  reference-reconstruction CPU for needing no extra disk (the
  read-decode `shard_pre_pass2` stage went 5 s → 48 s; everything
  downstream is identical).

## P4 — I/O polish

- [x] **TF made optional, ORT is now the default backend.** `cargo build`
  produces a `dv` that links neither libtensorflow nor libonnxruntime
  (ORT loads via `dlopen`). TF stays available for regression testing
  via `--features tf`. Backend selection is automatic from the
  `--checkpoint` path (`.onnx` → ORT, directory → SavedModel/TF).
  Verified by `tests/call_variants_backend_parity.rs`: 24 chr20 CVOs,
  max prob delta `1.7e-6`, post-processed VCFs byte-identical.

- [x] **Bundled ONNX Runtime fetcher.** `scripts/fetch_onnxruntime.sh`
  downloads the right pre-built tarball from the official Microsoft
  release page into `models/lib/` (Linux x86_64/aarch64, macOS
  arm64/x86_64). Idempotent. `dv-cli/build.rs` emits a one-line
  cargo warning on a fresh clone if the lib is missing, pointing at
  the script. No compilation needed.

- [x] **Auto `.tbi` index emission** alongside `.vcf.gz` outputs via
  `noodles-tabix`. `dv postprocess-variants` now writes
  `<output>.vcf.gz.tbi` (and `.g.vcf.gz.tbi`) automatically.

- [x] **`make_examples_call_variant_outputs.tfrecord` output.** Done
  alongside the small-model fast path — `dv make-examples
  --small-model-cvo <path>` writes the CVO shard for accepted
  candidates with `MID=small_model`, ready to be merged with the
  big-model CVO via `dv postprocess-variants --small-model-cvo`.

## P5 — Cross-compile (M4)

- [x] **WASM build target — three layers, end-to-end ORT inference.**
  - `cargo wasm-test` runs **279 unit tests** through wasmtime via
    `wasm32-wasip1` (alias in `.cargo/config.toml`): all of dv-core's
    pure-compute kernels (allelecounter, variant_calling, every channel
    encoder, realigner, direct_phasing, methylation_aware_phasing, vcf
    math, the new alt-aligned primitives) plus dv-wasm.
  - `wasm-node-test/` — Node + onnxruntime-node + dv-wasm via
    wasm-bindgen. 32/32 examples on the chr20 norealign fixture,
    Δmax = 0.0e+0 vs native (bit-exact).
  - `wasm-browser-test/` — Headless Chromium via Playwright +
    onnxruntime-web + dv-wasm. 32/32, Δmax ≈ 2.4e-7 (single-precision
    FMA rounding between onnxruntime-web's wasm SIMD kernels and
    libonnxruntime's CPU kernels — top-class predictions identical).
  - Same `dv-wasm` crate drives all three; ort upgraded rc.10 → rc.12
    (wasm-friendly), libonnxruntime bumped to 1.24.4 (matches rc.12's
    expected ABI; older libs deadlocked Session::commit_from_file).

- [ ] **iOS xcframework.** CoreML now works on macOS through the
  ORT CoreML execution provider (`dv-infer::ort`, `cfg(target_os =
  "ios"/"macos")`). What's left: build `libonnxruntime` for the
  iOS triples (`aarch64-apple-ios`, `aarch64-apple-ios-sim`),
  package as an xcframework, ship a thin Swift wrapper around
  `dv-cli`'s pipeline subcommand. No separate `dv-infer::coreml`
  backend is needed — ORT's CoreML EP handles it.

- [ ] **Android AAR.** TFLite conversion; new `dv-infer::tflite`
  backend; JNI bindings.

## P6 — out of scope for v1 (revisit later)

- DeepSomatic mode (somatic variant calling).
- DeepTrio mode (parents+child joint calling).
- Pangenome-aware mode (gbwt/gbwtgraph/sdsl/libdivsufsort).
- ~~`--stream_examples` Boost shared-memory pipeline~~ — superseded
  by the Rust `dv pipeline` in-process channel; if a multi-machine
  / cross-language streaming endpoint is ever needed, gRPC over the
  same channel format is the natural fit, not Boost SHM.
- Training pipeline (we're inference-only).

## Stretch / nice-to-have

- [ ] Replace classical SW with `block-aligner` SIMD for ~10× realigner
      speedup.
- [ ] Cargo-publish the crates with stable APIs (`dv-proto`,
      `dv-io`, `dv-core`, `dv-infer`, `dv-cli`).
- [ ] More end-to-end fixtures: HG002, HG003, HG004 from GIAB.
- [ ] Property-based tests via `proptest` on the genomic primitives
      (cigar walking, range arithmetic, etc.).
- [ ] Benchmarks (`criterion`) for the hot paths: pileup rendering,
      allelecounter, realigner SSW.
- [ ] Make `dv-cli` a single statically-linked binary (currently
      depends on dynamic libtensorflow / libonnxruntime).
- [ ] Documentation site (mdbook).
