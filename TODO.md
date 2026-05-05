# DeepVariant → Rust port — TODO

What's already shipped is in `STATUS.md`. This file lists the work
that's still outstanding, grouped by impact.

## P0 — closes the last accuracy gap on chr20 quickstart

- [ ] **Small-model fast path.** Upstream routes ~60 of 84 chr20
  candidates through a lightweight Keras model (`/opt/smallmodels/wgs/
  model.keras`) tagged `MID=small_model`, falling back to the large
  WGS CNN only for hard cases. We currently send everything through
  the large model. PASS classification still matches but PL/GQ values
  differ for those records. Needed: load the small model (likely via
  `tract`/`burn` or another keras→ort path), implement the fast-path
  decision logic from `make_examples_core.py`.

- [x] **Wire realigner into `dv make-examples` candidate flow.**
  Now called inline (window selector → de Bruijn assembly →
  `variants_from_haplotype`). Recovers indels that don't appear in any
  single read's CIGAR. Multi-allelic indels in long repeats still
  missing from byte-diff sweep — needs a haplotype-vs-haplotype
  realignment pass (or larger k for the assembler).

## P1 — pileup parity edge cases (currently 100% on chr20:10001019)

- [ ] **Alt-aligned pileups.** When `--alt_aligned_pileup=diff_channels`
  or `=base_channels`, upstream re-renders the pileup with reads
  aligned to the alt haplotype and stacks 2 extra channels (#9, #10
  for diff; #20, #21 for base). Adds two channels to the model input.

- [x] **Run pileup byte-diff on more variants.** Sweep harness now runs
  all 32 norealign upstream examples; current state is 100% on the SNV
  floor (chr20:10001019), 99.06% overall pixel match across 25
  compared records, with 7 multi-allelic / repeat-region indels still
  missing from our candidate set (deeper realigner work).

- [ ] **Indel-anchor edge cases.** The DELETE off-by-one + SOFT_CLIP
  fix landed on chr20 SNV. Verify it holds for reads with a leading
  hard-clip+soft-clip combo, multi-base indels, and reads aligned at
  contig start (where `ref_i == 0` short-circuits anchor painting).

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
- [ ] CH_DIFF_CHANNELS_ALTERNATE_ALLELE_{1,2} (#9, #10) — needs alt-aligned
      pileup re-render (re-aligns reads to alt haplotype, stacks 2 channels)
- [ ] CH_BASE_CHANNELS_ALTERNATE_ALLELE_{1,2} (#20, #21) — same alt-aligned
      dependency
- [x] CH_BASE_METHYLATION (#23), CH_BASE_6MA (#24) — encoder done; needs
      MM/ML BAM aux-tag parsing
- [x] CH_HOMOPOLYMER_WEIGHTED (#17), CH_IS_HOMOPOLYMER (#16) — done & wireable
- [x] CH_HOMOPOLYMER_INSERTION_QUALITY (#28),
      CH_HOMOPOLYMER_DELETION_QUALITY (#29) — encoder done; needs `tp`
      Ultima-tag parsing
- [x] CH_SUPPLEMENTARY_ALIGNMENT (#26) — encoder done; needs SAM flag wiring
- [x] CH_ALLELE_SAMPLE_PROBABILITY (#27) — encoder done; needs allele-support
      counts threaded from candidate to pileup row
- [x] CH_READ_SUPPORTS_VARIANT_FUZZY (#25) — encoder done; needs HP/PS/ALT_PS
      classifier (computes the support code from phasing info)

## P3 — broader coverage / robustness

- [ ] **Generalization tests.** Run our pipeline on chr1, chr22, X
  segments and confirm parity. Run on a different model (WES) and
  confirm. Add to CI.

- [ ] **`direct_phasing.cc` port** (~38k bytes). Read-based phasing.
  Needed for `MID=phased` annotations and the gVCF `PS` tag.

- [ ] **`methylation_aware_phasing.cc` port.** Methylation-tag phasing.

- [ ] **Multi-shard / multi-region parallelism.** Use `rayon` to run
  candidate calling per-region in parallel. Upstream uses GNU
  `parallel` to shard `make_examples` over CPU cores.

- [ ] **CRAM support.** `noodles-cram` instead of just BAM.

## P4 — I/O polish

- [x] **TF made optional, ORT is now the default backend.** `cargo build`
  produces a `dv` that links neither libtensorflow nor libonnxruntime
  (ORT loads via `dlopen`). TF stays available for regression testing
  via `--features tf`. Backend selection is automatic from the
  `--checkpoint` path (`.onnx` → ORT, directory → SavedModel/TF).
  Verified by `tests/call_variants_backend_parity.rs`: 24 chr20 CVOs,
  max prob delta `1.7e-6`, post-processed VCFs byte-identical.

- [x] **Auto `.tbi` index emission** alongside `.vcf.gz` outputs via
  `noodles-tabix`. `dv postprocess-variants` now writes
  `<output>.vcf.gz.tbi` (and `.g.vcf.gz.tbi`) automatically.

- [ ] **`make_examples_call_variant_outputs.tfrecord` output.** When
  the small-model fast path is implemented, also emit the small-model
  CVO shard upstream produces.

## P5 — Cross-compile (M4)

- [ ] **WASM build target.** Blocked on installing `rust-wasm` Arch
  package (or `rustup target add wasm32-unknown-unknown`). Once
  installed: `cargo build --target wasm32-unknown-unknown -p dv-core`
  should work; wasm-bindgen wrapper for inference; `ort` already has a
  wasm-bindgen feature.

- [ ] **iOS xcframework.** CoreML conversion of the SavedModel; new
  `dv-infer::coreml` backend; static lib + Swift bindings.

- [ ] **Android AAR.** TFLite conversion; new `dv-infer::tflite`
  backend; JNI bindings.

## P6 — out of scope for v1 (revisit later)

- DeepSomatic mode (somatic variant calling).
- DeepTrio mode (parents+child joint calling).
- Pangenome-aware mode (gbwt/gbwtgraph/sdsl/libdivsufsort).
- `--stream_examples` Boost shared-memory pipeline.
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
