# DeepVariant → Rust port

Goal: a Rust reimplementation of upstream DeepVariant r1.10.0 that
cross-compiles to Linux/macOS native, WASM, iOS, and Android from a single
codebase. Behavioral parity with upstream is the bar.

## Strategic decisions

- **Clean-room reimplementation.** No `cxx` bridge to upstream C++ anywhere
  (production or test). The Rust port is written from public specs, the
  `.proto` schemas, and the **upstream test suite as the parity oracle**.
  We do not lift C++ source verbatim.
- **Same model weights.** Use the published SavedModel artifacts unchanged;
  the only transformation is one-time format conversion (TF SavedModel →
  ONNX/TFLite/CoreML). Verify converted outputs against the original on the
  quickstart inputs.
- **Tests-as-spec workflow.** For each module: (1) copy upstream test
  fixtures (input data + expected outputs) verbatim from `testdata/`,
  (2) reauthor the test in Rust against the Rust API surface, (3) implement
  to pass. Fixtures are the contract; test logic is reauthored.
- **Standalone codebase** (sibling `dv-rs/`), not a fork. Confirmed by the
  clean-room directive.
- **Drop pangenome-aware mode** (gbwt/gbwtgraph/sdsl/libdivsufsort/
  libhandlegraph) for v1. Re-add later if needed.
- **Drop `--stream_examples` / `fast_pipeline`** — Boost.Interprocess shared
  memory doesn't fit WASM/iOS sandboxes. Default disk-TFRecord path covers
  the common case.

## Test-suite scope

Upstream has **88 Python tests + 47 C++ tests** under `deepvariant/` and
`third_party/nucleus/`. Categorize before porting:

- **Port (core parity)**: `allelecounter_test.cc`, `variant_calling_test.{cc,py}`,
  `pileup_image_native_test.cc`, `pileup_image_test.py`, `pileup_channel_lib_test.cc`,
  `make_examples_native_test.cc`, `make_examples_core_test.py`, `make_examples_test.py`,
  `realigner_test.py`, `fast_pass_aligner_test.cc`, `ssw_test.cc`, `ssw_*_test.py`,
  `window_selector_test.py`, `direct_phasing_test.cc`, `methylation_aware_phasing_test.cc`,
  `postprocess_variants_test.{cc,py}`, `haplotypes_test.py`, `merge_phased_reads_test.cc`,
  `alt_aligned_pileup_lib_test.cc`, `call_variants_test.py`, `dv_vcf_constants_test.py`,
  `variant_caller_test.py`, `very_sensitive_caller_test.py`, `allele_frequency_test.py`,
  `vcf_stats_test.py`, `exclude_contigs_test.py`, `calling_regions_utils_test.py`,
  `dv_utils_test.py`, `utils_test.cc`, `sampling_util_test.cc`,
  `distribution_functor_test.cc`.
- **Port (I/O parity)**: `sam_reader_test.cc`, `sam_test.py`, `vcf_reader_test.cc`,
  `vcf_writer_test.cc`, `vcf_test.py`, `vcf_roundtrip_test.cc`, `reference_test.cc`,
  `fasta_test.py`, `bed_*test*`, `fastq_*test*`, `gff_*test*`, `tabix_*test*`,
  `tfrecord_*test*`, `merge_variants_test.cc`, `variant_reader_test.cc`,
  `text_io_test.cc`, `reader_base_test.cc`, `sharded_file_utils_test.py`,
  `genomics_reader_test.py`, `genomics_io_noplugin_test.py`.
- **Port (utilities)**: `cigar_test.py`, `ranges_test.py`, `sequence_utils_test.py`,
  `struct_utils_test.py`, `variantcall_utils_test.py`, `variant_utils_test.py`,
  `vcf_constants_test.py`, `genomics_math_test.py`, `math_test.cc`, `samplers_test.cc`,
  `errors_test.py`, `vendor/timer_test.py`, `nucleus/util/utils_test.{cc,py}`.
- **Drop (training)**: `train_test.py`, `keras_modeling_test.py`, `data_providers_test.py`,
  `small_model/*_test.py`, `labeler/*_test.py` (training labels).
- **Drop (workflow / vis / env)**: `dashboard_utils_test.py`, `runtime_by_region_vis_test.py`,
  `vcf_stats_vis_test.py`, `vis_test.py`, `show_examples_test.py`,
  `vcf_candidate_importer_test.py`, `environment_tests/*`, `*_smoke_test*`,
  `*_implementation_test*`, `dv_utils_using_clif_test.py`, `resources_test.py`.
- **Drop (DeepSomatic / DeepTrio variants)**: `make_examples_somatic_test.py`,
  `variant_caller_trio_test.py`, `very_sensitive_caller_trio_test.py`,
  `variant_calling_multisample_*test*`, `variant_calling_multisample_somatic_test.cc`.
- **Drop (CLIF / wrap shims)**: `*_wrap_test.py`, `statusor_examples_test.{cc,py}`,
  `gunit_extras_test.cc`, `protobuf_implementation_test.py`, `tensorflow_smoke_test.py`,
  `test_utils_test.py`, `dv_utils_using_clif_test.py`. (We test the Rust API directly,
  not pybind shims that don't exist.)
- **Drop (pangenome)**: anything that imports `gbz_reader` or
  `make_examples_pangenome_aware_dv`.

After culls, ~70 test files form the parity contract.

## Dependency swap-out table

| Upstream | Rust replacement | Notes |
|---|---|---|
| htslib (C, autoconf) — BAM/CRAM/VCF/FASTA/BED/FASTQ | **`noodles`** (pure Rust) | Cross-compiles trivially. `rust-htslib` is fallback (FFI) if a feature is missing. |
| TensorFlow runtime (full) | Backend trait. `tensorflow` crate (Linux server), **`ort`** / ONNX Runtime (everywhere incl. WASM via wasm-bindgen), **TFLite** (Android), **CoreML** (iOS) | One-time SavedModel → ONNX/TFLite/CoreML conversion at build time. |
| protobuf 21.9 | **`prost`** + `prost-build` | Regenerate from upstream `.proto` files. |
| libssw (Striped Smith-Waterman, C++) | **`block-aligner`** or port the ~1k-line SSW to Rust | Small; clean-room is feasible. |
| abseil (via TF) | Rust std + `tokio` + `parking_lot` | Drop. |
| Boost.Interprocess + Boost.Process | (none) | Only used by `fast_pipeline` / `--stream_examples`; mode dropped. |
| GNU `parallel` + Python multiprocessing | **`rayon`** (CPU-parallel shard iteration) | In-process; no fork. |
| pysam, samtools, bcftools (shell-out) | Folded into `noodles` (tabix indexing, BCF concat) | No external binaries needed. |
| CLIF | (none) | Drop — legacy. |
| CCTZ, glog | `chrono`, `tracing` | Drop. |
| gbwt / gbwtgraph / sdsl_lite / libdivsufsort / libhandlegraph | (none in v1) | Pangenome mode dropped. |

## Crate layout

```
dv-rs/
├── Cargo.toml              # workspace
├── crates/
│   ├── dv-proto/           # prost-generated from upstream .proto
│   ├── dv-io/              # noodles BAM/VCF/FASTA wrappers + TFRecord codec
│   ├── dv-infer/           # inference backend trait + impls (tf, ort, tflite, coreml)
│   ├── dv-core/            # genomics compute: pileup, allele_count, candidate_call,
│   │                       #   realigner, image_encode, phasing, postprocess
│   └── dv-cli/             # native CLI binary (Linux/macOS)
├── testdata/               # mirrored fixtures from upstream testdata/ dirs
└── platforms/
    ├── wasm/               # wasm-bindgen wrapper (deferred to M4)
    ├── ios/                # xcframework (deferred to M4)
    └── android/            # JNI / AAR (deferred to M4)
```

## Milestones

### M1 — Inference-only slice (start here)

Minimum-viable end-to-end: take pre-built TFRecord examples (output of
upstream `make_examples`), run inference from Rust, emit
`CallVariantsOutput` protos. Byte-diff against upstream's `call_variants`
output on the chr20 quickstart.

Validates: (a) `prost` schema mapping for upstream `.proto`s, (b) TFRecord
codec, (c) at least one inference backend, (d) the SavedModel actually
behaves the same when called from Rust.

Steps:
1. Copy/symlink upstream `.proto`s into `dv-proto/proto/`; wire `prost-build`
   in `build.rs`.
2. Implement TFRecord reader/writer in `dv-io` (length + masked CRC32C
   framing — ~50 lines).
3. Extract a SavedModel from the `deepvariant:release` Docker image to a
   local cache dir (`models/wgs/`).
4. `dv-infer` — first backend = `tensorflow` crate (FFI to libtensorflow);
   load SavedModel, run `serving_default` signature on a batch.
5. `dv-cli call_variants` subcommand: TFRecord shards in → SavedModel →
   `CallVariantsOutput` TFRecord shards out.
6. Diff against upstream output from the chr20 quickstart. Goal: byte-equal
   (or float-near-equal) probabilities.

### M2 — `postprocess_variants` port

`CallVariantsOutput` TFRecord → VCF/gVCF via `noodles-vcf`. Port
`haplotypes.py` resolution and the C++ sort/stitch from
`postprocess_variants.cc`. Diff VCFs against upstream.

### M3 — `make_examples` port (the big one)

Port the C++ hot path module by module. Each module's ported test suite is
the parity gate (no runtime cxx bridge — fixtures are the oracle):

1. `nucleus/io/sam_reader` → `dv-io::bam` (noodles)
2. `allelecounter` → `dv_core::allele_count`
3. `variant_calling` → `dv_core::candidate_call`
4. `realigner/{ssw, debruijn_graph, fast_pass_aligner, window_selector}` →
   `dv_core::realigner`
5. `pileup_image_native` + `channels/*` → `dv_core::image_encode`
   (~25 channel encoders)
6. `make_examples_native` → `dv_core::examples` (tensor batch builder
   instead of tf.Example serializer)
7. `direct_phasing`, `methylation_aware_phasing` → `dv_core::phasing`

End of M3: full pipeline in Rust, byte-equal to upstream on quickstart.

### M4 — Cross-compile targets

- ONNX export of SavedModel → `ort` backend → WASM build via
  `wasm-bindgen` + `wasm-pack`. Browser demo over a sliced BAM via byte-range.
- TFLite conversion → Android JNI crate.
- CoreML conversion → iOS xcframework.
- Each target gets its own example app verifying inference correctness.

## Build / test reference

`compile.md` documents the Linux Docker build of upstream — used to
regenerate fixture expected-outputs and to run the chr20 end-to-end check
(78 variants = 64 SNPs + 14 indels, see `compile.md` step 8).

Per-module parity = ported test suite passes against fixture data copied
from `testdata/`. End-to-end parity = Rust pipeline matches upstream VCF
on the chr20 quickstart, byte-equal where deterministic / float-near-equal
on inference outputs.
