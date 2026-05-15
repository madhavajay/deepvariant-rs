# Benchmark: dv (Rust port) vs C++ DeepVariant on macOS Apple Silicon

This file tracks the optimization journey of the Rust port (`dv` /
`crates/`) and compares it to the C++ DeepVariant implementation
(both upstream and the Apple-Silicon-Metal fork in
`deepvariant-macos-arm64-metal/`).

## Test machine

```
Chip                          Apple M2 Max
hw.perflevel0.logicalcpu      8     # performance cores
hw.logicalcpu                 12    # 8P + 4E
hw.memsize                    64 GB
macOS                         26.4.1 (Tahoe, build 25E253)
```

## Test workload

Standard DeepVariant case study: **HG003 chr20** (NA12878-relative,
35× novaseq, pcr-free). Single-shard end-to-end pipeline
(`make_examples` → `call_variants` → `postprocess_variants`) on the
full 64.4 Mbp chr20 region.

Inputs come from `~/deepvariant-benchmark/data/` (cached by both the
C++ fork's `scripts/benchmark.sh` and our `./benchmark.sh`):
- GRCh38 reference (`GRCh38_no_alt_analysis_set.fasta`, ~3 GB)
- HG003 chr20 BAM (~1.6 GB)

Model:
- C++ fork: `~/.deepvariant/models/wgs/` (TF SavedModel + CoreML
  `.mlmodel`).
- Rust port: `models/wgs/model.onnx`, exported from the SavedModel
  via `tf2onnx --opset 17`.

ONNX Runtime: `1.24.4` (downloaded via
`scripts/fetch_onnxruntime.sh`).

## Reference C++ numbers (HG003 chr20, M2 Max)

From `deepvariant-macos-arm64-metal/benchmark.md`, run on the same
machine, same inputs:

| Mode | make_examples | call_variants | postprocess | Total |
|---|---|---|---|---|
| C++ Metal+CoreML+fast-pipeline | (concurrent) | (concurrent) | 16s | **3m06s (186s)** |

Earlier M1 Max published numbers from the fork README, sequential:

| Mode | make_examples | call_variants | postprocess | Total |
|---|---|---|---|---|
| CPU-only | ~263s | ~950s | ~16s | ~20m29s |
| Metal GPU | 263s | 224s | 16s | 8m23s |
| Metal GPU + CoreML (sequential) | 263s | 175s | 16s | 7m34s |
| Metal GPU + CoreML + fast-pipeline | (concurrent) | (concurrent) | 16s | 3m21s |

The C++ fork uses 8 parallel `make_examples` shards via GNU
`parallel`, plus C++ `fast_pipeline` to overlap make_examples and
call_variants. The Rust port runs single-process throughout.

## Quick-bench region (for fast iteration)

`chr20:10000000-11000000` (1 Mbp slice). One `make_examples` run is
≈ 1/64 of full chr20; large enough to exercise the BAM index, allele
counter, realigner, pileup, and write paths, but completes in a few
seconds for fast iteration.

```bash
RUNS=2 ./benchmark.sh --quick      # 1 Mbp, 2 runs
RUNS=1 ./benchmark.sh --full       # full chr20, 1 run
```

## Optimization journey

Each row reports per-stage timings (full chr20, single-shard) and
total wall time after that change landed. Numbers in **bold** are
end-to-end totals.

### make_examples

| # | Optimization | bam_read | realigner | pass2_render | pass3_write | total |
|---|---|---|---|---|---|---|
| 0 | Baseline (whole-BAM scan, serial realigner, level-6 gzip serial write) | 30s | 36 min | (got stuck, ~45 min projected) | — | **>40 min** |
| 1 | Indexed BAM `.bai` query | 0.4s | 36 min | 45 min projected | — | (still stuck on pass2) |
| 2 | Parallel realigner (rayon `par_iter` over windows) | 0.4s | ~12 min | 45 min projected | — | (still stuck on pass2) |
| 3 | Cache `ref_end` + sorted-read overlap index (binary-search slice instead of O(reads × queries) scan) | 30s | 42s | 7s | 297s | **6m35s (395s)** |
| 4 | Parallel-chunked gzip write (`Compression::fast()` per chunk, concat) | 30s | 63s | 11s | 11s | **2m13s (133s)** |

**make_examples total: >40 min → 133s = ≥18× faster.**

The single most impactful change was #3: the inner read-overlap
filter was O(reads × candidates) and recomputed `read_end` from the
CIGAR on every call. With 15.8 M chr20 reads and 205 k candidates,
that's 3.2 trillion comparisons per pass. Caching `ref_end` once at
BAM load + binary-searching a sorted prefix slice per query brought
both `realigner` and `pass2_render` from ~minutes down to seconds.

For the 1 Mbp quick bench, the same pipeline went **25s → ~2s**
(~12.5× faster).

### call_variants

| # | Optimization | predict (1 Mbp) | predict (full chr20) | full chr20 wall |
|---|---|---|---|---|
| 0 | ORT-CPU (default settings) | ~20 s | ~25 min projected | ~25 min |
| 1 | ORT-CPU `with_memory_pattern(true)` | ~20 s | (no measurable change; aggressive thread-tuning regressed wall time so reverted) | ~25 min |
| 2 | CoreML EP (NeuralNetwork format, ALL compute units, FastPrediction strategy) | **4.1 s** | **106 s predict / 143 s wall** | **2m23s** |

Key gotcha: the WGS ONNX export (`tf2onnx`) keeps a dynamic batch
dim. Setting `with_static_input_shapes(true)` makes CoreML reject
every node touching that dim and silently fall back to CPU EP for
the entire graph (we observed zero speedup before fixing). Leaving
the default (accept dynamic shapes) is the right call — the CoreML
EP routes static-shape sub-graphs to GPU+ANE and falls back per-op.

`MLProgram` format also fails to parse this particular tf2onnx
export ("`Required param 'pad' is missing`" on Conv2D); the older
`NeuralNetwork` format works fine.

CoreML model cache (`~/.cache/deepvariant-rs/coreml`, override with
`DV_COREML_CACHE`) is enabled — first run pays ~20s compile, every
subsequent run skips it.

### `dv pipeline` (ME + CV concurrent, in-process)

Single subcommand that drains rendered pileup-images through a
crossbeam-style bounded channel into an inference worker thread —
the Rust analogue of the C++ fork's `fast_pipeline` shared-memory
IPC, but all in one process and no disk roundtrip.

Pre-pass2 work (BAM read + allele counter) is also region-sharded
across N=perf-cores sub-regions in parallel, with reads at shard
boundaries contributing to allele counts on both sides but emitted
to the merged owned list exactly once.

Stage breakdown for full chr20 (M2 Max, CoreML enabled):

| Stage | sequential pre-pass2 | sharded pre-pass2 | sharded + MLProgram batch=128 |
|---|---|---|---|
| bam_read + allele_counter | 29.5 s + 7.7 s | 7.0 s (12 shards in parallel) | **8.4 s** |
| bam_sort | 0.2 s | 0.3 s | 0.3 s |
| candidate_caller | 3.3 s | 3.5 s | 2.8 s |
| realigner | 48.5 s | 32.8 s | 35.9 s |
| pass1_prepare_renders | 0.6 s | 0.4 s | 0.4 s |
| **pass2_render_streamed** (overlapped with inference) | 125.9 s | 124.8 s | **106.2 s** |
| inference_drain (final batch flush after pass2 done) | 1.8 s | 1.6 s | 0.2 s |
| **wall total** | **3 m 45 s** | **2 m 55 s** | **~2 m 34 s** |

Measured on the regenerated `models/wgs/model.onnx` (batch dim
pinned to 128 by `scripts/normalize_onnx_pads.py`, explicit Conv
pads, MLProgram-safe) with 208,882 CVOs emitted.

**Honest note on the MLProgram batch-pinning win.** The 1 Mbp
inference-only smoke bench showed a ~3× speedup (4.1 s → 1.3 s,
NeuralNetwork dynamic-batch → MLProgram batch=128). That does
**not** translate to a 3× drop in `pass2_render_streamed` at full
chr20: it fell 124.8 s → 106.2 s (≈15 %), not → ~50 s as earlier
projected. Reason: `pass2_render_streamed` overlaps pileup-image
rendering (CPU, rayon) with inference (GPU+ANE) through the bounded
channel. At full-chr20 scale rendering is the bottleneck, so faster
inference only shaves the portion of wall time where inference was
the long pole. The smoke bench isolated inference, so it saw the
full backend speedup; the streamed pipeline is render-bound. The
next pass2 win has to come from the renderer, not the model
backend.

### Side-by-side end-to-end (HG003 chr20, M2 Max)

| Configuration | make_examples | call_variants | postprocess | **Total** |
|---|---|---|---|---|
| Rust port — original baseline | (>40 min, never finished) | (~25 min projected) | — | (never measured) |
| Rust port — sequential (CoreML, ME→file→CV) | 133 s | 143 s | ~10 s | ~286 s (4 m 46 s) |
| Rust port — `dv pipeline` (concurrent, sequential pre-pass2) | concurrent | concurrent | ~10 s pending | ~225 s (3 m 45 s) |
| Rust port — `dv pipeline` w/ sharded pre-pass2 (NeuralNetwork) | concurrent | concurrent | ~10 s | 175 s (2 m 55 s) |
| **Rust port — `dv pipeline` sharded + MLProgram batch=128** | **concurrent** | **concurrent** | **1.4 s** | **~155 s (2 m 35 s)** |
| C++ fork — sequential Metal+CoreML (M1 Max published) | 263 s | 175 s | 16 s | 454 s (7 m 34 s) |
| C++ fork — fast-pipeline Metal+CoreML (M2 Max measured) | concurrent | concurrent | 16 s | 186 s (3 m 06 s) |

`postprocess` is 1.4 s for the full-chr20 CVO (205,198 VCF
records) without `--phase-reads`; the earlier "~10 s" was a
pre-measurement placeholder.

**Rust `dv pipeline` is ~17 % faster than the C++ Metal+CoreML
fast-pipeline on the same machine** (2 m 35 s vs 3 m 06 s),
single-process, no GNU `parallel`, no shared-memory IPC. The win
relative to C++ comes mostly from the sorted-read overlap index +
parallel-chunked gzip + single-process channel coordination
overhead being lower than the C++ fork's 8-shard `make_examples` +
shared-memory bridge, plus the MLProgram batch=128 pinning shaving
~20 s off `pass2_render_streamed`.

### Full WGS (chr1–chr22, X, Y) projection on this M2 Max

Linear scaling from chr20 (64.4 Mbp / 3088 Mbp full genome ≈ 48×):

| Configuration | chr20 measured | WGS projected |
|---|---|---|
| **Rust `dv pipeline` (sharded + MLProgram batch=128, BAM)** | 2 m 35 s | **~2 h 04 m** |
| Rust `dv pipeline` (sharded, BAM, NeuralNetwork) | 2 m 55 s | ~2 h 20 m |
| Rust `dv pipeline` (sharded, CRAM, indexed `.crai`) | est. +10 % | ~2 h 17 m |
| C++ Metal+CoreML+fast-pipeline (M2 Max) | 3 m 06 s | ~2 h 30 m |
| C++ Metal+CoreML+fast-pipeline (M1 Max, fork README) | 3 m 21 s | ~3 h (published) |
| C++ stock GCP n2-standard-96 (96 vCPU CPU-only) | 1 m 39 s | 1 h 19 m |
| C++ stock GCP L4 GPU Cloud Run | 9 m 44 s | ~3 h 30 m |
| C++ stock GCP n2-standard-16 (16 vCPU CPU-only) | 14 m 28 s | ~7 h |

Per-chromosome dispatch is the natural execution model: 24
invocations of `dv pipeline`, then concat the resulting CVO
shards through `dv postprocess-variants`. chr1 is ~5× chr20 in
length; at 35× depth it needs ~50 GB peak RAM during the merged
owned-reads phase — fits on a 64 GB M2 Max.

**Not yet measured end-to-end:** the local benchmark dataset is
chr20-only (`HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam`).
A real WGS run needs the whole-genome HG003 BAM (~100 GB+), not
yet downloaded. The WGS column above is a linear projection from
the measured chr20 wall, not a measured number.

### Remaining

| Target | Est. wall savings on full chr20 | Notes |
|---|---|---|
| ~~`MLProgram` CoreML format~~ — **landed**, batch=128 pinned | measured: 124.8 s → 106.2 s on pass2 (~20 s wall) | macOS/iOS only. Smaller than the 1.2–1.5× projected — pass2 is render-bound, not inference-bound, at full scale. |
| Renderer speedup for `pass2_render_streamed` (now the dominant stage at 106 s) | unknown; this is the new long pole | SIMD pileup encode / fewer per-pixel allocations / batch the rayon map. |
| Per-window haplotype cap (≤8 haplotypes per DBG) — the C++ fork's portable optimization | ~5 s on realigner | Portable, also reduces noisy assembly windows. |
| Realigner DBG SIMD/optimized hashing | ~10 s on realigner | Portable. |
| Multi-threaded BGZF write for the CVO output | small (~1–2 s, write is already minor) | Portable. |
| WASM target for `dv pipeline` core | n/a (offload target) | Portable; needs ORT wasm or onnxruntime-web. |

### Known issues

- ~~**`postprocess-variants` panics on full chr20**: assertion
  `n_alleles must be >= 2` in `add_call_to_variant`.~~ **Fixed.**
  Root cause: the candidate caller emitted a duplicate alt
  (`alternate_bases = ["GT","GT"]`) at chr20:35167420; the
  HashSet-based `to_remove` deduped it and the prune safety net
  misfired, leaving a 0-alt variant. Fixed at the source
  (`variant_calling::candidates_from_counts` collapses alts that
  project to the same string, summing counts) plus a defensive
  `group_cvos` split by `(range, sorted_alts)`. Full-chr20 now
  emits 205,198 VCF records cleanly. See `TODO.md`.

## Validation

Output VCF records:

| Run | record count |
|---|---|
| Rust ORT-CPU (1 Mbp) | 2601 |
| Rust CoreML (1 Mbp)  | 2601 |
| C++ fork (full chr20, separately measured) | 207,799 |
| Rust port (full chr20, current) | (pending end-to-end VCF gen) |

CoreML vs ORT-CPU on the same examples: identical chr/pos/ref/alt
and identical genotype calls; ±1 in `GQ` on a few records, expected
because CoreML uses fp16 GPU kernels and ORT-CPU uses fp32. No
classification differences.

## Fast pipeline (`dv pipeline`)

A single subcommand that runs make_examples and call_variants
concurrently in-process. Architecture:

```
   pre-pass2 work (BAM read, sort, allele counter, candidate caller,
   realigner expansion, pass-1 prepare-renders) — serial
                  │
                  ▼
   ┌─────────────────────────────┐    spawn  ┌────────────────────┐
   │  Pass 2: rayon par_iter     │  ───────▶ │  Inference worker  │
   │  renders pileup-images and  │ (channel) │  thread: loads ORT │
   │  pushes (variant, alt_idx,  │  bounded  │  + CoreML model,   │
   │  image_u8) onto a channel   │  256-slot │  drains channel,   │
   │                             │           │  batches, runs     │
   └─────────────────────────────┘           │  predict_batch,    │
              drop sender                    │  writes CVOs       │
                                             └────────────────────┘
                                                        │
                                                        ▼
                                                 join, return total
```

The inference worker loads CoreML (~20s cold, ~0s warm with cache)
**while** pass 2 is rendering — the channel back-pressure means
rendering blocks until the worker is ready, but if model load is
slower than the first 256 renders, those are queued for when the
model is up. Net result: rendering and inference overlap entirely,
similar to the C++ fork's `fast_pipeline` shared-memory IPC, but all
in a single Rust process.

```bash
dv pipeline \
    --reads HG003.bam \
    --ref-fasta GRCh38.fasta \
    --region chr20:1-64444167 \
    --checkpoint models/wgs/model.onnx \
    --output cvo.tfrecord.gz \
    --batch-size 128
```

Flags `dv make-examples` and `dv call-variants` still exist as
separate subcommands for cases where you want to inspect the
intermediate `examples.tfrecord.gz` (development, debugging,
out-of-process call-variants on a different machine, etc.). The
pipeline subcommand is the recommended path for end-to-end runs on
the same host.

## How to reproduce

```bash
# Quick (1 Mbp, ~30s per run, fast iteration loop)
./benchmark.sh --quick

# Full chr20 (~5 min with current optimizations)
RUNS=1 ./benchmark.sh --full

# A/B against CPU-only ORT (forces CoreML off)
DV_DISABLE_COREML=1 RUNS=1 ./benchmark.sh --full

# Pipeline (ME + CV concurrent in one process) on full chr20:
ORT_DYLIB_PATH=$PWD/models/lib/libonnxruntime.dylib \
  ./target/release/dv pipeline \
    --reads ~/deepvariant-benchmark/data/input/HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam \
    --ref-fasta ~/deepvariant-benchmark/data/reference/GRCh38_no_alt_analysis_set.fasta \
    --region chr20:1-64444167 \
    --checkpoint $PWD/models/wgs/model.onnx \
    --output /tmp/dv_cvo.tfrecord.gz \
    --batch-size 128
```

Stage-level timings are emitted via `tracing` at INFO level when
`RUST_LOG=info` is set. The benchmark script captures full per-stage
logs in `~/deepvariant-benchmark/rust_runs/runs/run_N/`.
