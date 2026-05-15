#!/usr/bin/env bash
# Benchmark the Rust dv-cli (ORT backend, release build) on HG003 chr20,
# mirroring deepvariant-macos-arm64-metal/scripts/benchmark.sh as closely
# as the current Rust port allows.
#
# Differences from the C++ harness:
#  - Single-shard (dv make-examples is not sharded yet — TODO P3).
#  - No accuracy step (rtg/hap.py); not the variable under test.
#  - Inference via ONNX Runtime CPU (no Metal/CoreML/GPU).
#
# Reuses cached BAM/FASTA from ~/deepvariant-benchmark/data/ if present.

set -euo pipefail

caffeinate -i -w $$ &

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DV_BIN="${DV_BIN:-$WORKSPACE_ROOT/target/release/dv}"
ORT_DYLIB_PATH="${ORT_DYLIB_PATH:-$WORKSPACE_ROOT/models/lib/libonnxruntime.dylib}"
MODEL_ONNX="${MODEL_ONNX:-$WORKSPACE_ROOT/models/wgs/model.onnx}"

BENCH_DIR="${BENCH_DIR:-$HOME/deepvariant-benchmark}"
DATA_DIR="$BENCH_DIR/data"
REF="$DATA_DIR/reference/GRCh38_no_alt_analysis_set.fasta"
BAM="$DATA_DIR/input/HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam"

OUTPUT_DIR="${OUTPUT_DIR:-$BENCH_DIR/rust_runs}"
NUM_RUNS="${NUM_RUNS:-2}"
REGION="${REGION:-chr20:1-64444167}"

GREEN='\033[0;32m'; RED='\033[0;31m'; BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'
info()   { echo -e "${BLUE}==>${NC} $*"; }
pass()   { echo -e "${GREEN}✓${NC}  $*"; }
fail()   { echo -e "${RED}✗${NC}  $*"; exit 1; }
banner() { echo -e "\n${BOLD}$*${NC}"; }

[[ -x "$DV_BIN" ]]        || fail "dv binary not found at $DV_BIN (run: cargo build --release -p dv-cli)"
[[ -f "$ORT_DYLIB_PATH" ]] || fail "libonnxruntime.dylib not found (run: scripts/fetch_onnxruntime.sh)"
[[ -f "$MODEL_ONNX" ]]    || fail "WGS ONNX model not found at $MODEL_ONNX"
[[ -f "$REF" ]]           || fail "reference FASTA not found at $REF"
[[ -f "$BAM" ]]           || fail "HG003 BAM not found at $BAM"

export ORT_DYLIB_PATH

CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple Silicon")
PERF_CORES=$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || echo "?")

# Git sha (+ -dirty suffix) so each run produces a stable, immutable
# per-commit snapshot we can graph over time / regression-check in CI.
GIT_SHA=$(git -C "$WORKSPACE_ROOT" rev-parse --short HEAD 2>/dev/null || echo "nogit")
if ! git -C "$WORKSPACE_ROOT" diff --quiet HEAD -- 2>/dev/null; then
    GIT_SHA="${GIT_SHA}-dirty"
fi

mkdir -p "$OUTPUT_DIR"
RESULTS_JSONL="$OUTPUT_DIR/benchmark_runs.jsonl"
RESULTS_JSON="$OUTPUT_DIR/benchmark_results.json"
: > "$RESULTS_JSONL"

echo
echo "================================================================"
echo "  dv (Rust) — HG003 chr20 benchmark"
echo "================================================================"
echo "  Chip:     $CHIP   (perf cores: $PERF_CORES)"
echo "  Binary:   $DV_BIN"
echo "  Model:    $MODEL_ONNX"
echo "  Region:   $REGION"
echo "  Output:   $OUTPUT_DIR"
echo "  Runs:     $NUM_RUNS  (single-shard; ORT CPU backend)"
echo

run_pipeline() {
    local run_num="$1"
    local run_dir="$OUTPUT_DIR/runs/run_${run_num}"
    rm -rf "$run_dir"; mkdir -p "$run_dir"

    local EXAMPLES="$run_dir/make_examples.tfrecord.gz"
    local CV_OUT="$run_dir/call_variants_output.tfrecord.gz"
    local OUT_VCF="$run_dir/output.vcf.gz"

    info "make_examples (run $run_num)"
    SECONDS=0
    "$DV_BIN" make-examples \
        --reads "$BAM" --ref-fasta "$REF" --region "$REGION" \
        --examples "$EXAMPLES" --sample-name HG003 \
        2>&1 | tee "$run_dir/make_examples.log" >/dev/null
    local me_seconds=$SECONDS

    info "call_variants (run $run_num)"
    SECONDS=0
    "$DV_BIN" call-variants \
        --examples "$EXAMPLES" --checkpoint "$MODEL_ONNX" --output "$CV_OUT" \
        2>&1 | tee "$run_dir/call_variants.log" >/dev/null
    local cv_seconds=$SECONDS

    info "postprocess_variants (run $run_num)"
    SECONDS=0
    "$DV_BIN" postprocess-variants \
        --cvo "$CV_OUT" --output-vcf "$OUT_VCF" \
        --contig chr20:64444167 --sample-name HG003 \
        2>&1 | tee "$run_dir/postprocess_variants.log" >/dev/null
    local pp_seconds=$SECONDS

    local total=$((me_seconds + cv_seconds + pp_seconds))
    pass "Run $run_num: make_examples=${me_seconds}s  call_variants=${cv_seconds}s  postprocess=${pp_seconds}s  total=${total}s"

    python3 -c "
import json
print(json.dumps({
    'run': $run_num,
    'stages': {
        'make_examples': $me_seconds,
        'call_variants': $cv_seconds,
        'postprocess_variants': $pp_seconds,
    },
    'total': $total,
}))" >> "$RESULTS_JSONL"
}

for r in $(seq 1 "$NUM_RUNS"); do
    banner "=== Run $r/$NUM_RUNS ==="
    run_pipeline "$r"
done

# Aggregate. Writes both the rolling `benchmark_results.json` and an
# immutable per-commit snapshot `rust_runs_<sha>.json` under BENCH_DIR.
SNAPSHOT_JSON="$BENCH_DIR/rust_runs_${GIT_SHA}.json"
python3 - "$RESULTS_JSONL" "$RESULTS_JSON" "$CHIP" "$PERF_CORES" "$GIT_SHA" "$REGION" "$SNAPSHOT_JSON" <<'PY'
import json, statistics, sys, datetime
jsonl, out, chip, perf_cores, git_sha, region, snapshot = sys.argv[1:8]
runs = [json.loads(l) for l in open(jsonl) if l.strip()]
def stat(vals):
    return {
        'mean': round(statistics.mean(vals), 1),
        'std': round(statistics.stdev(vals), 1) if len(vals) > 1 else 0.0,
        'min': min(vals), 'max': max(vals), 'values': vals,
    }
summary = {}
for stage in ['make_examples', 'call_variants', 'postprocess_variants']:
    summary[stage] = stat([r['stages'][stage] for r in runs])
summary['total'] = stat([r['total'] for r in runs])
out_blob = {
    'metadata': {
        'binary': 'dv (Rust, release, ORT CPU)',
        'chip': chip,
        'perf_cores': int(perf_cores) if perf_cores.isdigit() else None,
        'git_sha': git_sha,
        'sample': 'HG003',
        'region': region,
        'shards': 1,
        'runs': len(runs),
        'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
    },
    'runs': runs,
    'summary': summary,
}
json.dump(out_blob, open(out, 'w'), indent=2)
print(f'Wrote {out}')
# Per-commit snapshot. Clean commits get `rust_runs_<sha>.json`;
# uncommitted trees get `rust_runs_<sha>-dirty.json`, so a clean
# baseline is never clobbered by an ad-hoc dirty run.
json.dump(out_blob, open(snapshot, 'w'), indent=2)
print(f'Wrote {snapshot}')
PY

echo
echo "================================================================"
echo "  Done. Results: $RESULTS_JSON"
echo "================================================================"
