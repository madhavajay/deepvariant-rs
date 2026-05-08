#!/usr/bin/env bash
# One-shot benchmark wrapper for the Rust dv-cli on macOS arm64.
#
# What it does:
#   1. Builds `dv` in release mode if missing.
#   2. Fetches libonnxruntime into models/lib/ if missing.
#   3. Ensures a WGS ONNX model is at models/wgs/model.onnx, converting
#      from ~/.deepvariant/models/wgs/ (SavedModel) via tf2onnx if needed.
#   4. Downloads HG003 chr20 BAM + GRCh38 reference into
#      ~/deepvariant-benchmark/data/ if missing (~4.7 GB on first run).
#   5. Invokes scripts/benchmark_rust.sh.
#
# Usage:
#   ./benchmark.sh                  # full chr20, 2 runs (≈2 hr)
#   ./benchmark.sh --quick          # 1 Mbp slice, 2 runs (≈1 min) — for iterating
#   RUNS=1 ./benchmark.sh --quick   # single quick run (≈30 s)
#   REGION=chr20:1-5000000 ./benchmark.sh   # arbitrary region
#
# Pass-through flags: any args after --pass are forwarded to
# scripts/benchmark_rust.sh.

set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKSPACE_ROOT"

# ── Argv parse: --quick / --full / --pass ─────────────────────────────────────
QUICK_REGION="chr20:10000000-11000000"   # 1 Mbp slice, ~30 s per stage
FULL_REGION="chr20:1-64444167"
PRESET=""
PASSTHROUGH=()
seen_pass=0
for a in "$@"; do
    if (( seen_pass )); then PASSTHROUGH+=("$a"); continue; fi
    case "$a" in
        --quick) PRESET="quick" ;;
        --full)  PRESET="full" ;;
        --pass)  seen_pass=1 ;;
        *) echo "Unknown arg: $a" >&2; exit 2 ;;
    esac
done

case "$PRESET" in
    quick) export REGION="${REGION:-$QUICK_REGION}"; export OUTPUT_DIR="${OUTPUT_DIR:-${BENCH_DIR:-$HOME/deepvariant-benchmark}/rust_runs_quick}" ;;
    full|"") export REGION="${REGION:-$FULL_REGION}" ;;
esac

GREEN='\033[0;32m'; RED='\033[0;31m'; BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'
info()  { echo -e "${BLUE}==>${NC} $*"; }
pass()  { echo -e "${GREEN}✓${NC}  $*"; }
fail()  { echo -e "${RED}✗${NC}  $*"; exit 1; }

DV_BIN="$WORKSPACE_ROOT/target/release/dv"
ORT_LIB="$WORKSPACE_ROOT/models/lib/libonnxruntime.dylib"
MODEL_ONNX="$WORKSPACE_ROOT/models/wgs/model.onnx"
SAVEDMODEL="$HOME/.deepvariant/models/wgs"

BENCH_DIR="${BENCH_DIR:-$HOME/deepvariant-benchmark}"
DATA_DIR="$BENCH_DIR/data"
REF="$DATA_DIR/reference/GRCh38_no_alt_analysis_set.fasta"
REF_FAI="$DATA_DIR/reference/GRCh38_no_alt_analysis_set.fasta.fai"
BAM="$DATA_DIR/input/HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam"
BAM_BAI="$DATA_DIR/input/HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam.bai"

# ── 1. Release build ──────────────────────────────────────────────────────────
info "Build: cargo build --release -p dv-cli"
cargo build --release -p dv-cli >/dev/null
[[ -x "$DV_BIN" ]] || fail "build did not produce $DV_BIN"
pass "dv binary at $DV_BIN"

# ── 2. ONNX Runtime ───────────────────────────────────────────────────────────
if [[ ! -f "$ORT_LIB" ]]; then
    info "Fetching ONNX Runtime"
    bash "$WORKSPACE_ROOT/scripts/fetch_onnxruntime.sh"
fi
pass "libonnxruntime at $ORT_LIB"

# ── 3. WGS ONNX model ─────────────────────────────────────────────────────────
if [[ ! -f "$MODEL_ONNX" ]]; then
    info "ONNX model missing — converting from SavedModel"
    [[ -d "$SAVEDMODEL" && -f "$SAVEDMODEL/saved_model.pb" ]] \
        || fail "no SavedModel at $SAVEDMODEL — install the WGS model first
       (e.g. \`bash deepvariant-macos-arm64-metal/scripts/deepvariant-download-model WGS\`)"

    PY=""
    if [[ -x "$HOME/.deepvariant/venv/bin/python3" ]]; then
        PY="$HOME/.deepvariant/venv/bin/python3"
    elif command -v python3 >/dev/null; then
        PY="python3"
    else
        fail "python3 not found — needed to run tf2onnx"
    fi
    if ! "$PY" -c "import tf2onnx" 2>/dev/null; then
        info "Installing tf2onnx into $($PY -c 'import sys;print(sys.prefix)')"
        "$PY" -m pip install tf2onnx >/dev/null
    fi
    mkdir -p "$(dirname "$MODEL_ONNX")"
    "$PY" -m tf2onnx.convert \
        --saved-model "$SAVEDMODEL" \
        --output "$MODEL_ONNX" \
        --opset 17 2>&1 | tail -5
fi
pass "ONNX model at $MODEL_ONNX"

# ── 4. Inputs (HG003 chr20 BAM + GRCh38 reference) ────────────────────────────
mkdir -p "$DATA_DIR/reference" "$DATA_DIR/input"

REF_FTP="ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/GCA_000001405.15_GRCh38/seqs_for_alignment_pipelines.ucsc_ids"
BAM_URL="https://storage.googleapis.com/deepvariant/case-study-testdata"

download() {
    local url="$1" dst="$2"
    [[ -f "$dst" ]] && { echo "  cached: $(basename "$dst")"; return; }
    echo "  downloading: $(basename "$dst") ..."
    curl --fail --location --silent --show-error -o "$dst" "$url" \
        || fail "failed to download $url"
}

if [[ ! -f "$REF" ]]; then
    info "Downloading GRCh38 reference (~3 GB, gunzipped on the fly)"
    curl --fail --location --silent --show-error \
        "${REF_FTP}/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz" \
        | gunzip > "$REF"
fi
download "${REF_FTP}/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.fai" "$REF_FAI"
download "${BAM_URL}/HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam"     "$BAM"
download "${BAM_URL}/HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam.bai" "$BAM_BAI"
pass "Inputs ready under $DATA_DIR"

# ── 5. Run the benchmark ──────────────────────────────────────────────────────
export NUM_RUNS="${RUNS:-${NUM_RUNS:-2}}"
export REGION="${REGION:-chr20:1-64444167}"
export BENCH_DIR
export OUTPUT_DIR="${OUTPUT_DIR:-$BENCH_DIR/rust_runs}"

info "Launching scripts/benchmark_rust.sh (NUM_RUNS=$NUM_RUNS  REGION=$REGION)"

bash "$WORKSPACE_ROOT/scripts/benchmark_rust.sh" ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}
