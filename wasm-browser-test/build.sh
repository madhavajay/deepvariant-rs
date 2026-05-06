#!/usr/bin/env bash
# Build everything the browser test needs:
#   1. dv-wasm compiled for wasm32-unknown-unknown + JS glue (--target web)
#   2. onnxruntime-web runtime files copied next to public/
#   3. fixtures + model symlinked next to public/
#   4. expected.json — the native dv call-variants reference predictions,
#      generated once and reused across browser runs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
PUBLIC="${SCRIPT_DIR}/public"

cd "${REPO}"

echo "[1/5] cargo build dv-wasm (wasm32-unknown-unknown, release)"
cargo build -p dv-wasm --target wasm32-unknown-unknown --release >/dev/null

echo "[2/5] wasm-bindgen --target web → public/pkg/"
mkdir -p "${PUBLIC}/pkg"
wasm-bindgen \
    target/wasm32-unknown-unknown/release/dv_wasm.wasm \
    --target web \
    --out-dir "${PUBLIC}/pkg" \
    --no-typescript

echo "[3/5] copy onnxruntime-web into public/ort/"
ORT_SRC="${SCRIPT_DIR}/node_modules/onnxruntime-web/dist"
if [[ ! -d "${ORT_SRC}" ]]; then
    echo "  error: ${ORT_SRC} missing — run \`npm install\` first" >&2
    exit 1
fi
mkdir -p "${PUBLIC}/ort"
cp "${ORT_SRC}"/*.mjs "${PUBLIC}/ort/" 2>/dev/null || true
cp "${ORT_SRC}"/*.js "${PUBLIC}/ort/" 2>/dev/null || true
cp "${ORT_SRC}"/*.wasm "${PUBLIC}/ort/" 2>/dev/null || true

echo "[4/5] symlink testdata + model into public/"
mkdir -p "${PUBLIC}/testdata" "${PUBLIC}/models"
ln -sfn "${REPO}/testdata/quickstart_chr20_norealign/examples.tfrecord.gz" "${PUBLIC}/testdata/examples.tfrecord.gz"
ln -sfn "${REPO}/models/wgs.onnx" "${PUBLIC}/models/wgs.onnx"

echo "[5/5] generate expected.json (golden predictions from native dv-cli)"
EXPECTED="${PUBLIC}/expected.json"
GOLDEN_CVOS="${SCRIPT_DIR}/cvos.fresh.tfrecord.gz"
if [[ ! -f "${GOLDEN_CVOS}" ]]; then
    if [[ ! -x "${REPO}/target/release/dv" ]]; then
        echo "  building dv (release)…"
        cargo build -p dv-cli --release >/dev/null
    fi
    "${REPO}/target/release/dv" call-variants \
        --examples "${PUBLIC}/testdata/examples.tfrecord.gz" \
        --checkpoint "${PUBLIC}/models/wgs.onnx" \
        --output "${GOLDEN_CVOS}" >/dev/null
fi
node "${SCRIPT_DIR}/scripts/extract-expected.mjs" "${GOLDEN_CVOS}" "${EXPECTED}"

echo
echo "done."
echo "  public/pkg/      $(ls "${PUBLIC}/pkg" | wc -l) files"
echo "  public/ort/      $(ls "${PUBLIC}/ort" | wc -l) files"
echo "  public/expected.json  $(wc -l < "${EXPECTED}") lines"
