#!/usr/bin/env bash
# Populate wasm-frontend/public/ with everything the Vite dev server
# needs to serve:
#   1. dv-wasm compiled for wasm32-unknown-unknown + JS glue (--target web)
#   2. onnxruntime-web runtime files (.mjs + .wasm)
#   3. testdata + model symlinks (so the dev server can stream them)
#
# This is intentionally a straight clone of wasm-browser-test/build.sh,
# minus the golden-prediction extraction step (the user-facing app
# doesn't need expected.json).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
PUBLIC="${SCRIPT_DIR}/public"

cd "${REPO}"

echo "[1/4] cargo build dv-wasm (wasm32-unknown-unknown, release)"
cargo build -p dv-wasm --target wasm32-unknown-unknown --release >/dev/null

echo "[2/4] wasm-bindgen --target web → public/pkg/"
mkdir -p "${PUBLIC}/pkg"
wasm-bindgen \
    target/wasm32-unknown-unknown/release/dv_wasm.wasm \
    --target web \
    --out-dir "${PUBLIC}/pkg" \
    --no-typescript

echo "[3/4] copy onnxruntime-web into public/ort/"
ORT_SRC="${SCRIPT_DIR}/node_modules/onnxruntime-web/dist"
if [[ ! -d "${ORT_SRC}" ]]; then
    echo "  error: ${ORT_SRC} missing — run \`npm install\` first" >&2
    exit 1
fi
mkdir -p "${PUBLIC}/ort"
cp "${ORT_SRC}"/*.mjs "${PUBLIC}/ort/" 2>/dev/null || true
cp "${ORT_SRC}"/*.js "${PUBLIC}/ort/" 2>/dev/null || true
cp "${ORT_SRC}"/*.wasm "${PUBLIC}/ort/" 2>/dev/null || true

echo "[4/4] symlink testdata + model into public/"
mkdir -p "${PUBLIC}/testdata" "${PUBLIC}/models"
ln -sfn "${REPO}/testdata/quickstart_chr20_norealign/examples.tfrecord.gz" "${PUBLIC}/testdata/examples.tfrecord.gz"
if [[ -e "${REPO}/models/wgs.onnx" ]]; then
    ln -sfn "${REPO}/models/wgs.onnx" "${PUBLIC}/models/wgs.onnx"
else
    echo "  warning: ${REPO}/models/wgs.onnx not found — the in-app 'Download model' button will 404 until you place it there" >&2
fi

echo
echo "done."
echo "  public/pkg/     $(ls "${PUBLIC}/pkg" 2>/dev/null | wc -l) files"
echo "  public/ort/     $(ls "${PUBLIC}/ort" 2>/dev/null | wc -l) files"
echo "  public/models/  $(ls -L "${PUBLIC}/models" 2>/dev/null | wc -l) files"
