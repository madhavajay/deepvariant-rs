#!/usr/bin/env bash
# Build the static, no-backend browser bundle:
#   1. dv-wasm → wasm32 + wasm-bindgen (--target web) → public/pkg/
#   2. onnxruntime-web → public/ort/
#   3. dynamic-batch WGS ONNX → public/models/wgs.onnx
# Everything the page needs is then static files.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
PUBLIC="${SCRIPT_DIR}/public"
cd "${REPO}"

# wasm-bindgen-cli must match the wasm-bindgen crate version.
WB_VER="$(cargo metadata --format-version 1 --no-deps 2>/dev/null \
  | grep -o '"wasm-bindgen","version":"[0-9.]*"' | head -1 \
  | grep -o '[0-9][0-9.]*' || true)"
WB_VER="${WB_VER:-0.2.120}"
if ! command -v wasm-bindgen >/dev/null || \
   [[ "$(wasm-bindgen --version | awk '{print $2}')" != "${WB_VER}" ]]; then
    echo "[*] installing wasm-bindgen-cli ${WB_VER} (one-time, a few min)…"
    cargo install wasm-bindgen-cli --version "${WB_VER}" --locked
fi

echo "[1/3] cargo build dv-wasm (wasm32, release)"
cargo build -p dv-wasm --target wasm32-unknown-unknown --release >/dev/null

echo "[2/3] wasm-bindgen --target web → public/pkg/"
mkdir -p "${PUBLIC}/pkg"
wasm-bindgen target/wasm32-unknown-unknown/release/dv_wasm.wasm \
    --target web --out-dir "${PUBLIC}/pkg" --no-typescript

echo "[3/3] onnxruntime-web + model"
if [[ ! -d "${SCRIPT_DIR}/node_modules/onnxruntime-web" ]]; then
    (cd "${SCRIPT_DIR}" && npm install --silent)
fi
mkdir -p "${PUBLIC}/ort" "${PUBLIC}/models"
cp "${SCRIPT_DIR}"/node_modules/onnxruntime-web/dist/*.{mjs,wasm,js} \
    "${PUBLIC}/ort/" 2>/dev/null || true
# Browser inference is per-batch with a variable batch size, so use
# the DYNAMIC-batch export (model.onnx is pinned to batch=128 and
# would reject it). The file is just confusingly named.
ln -sfn "${REPO}/models/wgs/model_mlprogram.onnx" "${PUBLIC}/models/wgs.onnx"

echo "done — static bundle in ${PUBLIC}/"
