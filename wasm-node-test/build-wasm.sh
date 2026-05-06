#!/usr/bin/env bash
# Rebuild dv-wasm and regenerate the Node-targeted JS glue.
# Run from the repo root.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO}"
cargo build -p dv-wasm --target wasm32-unknown-unknown --release
wasm-bindgen \
    target/wasm32-unknown-unknown/release/dv_wasm.wasm \
    --target nodejs \
    --out-dir wasm-node-test/pkg \
    --no-typescript

# Node treats this directory as ESM (`"type": "module"` in package.json).
# wasm-bindgen --target nodejs produces CommonJS, so rename the JS glue
# from .js to .cjs to opt it back into CommonJS.
if [[ -f wasm-node-test/pkg/dv_wasm.js ]]; then
    mv wasm-node-test/pkg/dv_wasm.js wasm-node-test/pkg/dv_wasm.cjs
fi

ls -lh wasm-node-test/pkg/
