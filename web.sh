#!/usr/bin/env bash
# Launch the in-browser DeepVariant-rs UI — fully client-side WASM,
# NO backend, NO data upload. Builds the static bundle (dv-wasm +
# onnxruntime-web + model) then serves it as plain static files.
#
#   ./web.sh                 # build + serve on :8080
#   PORT=9000 ./web.sh
#   ./web.sh --no-build      # skip rebuild, just serve
#
# Then open http://localhost:$PORT/ and drag in a BAM + reference
# FASTA. Everything runs in the browser (WebGPU; wasm fallback).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8080}"

if [[ "${1:-}" != "--no-build" ]]; then
    bash "$ROOT/web/build.sh"
fi
[[ -f "$ROOT/web/public/pkg/dv_wasm.js" ]] || {
    echo "✗ bundle not built — run ./web.sh (without --no-build)" >&2; exit 1; }

exec python3 "$ROOT/web/server.py" "$PORT"
