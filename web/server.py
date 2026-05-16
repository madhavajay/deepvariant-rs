#!/usr/bin/env python3
"""Dumb static file server for the no-backend browser bundle.

No data ever reaches this process — it only serves JS/HTML/WASM/the
ONNX model. Adds COOP/COEP so the page is cross-origin isolated
(SharedArrayBuffer → onnxruntime-web wasm threads; harmless for the
WebGPU path) and the right wasm MIME type.
"""
import http.server
import socketserver
import sys
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
ROOT = Path(__file__).parent / "public"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def guess_type(self, path):
        if str(path).endswith(".wasm"):
            return "application/wasm"
        if str(path).endswith(".mjs"):
            return "text/javascript"
        return super().guess_type(path)


class TCP(socketserver.TCPServer):
    allow_reuse_address = True


with TCP(("0.0.0.0", PORT), Handler) as httpd:
    print(f"\n  deepvariant-rs (in-browser, no backend) → http://localhost:{PORT}/\n")
    httpd.serve_forever()
