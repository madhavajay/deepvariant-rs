import { defineConfig } from 'vite';

// Vite serves `public/` at the site root, which is where build.sh
// drops:
//   public/pkg/      ← dv-wasm + JS glue (wasm-bindgen --target web)
//   public/ort/      ← onnxruntime-web runtime (mjs + .wasm)
//   public/models/   ← wgs.onnx (symlinked from <repo>/models)
//   public/testdata/ ← examples.tfrecord.gz (symlinked)
//
// We import onnxruntime-web *from public/ort/* (not from npm) so the
// runtime files and the JS glue are guaranteed to come from the same
// build, exactly the same way wasm-browser-test does it. That avoids
// version-skew issues where Vite's optimizer rewrites the entry but
// the .wasm files don't get bundled.
export default defineConfig({
    root: 'src',
    publicDir: '../public',
    server: {
        port: 5173,
        // Serving 87MB+ model + .wasm requires the right MIME types.
        // Vite handles those by default, but we want strict cache
        // headers off in dev so the model fetch button actually
        // re-fetches when the user clears the cache.
        headers: {
            'Cache-Control': 'no-store',
        },
    },
    build: {
        outDir: '../dist',
        emptyOutDir: true,
        sourcemap: true,
    },
});
