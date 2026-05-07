// DeepVariant in the browser — orchestrator.
//
//   Step 1 (button):  init dv-wasm                            ← wasm-bindgen --target web
//   Step 2 (button):  fetch wgs.onnx, store in CacheStorage   ← progress + persistent
//   Step 3 (button):  run on bundled sample OR user file      ← drag/drop + picker
//   Step 4 (button):  download .tar.gz of predictions         ← USTAR + CompressionStream
//
// All in-browser. Inference goes through onnxruntime-web's wasm
// backend; image decoding and variant proto parsing go through the
// Rust → wasm32 dv-wasm crate.

import { tarGz } from './tar.mjs';

// dv-wasm and onnxruntime-web are served from /pkg/ and /ort/ in the
// public dir. Vite's static import-analysis refuses to resolve files
// inside /public, so we load them at runtime via runtime-computed
// URLs (which Vite can't analyze) — same files, same URLs, just lazy.
let dv, ort;
async function ensureModules() {
    if (!dv) {
        const url = new URL('/pkg/dv_wasm.js', window.location.origin).href;
        dv = await import(/* @vite-ignore */ url);
    }
    if (!ort) {
        const url = new URL('/ort/ort.bundle.min.mjs', window.location.origin).href;
        ort = await import(/* @vite-ignore */ url);
        // onnxruntime-web's runtime .wasm files live next to its mjs bundle.
        ort.env.wasm.wasmPaths = new URL('/ort/', window.location.origin).href;
    }
}

const MODEL_URL = './models/wgs.onnx';
const MODEL_CACHE = 'dv-model-v1';
const MODEL_CACHE_KEY = 'wgs.onnx';

// ─────────────── DOM helpers ───────────────
const $ = (id) => document.getElementById(id);

const ui = {
    wasmStatus: $('wasm-status'),
    wasmBadge: $('wasm-badge'),
    btnLoadWasm: $('btn-load-wasm'),

    modelStatus: $('model-status'),
    modelBadge: $('model-badge'),
    modelProgress: $('model-progress'),
    btnDownloadModel: $('btn-download-model'),
    btnClearModel: $('btn-clear-model'),

    runStatus: $('run-status'),
    runProgress: $('run-progress'),
    btnRunSample: $('btn-run-sample'),
    drop: $('drop'),
    file: $('file'),

    panelResults: $('panel-results'),
    btnDownload: $('btn-download'),
    stats: $('stats'),
    resultsBody: document.querySelector('#results tbody'),
};

function setBadge(el, text, kind) {
    el.textContent = text;
    el.className = 'badge' + (kind ? ' ' + kind : '');
}
function setStatus(el, text) { el.textContent = text; el.dataset.value = text; }
function setProgress(panel, frac) {
    const wrap = panel === 'model' ? ui.modelProgress : ui.runProgress;
    wrap.hidden = frac == null;
    if (frac != null) {
        wrap.firstElementChild.style.width = `${(frac * 100).toFixed(1)}%`;
    }
}

// ─────────────── Step 1: dv-wasm ───────────────
let wasmReady = false;
async function loadWasm() {
    if (wasmReady) return;
    setStatus(ui.wasmStatus, 'loading dv_wasm_bg.wasm…');
    setBadge(ui.wasmBadge, 'loading', 'warn');
    ui.btnLoadWasm.disabled = true;
    try {
        await ensureModules();
        await dv.default();
        const shape = dv.wgs_image_shape();
        wasmReady = true;
        setStatus(ui.wasmStatus, `loaded · WGS image shape [H=${shape[0]}, W=${shape[1]}, C=${shape[2]}]`);
        setBadge(ui.wasmBadge, 'loaded', 'ok');
        ui.btnLoadWasm.textContent = 'Reload WASM';
    } catch (err) {
        setStatus(ui.wasmStatus, `error: ${err.message}`);
        setBadge(ui.wasmBadge, 'error', 'err');
        throw err;
    } finally {
        ui.btnLoadWasm.disabled = false;
    }
}

// ─────────────── Step 2: model fetch + cache ───────────────
async function getCachedModel() {
    if (!('caches' in window)) return null;
    const cache = await caches.open(MODEL_CACHE);
    const hit = await cache.match(MODEL_CACHE_KEY);
    return hit || null;
}

async function refreshModelBadge() {
    const hit = await getCachedModel();
    if (hit) {
        const blob = await hit.clone().blob();
        const mb = (blob.size / (1024 * 1024)).toFixed(1);
        setBadge(ui.modelBadge, `cached · ${mb} MB`, 'ok');
        setStatus(ui.modelStatus, `cached in CacheStorage · ${blob.size} bytes`);
        ui.btnDownloadModel.textContent = 'Re-download model';
        return blob;
    }
    setBadge(ui.modelBadge, 'not cached', 'warn');
    setStatus(ui.modelStatus, 'idle — click "Download model" to fetch and cache');
    ui.btnDownloadModel.textContent = 'Download model (~87MB)';
    return null;
}

async function downloadAndCacheModel() {
    ui.btnDownloadModel.disabled = true;
    ui.btnClearModel.disabled = true;
    setStatus(ui.modelStatus, `fetching ${MODEL_URL}…`);
    setProgress('model', 0);
    try {
        const resp = await fetch(MODEL_URL);
        if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching ${MODEL_URL}`);
        const total = Number(resp.headers.get('content-length')) || 0;
        const reader = resp.body.getReader();
        const chunks = [];
        let received = 0;
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            chunks.push(value);
            received += value.length;
            if (total) {
                setProgress('model', received / total);
                setStatus(ui.modelStatus, `${(received / 1024 / 1024).toFixed(1)} / ${(total / 1024 / 1024).toFixed(1)} MB`);
            } else {
                setStatus(ui.modelStatus, `${(received / 1024 / 1024).toFixed(1)} MB (no content-length)`);
            }
        }
        const blob = new Blob(chunks, { type: 'application/octet-stream' });
        const cache = await caches.open(MODEL_CACHE);
        await cache.put(MODEL_CACHE_KEY, new Response(blob, {
            headers: { 'Content-Type': 'application/octet-stream', 'Content-Length': String(blob.size) },
        }));
        setProgress('model', null);
        await refreshModelBadge();
    } catch (err) {
        setStatus(ui.modelStatus, `error: ${err.message}`);
        setBadge(ui.modelBadge, 'error', 'err');
        setProgress('model', null);
    } finally {
        ui.btnDownloadModel.disabled = false;
        ui.btnClearModel.disabled = false;
    }
}

async function clearModelCache() {
    if ('caches' in window) await caches.delete(MODEL_CACHE);
    session = null; // force re-create on next run
    await refreshModelBadge();
}

// Lazily build an InferenceSession from the cached model.
let session = null;
let inputName = null;
let outputName = null;
async function ensureSession() {
    if (session) return session;
    let blob = await getCachedModel().then((r) => r && r.blob());
    if (!blob) {
        await downloadAndCacheModel();
        blob = await getCachedModel().then((r) => r && r.blob());
        if (!blob) throw new Error('model not available after download attempt');
    }
    setStatus(ui.runStatus, 'creating onnxruntime-web session…');
    const buf = new Uint8Array(await blob.arrayBuffer());
    session = await ort.InferenceSession.create(buf, { executionProviders: ['wasm'] });
    inputName = session.inputNames[0];
    outputName = session.outputNames[0];
    return session;
}

// ─────────────── Step 3: run inference ───────────────
async function maybeGunzipToBytes(arrayBuffer) {
    // Vite's dev server (and many production hosts) auto-add
    // `Content-Encoding: gzip` for .gz files, which means the browser
    // has already transparently decompressed the response body before
    // our code sees it. Detect the gzip magic bytes (0x1f 0x8b) and
    // only run DecompressionStream when the data is actually still
    // gzipped.
    const head = new Uint8Array(arrayBuffer, 0, Math.min(2, arrayBuffer.byteLength));
    if (head.length >= 2 && head[0] === 0x1f && head[1] === 0x8b) {
        const stream = new Blob([arrayBuffer]).stream().pipeThrough(new DecompressionStream('gzip'));
        return new Uint8Array(await new Response(stream).arrayBuffer());
    }
    return new Uint8Array(arrayBuffer);
}

function readTfRecords(bytes) {
    const records = [];
    let off = 0;
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    while (off + 12 < bytes.length) {
        const len = Number(view.getBigUint64(off, true));
        off += 8 + 4; // length + length-crc
        records.push(bytes.subarray(off, off + len));
        off += len + 4; // payload + payload-crc
    }
    return records;
}

let lastRun = null; // { sourceName, results, ranAt }

async function runOnArrayBuffer(arrayBuffer, sourceName) {
    if (!wasmReady) await loadWasm();
    await ensureSession();

    setStatus(ui.runStatus, `gunzip + framing (${sourceName})…`);
    const bytes = await maybeGunzipToBytes(arrayBuffer);
    const examples = readTfRecords(bytes);
    setStatus(ui.runStatus, `decoded ${examples.length} examples; running inference…`);
    setProgress('run', 0);

    const shape = dv.wgs_image_shape();
    const results = [];
    ui.resultsBody.innerHTML = '';
    ui.panelResults.hidden = true;

    const t0 = performance.now();
    for (let i = 0; i < examples.length; i++) {
        const features = dv.example_to_model_input(examples[i]);
        const summary = JSON.parse(dv.example_variant_summary(examples[i]));

        const input = new ort.Tensor('float32', features, [1, shape[0], shape[1], shape[2]]);
        const out = await session.run({ [inputName]: input });
        const probs = Array.from(out[outputName].data);

        const top = probs.indexOf(Math.max(...probs));
        results.push({ index: i, ...summary, top, probs });

        if ((i & 3) === 0 || i === examples.length - 1) {
            setProgress('run', (i + 1) / examples.length);
            setStatus(ui.runStatus, `inference ${i + 1}/${examples.length}`);
            await new Promise((r) => setTimeout(r, 0)); // yield to UI
        }
    }
    const elapsed = performance.now() - t0;

    renderResults(results, sourceName, elapsed);
    lastRun = { sourceName, results, ranAt: new Date().toISOString(), elapsedMs: elapsed };
    setProgress('run', null);
    setStatus(ui.runStatus, `done — ${examples.length} examples in ${(elapsed / 1000).toFixed(2)}s`);
}

function renderResults(results, sourceName, elapsedMs) {
    const total = results.length;
    const counts = [0, 0, 0];
    for (const r of results) counts[r.top]++;
    const variantCount = counts[1] + counts[2]; // het + hom-alt

    ui.stats.innerHTML = `
        <div class="stat"><div class="v">${total}</div><div class="l">examples</div></div>
        <div class="stat"><div class="v">${variantCount}</div><div class="l">variant calls</div></div>
        <div class="stat"><div class="v">${counts[0]}</div><div class="l">hom-ref</div></div>
        <div class="stat"><div class="v">${counts[1]}</div><div class="l">het</div></div>
        <div class="stat"><div class="v">${counts[2]}</div><div class="l">hom-alt</div></div>
        <div class="stat"><div class="v">${(elapsedMs / total).toFixed(0)}<small style="font-size:0.6em; color:var(--muted);">ms</small></div><div class="l">avg / example</div></div>
    `;

    const TOP_LABELS = ['hom-ref', 'het', 'hom-alt'];
    const frag = document.createDocumentFragment();
    for (const r of results) {
        const tr = document.createElement('tr');
        const altsStr = r.alts.join(',') || '·';
        tr.innerHTML = `
            <td>${r.index}</td>
            <td>${r.chrom}:${r.start + 1}</td>
            <td>${r.ref}→${altsStr}</td>
            <td class="${r.top > 0 ? 'pass' : ''}">${TOP_LABELS[r.top]}</td>
            <td>${r.probs[0].toFixed(4)}</td>
            <td>${r.probs[1].toFixed(4)}</td>
            <td>${r.probs[2].toFixed(4)}</td>
        `;
        frag.append(tr);
    }
    ui.resultsBody.append(frag);
    ui.panelResults.hidden = false;
}

// ─────────────── Step 4: download bundle ───────────────
async function downloadBundle() {
    if (!lastRun) return;
    const { sourceName, results, ranAt, elapsedMs } = lastRun;
    const TOP_LABELS = ['hom-ref', 'het', 'hom-alt'];

    const predictions = {
        meta: {
            tool: 'deepvariant-rs / dv-wasm + onnxruntime-web',
            ran_at: ranAt,
            source: sourceName,
            elapsed_ms: Math.round(elapsedMs),
            example_count: results.length,
            classes: TOP_LABELS,
            note: 'These are call-variants outputs; postprocess-variants (VCF emission) is not yet wired in WASM. See README.md.',
        },
        predictions: results.map((r) => ({
            index: r.index,
            chrom: r.chrom,
            start: r.start,
            end: r.end,
            ref: r.ref,
            alts: r.alts,
            top_class: TOP_LABELS[r.top],
            probabilities: { hom_ref: r.probs[0], het: r.probs[1], hom_alt: r.probs[2] },
        })),
    };

    const summaryLines = [
        `DeepVariant in the browser — predictions summary`,
        ``,
        `source:        ${sourceName}`,
        `ran_at:        ${ranAt}`,
        `elapsed:       ${(elapsedMs / 1000).toFixed(2)}s (${(elapsedMs / results.length).toFixed(1)} ms/example)`,
        `examples:      ${results.length}`,
        `hom-ref calls: ${results.filter((r) => r.top === 0).length}`,
        `het calls:     ${results.filter((r) => r.top === 1).length}`,
        `hom-alt calls: ${results.filter((r) => r.top === 2).length}`,
        ``,
        `# index  chrom:pos       ref→alt        top       p(hom-ref)  p(het)      p(hom-alt)`,
    ];
    for (const r of results) {
        const altsStr = r.alts.join(',') || '·';
        summaryLines.push(
            `${String(r.index).padStart(5)}  ` +
            `${(r.chrom + ':' + (r.start + 1)).padEnd(15)} ` +
            `${(r.ref + '→' + altsStr).padEnd(14)} ` +
            `${TOP_LABELS[r.top].padEnd(9)} ` +
            `${r.probs[0].toFixed(6).padStart(10)}  ${r.probs[1].toFixed(6).padStart(10)}  ${r.probs[2].toFixed(6).padStart(10)}`
        );
    }

    const readme = `# DeepVariant in the browser — output bundle

Generated by the dv-wasm + onnxruntime-web dev frontend.

## Files
- predictions.json  Structured per-example predictions (machine-readable).
- summary.txt       Same data, human-readable / grep-friendly.
- README.md         This file.

## What's *not* in this bundle yet
- **VCF.gz / .tbi** — the postprocess-variants step (CallVariantsOutput records
  → bgzipped VCF + tabix index) is not yet ported to WASM. The native
  \`dv postprocess-variants\` does this server-side today; the same noodles-vcf
  + noodles-bgzf path will work in WASM once wired in. predictions.json carries
  enough information to reproduce the post-processing step offline:

      $ dv postprocess-variants --cvo <generated-cvo.tfrecord> --output-vcf out.vcf.gz

  The \`predictions.json\` here matches the per-row CVO probabilities exactly.

## Pipeline
  tf.Example shard (.tfrecord.gz)
    → DecompressionStream('gzip')                     [browser-native]
    → TFRecord framing                                [browser-native]
    → dv_wasm.example_to_model_input()                [Rust → wasm32]
    → onnxruntime-web InferenceSession.run()          [wasm SIMD]
    → predictions.json (this file)
`;

    const baseName = sourceName.replace(/\.tfrecord\.gz$|\.gz$|\.tfrecord$/i, '');
    const stem = (baseName || 'predictions').replace(/[^a-zA-Z0-9._-]/g, '_');
    const archiveName = `${stem}-deepvariant.tar.gz`;

    setStatus(ui.runStatus, `building ${archiveName}…`);
    const bytes = await tarGz([
        { name: `${stem}/predictions.json`, data: JSON.stringify(predictions, null, 2) },
        { name: `${stem}/summary.txt`, data: summaryLines.join('\n') + '\n' },
        { name: `${stem}/README.md`, data: readme },
    ]);

    const url = URL.createObjectURL(new Blob([bytes], { type: 'application/gzip' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = archiveName;
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 30_000);
    setStatus(ui.runStatus, `downloaded ${archiveName} (${(bytes.length / 1024).toFixed(1)} KB)`);
}

// ─────────────── Wire-up ───────────────
ui.btnLoadWasm.addEventListener('click', () => loadWasm());
ui.btnDownloadModel.addEventListener('click', () => downloadAndCacheModel());
ui.btnClearModel.addEventListener('click', () => clearModelCache());
ui.btnDownload.addEventListener('click', () => downloadBundle());

ui.btnRunSample.addEventListener('click', async () => {
    setStatus(ui.runStatus, 'fetching testdata/examples.tfrecord.gz…');
    const r = await fetch('./testdata/examples.tfrecord.gz');
    if (!r.ok) {
        setStatus(ui.runStatus, `fetch failed: ${r.status}`);
        return;
    }
    await runOnArrayBuffer(await r.arrayBuffer(), 'testdata/examples.tfrecord.gz');
});

ui.file.addEventListener('change', async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    await runOnArrayBuffer(await f.arrayBuffer(), f.name);
});

// Drag-and-drop on the .drop region.
['dragenter', 'dragover'].forEach((ev) =>
    ui.drop.addEventListener(ev, (e) => {
        e.preventDefault();
        ui.drop.classList.add('over');
    })
);
['dragleave', 'drop'].forEach((ev) =>
    ui.drop.addEventListener(ev, (e) => {
        e.preventDefault();
        ui.drop.classList.remove('over');
    })
);
ui.drop.addEventListener('drop', async (e) => {
    const f = e.dataTransfer?.files?.[0];
    if (!f) return;
    await runOnArrayBuffer(await f.arrayBuffer(), f.name);
});

// On boot: surface model cache state (no auto-download).
refreshModelBadge();
setStatus(ui.wasmStatus, 'idle — click "Load WASM" to initialize');
setStatus(ui.runStatus, 'idle');
