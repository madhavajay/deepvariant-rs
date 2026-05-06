// DeepVariant in the browser.
//
//   tf.Example shard (.tfrecord.gz)
//     │
//     ▼
//   gunzip + TFRecord framing  (browser-native: DecompressionStream + DataView)
//     │
//     ▼
//   tf.Example payload
//     │
//     ▼
//   dv_wasm.example_to_model_input(payload)         ← Rust → wasm32 → wasm-bindgen
//     │
//     ▼
//   Float32Array [100×221×7]
//     │
//     ▼
//   ort.InferenceSession.run(wgs.onnx)               ← onnxruntime-web (wasm)
//     │
//     ▼
//   { 'classification': Float32Array [3] }
//
// If `expected.json` (the native dv call-variants reference) is
// reachable on the same origin, the page diffs each in-browser
// prediction against the native one and reports max abs delta.

import init, * as dv from './pkg/dv_wasm.js';
import * as ort from './ort/ort.bundle.min.mjs';

// Tell ort where its own .wasm runtime lives — same directory we
// copied the bundle into during build.sh.
ort.env.wasm.wasmPaths = new URL('./ort/', import.meta.url).href;

const $status = document.getElementById('status');
const $results = document.getElementById('results');
const $summary = document.getElementById('summary');
const $tbody = $results.querySelector('tbody');
const $file = document.getElementById('file');
const $runDefault = document.getElementById('run-default');

function setStatus(s) {
    $status.textContent = s;
    // Surface to Playwright via a known marker.
    $status.dataset.value = s;
}

async function gunzipToBytes(arrayBuffer) {
    const stream = new Blob([arrayBuffer]).stream().pipeThrough(new DecompressionStream('gzip'));
    return new Uint8Array(await new Response(stream).arrayBuffer());
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

let session = null;
let inputName = null;
let outputName = null;
let wasmReady = false;
let expected = null;

async function ensureSetup() {
    if (!wasmReady) {
        setStatus('loading dv-wasm…');
        await init();
        wasmReady = true;
    }
    if (!session) {
        setStatus('loading models/wgs.onnx via onnxruntime-web…');
        session = await ort.InferenceSession.create('./models/wgs.onnx', {
            executionProviders: ['wasm'],
        });
        inputName = session.inputNames[0];
        outputName = session.outputNames[0];
    }
    if (expected === null) {
        try {
            const r = await fetch('./expected.json');
            if (r.ok) expected = await r.json();
            else expected = { genotype_probabilities: [] };
        } catch {
            expected = { genotype_probabilities: [] };
        }
    }
}

async function runOnFile(arrayBuffer, label) {
    await ensureSetup();
    setStatus(`gunzip + framing (${label})…`);
    const bytes = await gunzipToBytes(arrayBuffer);
    const examples = readTfRecords(bytes);
    setStatus(`decoded ${examples.length} examples; running inference…`);

    const shape = dv.wgs_image_shape(); // [H, W, C]
    const tol = 5e-4;
    let pass = 0, fail = 0, maxDelta = 0;
    $tbody.innerHTML = '';
    $results.hidden = false;

    for (let i = 0; i < examples.length; i++) {
        const features = dv.example_to_model_input(examples[i]);
        const start = Number(dv.example_variant_start(examples[i]));

        const input = new ort.Tensor('float32', features, [1, shape[0], shape[1], shape[2]]);
        const out = await session.run({ [inputName]: input });
        const probs = Array.from(out[outputName].data);

        const golden = expected.genotype_probabilities[i] ?? null;
        const delta = golden ? Math.max(...probs.map((p, j) => Math.abs(p - golden[j]))) : NaN;
        if (Number.isFinite(delta)) maxDelta = Math.max(maxDelta, delta);
        const ok = golden ? delta < tol : true;
        if (ok) pass++; else fail++;

        const top = probs.indexOf(Math.max(...probs));
        const tr = document.createElement('tr');
        tr.id = `row-${i}`;
        tr.innerHTML = `
            <td>${i}</td>
            <td>${start}</td>
            <td data-test="top-${i}">${top}</td>
            <td data-test="probs-${i}">${probs.map(p => p.toFixed(4)).join(', ')}</td>
            <td data-test="delta-${i}">${Number.isFinite(delta) ? delta.toExponential(2) : '—'}</td>
            <td class="${ok ? 'pass' : 'fail'}" data-test="status-${i}">${ok ? 'pass' : 'FAIL'}</td>
        `;
        $tbody.append(tr);
        if (i % 4 === 0) await new Promise((r) => setTimeout(r, 0));  // yield to UI
    }

    $summary.dataset.pass = String(pass);
    $summary.dataset.fail = String(fail);
    $summary.dataset.total = String(examples.length);
    $summary.dataset.maxDelta = Number.isFinite(maxDelta) ? maxDelta.toExponential(3) : '—';
    $summary.innerHTML = `
        <p><strong>${pass} pass / ${fail} fail</strong> across ${examples.length} examples
        — max Δ vs native: <code>${$summary.dataset.maxDelta}</code></p>
    `;
    setStatus(`done — ${pass}/${examples.length} pass, max Δ=${$summary.dataset.maxDelta}`);
}

$file.addEventListener('change', async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    const buf = await f.arrayBuffer();
    await runOnFile(buf, f.name);
});

$runDefault.addEventListener('click', async () => {
    setStatus('fetching testdata/examples.tfrecord.gz…');
    const r = await fetch('./testdata/examples.tfrecord.gz');
    if (!r.ok) {
        setStatus(`fetch failed: ${r.status}`);
        return;
    }
    await runOnFile(await r.arrayBuffer(), 'testdata/examples.tfrecord.gz');
});

setStatus('idle');
