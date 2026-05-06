// End-to-end ORT-through-wasm test, command-line edition.
//
// Pipeline:
//   1. Read tf.Example records from a gzipped TFRecord shard
//      (testdata/quickstart_chr20_norealign/examples.tfrecord.gz).
//   2. For each example: call dv-wasm via wasm-bindgen to decode it
//      into the 100×221×7 float32 model input.
//   3. Run that input through models/wgs.onnx via onnxruntime-node.
//   4. Validate the predictions look like real probability vectors
//      (3 classes, sum ≈ 1, all in [0, 1]).
//
// What this proves: the dv-wasm Rust port produces a model input that
// the ONNX Runtime accepts and turns into well-formed predictions.
// Same dv-wasm crate (134KB .wasm) used here will also drive the
// browser path with onnxruntime-web — only the JS host changes.

import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import zlib from 'node:zlib';
import process from 'node:process';
import { createRequire } from 'node:module';
import { spawnSync } from 'node:child_process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(__dirname, '..');
const EXAMPLES = path.join(REPO, 'testdata/quickstart_chr20_norealign/examples.tfrecord.gz');
// Generated on demand by `dv call-variants` so example count and CVO
// count are guaranteed to match 1:1. The pre-existing testdata file
// has multi-allelic CVOs split into separate records, which breaks the
// simple by-index lookup we want here.
const GOLDEN_CVOS = path.join(REPO, 'wasm-node-test/cvos.fresh.tfrecord.gz');
const MODEL = path.join(REPO, 'models/wgs.onnx');
// Tolerance for wasm/onnxruntime-node vs golden native ORT predictions.
// Both pathways run the same ONNX graph, so disagreement is float-rounding only.
const TOL = 5e-4;

// wasm-bindgen --target nodejs emits CommonJS; load it through
// createRequire so this ESM file can use it directly.
const require = createRequire(import.meta.url);
const wasm = require('./pkg/dv_wasm.cjs');
const ort = await import('onnxruntime-node');

// --- Minimal TFRecord reader -------------------------------------------------
// TFRecord layout per record:
//   uint64 length (LE)
//   uint32 length-CRC (LE) — skipped, fixture is trusted
//   bytes  payload
//   uint32 payload-CRC (LE) — skipped
function readTfRecords(gzPath) {
    const raw = zlib.gunzipSync(readFileSync(gzPath));
    const records = [];
    let off = 0;
    const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    while (off + 12 < raw.length) {
        const len = Number(view.getBigUint64(off, true));
        off += 8 + 4; // length + length-crc
        records.push(raw.subarray(off, off + len));
        off += len + 4; // payload + payload-crc
    }
    return records;
}

// --- Minimal CallVariantsOutput proto reader --------------------------------
// We only need genotype_probabilities (field 3, packed `repeated double`).
// Wire format: tag = (3<<3)|2 = 0x1A, then varint length, then 8-byte
// little-endian doubles.
function readVarint(buf, off) {
    let result = 0;
    let shift = 0;
    let i = off;
    while (true) {
        const b = buf[i++];
        result |= (b & 0x7f) << shift;
        if ((b & 0x80) === 0) return [result, i];
        shift += 7;
    }
}
function extractGenotypeProbs(payload) {
    const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
    let off = 0;
    while (off < payload.length) {
        const tag = payload[off++];
        const fieldNum = tag >> 3;
        const wireType = tag & 7;
        if (fieldNum === 3 && wireType === 2) {
            const [len, after] = readVarint(payload, off);
            off = after;
            const out = [];
            for (let i = 0; i < len; i += 8) {
                out.push(view.getFloat64(off + i, true));
            }
            return out;
        }
        // skip other fields by wire type
        if (wireType === 0) {
            const [, n] = readVarint(payload, off);
            off = n;
        } else if (wireType === 2) {
            const [len, after] = readVarint(payload, off);
            off = after + len;
        } else if (wireType === 1) {
            off += 8;
        } else if (wireType === 5) {
            off += 4;
        } else {
            throw new Error(`unsupported wire type ${wireType}`);
        }
    }
    return null;
}

// --- Generate the native ORT reference if missing ----------------------------
// Reuses the shipped `dv` binary. This is the side-by-side native run we
// diff our wasm + onnxruntime-node predictions against. Idempotent — only
// runs if cvos.fresh.tfrecord.gz is missing.
function ensureGoldenCvos() {
    if (existsSync(GOLDEN_CVOS)) return;
    const dv = path.join(REPO, 'target/release/dv');
    if (!existsSync(dv)) {
        console.error(`error: ${dv} not found — run \`cargo build -p dv-cli --release\` first`);
        process.exit(2);
    }
    console.log(`[setup] running native dv call-variants for golden CVOs…`);
    const res = spawnSync(
        dv,
        [
            'call-variants',
            '--examples', EXAMPLES,
            '--checkpoint', MODEL,
            '--output', GOLDEN_CVOS,
        ],
        { stdio: 'inherit' }
    );
    if (res.status !== 0) {
        console.error('error: dv call-variants failed');
        process.exit(2);
    }
}

// --- Run the test ------------------------------------------------------------
const t0 = Date.now();
ensureGoldenCvos();
const examples = readTfRecords(EXAMPLES);
const goldenCvoBytes = readTfRecords(GOLDEN_CVOS);
const goldenProbs = goldenCvoBytes.map(extractGenotypeProbs);
if (goldenProbs.length !== examples.length) {
    console.error(`example count ${examples.length} != golden CVO count ${goldenProbs.length}`);
    process.exit(2);
}
console.log(`Read ${examples.length} tf.Example records from ${path.relative(REPO, EXAMPLES)}`);

// Process every example by default; allow shrinking via WASM_TEST_SAMPLE for
// quick iteration (e.g. during interactive debugging).
const SAMPLE = process.env.WASM_TEST_SAMPLE
    ? Math.min(parseInt(process.env.WASM_TEST_SAMPLE, 10), examples.length)
    : examples.length;
const indices = Array.from({ length: SAMPLE }, (_, i) =>
    SAMPLE === examples.length ? i : Math.floor((i * examples.length) / SAMPLE)
);

const session = await ort.InferenceSession.create(MODEL);
const inputName = session.inputNames[0];
const outputName = session.outputNames[0];
const shape = wasm.wgs_image_shape();
console.log(`Model loaded: ${path.relative(REPO, MODEL)}`);
console.log(`  input '${inputName}'  shape from dv-wasm: [${shape.join(', ')}]`);
console.log(`  output '${outputName}'`);
console.log();

let pass = 0;
let fail = 0;
for (const i of indices) {
    const payload = examples[i];

    // 1. dv-wasm: decode the tf.Example payload to the float32 model input.
    let features;
    try {
        features = wasm.example_to_model_input(payload);
    } catch (err) {
        console.log(`  example #${i.toString().padStart(4)}: wasm decode failed: ${err}`);
        fail++;
        continue;
    }
    const expectedLen = shape[0] * shape[1] * shape[2];
    if (features.length !== expectedLen) {
        console.log(`  example #${i.toString().padStart(4)}: feature len ${features.length} != ${expectedLen}`);
        fail++;
        continue;
    }
    const start = wasm.example_variant_start(payload);

    // 2. onnxruntime-node: run inference on the wasm-produced features.
    const tensor = new ort.Tensor('float32', features, [1, shape[0], shape[1], shape[2]]);
    const outputs = await session.run({ [inputName]: tensor });
    const probs = Array.from(outputs[outputName].data);

    // 3. Compare against the golden native-ORT predictions.
    const golden = goldenProbs[i];
    if (!golden || golden.length !== probs.length) {
        console.log(
            `  example #${i.toString().padStart(4)}: missing/short golden CVO (${golden?.length ?? 'null'} vs ${probs.length})`
        );
        fail++;
        continue;
    }
    const maxAbsDiff = Math.max(...probs.map((p, j) => Math.abs(p - golden[j])));
    const top = probs.indexOf(Math.max(...probs));
    const goldenTop = golden.indexOf(Math.max(...golden));
    const ok = maxAbsDiff < TOL && top === goldenTop;

    const probsStr = probs.map((p) => p.toFixed(4)).join(', ');
    const goldStr = golden.map((p) => p.toFixed(4)).join(', ');
    const status = ok ? 'pass' : 'FAIL';
    console.log(
        `  example #${i.toString().padStart(4)}  pos=${start}` +
        `  wasm=[${probsStr}]  golden=[${goldStr}]` +
        `  Δmax=${maxAbsDiff.toExponential(2)}  ${status}`
    );
    if (ok) pass++;
    else fail++;
}

const elapsed = ((Date.now() - t0) / 1000).toFixed(2);
console.log();
console.log(`Sampled ${SAMPLE}/${examples.length} examples in ${elapsed}s — ${pass} pass / ${fail} fail`);
process.exit(fail === 0 ? 0 : 1);
