// DeepVariant-rs, entirely in the browser. No backend, no upload.
//
//   BAM bytes + reference FASTA bytes
//     → dv_wasm.Pipeline (Rust→wasm: parse BAM, allele count,
//       candidates, realigner ≤8 hap/window, pileup render)
//     → per-batch onnxruntime-web inference (WebGPU, wasm fallback)
//     → Pipeline.to_vcf (Rust→wasm: postprocess, dup-alt fix)
//     → download .vcf
//
// Parity with the native `dv pipeline` is byte-identical (validated
// on NA06985 chr22 — see commit d82d894).

import init, * as dv from './pkg/dv_wasm.js';
import * as ort from './ort/ort.bundle.min.mjs';

ort.env.wasm.wasmPaths = new URL('./ort/', import.meta.url).href;

const $ = (id) => document.getElementById(id);
const bar = $('bar').firstElementChild;
let bam = null, fa = null, session = null, inputName = null, outputName = null;

function log(s) { $('log').textContent += s + '\n'; $('log').scrollTop = 1e9; }
function setBar(f) { bar.style.width = (f * 100).toFixed(1) + '%'; }

function wireDrop(dropId, inputId, onPick) {
  const d = $(dropId), inp = $(inputId);
  d.onclick = () => inp.click();
  inp.onchange = (e) => e.target.files[0] && onPick(e.target.files[0]);
  ['dragover', 'dragenter'].forEach(ev =>
    d.addEventListener(ev, e => { e.preventDefault(); d.classList.add('over'); }));
  ['dragleave', 'drop'].forEach(ev =>
    d.addEventListener(ev, e => { e.preventDefault(); d.classList.remove('over'); }));
  d.addEventListener('drop', e =>
    e.dataTransfer.files[0] && onPick(e.dataTransfer.files[0]));
}
function ready() { $('run').disabled = !(bam && fa); }
wireDrop('dropBam', 'fileBam', f => {
  bam = f; const d = $('dropBam');
  d.classList.add('has'); d.textContent = `✓ ${f.name} (${(f.size/1e6).toFixed(1)} MB)`;
  ready();
});
wireDrop('dropFa', 'fileFa', f => {
  fa = f; const d = $('dropFa');
  d.classList.add('has'); d.textContent = `✓ ${f.name} (${(f.size/1e6).toFixed(1)} MB)`;
  ready();
});

async function readBytes(file) {
  return new Uint8Array(await file.arrayBuffer());
}

async function makeSession() {
  // WebGPU first (the fast path the user asked for); fall back to
  // wasm (SIMD+threads) so it still runs everywhere.
  for (const ep of ['webgpu', 'wasm']) {
    try {
      const s = await ort.InferenceSession.create('./models/wgs.onnx', {
        executionProviders: [ep],
        graphOptimizationLevel: 'all',
      });
      $('ep').textContent = 'execution provider: ' + ep.toUpperCase();
      log('onnxruntime-web ready (' + ep + ')');
      return s;
    } catch (e) {
      log(`EP ${ep} unavailable: ${e}`);
    }
  }
  throw new Error('no usable onnxruntime-web execution provider');
}

$('run').onclick = async () => {
  $('run').disabled = true;
  $('log').textContent = '';
  $('done').innerHTML = '';
  setBar(0);
  const region = $('region').value.trim();
  const t0 = performance.now();
  try {
    await init();
    if (!session) {
      session = await makeSession();
      inputName = session.inputNames[0];
      outputName = session.outputNames[0];
    }
    log(`reading ${bam.name} + ${fa.name}…`);
    const [bamBytes, faBytes] = await Promise.all([readBytes(bam), readBytes(fa)]);

    log('make-examples (wasm)…');
    const pipe = new dv.Pipeline(bamBytes, faBytes, region);
    const n = pipe.len();
    log(`${n} examples`);
    if (n === 0) { throw new Error('no candidate examples in region'); }

    const [H, W, C] = dv.wgs_image_shape();
    const PIX = H * W * C;
    const BATCH = 32;
    const probs = new Float32Array(n * 3);
    let done = 0;
    for (let s = 0; s < n; s += BATCH) {
      const b = Math.min(BATCH, n - s);
      const buf = new Float32Array(b * PIX);
      for (let k = 0; k < b; k++) {
        buf.set(dv.example_to_model_input(pipe.example(s + k)), k * PIX);
      }
      const t = new ort.Tensor('float32', buf, [b, H, W, C]);
      const out = await session.run({ [inputName]: t });
      probs.set(out[outputName].data.subarray(0, b * 3), s * 3);
      done += b;
      setBar(done / n);
      const el = (performance.now() - t0) / 1000;
      const eta = el / done * (n - done);
      $('eta').textContent =
        `${done}/${n} examples · ${el.toFixed(0)}s elapsed · ~${eta.toFixed(0)}s left`;
    }

    log('postprocess (wasm)…');
    const contig = region.split(':')[0];
    const end = parseInt(region.split('-')[1] || '300000000', 10);
    const vcf = pipe.to_vcf(probs, JSON.stringify([[contig, end]]), 'SAMPLE');
    const recs = vcf.split('\n').filter(l => l && l[0] !== '#').length;
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    $('eta').textContent = `done in ${secs}s`;
    log(`VCF: ${recs} variant records`);

    const url = URL.createObjectURL(new Blob([vcf], { type: 'text/plain' }));
    $('done').innerHTML =
      `<a class="dl" href="${url}" download="deepvariant.vcf">⬇ download VCF (${recs} records)</a>`;
  } catch (e) {
    log('❌ ' + (e && e.stack ? e.stack : e));
  } finally {
    $('run').disabled = false;
  }
};
