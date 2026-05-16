// DeepVariant-rs, entirely in the browser. No backend, no upload.
//
//   BAM bytes + reference FASTA bytes
//     → dv_wasm.Pipeline (parse BAM, make-examples)
//     → per-batch onnxruntime-web inference (WebGPU, wasm fallback)
//     → Pipeline.to_vcf (postprocess) → download .vcf
//
// Byte-identical to native `dv pipeline` (validated, commit d82d894).

const log = (s) => window.__log(s);
const $ = (id) => document.getElementById(id);

let dv = null, ort = null, session = null, inputName = null, outputName = null;
let bam = null, fa = null;

// Eagerly load wasm + onnxruntime-web and detect the execution
// provider ON PAGE LOAD, so "detecting…" resolves immediately and
// any load failure is visible instead of a dead page.
(async () => {
  try {
    log('loading wasm + onnxruntime-web…');
    const dvmod = await import('./pkg/dv_wasm.js');
    await dvmod.default();           // wasm-bindgen init
    dv = dvmod;
    ort = await import('./ort/ort.bundle.min.mjs');
    ort.env.wasm.wasmPaths = new URL('./ort/', import.meta.url).href;

    for (const ep of ['webgpu', 'wasm']) {
      try {
        session = await ort.InferenceSession.create('./models/wgs.onnx', {
          executionProviders: [ep],
          graphOptimizationLevel: 'all',
        });
        inputName = session.inputNames[0];
        outputName = session.outputNames[0];
        $('ep').textContent = 'execution provider: ' + ep.toUpperCase()
          + ' · model loaded ✓';
        log('onnxruntime-web ready (' + ep + ')');
        break;
      } catch (e) {
        log(`EP ${ep} unavailable: ${e}`);
      }
    }
    if (!session) {
      $('ep').textContent = 'execution provider: NONE — inference unavailable';
    }
    refresh();
  } catch (e) {
    $('ep').textContent = 'failed to initialise — see log';
    log('❌ init: ' + (e && e.stack ? e.stack : e));
  }
})();

const bar = $('bar').firstElementChild;
const setBar = (f) => { bar.style.width = (f * 100).toFixed(1) + '%'; };

function refresh() {
  const need = [];
  if (!bam) need.push('BAM (.bam)');
  if (!fa) need.push('reference FASTA (.fa) — required, see note below');
  if (!session) need.push('onnxruntime-web (still loading / unavailable)');
  const r = $('run');
  if (need.length === 0) {
    r.disabled = false;
    $('eta').textContent = 'ready — click Run';
  } else {
    r.disabled = true;
    $('eta').textContent = 'waiting for: ' + need.join('  ·  ');
  }
}

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
wireDrop('dropBam', 'fileBam', f => {
  bam = f; const d = $('dropBam');
  d.classList.add('has'); d.textContent = `✓ ${f.name} (${(f.size/1e6).toFixed(1)} MB)`;
  refresh();
});
wireDrop('dropFa', 'fileFa', f => {
  fa = f; const d = $('dropFa');
  d.classList.add('has'); d.textContent = `✓ ${f.name} (${(f.size/1e6).toFixed(1)} MB)`;
  refresh();
});

const readBytes = async (file) => new Uint8Array(await file.arrayBuffer());

$('run').onclick = async () => {
  if (!bam || !fa) { log('need both a BAM and a reference FASTA'); return; }
  if (!session) { log('onnxruntime-web not ready'); return; }
  $('run').disabled = true;
  $('done').innerHTML = '';
  setBar(0);
  const region = $('region').value.trim();
  const t0 = performance.now();
  try {
    log(`reading ${bam.name} + ${fa.name}…`);
    const [bamBytes, faBytes] = await Promise.all([readBytes(bam), readBytes(fa)]);

    log('make-examples (wasm)…');
    const pipe = new dv.Pipeline(bamBytes, faBytes, region);
    const n = pipe.len();
    log(`${n} examples`);
    if (n === 0) throw new Error('no candidate examples in region (check the region matches the BAM/reference)');

    const [H, W, C] = dv.wgs_image_shape();
    const PIX = H * W * C, BATCH = 32;
    const probs = new Float32Array(n * 3);
    let done = 0;
    for (let s = 0; s < n; s += BATCH) {
      const b = Math.min(BATCH, n - s);
      const buf = new Float32Array(b * PIX);
      for (let k = 0; k < b; k++)
        buf.set(dv.example_to_model_input(pipe.example(s + k)), k * PIX);
      const out = await session.run(
        { [inputName]: new ort.Tensor('float32', buf, [b, H, W, C]) });
      probs.set(out[outputName].data.subarray(0, b * 3), s * 3);
      done += b;
      setBar(done / n);
      const el = (performance.now() - t0) / 1000;
      $('eta').textContent =
        `${done}/${n} · ${el.toFixed(0)}s elapsed · ~${(el/done*(n-done)).toFixed(0)}s left`;
    }

    log('postprocess (wasm)…');
    const contig = region.split(':')[0];
    const end = parseInt((region.split('-')[1] || '300000000'), 10);
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
