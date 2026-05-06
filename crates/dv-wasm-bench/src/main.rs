//! Native-vs-wasm microbenchmark for the pure-compute kernel
//! `benchmark_normalize_image` from `dv-wasm`. Loads the wasm module
//! via wasmtime and times the same workload native and via wasm,
//! reports the speed ratio.
//!
//! Build the wasm module first (no default features so we don't pull
//! in the wasm-bindgen JS shim — wasmtime has no JS host to satisfy
//! those imports):
//!
//!   cargo build -p dv-wasm --target wasm32-unknown-unknown --release \
//!       --no-default-features
//!
//! Then run the bench:
//!
//!   cargo run -p dv-wasm-bench --release
//!
//! The kernel itself does the `(b - 128) / 128.0` per-pixel
//! normalisation on a 100×221×7 byte image — the same step the real
//! pipeline does between the make-examples shard and the ORT model
//! input. It's the kernel most affected by wasm vs native execution
//! since it's a tight per-byte loop.

use std::time::Instant;

use anyhow::Result;
use wasmtime::{Engine, Instance, Module, Store};

const ITERATIONS: u32 = 1000;
const WASM_PATH: &str = "target/wasm32-unknown-unknown/release/dv_wasm.wasm";

fn run_native(iterations: u32) -> (u64, std::time::Duration) {
    let t0 = Instant::now();
    let acc = dv_wasm::benchmark_normalize_image(iterations);
    (acc, t0.elapsed())
}

fn run_wasm(iterations: u32) -> Result<(u64, std::time::Duration)> {
    let engine = Engine::default();
    let module = Module::from_file(&engine, WASM_PATH).map_err(|e| {
        anyhow::anyhow!(
            "load {WASM_PATH}: {e}; did you `cargo build -p dv-wasm --target wasm32-unknown-unknown --release`?"
        )
    })?;
    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[])
        .map_err(|e| anyhow::anyhow!("instantiate wasm module: {e}"))?;
    let func = instance
        .get_typed_func::<u32, u64>(&mut store, "benchmark_normalize_image")
        .map_err(|e| anyhow::anyhow!("find benchmark_normalize_image export: {e}"))?;
    // Warm-up so JIT/code-gen time isn't included in the measurement.
    func.call(&mut store, 1)
        .map_err(|e| anyhow::anyhow!("warm-up call: {e}"))?;
    let t0 = Instant::now();
    let acc = func
        .call(&mut store, iterations)
        .map_err(|e| anyhow::anyhow!("benchmark call: {e}"))?;
    Ok((acc, t0.elapsed()))
}

fn main() -> Result<()> {
    println!("Benchmark: per-pixel image normalisation");
    println!("  image:      100 × 221 × 7 = 154 700 bytes");
    println!("  iterations: {ITERATIONS}");
    println!("  workload:   {} bytes total", 154_700u64 * ITERATIONS as u64);
    println!();

    let (native_acc, native_dur) = run_native(ITERATIONS);
    let native_per_iter = native_dur / ITERATIONS;
    println!(
        "Native:  {:>9.3?} total, {:>7.1?}/iter, acc=0x{:016x}",
        native_dur, native_per_iter, native_acc
    );

    let (wasm_acc, wasm_dur) = run_wasm(ITERATIONS)?;
    let wasm_per_iter = wasm_dur / ITERATIONS;
    println!(
        "Wasm:    {:>9.3?} total, {:>7.1?}/iter, acc=0x{:016x}",
        wasm_dur, wasm_per_iter, wasm_acc
    );

    let ratio = wasm_dur.as_secs_f64() / native_dur.as_secs_f64();
    println!();
    println!("Wasm/native ratio: {:.2}× (lower = wasm closer to native)", ratio);

    // Sanity: both paths should produce the same checksum.
    if native_acc == wasm_acc {
        println!("✓ Output checksums match — native and wasm computed identical results.");
    } else {
        println!(
            "✗ MISMATCH: native acc = 0x{:016x}, wasm acc = 0x{:016x}",
            native_acc, wasm_acc
        );
        anyhow::bail!("native and wasm computed different results");
    }
    Ok(())
}
