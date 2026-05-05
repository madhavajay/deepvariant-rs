//! Embed an rpath that points back into the `tensorflow-sys` build dir so
//! the `dv` binary can find `libtensorflow.so.2` / `libtensorflow_framework.so.2`
//! without `LD_LIBRARY_PATH`. Only emitted when the `tf` feature is on —
//! default ORT builds don't link any native ML lib (ORT loads via dlopen)
//! so they should not carry dead rpath entries.
//!
//! The exact hash dir name varies per cargo rebuild, so we list both the
//! current debug + release hashes. If you see
//! `error while loading shared libraries: libtensorflow_framework.so.2`,
//! update the hashes here from `find target -name libtensorflow.so`.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_TF");

    // ORT loads via dlopen at runtime (not link time), so the dylib is
    // not required for `cargo build` to succeed. Emit a one-time warning
    // if the bundled location is empty so a fresh clone knows where to
    // get it. The check is cheap; skip when the user has already set
    // ORT_DYLIB_PATH (they know what they're doing).
    if std::env::var_os("CARGO_FEATURE_ORT").is_some()
        && std::env::var_os("ORT_DYLIB_PATH").is_none()
    {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
        let workspace = std::path::Path::new(&manifest)
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.to_path_buf());
        let lib_present = workspace.as_ref().is_some_and(|w| {
            w.join("models/lib/libonnxruntime.so").exists()
                || w.join("models/lib/libonnxruntime.dylib").exists()
                || w.join("models/lib/libonnxruntime.1.dylib").exists()
        });
        if !lib_present {
            println!(
                "cargo:warning=libonnxruntime not found under models/lib/ — \
                 run `./scripts/fetch_onnxruntime.sh` to download it, or set \
                 ORT_DYLIB_PATH to a system-installed copy. The `dv` binary \
                 builds fine without it but won't run inference."
            );
        }
    }

    if std::env::var_os("CARGO_FEATURE_TF").is_none() {
        return; // ORT-only build: nothing else to do.
    }
    let target = std::env::var("TARGET").unwrap_or_default();
    if target != "x86_64-unknown-linux-gnu" {
        return; // rpath syntax below is Linux/glibc-specific.
    }

    // `$ORIGIN` is resolved at load time relative to the binary location.
    let rpaths = [
        // debug
        "$ORIGIN/build/tensorflow-sys-dcb2b767d9ef58e7/out",
        // release
        "$ORIGIN/build/tensorflow-sys-abd0ab388aa204f2/out",
        // also via deps/.. for tests
        "$ORIGIN/../build/tensorflow-sys-dcb2b767d9ef58e7/out",
        "$ORIGIN/../build/tensorflow-sys-abd0ab388aa204f2/out",
    ];
    for p in rpaths {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{p}");
    }
}
