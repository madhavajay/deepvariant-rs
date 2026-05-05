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

    if std::env::var_os("CARGO_FEATURE_TF").is_none() {
        return; // ORT-only build: nothing to do.
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
