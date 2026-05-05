# Upstream DeepVariant build & test on this Arch host

Goal: get a reliably-buildable, fully-tested copy of upstream `deepvariant/`
running on this machine, so it can serve as the behavioral reference for the
Rust port.

Strategy: build inside Docker (matches Google's maintained path); avoid native
Arch builds (Python/GCC/apt mismatches).

---

## Checklist

- [x] **1. Confirm Docker works without sudo** for this user.
      → user is in `docker` group, `dockerd` active, client/server 29.4.1.
- [x] **2. Pre-flight checks**: disk free, RAM, image base reachable, no stale
      containers/volumes from earlier attempts.
      → 90 GB free on `/` (Docker root is `/var/lib/docker` on same LV);
        20 GB RAM available; `overlayfs`; clean slate; `ubuntu:22.04` pulled OK.
      → Watch disk: builder image will be ~30–50 GB; if it gets tight, prune
        unused images with `docker image prune -a` before proceeding.
- [x] **3. Build the `builder` stage image** (`docker build --target builder
      -t deepvariant:local .` from inside `deepvariant/`). This stops after the
      full source + Bazel cache + TF build, which is what we want for iteration.
      → image `deepvariant:local`, 6.49 GB; `make_examples`, `call_variants`,
        `convert_to_saved_model`, `load_gbz_into_shared_memory`,
        `examples_from_stream.so` all present under
        `/opt/deepvariant/bazel-out/k8-opt/bin/deepvariant/`.
      → wall time: ~12 min total (prereq layer ~6.5 min, builder layer ~5.5 min,
        export ~3 min). Faster than the 30–90 min I projected — the `:binaries`
        bazel target only runs ~2400 actions, not the full TF tree.
- [x] **4. Create a persistent Bazel cache volume** (`docker volume create
      dv-bazel-cache`) so subsequent test runs don't recompile from scratch.
      → created at `/var/lib/docker/volumes/dv-bazel-cache/_data`.
- [x] **5. Run the upstream test suite** inside the container via
      `./build_and_test.sh` (which runs `bazel test -c opt deepvariant/...`).
      → **59 of 59 tests pass.** Total wall time ~few minutes (cache hot).
      → Used `bazel test -c opt ${DV_COPT_FLAGS} --test_output=errors
        --build_tests_only deepvariant/...` directly, since the binaries were
        already built in the image.
- [x] **6. Triage any failing tests** — distinguish "real upstream test issue
      under our toolchain" vs "infrastructure (network/missing data/etc.)".
      → N/A: no failures.
- [x] **7. Build the final release image** (`docker build -t
      deepvariant:release .`) — this is what end-users would actually run.
      → `deepvariant:release` 7.26 GB, prints `DeepVariant version 1.10.0`.
      → Multi-stage cache reused builder/prereq layers; only built
        `hts_utils` (samtools+bcftools), `download_models` (~7 GB), and
        the final `integrate` stage.
- [x] **8. End-to-end sanity check**: run `run_deepvariant` on a small published
      sample (per `docs/deepvariant-quick-start.md`) and verify it produces a
      VCF. This is the "known-good behavioral baseline" for the Rust port.
      → input: chr20 10–10.01 MB region, NA12878. Output: `output.vcf.gz`,
        `output.g.vcf.gz`, `output.visual_report.html` etc. in
        `quickstart/output/`. **78 variants** (64 SNPs + 14 indels).
      → Pipeline ran in ~20 s (`make_examples` ~10 s, `call_variants` ~3 s,
        `postprocess_variants` ~3 s, vcf_stats_report ~3 s).
- [ ] **9. Document the working incantations** in this file so they're trivially
      reproducible later (commands, env vars, gotchas hit).

---

## Notes & gotchas (filled in as we go)

_(empty — to be populated during execution)_
