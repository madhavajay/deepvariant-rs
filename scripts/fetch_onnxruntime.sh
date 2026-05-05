#!/usr/bin/env bash
#
# Download a pre-built ONNX Runtime release tarball into models/lib/ so
# `dv` (which loads libonnxruntime via dlopen) can find it.
#
# Idempotent: if models/lib/libonnxruntime.so.1 (Linux) or
# .../libonnxruntime.dylib (macOS) already exists, the script is a no-op.
#
# Override the version with ORT_VERSION=1.22.0 ./scripts/fetch_onnxruntime.sh
#
# Pre-built artifacts come from the official Microsoft release page:
# https://github.com/microsoft/onnxruntime/releases — no compilation needed.

set -euo pipefail

ORT_VERSION="${ORT_VERSION:-1.22.0}"

# Resolve the workspace root from this script's own location so the
# script works regardless of where it's invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LIB_DIR="${WORKSPACE_ROOT}/models/lib"

uname_s="$(uname -s)"
uname_m="$(uname -m)"

case "${uname_s}-${uname_m}" in
    Linux-x86_64)
        slug="onnxruntime-linux-x64-${ORT_VERSION}"
        archive="${slug}.tgz"
        sentinel="libonnxruntime.so.1"
        ;;
    Linux-aarch64|Linux-arm64)
        slug="onnxruntime-linux-aarch64-${ORT_VERSION}"
        archive="${slug}.tgz"
        sentinel="libonnxruntime.so.1"
        ;;
    Darwin-arm64)
        slug="onnxruntime-osx-arm64-${ORT_VERSION}"
        archive="${slug}.tgz"
        sentinel="libonnxruntime.dylib"
        ;;
    Darwin-x86_64)
        slug="onnxruntime-osx-x86_64-${ORT_VERSION}"
        archive="${slug}.tgz"
        sentinel="libonnxruntime.dylib"
        ;;
    *)
        echo "error: unsupported platform ${uname_s}-${uname_m}" >&2
        echo "       set ORT_DYLIB_PATH manually to a libonnxruntime built for your" >&2
        echo "       target, or extend this script with a new branch." >&2
        exit 1
        ;;
esac

mkdir -p "${LIB_DIR}"
if [[ -e "${LIB_DIR}/${sentinel}" ]]; then
    echo "ONNX Runtime ${ORT_VERSION} already present at ${LIB_DIR}/${sentinel}"
    exit 0
fi

url="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${archive}"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

echo "Fetching ${url}"
if command -v curl >/dev/null 2>&1; then
    curl --fail --location --silent --show-error --output "${tmp}/${archive}" "${url}"
elif command -v wget >/dev/null 2>&1; then
    wget --quiet --output-document="${tmp}/${archive}" "${url}"
else
    echo "error: neither curl nor wget is installed" >&2
    exit 1
fi

echo "Extracting ${archive}"
tar -xf "${tmp}/${archive}" -C "${tmp}"

# Copy lib/ contents and preserve the symlink chain so .so / .so.1 /
# .so.1.22.0 all resolve correctly.
cp -a "${tmp}/${slug}/lib/." "${LIB_DIR}/"

echo "Installed ONNX Runtime ${ORT_VERSION} → ${LIB_DIR}"
ls -lh "${LIB_DIR}" | grep -E "libonnxruntime" || true
