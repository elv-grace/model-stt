#!/usr/bin/env bash
# Build stt-diarize and stage the ONNX Runtime it loads at run time.
#
# Two things have to line up for CUDA to work:
#
#   1. ort 2.0.0-rc.12 (what speakrs 0.5 declares, and what upstream's own GPU
#      image is built against) speaks ONNX Runtime 1.24's C API. ORT_VERSION is
#      a pin, not a default: a mismatched runtime loads and then refuses, e.g.
#      "too old; expected >= 1.27.x, got 1.24.2" from ort rc.13. Cargo.lock
#      holds ort at rc.12 for the same reason -- the version requirement alone
#      allows rc.13, so the lock file is what keeps the two ends agreeing.
#   2. The CUDA execution provider lives in libonnxruntime_providers_cuda.so,
#      which ships only in Microsoft's -gpu build. ort's own download does not
#      include it, which is why the crate is built with `load-dynamic` and
#      pointed at these libraries instead.
#
# The provider then dlopens libcublasLt.so.12 and libcudnn.so.9 off
# LD_LIBRARY_PATH -- the same wheels CTranslate2 already loads there, so the
# Containerfile's existing LD_LIBRARY_PATH covers this binary too.
set -euo pipefail

ORT_VERSION=${ORT_VERSION:-1.24.2}
CACHE_DIR=${DIARIZE_CACHE:-$HOME/.cache/model-stt}
ORT_DIR="$CACHE_DIR/onnxruntime"

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$here"

if [ ! -f "$ORT_DIR/libonnxruntime.so" ]; then
    echo "staging onnxruntime-gpu $ORT_VERSION in $ORT_DIR"
    mkdir -p "$ORT_DIR"
    curl -fSL -o "$ORT_DIR/ort.tgz" \
        "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz"
    tar xzf "$ORT_DIR/ort.tgz" --strip-components=2 -C "$ORT_DIR" --wildcards "*/lib/*.so*"
    rm "$ORT_DIR/ort.tgz"
fi

cargo build --release "$@"

echo
echo "binary:   $here/target/release/stt-diarize"
echo "ORT libs: $ORT_DIR"
echo
echo "config.yml's diarization.binary and diarization.ort_lib point at these."
