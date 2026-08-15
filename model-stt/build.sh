#!/bin/bash
#
# Build the model-stt container.
#
#   WEIGHTS=turbo-ct2 ./build.sh     the shipped image: CT2 + turbo   (~1.6 GB weights)
#   WEIGHTS=none      ./build.sh     no baked weights; mount the cache at run time
#
# full-ct2 and all additionally stage large-v3 and/or the openai checkpoints. They
# exist for bench/ only -- neither is needed by the transcription-only image, and
# both require the matching entries in config.yml to be uncommented.
#
# Weights are staged into the build context from the local cache populated by
# download_weights.py -- podman COPY can only read from the build context, so they
# have to be rsynced in rather than referenced in place.

set -e

git submodule update --init --recursive

SCRIPT_PATH="$(dirname "$(realpath "$0")")"
WEIGHTS="${WEIGHTS:-turbo-ct2}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
STAGE="$SCRIPT_PATH/models/whisper"

CACHE=$(yq -r .storage.weights_dir "$SCRIPT_PATH/config.yml")
CACHE="${CACHE/#\~/$HOME}"

# pip extras, and therefore whether torch lands in the image at all. CT2-only
# images skip the multi-GB torch/nvidia wheel stack entirely.
case "$WEIGHTS" in
  all)  EXTRAS="${EXTRAS:-all}" ;;
  *)    EXTRAS="${EXTRAS:-ct2}" ;;
esac

stage() {
  # stage <relative-path-under-cache>
  local rel="$1"
  if [ ! -e "$CACHE/$rel" ]; then
    echo "ERROR: $CACHE/$rel is missing. Run: python download_weights.py" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$STAGE/$rel")"
  rsync --archive --update --delete "$CACHE/$rel" "$(dirname "$STAGE/$rel")/"
}

rm -rf "$STAGE"
mkdir -p "$STAGE"

case "$WEIGHTS" in
  none)
    echo "baking no weights; mount $CACHE into the container at run time"
    ;;
  turbo-ct2)
    stage faster-whisper/large-v3-turbo
    ;;
  full-ct2)
    stage faster-whisper/large-v3-turbo
    stage faster-whisper/large-v3
    ;;
  all)
    stage faster-whisper/large-v3-turbo
    stage faster-whisper/large-v3
    stage openai/large-v3-turbo.pt
    stage openai/large-v3.pt
    ;;
  *)
    echo "ERROR: unknown WEIGHTS=$WEIGHTS (none|turbo-ct2|full-ct2|all)" >&2
    exit 1
    ;;
esac

echo "staged $(du -sh "$STAGE" | cut -f1) of weights for WEIGHTS=$WEIGHTS (extras: $EXTRAS)"

# WEIGHTS=none leaves WEIGHTS_DIR empty so config.yml's ~/.cache path applies
if [ "$WEIGHTS" = "none" ]; then
  BAKED_DIR=""
else
  BAKED_DIR="/elv/models/whisper"
fi

# build_container.bash forwards everything after its own flags to the container
# build, so EXTRAS and WEIGHTS_DIR have to be passed here: without WEIGHTS_DIR the
# Containerfile's empty default applies and the image looks for weights under
# config.yml's ~/.cache path instead of the copy baked at /elv/models/whisper.
exec buildscripts/build_container.bash \
  -t "${IMAGE_NAME:-model-whisper-stt}:${IMAGE_TAG:-latest}" \
  . -f Containerfile \
  --build-arg "EXTRAS=${EXTRAS}" \
  --build-arg "WEIGHTS_DIR=${BAKED_DIR}"
