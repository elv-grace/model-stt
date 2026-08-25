"""Populate the local whisper weight cache.

Run once on a host with network access (or after adding a model to config.yml):

    python download_weights.py                              # everything
    python download_weights.py --models large-v3-turbo --backends ct2

The cache is then either baked into an image by build.sh or mounted into the
container at run time, so container starts never touch the network.

Sizes: large-v3-turbo 1.62 GB per backend, large-v3 3.09 GB per backend
(9.42 GB for all four combinations), plus 2.2 GB for the punctuation model, which
is fetched by default and skipped with --no-punctuation.
"""
from __future__ import annotations

import argparse
import os
import sys

from config import config


def download_openai(model_name: str, entry: dict, dest: str) -> None:
    import whisper

    os.makedirs(dest, exist_ok=True)
    target = os.path.join(dest, entry["openai"])
    if os.path.isfile(target):
        print(f"  openai/{entry['openai']} already present")
        return
    # _download writes to <dest>/<basename(url)> and verifies sha256 -- the same
    # filename load_model() looks for when given download_root at runtime
    print(f"  downloading openai/{entry['openai']} ...")
    whisper._download(whisper._MODELS[model_name], dest, in_memory=False)


def download_ct2(entry: dict, dest: str) -> None:
    from huggingface_hub import snapshot_download

    target = os.path.join(dest, entry["ct2"])
    if os.path.isfile(os.path.join(target, "model.bin")):
        print(f"  faster-whisper/{entry['ct2']} already present")
        return
    # pin the repo explicitly: faster-whisper's size-name -> repo mapping changes
    # between library versions, so resolving by name is not reproducible
    print(f"  downloading faster-whisper/{entry['ct2']} from {entry['ct2_repo']} ...")
    snapshot_download(repo_id=entry["ct2_repo"], local_dir=target)


def download_punctuation(model: str, dest: str) -> None:
    from huggingface_hub import snapshot_download

    target = os.path.join(dest, model.replace("/", "--"))
    if os.path.isfile(os.path.join(target, "config.json")):
        print(f"  punctuation/{model} already present")
        return
    print(f"  downloading punctuation/{model} ...")
    # flattened repo id, matching src/punctuate.py's _local_weights, so the model
    # loads from the cache with no network at container start
    snapshot_download(repo_id=model, local_dir=target, ignore_patterns=["*.onnx", "*.h5"])


# speakrs' model bundle, for src/diarize.py. Listed file by file rather than
# snapshot_download'ed whole because the repo also carries the CoreML and
# int8/fp16 variants, which are several GB this never loads.
#
# The first three groups are what speakrs' own required_files() fetches for CUDA.
# The last entry is not: speakrs 0.5.0 downloads the multi-mask tail for CUDA but
# then loads wespeaker-voxceleb-resnet34-tail.onnx unconditionally once a split
# backend is available (src/inference/embedding/load/sessions.rs:111 -- the
# batched variants beside it are guarded by .exists(), this one is not), so
# from_pretrained on CUDA dies with "does not exist". Staging the file it wants
# is the fix that does not involve patching speakrs.
DIARIZATION_FILES = [
    # PLDA transform
    "plda_lda.npy", "plda_tr.npy", "plda_mu.npy", "plda_psi.npy",
    "plda_mean1.npy", "plda_mean2.npy",
    "wespeaker-voxceleb-resnet34.min_num_samples.txt",
    # segmentation and embedding
    "segmentation-3.0.onnx", "segmentation-3.0-b32.onnx",
    "wespeaker-voxceleb-resnet34.onnx", "wespeaker-voxceleb-resnet34.onnx.data",
    "wespeaker-voxceleb-resnet34-b64.onnx",
    # split fbank + multi-mask tail, the CUDA embedding path
    "wespeaker-fbank.onnx", "wespeaker-fbank-b32.onnx",
    "wespeaker-multimask-tail.onnx", "wespeaker-multimask-tail-b32.onnx",
    # loaded unconditionally by speakrs 0.5.0; see above
    "wespeaker-voxceleb-resnet34-tail.onnx",
]

DIARIZATION_REPO = "avencera/speakrs-models"


def download_diarization(dest: str) -> None:
    from huggingface_hub import hf_hub_download

    os.makedirs(dest, exist_ok=True)
    for filename in DIARIZATION_FILES:
        if os.path.isfile(os.path.join(dest, filename)):
            print(f"  speakrs/{filename} already present")
            continue
        print(f"  downloading speakrs/{filename} ...")
        # local_dir keeps the flat layout ModelBundle::from_dir expects
        hf_hub_download(repo_id=DIARIZATION_REPO, filename=filename, local_dir=dest)


def main() -> int:
    models = config["models"]
    parser = argparse.ArgumentParser()
    parser.add_argument('--dest', default=config["storage"]["weights_dir"],
                        help='weight cache root (default: storage.weights_dir)')
    parser.add_argument('--models', nargs='+', default=sorted(models), choices=sorted(models))
    # ct2 only by default: the image ships the CTranslate2 backend, and the openai
    # checkpoints are needed only to reproduce bench comparisons
    parser.add_argument('--backends', nargs='+', default=['ct2'],
                        choices=['openai', 'ct2'])
    parser.add_argument('--no-punctuation', action='store_true',
                        help='skip the punctuation model (2.2 GB); the tagger then '
                             'falls back to whisper\'s own punctuation')
    # Diarization ships disabled, so unlike punctuation the useful flag is the
    # one that turns staging ON: you stage the weights, then enable it. With it
    # enabled in config.yml the weights are fetched without asking, because a
    # tagger configured to diarize and missing its models is not a state worth
    # supporting.
    parser.add_argument('--diarization', action='store_true',
                        help='stage speakrs\' diarization models (~170 MB) even if '
                             'config.yml has diarization disabled')
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    for model_name in args.models:
        entry = models[model_name]
        print(f"{model_name}:")
        if 'openai' in args.backends:
            download_openai(model_name, entry, os.path.join(args.dest, "openai"))
        if 'ct2' in args.backends:
            download_ct2(entry, os.path.join(args.dest, "faster-whisper"))

    punctuation = config.get("postprocessing", {}).get("punctuation", {})
    if not args.no_punctuation and punctuation.get("enabled", True):
        print("punctuation:")
        download_punctuation(punctuation["model"], os.path.join(args.dest, "punctuation"))

    diarization = config.get("diarization", {})
    if args.diarization or diarization.get("enabled", False):
        print("diarization:")
        # matches diarization.models_dir in config.yml, which resolves relative
        # paths against this same weights root
        download_diarization(os.path.join(args.dest, diarization.get("models_dir") or "speakrs"))

    print(f"\nweights staged under {args.dest}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
