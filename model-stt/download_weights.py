"""Populate the local whisper weight cache.

Run once on a host with network access (or after adding a model to config.yml):

    python download_weights.py                              # everything
    python download_weights.py --models large-v3-turbo --backends ct2

The cache is then either baked into an image by build.sh or mounted into the
container at run time, so container starts never touch the network.

Sizes: large-v3-turbo 1.62 GB per backend, large-v3 3.09 GB per backend
(9.42 GB for all four combinations).
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


def main() -> int:
    models = config["models"]
    parser = argparse.ArgumentParser()
    parser.add_argument('--dest', default=config["storage"]["weights_dir"],
                        help='weight cache root (default: storage.weights_dir)')
    parser.add_argument('--models', nargs='+', default=sorted(models), choices=sorted(models))
    parser.add_argument('--backends', nargs='+', default=['openai', 'ct2'],
                        choices=['openai', 'ct2'])
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    for model_name in args.models:
        entry = models[model_name]
        print(f"{model_name}:")
        if 'openai' in args.backends:
            download_openai(model_name, entry, os.path.join(args.dest, "openai"))
        if 'ct2' in args.backends:
            download_ct2(entry, os.path.join(args.dest, "faster-whisper"))

    print(f"\nweights staged under {args.dest}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
