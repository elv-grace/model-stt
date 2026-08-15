"""Build a scoring set from FLEURS.

FLEURS is n-way parallel (it is FLoRes-101 read aloud), so the English transcript
for a given sentence id is a valid X->English *translation* reference for the
same id in any other language. One fetch therefore yields both:

  <out>/audio/<lang>/<id>.wav          audio to tag
  <out>/refs-asr/<lang>/<id>.txt       transcript in the source language
  <out>/refs-en/<lang>/<id>.txt        English translation reference

Only ids present in both the source language and English are kept, so every
sample is scoreable on both tasks.

    python -m bench.fetch_fleurs --langs fr_fr ko_kr cmn_hans_cn --limit 50
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

from datasets import Audio, load_dataset
from loguru import logger

# FLEURS config -> the whisper language code its audio should be detected as
LANG_CODES = {
    "en_us": "en",
    "fr_fr": "fr",
    "de_de": "de",
    "es_419": "es",
    "it_it": "it",
    "ko_kr": "ko",
    "ja_jp": "ja",
    "cmn_hans_cn": "zh",
    "ta_in": "ta",
    "hi_in": "hi",
}


def load_split(config: str, split: str, limit: int) -> Dict[int, dict]:
    """Return {sentence_id: row} for the first `limit` rows of a FLEURS config."""
    logger.info(f"loading fleurs/{config}:{split} (limit {limit})")
    ds = load_dataset("google/fleurs", config, split=split, streaming=True)
    # keep the raw WAV bytes rather than decoding: datasets>=5 routes decoding
    # through torchcodec, and we only need to write the file back out anyway
    ds = ds.cast_column("audio", Audio(decode=False))
    rows = {}
    for row in ds:
        rows[row["id"]] = row
        if len(rows) >= limit:
            break
    return rows


def collect_pairs(config: str, split: str, english: Dict[int, dict], limit: int):
    """Stream a config, keeping rows whose sentence id also exists in English."""
    logger.info(f"pairing fleurs/{config}:{split} against the english pool")
    ds = load_dataset("google/fleurs", config, split=split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    paired = []
    for row in ds:
        if row["id"] in english:
            paired.append((row["id"], row))
            if len(paired) >= limit:
                break
    return paired


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--langs', nargs='+', default=['fr_fr', 'ko_kr', 'cmn_hans_cn'],
                        help=f'FLEURS configs; known: {sorted(LANG_CODES)}')
    parser.add_argument('--split', default='test')
    parser.add_argument('--limit', type=int, default=50, help='samples per language')
    parser.add_argument('--out', default='fleurs')
    # English is pulled with a wider net so parallel ids can be matched: the
    # first N rows of two configs are not the same N sentences
    parser.add_argument('--en-pool', type=int, default=2000)
    args = parser.parse_args()

    english = load_split('en_us', args.split, args.en_pool)
    logger.info(f"english pool: {len(english)} sentences")

    for lang in args.langs:
        code = LANG_CODES.get(lang, lang.split('_')[0])
        # over-fetch so that after intersecting with the English pool we still
        # have close to --limit usable samples
        # stream and keep only ids that exist in English, stopping at --limit:
        # the two configs are not ordered alike, so a fixed over-fetch either
        # wastes downloads or silently returns fewer samples than asked for
        paired = collect_pairs(lang, args.split, english, args.limit)
        if not paired:
            logger.error(f"{lang}: no ids shared with the english pool, skipping")
            continue

        audio_dir = os.path.join(args.out, 'audio', code)
        asr_dir = os.path.join(args.out, 'refs-asr', code)
        en_dir = os.path.join(args.out, 'refs-en', code)
        for d in (audio_dir, asr_dir, en_dir):
            os.makedirs(d, exist_ok=True)

        for sid, row in paired:
            audio = row['audio']
            data = audio.get('bytes')
            if data is None:
                # some builds hand back only a local path to the extracted file
                with open(audio['path'], 'rb') as src:
                    data = src.read()
            with open(os.path.join(audio_dir, f'{sid}.wav'), 'wb') as f:
                f.write(data)
            with open(os.path.join(asr_dir, f'{sid}.txt'), 'w') as f:
                f.write(row['transcription'])
            with open(os.path.join(en_dir, f'{sid}.txt'), 'w') as f:
                f.write(english[sid]['transcription'])

        logger.info(f"{lang} ({code}): wrote {len(paired)} samples to {audio_dir}")

    print(f"\nscoring set written to {args.out}/")
    return 0


if __name__ == '__main__':
    sys.exit(main())
