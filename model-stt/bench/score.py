"""Quality scoring for tagger output.

Reads .jsonl tag output in the common-ml message format, so it scores model-stt,
model-asr and model-multilingual-stt identically regardless of which container
produced the file.

    python -m bench.score --hyp bench-output/turbo-ct2.jsonl --ref refs/ --task asr
    python -m bench.score --hyp bench-output/translate-ct2.jsonl --ref refs-en/ --task translation

Normalization is the part that decides whether the numbers mean anything.
Whisper emits punctuated, cased text and model-asr does not, so WER computed on
raw strings mostly measures punctuation. Both sides go through whisper's own
EnglishTextNormalizer (or BasicTextNormalizer for other languages), which is the
normalization the published whisper WER figures use.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Callable, Dict, List

from loguru import logger


def load_normalizer(language: str) -> Callable[[str], str]:
    try:
        from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer
    except ImportError:
        logger.error(
            "openai-whisper is not installed, falling back to naive lowercasing. "
            "Numbers will not be comparable to published whisper WER. "
            "Install with: pip install '.[bench]'"
        )
        return lambda s: " ".join(s.lower().split())

    return EnglishTextNormalizer() if language == "en" else BasicTextNormalizer()


def load_tags(path: str, track: str) -> Dict[str, List[dict]]:
    """Group a tagger's .jsonl output by source file, keeping one track."""
    by_file: Dict[str, List[dict]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            if msg.get("type") != "tag":
                continue
            data = msg["data"]
            if (data.get("track") or "") != track:
                continue
            by_file[data["source_media"]].append(data)

    for tags in by_file.values():
        tags.sort(key=lambda t: t["start_time"])
    return by_file


def hypothesis_text(by_file: Dict[str, List[dict]]) -> Dict[str, str]:
    return {
        os.path.basename(path): " ".join(t["tag"] for t in tags).strip()
        for path, tags in by_file.items()
    }


def load_references(ref: str) -> Dict[str, str]:
    """A directory of <media-basename>.txt files, or a single JSON mapping."""
    if os.path.isdir(ref):
        refs = {}
        for name in os.listdir(ref):
            if name.endswith(".txt"):
                with open(os.path.join(ref, name)) as f:
                    refs[os.path.splitext(name)[0]] = f.read().strip()
        return refs
    with open(ref) as f:
        return {os.path.splitext(k)[0]: v for k, v in json.load(f).items()}


def pair(hyps: Dict[str, str], refs: Dict[str, str]) -> List[tuple]:
    pairs = []
    for name, hyp in sorted(hyps.items()):
        stem = os.path.splitext(name)[0]
        if stem not in refs:
            logger.warning(f"no reference for {name}, skipping")
            continue
        pairs.append((stem, hyp, refs[stem]))
    if not pairs:
        raise SystemExit("no hypothesis/reference pairs matched")
    return pairs


# Scripts with no whitespace word delimiter. WER is meaningless for these (the
# whole utterance is one "word"), and whitespace must be stripped before CER:
# FLEURS ships its Chinese references space-separated per character, which
# otherwise inflates CER to ~50% on a perfect transcription, since the reference
# is about twice the character length of the hypothesis.
NO_WORD_BOUNDARY = {"zh", "ja", "yue", "th", "lo", "my", "km"}


def score_asr(pairs: List[tuple], normalize: Callable[[str], str], language: str = "en") -> dict:
    import jiwer

    hyps = [normalize(h) for _, h, _ in pairs]
    refs = [normalize(r) for _, _, r in pairs]

    if language in NO_WORD_BOUNDARY:
        hyps = [re.sub(r"\s+", "", h) for h in hyps]
        refs = [re.sub(r"\s+", "", r) for r in refs]
        kept = [(h, r) for h, r in zip(hyps, refs) if r]
        return {
            "n": len(kept),
            "cer": round(jiwer.cer([r for _, r in kept], [h for h, _ in kept]), 4),
            "wer": None,
            "note": f"WER omitted: {language} has no whitespace word boundary; read CER",
        }
    # drop pairs the normalizer emptied, which jiwer rejects
    kept = [(h, r) for h, r in zip(hyps, refs) if r]
    if len(kept) < len(pairs):
        logger.warning(f"{len(pairs) - len(kept)} references normalized to empty, skipped")
    hyps, refs = [h for h, _ in kept], [r for _, r in kept]

    return {
        "n": len(kept),
        "wer": round(jiwer.wer(refs, hyps), 4),
        "cer": round(jiwer.cer(refs, hyps), 4),
    }


def score_translation(pairs: List[tuple], normalize: Callable[[str], str]) -> dict:
    import sacrebleu

    hyps = [normalize(h) for _, h, _ in pairs]
    refs = [normalize(r) for _, _, r in pairs]
    return {
        "n": len(pairs),
        # chrF is listed first: it is more robust than BLEU on the short,
        # sentence-level segments this pipeline produces
        "chrf2": round(sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score, 2),
        "bleu": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyp', required=True, help='tagger .jsonl output')
    parser.add_argument('--ref', required=True, help='reference dir of .txt, or a .json mapping')
    parser.add_argument('--task', choices=['asr', 'translation'], default='asr')
    parser.add_argument('--track', default=None,
                        help="track to score (default: auto_captions for asr, "
                             "translation for path C output)")
    parser.add_argument('--language', default='en',
                        help='reference language, selects the normalizer')
    parser.add_argument('--per-file', action='store_true')
    args = parser.parse_args()

    track = args.track if args.track is not None else "auto_captions"
    normalize = load_normalizer(args.language)

    by_file = load_tags(args.hyp, track)
    if not by_file:
        raise SystemExit(f"no tags on track {track!r} in {args.hyp}")

    pairs = pair(hypothesis_text(by_file), load_references(args.ref))
    if args.task == 'asr':
        result = score_asr(pairs, normalize, args.language)
    else:
        result = score_translation(pairs, normalize)

    print(json.dumps({"hyp": args.hyp, "track": track, "task": args.task, **result}, indent=2))

    if args.per_file:
        print()
        for stem, hyp, ref in pairs:
            single = (score_asr([(stem, hyp, ref)], normalize, args.language)
                      if args.task == 'asr' else score_translation([(stem, hyp, ref)], normalize))
            metric = single.get('wer', single.get('chrf2'))
            print(f"  {stem:<40} {metric}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
