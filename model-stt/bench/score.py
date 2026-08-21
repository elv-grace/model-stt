"""Quality scoring for tagger output.

Reads .jsonl tag output in the common-ml message format, so it scores model-stt,
model-asr and model-multilingual-stt identically regardless of which container
produced the file.

    python -m bench.score --hyp bench-output/turbo-ct2.jsonl --ref refs/ --task asr
    (disabled) python -m bench.score --hyp bench-output/translate-ct2.jsonl --ref refs-en/ --task translation

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


# A cue index line, and a cue timing line. Presence of the latter is what marks a
# reference as SRT rather than plain text.
SRT_TIMING = re.compile(r"\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}\s*-->")
SRT_INDEX = re.compile(r"^\s*\d+\s*$")


# Non-speech content a subtitle carries that a transcript never should. Stripped
# here rather than left to the normalizer: EnglishTextNormalizer happens to remove
# brackets and parentheses too, but the reference should not depend on which
# normalizer is selected.
SUBTITLE_MARKUP = [
    (re.compile(r"<[^>]*>"), " "),            # <i>italics</i>, <font ...>
    (re.compile(r"\{[^}]*\}"), " "),          # {\an8} positioning
    (re.compile(r"\[[^\]]*\]"), " "),         # [GUNSHOT], [indistinct]
    (re.compile(r"\([^)]*\)"), " "),          # (laughing)
    # The note goes, the lyric stays: whisper transcribes singing, so dropping the
    # words would book real output as insertions.
    (re.compile(r"[♪♫♩♬]"), " "),
    (re.compile(r"^\s*[-–—]\s*", re.M), " "),  # dialogue dashes, per line
    # Speaker labels, all-caps (MAN:) or title case (Peter:, Aunt May:). At most
    # two words, so "Rule number one: never..." is left alone. A one-word
    # "Question:" is still caught, which is the accepted false positive -- it is
    # applied to hypothesis and reference alike, so it cannot bias the score.
    (re.compile(r"^\s*[A-Z][A-Za-z0-9.'-]{0,15}(?: [A-Z][A-Za-z0-9.'-]{0,15})?\s*:", re.M), " "),
]


def parse_srt(text: str) -> str:
    """Flatten an SRT subtitle file to the words a speaker actually says.

    Parsed by cue block rather than line by line, so a digits-only *subtitle* line
    ("1985", a shouted count) survives: an index line is only dropped when it
    opens a block and is followed by a timing line. A line-based filter deletes
    such dialogue silently."""
    body = text.replace("﻿", "").replace("\r\n", "\n").replace("\r", "\n")

    spoken: List[str] = []
    for block in re.split(r"\n\s*\n", body):
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        if SRT_INDEX.match(lines[0]) and len(lines) > 1 and SRT_TIMING.search(lines[1]):
            lines = lines[2:]
        else:
            lines = [line for line in lines if not SRT_TIMING.search(line)]
        spoken.extend(lines)

    out = "\n".join(spoken)
    for pattern, replacement in SUBTITLE_MARKUP:
        out = pattern.sub(replacement, out)
    return " ".join(out.split())


def load_references(ref: str) -> Dict[str, str]:
    """A directory of <media-basename>.txt files, or a single JSON mapping.

    A .txt holding SRT cue timings is parsed as SRT; anything else is read as a
    plain transcript.
    """
    if os.path.isdir(ref):
        refs = {}
        for name in sorted(os.listdir(ref)):
            if not name.endswith(".txt"):
                continue
            with open(os.path.join(ref, name), encoding="utf-8-sig") as f:
                body = f.read()
            if SRT_TIMING.search(body[:8000]) or " --> " in body[:8000]:
                body = parse_srt(body)
            refs[os.path.splitext(name)[0]] = body.strip()
        return refs
    with open(ref) as f:
        return {os.path.splitext(k)[0]: v for k, v in json.load(f).items()}


def pair(hyps: Dict[str, str], refs: Dict[str, str]) -> List[tuple]:
    """Match each hypothesis to its reference by media stem.

    Exact stem first. Falling back to the reference's leading id token lets a
    reference carry a human-readable suffix -- "AFGM1_jessep_cross.txt" against
    "afgm1.mp4" -- which is how the hand-verified set is named. The fallback is
    rejected if the id is ambiguous, so a silent mispairing cannot slip through
    and score one clip against another's transcript.
    """
    by_id: Dict[str, List[str]] = defaultdict(list)
    for key in refs:
        by_id[key.split("_")[0].lower()].append(key)

    pairs = []
    for name, hyp in sorted(hyps.items()):
        stem = os.path.splitext(name)[0]
        if stem in refs:
            pairs.append((stem, hyp, refs[stem]))
            continue
        candidates = by_id.get(stem.lower(), [])
        if len(candidates) == 1:
            pairs.append((stem, hyp, refs[candidates[0]]))
        elif len(candidates) > 1:
            logger.warning(f"{name} matches several references {candidates}, skipping")
        else:
            logger.warning(f"no reference for {name}, skipping")
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
    # DISABLED translation task
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
