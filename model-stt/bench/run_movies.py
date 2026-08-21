"""Feature-film benchmark against subtitle .txt references.

Unlike bench/run_bench.py, which compares *backends* on short clean fixtures, this
measures the profile that actually ships -- punctuation restoration on, VAD off,
self-calibrated artifact guard -- on full-length features, with and without the
deterministic temperature fallback retry ladder.

    python -m bench.run_movies --system default
    python -m bench.run_movies --system stochastic
    python -m bench.run_movies --score-only

Each system runs in its own process so peak GPU is attributable to one of them;
--score-only rescores whatever .jsonl output is already on disk.

References are a proxy, not a verbatim transcript.
Subtitles condense and paraphrase to fit reading speed, and omit some speech entirely, so absolute WER is inflated.
Both systems (deterministic and stochastic, current and whisper) are scored against the same references, so the *comparison* is unaffected.
Read the deltas, not the absolute numbers.
Only WER and CER are reported. BLEU and chrF2 are translation metrics.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, fields
from typing import List, Optional

from dacite import from_dict
from loguru import logger

import resource

from bench.run_bench import audio_duration, gpu_memory_mib
from bench.score import hypothesis_text, load_normalizer, load_references, load_tags, pair
from config import config
from src.model import RuntimeConfig, WhisperSTT
from src.punctuate import PunctuationConfig

OUTDIR = "test-output-sony-movies"
MEDIA_DIR = "test-files"
OUTPUTS = {
    # `default` now carries the deterministic ladder; `stochastic` is whisper's
    # sampled retries, kept as the comparison that justified making it default.
    "default": os.path.join(OUTDIR, "out-reproducible.jsonl"),
    "stochastic": os.path.join(OUTDIR, "out.jsonl"),
}
SUMMARY = os.path.join(OUTDIR, "out-summary.json")


def use_dirs(media_dir: str, outdir: str) -> None:
    """Point the benchmark at another corpus, e.g. the hand-verified clip set."""
    global MEDIA_DIR, OUTDIR, OUTPUTS, SUMMARY
    MEDIA_DIR, OUTDIR = media_dir, outdir
    OUTPUTS = {"default": os.path.join(outdir, "out-reproducible.jsonl"),
               "stochastic": os.path.join(outdir, "out.jsonl")}
    SUMMARY = os.path.join(outdir, "out-summary.json")


@dataclass
class FileResult:
    file: str
    audio_seconds: float
    wall_seconds: float
    rtf: float
    n_tags: int
    n_captions: int


@dataclass
class SystemResult:
    system: str
    deterministic_fallback: bool
    punctuation: bool
    load_seconds: float
    total_audio_seconds: float
    total_wall_seconds: float
    rtf: float
    peak_gpu_mib: Optional[int]
    # High-water mark of resident host memory for this process. ru_maxrss is a
    # peak the kernel maintains, so it needs no sampling and cannot be missed
    # between files the way a polled GPU reading can.
    peak_rss_mib: Optional[int]
    files: List[FileResult]


def media_files() -> List[str]:
    """Every .mp4 in the corpus that has a reference beside it.

    Matched the same way bench.score.pair does -- exact stem, or the reference's
    leading id token -- so a reference may carry a readable suffix
    ("AFGM1_jessep_cross.txt" for "afgm1.mp4") without going unpaired here and
    then silently pairing at scoring time.
    """
    ids = {os.path.basename(p).split("_")[0].lower()
           for p in glob.glob(os.path.join(MEDIA_DIR, "*.txt"))}
    paired = []
    for path in sorted(glob.glob(os.path.join(MEDIA_DIR, "*.mp4"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        if os.path.isfile(os.path.splitext(path)[0] + ".txt") or stem.lower() in ids:
            paired.append(path)
        else:
            logger.warning(f"no .txt reference for {path}, skipping")
    if not paired:
        raise SystemExit(f"no paired .mp4/.txt found under {MEDIA_DIR}/")
    return paired


OVERRIDES: dict = {}


def parse_overrides(specs: List[str]) -> dict:
    """FIELD=VALUE pairs, typed by json where possible ("5.0" -> 5.0, "true" -> True)."""
    out = {}
    valid = {f.name for f in fields(RuntimeConfig)}
    for spec in specs:
        field_name, _, raw = spec.partition("=")
        if not raw:
            raise SystemExit(f"--param expects FIELD=VALUE, got {spec!r}")
        if field_name not in valid:
            raise SystemExit(f"unknown RuntimeConfig field {field_name!r}")
        try:
            out[field_name] = json.loads(raw)
        except json.JSONDecodeError:
            out[field_name] = raw
    return out


def build(system: str):
    merged = dict(config["runtime"]["default"])
    if system == "stochastic":
        merged["deterministic_fallback"] = False
    merged.update(OVERRIDES)
    cfg = from_dict(RuntimeConfig, merged)
    punctuation = from_dict(PunctuationConfig, config["postprocessing"]["punctuation"])

    load_start = time.perf_counter()
    model = WhisperSTT(
        cfg,
        models=config["models"],
        weights_dir=config["storage"]["weights_dir"],
        sentence_gap_ms=config["postprocessing"]["sentence_gap"],
        max_caption_words=config["postprocessing"]["max_caption_words"],
        punctuation=punctuation,
    )
    load_seconds = time.perf_counter() - load_start
    if punctuation.enabled and model.punctuator is None:
        raise SystemExit(
            "punctuation restoration is enabled but failed to load, so this would "
            "not measure the shipped profile. Run download_weights.py."
        )
    return model, cfg, load_seconds


def run(system: str) -> SystemResult:
    files = media_files()
    model, cfg, load_seconds = build(system)
    logger.info(f"{system}: loaded in {load_seconds:.1f}s, {len(files)} files")

    os.makedirs(OUTDIR, exist_ok=True)
    results: List[FileResult] = []
    peak_gpu = gpu_memory_mib()

    with open(OUTPUTS[system], "w") as fh:
        for path in files:
            seconds = audio_duration(path)
            start = time.perf_counter()
            tags = model.tag(path)
            wall = time.perf_counter() - start

            captions = 0
            for tag in tags:
                if tag.track == "auto_captions":
                    captions += 1
                fh.write(json.dumps({"type": "tag", "data": {
                    "start_time": tag.start_time, "end_time": tag.end_time,
                    "tag": tag.tag, "vector": None, "source_media": path,
                    "track": tag.track, "additional_info": tag.additional_info,
                    "frame_info": None,
                }}) + "\n")

            used = gpu_memory_mib()
            if used is not None:
                peak_gpu = max(peak_gpu or 0, used)
            results.append(FileResult(
                file=os.path.basename(path), audio_seconds=round(seconds, 1),
                wall_seconds=round(wall, 1), rtf=round(seconds / wall, 1),
                n_tags=len(tags), n_captions=captions,
            ))
            logger.info(
                f"{system}: {os.path.basename(path)[:34]:36s} "
                f"{seconds/60:6.1f}min {wall:6.1f}s {seconds/wall:5.1f}x "
                f"{captions:5d} captions"
            )

    total_audio = sum(r.audio_seconds for r in results)
    total_wall = sum(r.wall_seconds for r in results)
    return SystemResult(
        system=system,
        deterministic_fallback=cfg.deterministic_fallback,
        punctuation=model.punctuator is not None,
        load_seconds=round(load_seconds, 1),
        total_audio_seconds=round(total_audio, 1),
        total_wall_seconds=round(total_wall, 1),
        rtf=round(total_audio / total_wall, 1),
        peak_gpu_mib=peak_gpu,
        # ru_maxrss is KiB on Linux, bytes on macOS; this benchmark targets Linux
        peak_rss_mib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        files=results,
    )


def error_breakdown(refs, hyps) -> dict:
    """WER with its three error classes separated, plus MER.

    WER is (S + D + I) / N_ref, so its denominator counts only reference words --
    which means insertions are unbounded and WER can exceed 100% when the
    hypothesis is longer than the reference.

    MER = (S + D + I) / (S + D + I + H) divides by the alignment length instead,
    so it is bounded at 1.0 and does not inflate with hypothesis length. Reading
    the two together separates "the model got words wrong" from "the model
    transcribed more than the subtitler wrote down": if WER rises while MER holds
    steady, the extra errors are insertions against a condensed reference, not
    new mistakes.

    Rates are per reference word, the same denominator WER uses, so sub + del +
    ins rates sum to WER exactly.
    """
    import jiwer

    out = jiwer.process_words(refs, hyps)
    n_ref = out.substitutions + out.deletions + out.hits
    return {
        "wer": round(out.wer, 4),
        "mer": round(out.mer, 4),
        "hits": out.hits,
        "substitutions": out.substitutions,
        "deletions": out.deletions,
        "insertions": out.insertions,
        "sub_rate": round(out.substitutions / n_ref, 4) if n_ref else None,
        "del_rate": round(out.deletions / n_ref, 4) if n_ref else None,
        "ins_rate": round(out.insertions / n_ref, 4) if n_ref else None,
    }


def score(system: str) -> dict:
    return score_file(OUTPUTS[system])


def score_file(path: str) -> dict:
    """Score every track the output carries.

    Both tracks are scored because they answer different questions. The word track
    is the transcription itself; auto_captions is that same text after grouping,
    punctuation restoration and the artifact guard. Scoring only the captions
    conflates "the model misheard a word" with "the caption layer dropped or
    reshaped something", and those have different fixes. They should land within
    noise of each other -- a gap between them is a bug in the caption layer, not
    in the decoder.
    """
    if not os.path.isfile(path):
        return {"error": f"{path} not found"}

    out = {}
    for label, track in (("auto_captions", "auto_captions"), ("word_level", "")):
        scored = score_track(path, track)
        if scored is not None:
            out[label] = scored
    return out or {"error": f"no scorable tracks in {path}"}


def score_track(path: str, track: str) -> Optional[dict]:
    import jiwer

    normalize = load_normalizer("en")
    by_file = load_tags(path, track)
    if not by_file:
        return None
    pairs = pair(hypothesis_text(by_file), load_references(MEDIA_DIR))

    hyps = [normalize(h) for _, h, _ in pairs]
    refs = [normalize(r) for _, _, r in pairs]
    per_file = {
        name: {
            **error_breakdown([r], [h]),
            "cer": round(jiwer.cer(r, h), 4),
            "hyp_words": len(h.split()),
            "ref_words": len(r.split()),
        }
        for (name, _, _), h, r in zip(pairs, hyps, refs)
    }
    return {
        "n": len(pairs),
        **error_breakdown(refs, hyps),
        "cer": round(jiwer.cer(refs, hyps), 4),
        "hyp_words": sum(len(h.split()) for h in hyps),
        "ref_words": sum(len(r.split()) for r in refs),
        "per_file": per_file,
    }


def report(summary: dict) -> None:
    print("\n=== speed / memory ===")
    print(f"{'system':16s} {'load':>7s} {'audio':>8s} {'wall':>9s} {'xRT':>7s} "
          f"{'peak GPU':>10s} {'peak RSS':>10s}")
    for name, block in summary.items():
        r = block.get("run")
        if not r:
            continue
        print(f"{name:16s} {r['load_seconds']:6.1f}s {r['total_audio_seconds']/3600:7.2f}h "
              f"{r['total_wall_seconds']:8.1f}s {r['rtf']:6.1f}x "
              f"{str(r['peak_gpu_mib'] or '-'):>6} MiB {str(r.get('peak_rss_mib') or '-'):>6} MiB")

    def tracks_of(block):
        s = block.get("score") or {}
        return {k: v for k, v in s.items() if isinstance(v, dict) and "wer" in v}

    print("\n=== quality vs subtitle references (proxy; read deltas) ===")
    print(f"{'system':16s} {'track':14s} {'n':>3s} {'WER':>8s} {'MER':>8s} {'CER':>8s} "
          f"{'hyp words':>10s} {'ref words':>10s}")
    for name, block in summary.items():
        for track, s in sorted(tracks_of(block).items()):
            print(f"{name:16s} {track:14s} {s['n']:3d} {s['wer']*100:7.2f}% "
                  f"{s['mer']*100:7.2f}% {s['cer']*100:7.2f}% "
                  f"{s['hyp_words']:10d} {s['ref_words']:10d}")

    print("\n=== error breakdown (counts, and rates per reference word) ===")
    print(f"{'system':16s} {'track':14s} {'sub':>8s} {'del':>8s} {'ins':>8s} {'hits':>8s}   "
          f"{'sub%':>7s} {'del%':>7s} {'ins%':>7s}")
    for name, block in summary.items():
        for track, s in sorted(tracks_of(block).items()):
            print(f"{name:16s} {track:14s} {s['substitutions']:8d} {s['deletions']:8d} "
                  f"{s['insertions']:8d} {s['hits']:8d}   {s['sub_rate']*100:6.2f}% "
                  f"{s['del_rate']*100:6.2f}% {s['ins_rate']*100:6.2f}%")

    names = [n for n in ("default", "stochastic")
             if "auto_captions" in tracks_of(summary.get(n, {}))]
    if len(names) == 2:
        a, b = (summary[n]["score"]["auto_captions"] for n in names)
        print(f"\n{names[1]} - {names[0]} (auto_captions):  "
              f"WER {100*(b['wer']-a['wer']):+.2f}pp   MER {100*(b['mer']-a['mer']):+.2f}pp   "
              f"CER {100*(b['cer']-a['cer']):+.2f}pp   "
              f"sub {b['substitutions']-a['substitutions']:+d}  "
              f"del {b['deletions']-a['deletions']:+d}  "
              f"ins {b['insertions']-a['insertions']:+d}")

    if names:
        print("\n=== per file (auto_captions) ===")
        header = "".join(f"{n[:6] + ' WER':>13s}{n[:6] + ' MER':>13s}" for n in names)
        print(f"{'file':34s}{header}{'ins%':>8s}{'ref words':>11s}")
        base = summary[names[0]]["score"]["auto_captions"]["per_file"]
        for f in sorted(base):
            cells = "".join(
                f"{summary[n]['score']['auto_captions']['per_file'][f]['wer']*100:12.2f}%"
                f"{summary[n]['score']['auto_captions']['per_file'][f]['mer']*100:12.2f}%"
                for n in names
            )
            print(f"{f[:32]:34s}{cells}{base[f]['ins_rate']*100:7.2f}%{base[f]['ref_words']:11d}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=sorted(OUTPUTS), help="run one system")
    parser.add_argument("--score-only", action="store_true",
                        help="score existing .jsonl output without re-running")
    parser.add_argument("--media-dir", help="corpus of .mp4 + .txt references "
                        "(default: test-files)")
    parser.add_argument("--outdir", help="where to write .jsonl and the summary")
    parser.add_argument("--param", nargs="*", default=[], metavar="FIELD=VALUE",
                        help="override a RuntimeConfig field for this run, e.g. "
                             "--param isolated_artifact_gap=5.0. Recorded in the "
                             "summary so a run's settings can be read back off it.")
    parser.add_argument("--compare", nargs="*", default=[], metavar="NAME=PATH",
                        help="score another tagger's .jsonl through this same "
                             "normalizer, e.g. "
                             "model-asr=../model-asr/test-output/out.jsonl")
    args = parser.parse_args()

    if args.media_dir or args.outdir:
        use_dirs(args.media_dir or MEDIA_DIR, args.outdir or OUTDIR)
    if args.param:
        OVERRIDES.update(parse_overrides(args.param))
        logger.info(f"RuntimeConfig overrides: {OVERRIDES}")

    os.makedirs(OUTDIR, exist_ok=True)
    summary = {}
    if os.path.isfile(SUMMARY):
        with open(SUMMARY) as f:
            summary = json.load(f)

    if args.system and not args.score_only:
        summary.setdefault(args.system, {})["run"] = asdict(run(args.system))
        if OVERRIDES:
            summary[args.system]["overrides"] = dict(OVERRIDES)

    # Rescore every system with output on disk, not just the one that just ran:
    # a scoring change would otherwise leave the other system's entry in the old
    # format, and the two would be reported side by side as if comparable.
    for name in sorted(OUTPUTS):
        if os.path.isfile(OUTPUTS[name]):
            summary.setdefault(name, {})["score"] = score(name)

    # Another tagger's output, scored through the same normalizer and the same
    # references. Comparing two systems means scoring both the same way; running
    # each through its own benchmark measures the scorers as much as the models.
    for spec in args.compare:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--compare expects NAME=PATH, got {spec!r}")
        summary.setdefault(name, {})["score"] = score_file(path)
        summary[name]["source"] = path

    with open(SUMMARY, "w") as f:
        json.dump(summary, f, indent=1)
    report(summary)
    print(f"\nsummary -> {SUMMARY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
