#!/usr/bin/env python3
"""
Extract the subtitle text for one excerpt span from an SRT file, so you can
score models against the subtitle on the SAME span the verbatim reference covers
(that pairing is what makes the calibration constant meaningful).

The verbatim reference is what you TYPE from the audio. The subtitle reference is
produced here mechanically from the SRT you downloaded -- never hand-edit it, or
you contaminate the very artifact whose bias you are trying to measure.

Usage:
    python extract_subtitle_ref.py --srt /path/Equalizer.srt \
        --start 01:42:58 --end 01:45:14 --out ../subtitle_refs/EQ4_homemart_music.txt

    # or drive every excerpt straight from excerpts.json, giving a folder of SRTs
    python extract_subtitle_ref.py --from-spec ../excerpts.json --srt-dir /path/to/srts

Cues that OVERLAP the [start, end] window are included whole. Formatting tags
(<i>, <b>, {\\an8}), speaker dashes, and known watermark/credit lines are stripped.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

_TS = re.compile(r"(\d\d):(\d\d):(\d\d)[,.](\d\d\d)")
_TAG = re.compile(r"<[^>]+>|\{[^}]*\}")
_WATERMARK = re.compile(r"(subtitle|www\.|http|\.me|\.com|opensubtitles|movieddl|"
                        r"downloaded from|api\.|encoded by|sync(ed)? by|corrected by)",
                        re.IGNORECASE)


def to_sec(ts: str) -> float:
    m = _TS.search(ts)
    if not m:
        # accept HH:MM:SS without ms
        parts = ts.split(":")
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    h, mi, s, ms = map(int, m.groups())
    return h * 3600 + mi * 60 + s + ms / 1000.0


def parse_srt(text: str):
    """Yield (start_sec, end_sec, lines[]) per cue."""
    blocks = re.split(r"\r?\n\r?\n", text.strip())
    for blk in blocks:
        lines = [l for l in blk.splitlines() if l.strip()]
        if not lines:
            continue
        # find the timing line
        tline_idx = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if tline_idx is None:
            continue
        a, _, b = lines[tline_idx].partition("-->")
        try:
            start, end = to_sec(a.strip()), to_sec(b.strip())
        except Exception:
            continue
        yield start, end, lines[tline_idx + 1:]


def clean_line(line: str) -> str:
    line = _TAG.sub("", line)
    line = line.replace("\\h", " ")
    line = re.sub(r"^\s*[-–—]\s*", "", line)  # leading speaker dash
    return line.strip()


def extract(srt_text: str, start: float, end: float) -> str:
    out = []
    for cs, ce, lines in parse_srt(srt_text):
        if ce < start or cs > end:      # no overlap
            continue
        for line in lines:
            cl = clean_line(line)
            if cl and not _WATERMARK.search(cl):
                out.append(cl)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--srt")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--out")
    ap.add_argument("--from-spec", help="excerpts.json to batch-extract from")
    ap.add_argument("--srt-dir", help="folder of SRTs (used with --from-spec)")
    ap.add_argument("--srt-suffix", default=".srt")
    args = ap.parse_args()

    if args.from_spec:
        import json
        spec = json.loads(Path(args.from_spec).read_text())
        srt_dir = Path(args.srt_dir)
        outroot = Path(args.from_spec).resolve().parent / "subtitle_refs"
        outroot.mkdir(exist_ok=True)
        print("Batch mode: map each excerpt's srt_project_doc to a local SRT in --srt-dir.")
        for e in spec["excerpts"]:
            # Heuristic: match by film keyword; you may need to rename SRTs.
            cand = list(srt_dir.glob("*"))
            key = e["id"].split("_")[0].lower()
            match = next((c for c in cand if key[:2] in c.name.lower()), None)
            if not match:
                print(f"  ! no SRT found for {e['id']} (looked for '{key}') -- skipping")
                continue
            txt = match.read_text(encoding="utf-8", errors="replace")
            res = extract(txt, to_sec(e["start"]), to_sec(e["end"]))
            (outroot / f"{e['id']}.txt").write_text(res, encoding="utf-8")
            print(f"  {e['id']:26} <- {match.name}  ({len(res.split())} words)")
        return

    srt_text = Path(args.srt).read_text(encoding="utf-8", errors="replace")
    res = extract(srt_text, to_sec(args.start), to_sec(args.end))
    if args.out:
        Path(args.out).write_text(res, encoding="utf-8")
        print(f"Wrote {args.out} ({len(res.split())} words)")
    else:
        print(res)


if __name__ == "__main__":
    main()
