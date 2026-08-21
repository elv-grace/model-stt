#!/usr/bin/env python3
"""Generate one blind-first transcription worksheet per excerpt from excerpts.json.

    python make_worksheets.py --root ..
"""
from __future__ import annotations
import argparse, json
from pathlib import Path


def to_sec(ts: str) -> int:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def fmt(sec: int) -> str:
    return f"{sec//3600:02d}:{(sec%3600)//60:02d}:{sec%60:02d}"


def checkpoints(start: str, end: str, step: int = 30):
    a, b = to_sec(start), to_sec(end)
    out, t = [], a
    while t < b:
        out.append((fmt(t), fmt(min(t + step, b))))
        t += step
    return out


def worksheet(e: dict) -> str:
    dur = to_sec(e["end"]) - to_sec(e["start"])
    L = []
    L.append(f"# Worksheet — {e['id']}")
    L.append("")
    L.append(f"- **Film:** {e['film']}")
    L.append(f"- **Excerpt span:** {e['start']} → {e['end']}  "
             f"(~{dur//60}:{dur%60:02d}, SRT cues {e.get('cue_range','?')})")
    L.append(f"- **Primary category:** `{e['primary_category']}`"
             + (f"   secondary: {', '.join('`%s`'%c for c in e['secondary_categories'])}"
                if e.get("secondary_categories") else ""))
    L.append(f"- **Scene:** {e['scene']}")
    L.append(f"- **Why this excerpt:** {e['why']}")
    L.append("")
    L.append("> **VERIFY THE TIMECODES against your own media first.** Different "
             "rips differ by seconds to minutes. Trim to the nearest natural "
             "sentence boundary in the audio.")
    L.append("")
    L.append("## Proper-noun / hard-vocab watchlist (spelling aid only)")
    L.append("Use these ONLY in Pass 2 to fix spelling — never to decide what was said.")
    L.append("")
    L.append(", ".join(f"`{w}`" for w in e.get("watch_vocab", [])) or "_(none noted)_")
    L.append("")
    L.append("## Pass 1 — BLIND transcription (subtitle & script CLOSED)")
    L.append("Type exactly what you hear in each ~30s window: false starts, "
             "repeats, fillers (`um`,`uh`), stammers. Numbers as words. Non-speech "
             "in `[brackets]`. One utterance per line when you move it to the "
             "reference file.")
    L.append("")
    for a, b in checkpoints(e["start"], e["end"]):
        L.append(f"**[{a}–{b}]**")
        L.append("")
        L.append("```")
        L.append("")
        L.append("```")
        L.append("")
    L.append("## Pass 2 — QC (aids OPEN): divergence log")
    L.append("Log every place the AUDIO differs from the script/subtitle. These "
             "divergences are the evidence the reference set is working. Your ears win.")
    L.append("")
    L.append("| timecode | what the AUDIO says | what the SCRIPT/SUBTITLE says | note |")
    L.append("|---|---|---|---|")
    L.append("|  |  |  |  |")
    L.append("|  |  |  |  |")
    L.append("|  |  |  |  |")
    L.append("")
    L.append("## QC checklist (tick before marking this excerpt done)")
    for item in [
        "Every proper noun on the watchlist spelled consistently",
        "Fillers (um/uh/etc.) kept, not cleaned out",
        "False starts and repeated words kept verbatim",
        "Numbers written as spoken words, not digits",
        "Non-speech marked in [brackets]; song lyrics NOT transcribed as speech",
        "One utterance per line in references/%s.txt" % e["id"],
        "Did a final listen with my transcript hidden and the audio playing, and it matches",
        "subtitle_refs/%s.txt generated with extract_subtitle_ref.py (NOT hand-edited)" % e["id"],
    ]:
        L.append(f"- [ ] {item}")
    L.append("")
    L.append(f"→ Save the verbatim transcript to `references/{e['id']}.txt`.")
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    args = ap.parse_args()
    root = Path(args.root)
    spec = json.loads((root / "excerpts.json").read_text())
    wdir = root / "worksheets"; wdir.mkdir(exist_ok=True)
    for e in spec["excerpts"]:
        (wdir / f"{e['id']}.md").write_text(worksheet(e), encoding="utf-8")
        print(f"  wrote worksheets/{e['id']}.md")
    # index
    idx = ["# Worksheets index", "",
           "Work top-to-bottom; the set spans all four categories. "
           "Do the blind pass first (see ../protocol/transcription_protocol.md).", ""]
    total = 0
    idx.append("| # | excerpt | film | category | span | ~dur |")
    idx.append("|---|---|---|---|---|---|")
    for i, e in enumerate(spec["excerpts"], 1):
        d = to_sec(e["end"]) - to_sec(e["start"]); total += d
        idx.append(f"| {i} | [{e['id']}]({e['id']}.md) | {e['film']} | "
                   f"`{e['primary_category']}` | {e['start']}–{e['end']} | {d//60}:{d%60:02d} |")
    idx.append(f"\n**Total ≈ {total//60}:{total%60:02d}** across {len(spec['excerpts'])} excerpts.")
    (wdir / "INDEX.md").write_text("\n".join(idx) + "\n", encoding="utf-8")
    print(f"  wrote worksheets/INDEX.md (total {total//60}:{total%60:02d})")


if __name__ == "__main__":
    main()
