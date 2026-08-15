"""Side-by-side transcript diff between two tagger runs, for eyeballing.

Built for material with no reference transcript, where corpus metrics are not
available and the only honest check is reading the output. Emits a markdown file
pairing each file's sentences from two systems.

    python -m bench.compare_outputs --a bench-output/turbo-ct2.jsonl \
        --b bench-output/large-v3-ct2.jsonl --labels turbo large-v3 \
        --out bench-output/turbo-vs-large-v3.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List


def load(path: str, track: str = "auto_captions") -> Dict[str, List[dict]]:
    by_file = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            if msg.get("type") != "tag":
                continue
            d = msg["data"]
            if (d.get("track") or "") != track:
                continue
            by_file[os.path.basename(d["source_media"])].append(d)
    for v in by_file.values():
        v.sort(key=lambda t: t["start_time"])
    return by_file


def words_of(path: str) -> Dict[str, int]:
    counts = defaultdict(int)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            if msg.get("type") == "tag" and (msg["data"].get("track") or "") == "":
                counts[os.path.basename(msg["data"]["source_media"])] += 1
    return counts


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--a', required=True)
    p.add_argument('--b', required=True)
    p.add_argument('--labels', nargs=2, default=['A', 'B'])
    p.add_argument('--out', required=True)
    p.add_argument('--max-sentences', type=int, default=12,
                   help='per file, to keep the document readable')
    args = p.parse_args()

    la, lb = args.labels
    A, B = load(args.a), load(args.b)
    WA, WB = words_of(args.a), words_of(args.b)
    files = sorted(set(A) | set(B))

    out = [f"# {la} vs {lb} — transcription\n",
           "No reference transcripts for this material, so this is for reading, not scoring.\n",
           "## Summary\n",
           f"| file | {la} words | {lb} words | {la} sents | {lb} sents |",
           "|---|---|---|---|---|"]
    for f in files:
        out.append(f"| {f} | {WA.get(f,0)} | {WB.get(f,0)} | {len(A.get(f,[]))} | {len(B.get(f,[]))} |")

    for f in files:
        out.append(f"\n## {f}\n")
        sa, sb = A.get(f, []), B.get(f, [])
        n = min(max(len(sa), len(sb)), args.max_sentences)
        out.append(f"| # | {la} | {lb} |")
        out.append("|---|---|---|")
        for i in range(n):
            ta = sa[i]["tag"].replace("|", "\\|") if i < len(sa) else "—"
            tb = sb[i]["tag"].replace("|", "\\|") if i < len(sb) else "—"
            out.append(f"| {i+1} | {ta} | {tb} |")
        if max(len(sa), len(sb)) > n:
            out.append(f"\n_({max(len(sa), len(sb)) - n} further sentences omitted)_")

    with open(args.out, 'w') as fh:
        fh.write("\n".join(out) + "\n")
    print(f"wrote {args.out} ({len(files)} files)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
