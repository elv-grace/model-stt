#!/usr/bin/env python3
"""
Score STT hypotheses against the hand-verified reference set AND against the
downloaded subtitles, then report the calibration constant

    calibration = subtitle_WER - true_WER

per excerpt and per failure-mode category, with bootstrap confidence intervals.

The calibration constant is the payoff of this whole exercise: it tells you how
much your cheap 28-hour subtitle-based WER over/under-states the truth, so every
subtitle-scored number becomes interpretable ("subtitle WER overstates by X pp").

Directory layout (relative to the refset root, or pass --root):

    excerpts.json
    references/<excerpt_id>.txt        # HAND-VERIFIED verbatim, one utterance per line
    subtitle_refs/<excerpt_id>.txt     # subtitle text for the same span (use extract_subtitle_ref.py)
    hypotheses/<model>/<excerpt_id>.txt# each model's transcription of the excerpt span

Missing files are skipped with a note, so you can run this incrementally as you
finish transcribing. Requires only the Python standard library.

Usage:
    python score.py --root .. --bootstrap 2000 --out results
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from normalize import NormConfig, normalize_lines, normalize_ref, normalize_hyp


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #
def align(ref: list[str], hyp: list[str]):
    """Word-level Levenshtein with backtrace.

    Returns (ops, counts) where ops is a list of (tag, ref_idx, hyp_idx) with
    tag in {match, sub, del, ins}, and counts is dict(S, D, I, C).
    """
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        ri = ref[i - 1]
        for j in range(1, m + 1):
            if ri == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j - 1],  # sub
                                  d[i - 1][j],       # del
                                  d[i][j - 1])       # ins
    # Backtrace (prefer diagonal, then deletion, then insertion for determinism)
    i, j = n, m
    ops = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and d[i][j] == d[i - 1][j - 1]:
            ops.append(("match", i - 1, j - 1)); i -= 1; j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            ops.append(("sub", i - 1, j - 1)); i -= 1; j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", i - 1, None)); i -= 1
        else:
            ops.append(("ins", None, j - 1)); j -= 1
    ops.reverse()
    counts = {"S": 0, "D": 0, "I": 0, "C": 0}
    for tag, _, _ in ops:
        counts[{"match": "C", "sub": "S", "del": "D", "ins": "I"}[tag]] += 1
    return ops, counts


def char_distance(a: str, b: str) -> int:
    """Levenshtein distance on characters (for CER)."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(prev[j - 1] if ca == cb else 1 + min(prev[j - 1], prev[j], cur[-1]))
        prev = cur
    return prev[-1]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def wer_from_counts(c) -> float:
    n_ref = c["S"] + c["D"] + c["C"]
    return (c["S"] + c["D"] + c["I"]) / n_ref if n_ref else 0.0


def mer_from_counts(c) -> float:
    denom = c["S"] + c["D"] + c["I"] + c["C"]
    return (c["S"] + c["D"] + c["I"]) / denom if denom else 0.0


def utterance_buckets(ref_lines, hyp_tokens):
    """Align the full excerpt, then attribute each op to a reference utterance.

    Returns (total_counts, per_utterance_counts). Insertions are attributed to
    the utterance of the next reference token (or the last utterance at the end).
    """
    ref_flat, ref_uidx = [], []
    for u, toks in enumerate(ref_lines):
        for t in toks:
            ref_flat.append(t); ref_uidx.append(u)
    ops, total = align(ref_flat, hyp_tokens)
    n_utt = max(1, len(ref_lines))
    buckets = [{"S": 0, "D": 0, "I": 0, "C": 0} for _ in range(n_utt)]
    for idx, (tag, ri, _hj) in enumerate(ops):
        if ri is not None:
            buckets[ref_uidx[ri]][{"match": "C", "sub": "S", "del": "D"}[tag]] += 1
        else:  # insertion: attribute to the NEXT reference token's utterance
            nxt = None
            for t2, r2, _2 in ops[idx + 1:]:
                if r2 is not None:
                    nxt = ref_uidx[r2]
                    break
            buckets[nxt if nxt is not None else n_utt - 1]["I"] += 1
    return total, buckets


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def bootstrap_wer(all_buckets, n_boot, rng):
    """Bootstrap WER + MER over utterance buckets. all_buckets: list of dicts."""
    if not all_buckets:
        return None
    k = len(all_buckets)
    wers, mers = [], []
    for _ in range(n_boot):
        agg = {"S": 0, "D": 0, "I": 0, "C": 0}
        for _ in range(k):
            b = all_buckets[rng.randrange(k)]
            for key in agg:
                agg[key] += b[key]
        wers.append(wer_from_counts(agg)); mers.append(mer_from_counts(agg))
    wers.sort(); mers.sort()
    lo, hi = int(0.025 * n_boot), int(0.975 * n_boot) - 1
    return {"wer_lo": wers[lo], "wer_hi": wers[max(lo, hi)],
            "mer_lo": mers[lo], "mer_hi": mers[max(lo, hi)]}


def bootstrap_constant(true_buckets, sub_buckets, n_boot, rng):
    """CI for subtitle_WER - true_WER.

    The two WERs use different references (verbatim vs subtitle) so their
    utterances don't correspond; we resample each independently and subtract.
    Independence OVERSTATES variance (the errors are positively correlated), so
    this CI is conservative -- if it excludes zero, the gap is real.
    """
    if not true_buckets or not sub_buckets:
        return None
    kt, ks = len(true_buckets), len(sub_buckets)
    diffs = []
    for _ in range(n_boot):
        at = {"S": 0, "D": 0, "I": 0, "C": 0}
        for _ in range(kt):
            b = true_buckets[rng.randrange(kt)]
            for key in at:
                at[key] += b[key]
        asb = {"S": 0, "D": 0, "I": 0, "C": 0}
        for _ in range(ks):
            b = sub_buckets[rng.randrange(ks)]
            for key in asb:
                asb[key] += b[key]
        diffs.append(wer_from_counts(asb) - wer_from_counts(at))
    diffs.sort()
    lo, hi = int(0.025 * n_boot), int(0.975 * n_boot) - 1
    return {"lo": diffs[lo], "hi": diffs[max(lo, hi)],
            "mean": sum(diffs) / len(diffs)}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def read(path: Path):
    return path.read_text(encoding="utf-8") if path.exists() else None


def score_pair(ref_text, hyp_text, cfg, is_subtitle=False):
    """Return metrics dict + utterance buckets for one (ref, hyp) pair."""
    ref_lines = normalize_lines(ref_text, cfg, is_ref=not is_subtitle)
    hyp_tokens = normalize_hyp(hyp_text, cfg)
    if not ref_lines:
        return None
    total, buckets = utterance_buckets(ref_lines, hyp_tokens)
    ref_flat = [t for line in ref_lines for t in line]
    cer = char_distance(" ".join(ref_flat), " ".join(hyp_tokens)) / max(1, len(" ".join(ref_flat)))
    return {
        "wer": wer_from_counts(total), "mer": mer_from_counts(total), "cer": cer,
        "S": total["S"], "D": total["D"], "I": total["I"], "C": total["C"],
        "n_ref": total["S"] + total["D"] + total["C"],
    }, buckets


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--drop-fillers", action="store_true",
                    help="score with fillers removed (default keeps them)")
    ap.add_argument("--exclude", default="",
                    help="comma-separated excerpt ids to skip (e.g. BD1_coffee_run_music)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    exclude = {x.strip() for x in args.exclude.split(",") if x.strip()}

    root = Path(args.root)
    cfg = NormConfig(drop_fillers=args.drop_fillers)
    rng = random.Random(args.seed)
    spec = json.loads((root / "excerpts.json").read_text())
    excerpts = spec["excerpts"]
    by_id = {e["id"]: e for e in excerpts}

    models = sorted(p.name for p in (root / "hypotheses").glob("*") if p.is_dir())
    if not models:
        print("No model folders found under hypotheses/. Create hypotheses/<model>/ "
              "and drop <excerpt_id>.txt files in it.")
    rows = []           # per (model, excerpt)
    # category -> model -> {'true': [buckets], 'sub': [buckets]}
    cat_buckets = {}

    for model in models:
        for e in excerpts:
            eid = e["id"]
            if eid in exclude:
                continue
            ref = read(root / "references" / f"{eid}.txt")
            hyp = read(root / "hypotheses" / model / f"{eid}.txt")
            sub = read(root / "subtitle_refs" / f"{eid}.txt")
            if ref is None or hyp is None:
                continue
            tstats, tbuckets = score_pair(ref, hyp, cfg, is_subtitle=False)
            row = {"model": model, "excerpt": eid, "film": e["film"],
                   "category": e["primary_category"],
                   "true_wer": tstats["wer"], "S": tstats["S"], "D": tstats["D"],
                   "I": tstats["I"], "C": tstats["C"], "n_ref": tstats["n_ref"],
                   "true_mer": tstats["mer"], "true_cer": tstats["cer"],
                   "sub_wer": None, "calibration": None}
            sbuckets = None
            if sub is not None:
                sstats, sbuckets = score_pair(sub, hyp, cfg, is_subtitle=True)
                row["sub_wer"] = sstats["wer"]
                row["calibration"] = sstats["wer"] - tstats["wer"]
            rows.append(row)

            cat = e["primary_category"]
            cat_buckets.setdefault(cat, {}).setdefault(model, {"true": [], "sub": []})
            cat_buckets[cat][model]["true"].extend(tbuckets)
            if sbuckets is not None:
                cat_buckets[cat][model]["sub"].extend(sbuckets)

    # Category aggregates with bootstrap
    cat_rows = []
    for cat, per_model in sorted(cat_buckets.items()):
        for model, d in sorted(per_model.items()):
            tb, sb = d["true"], d["sub"]
            agg_t = {k: sum(b[k] for b in tb) for k in ("S", "D", "I", "C")}
            true_wer = wer_from_counts(agg_t)
            cr = {"category": cat, "model": model, "true_wer": true_wer,
                  "n_ref": agg_t["S"] + agg_t["D"] + agg_t["C"],
                  "sub_wer": None, "calibration": None,
                  "cal_lo": None, "cal_hi": None,
                  "true_wer_lo": None, "true_wer_hi": None}
            bt = bootstrap_wer(tb, args.bootstrap, rng)
            if bt:
                cr["true_wer_lo"], cr["true_wer_hi"] = bt["wer_lo"], bt["wer_hi"]
            if sb:
                agg_s = {k: sum(b[k] for b in sb) for k in ("S", "D", "I", "C")}
                sub_wer = wer_from_counts(agg_s)
                cr["sub_wer"] = sub_wer
                cr["calibration"] = sub_wer - true_wer
                bc = bootstrap_constant(tb, sb, args.bootstrap, rng)
                if bc:
                    cr["cal_lo"], cr["cal_hi"] = bc["lo"], bc["hi"]
            cat_rows.append(cr)

    # ------------------------------------------------------------------ output
    outdir = root / args.out
    outdir.mkdir(exist_ok=True)
    with (outdir / "per_excerpt.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["model", "excerpt", "film", "category", "true_wer"])
        w.writeheader(); w.writerows(rows)
    with (outdir / "per_category.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cat_rows[0].keys()) if cat_rows else
                           ["category", "model", "true_wer"])
        w.writeheader(); w.writerows(cat_rows)
    (outdir / "results.json").write_text(json.dumps(
        {"per_excerpt": rows, "per_category": cat_rows}, indent=2))

    # console
    def pct(x):
        return "   n/a" if x is None else f"{100 * x:6.2f}"
    if exclude:
        print(f"(excluding excerpts: {', '.join(sorted(exclude))})")
    print("\n=== PER EXCERPT ===")
    print(f"{'model':10} {'excerpt':26} {'trueWER':>8} {'subWER':>8} "
          f"{'calib(pp)':>10} {'MER':>7} {'CER':>7} {'S/D/I':>12}")
    for r in rows:
        cal = "   n/a" if r["calibration"] is None else f"{100 * r['calibration']:+6.2f}"
        print(f"{r['model'][:10]:10} {r['excerpt'][:26]:26} {pct(r['true_wer'])} "
              f"{pct(r['sub_wer'])} {cal:>10} {pct(r['true_mer'])} {pct(r['true_cer'])} "
              f"{r['S']}/{r['D']}/{r['I']:>3}")
    print("\n=== PER CATEGORY (calibration = subtitle_WER - true_WER, +pp = subtitle OVERSTATES) ===")
    print(f"{'category':14} {'model':10} {'trueWER':>8} [{'95% CI':^13}] {'subWER':>8} "
          f"{'calib(pp)':>10} [{'95% CI (pp)':^15}]")
    for r in cat_rows:
        ci = (f"[{100*r['true_wer_lo']:5.2f},{100*r['true_wer_hi']:5.2f}]"
              if r['true_wer_lo'] is not None else " " * 15)
        cal = "   n/a" if r["calibration"] is None else f"{100 * r['calibration']:+6.2f}"
        calci = (f"[{100*r['cal_lo']:+5.2f},{100*r['cal_hi']:+5.2f}]"
                 if r['cal_lo'] is not None else "")
        print(f"{r['category'][:14]:14} {r['model'][:10]:10} {pct(r['true_wer'])} {ci:15} "
              f"{pct(r['sub_wer'])} {cal:>10} {calci}")
    print(f"\nWrote {outdir}/per_excerpt.csv, per_category.csv, results.json")


if __name__ == "__main__":
    main()
