#!/usr/bin/env python3
"""Render the findings figure (PNG) from results/results.json."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
data = json.loads((ROOT / "results" / "results.json").read_text())
pe = {(r["model"], r["excerpt"]): r for r in data["per_excerpt"]}

# palette (validated)
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10,
                     "svg.fonttype": "none", "axes.edgecolor": BASE})

# ---- Panel A data: model-stt dumbbell, grouped by category (BD1 excluded, annotated)
order = [("clean", ["GD1_cafe_baseline", "AFGM1_jessep_cross", "EQ1_diner_clean"]),
         ("fast_overlap", ["SN1_bar_breakup", "EQ2_mobmeeting_overlap"]),
         ("music", ["WH1_rushing_dragging", "BD2_diner_baby", "EQ4_homemart_music"]),
         ("noise", ["EQ3_store_robbery_noise"])]
short = {"GD1_cafe_baseline": "GD1 cafe", "AFGM1_jessep_cross": "AFGM1 Jessep",
         "EQ1_diner_clean": "EQ1 diner (clean)", "SN1_bar_breakup": "SN1 bar",
         "EQ2_mobmeeting_overlap": "EQ2 mob mtg", "WH1_rushing_dragging": "WH1 rushing",
         "BD2_diner_baby": "BD2 diner", "EQ4_homemart_music": "EQ4 PA-music",
         "EQ3_store_robbery_noise": "EQ3 robbery"}

rows, labels, cat_spans = [], [], []
y = 0
for cat, eids in order:
    start = y
    for eid in eids:
        r = pe[("model-stt", eid)]
        rows.append((y, 100 * r["true_wer"], 100 * r["sub_wer"]))
        labels.append((y, short[eid]))
        y += 1
    cat_spans.append((cat, start, y - 1))
    y += 0.6  # gap between categories

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.55, 1]})
fig.patch.set_facecolor(SURF)
for ax in (axA, axB):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

# Panel A: dumbbell
for yy, t, s in rows:
    axA.plot([t, s], [yy, yy], color=GRID, lw=2, zorder=1, solid_capstyle="round")
    axA.scatter([t], [yy], s=70, color=BLUE, zorder=3)
    axA.scatter([s], [yy], s=70, color=ORANGE, zorder=3)
    gap = s - t
    xlab = max(t, s) + 1.5
    axA.text(xlab, yy, f"{'+' if gap>=0 else ''}{gap:.0f} pp", va="center", ha="left",
             fontsize=8.5, color=(INK if abs(gap) >= 5 else MUTED))
axA.set_yticks([yy for yy, _ in labels])
axA.set_yticklabels([lab for _, lab in labels], fontsize=9, color=INK)
axA.invert_yaxis()
axA.set_xlim(0, 62)
axA.set_xlabel("WER (%)", color=INK2)
axA.xaxis.grid(True, color=GRID, lw=0.8)
axA.set_axisbelow(True)
# category brackets on the right
for cat, a, b in cat_spans:
    axA.text(-0.5, (a + b) / 2, cat.replace("_", "/"), rotation=90, va="center", ha="right",
             fontsize=8, color=MUTED, transform=axA.get_yaxis_transform())
axA.set_title("A  Subtitle WER vs hand-verified WER  (model-stt)", fontsize=11,
              color=INK, loc="left", pad=26)
axA.annotate("BD1 coffee-run off-chart: true WER 138%\n(model transcribes song lyrics as speech)",
             xy=(0.5, 1.015), xycoords="axes fraction", fontsize=8, color=MUTED, va="bottom")
legA = [Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE, markersize=9,
               label="hand-verified (true) WER"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ORANGE, markersize=9,
               label="subtitle WER")]
axA.legend(handles=legA, loc="lower right", frameon=False, fontsize=8.5)

# Panel B: Equalizer EQ1 vs EQ4 true WER, both models
eq = ["EQ1_diner_clean", "EQ4_homemart_music"]
eqlab = ["EQ1 diner\n(clean)", "EQ4 PA-music\n(dense music)"]
asr = [100 * pe[("model-asr", e)]["true_wer"] for e in eq]
stt = [100 * pe[("model-stt", e)]["true_wer"] for e in eq]
x = [0, 1]; w = 0.36
b1 = axB.bar([xi - w / 2 for xi in x], asr, w, color=AQUA, label="model-asr", zorder=3)
b2 = axB.bar([xi + w / 2 for xi in x], stt, w, color=VIOLET, label="model-stt", zorder=3)
for bars in (b1, b2):
    for rect in bars:
        h = rect.get_height()
        axB.text(rect.get_x() + rect.get_width() / 2, h + 1.2, f"{h:.0f}%",
                 ha="center", va="bottom", fontsize=9, color=INK)
axB.set_xticks(x); axB.set_xticklabels(eqlab, fontsize=9.5, color=INK)
axB.set_ylim(0, 62)
axB.set_ylabel("hand-verified WER (%)", color=INK2)
axB.yaxis.grid(True, color=GRID, lw=0.8); axB.set_axisbelow(True)
axB.set_title("B  Same film, clean vs dense-music\nWER roughly triples for both models",
              fontsize=11, color=INK, loc="left", pad=10)
axB.legend(frameon=False, fontsize=8.5, loc="upper left")
axB.annotate("EQ4: mostly deletions,\nnot misrecognitions\n(stt 36 del vs 14 sub)",
             xy=(0.63, 49), xytext=(0.02, 34), fontsize=8, color=MUTED, va="center", ha="left",
             arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))

fig.suptitle("Hand-verified reference set — calibration & the Equalizer read",
             fontsize=13, color=INK, x=0.02, ha="left", weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = ROOT / "results" / "findings_figure.png"
fig.savefig(out, dpi=180, facecolor=SURF, bbox_inches="tight")
print("wrote", out)
