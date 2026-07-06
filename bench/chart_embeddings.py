#!/usr/bin/env python3
"""Render embeddings semantic-separation chart from embeddings_compare_*.json."""
import json, sys, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, PAGE = "#fcfcfb", "#f9f9f7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
S1, S2 = "#2a78d6", "#1baf7a"  # categorical slots 1, 2

path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("bench/data/embeddings_compare_*.json"))[-1]
rep = json.load(open(path))
models = list(rep["models"].items())  # [(name, {dims, pairs}), ...]
pairs = [p["label"] for p in models[0][1]["pairs"]]

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                     "axes.edgecolor": BASELINE, "xtick.color": MUTED, "ytick.color": INK2})
fig, ax = plt.subplots(figsize=(10, 4.4), dpi=150, facecolor=PAGE)
fig.subplots_adjust(left=0.30, right=0.95, top=0.70, bottom=0.14)
ax.set_facecolor(SURFACE)

h = 0.32
colors = [S1, S2]
for mi, (name, d) in enumerate(models):
    margins = [p["related"] - p["unrelated"] for p in d["pairs"]]
    ys = [i + (mi - 0.5) * (h + 0.04) for i in range(len(pairs))]
    ax.barh(ys, margins, height=h, color=colors[mi], zorder=3, label=f"{name} · {d['dims']}d")
    for y, m in zip(ys, margins):
        ax.text(m + (0.012 if m >= 0 else -0.012), y, f"{m:+.3f}",
                va="center", ha="left" if m >= 0 else "right", fontsize=8.5, color=INK2, zorder=4)

ax.set_yticks(range(len(pairs)), pairs, fontsize=9)
ax.invert_yaxis()
ax.axvline(0, color=BASELINE, lw=1.2, zorder=2)
ax.set_xlim(-0.12, 0.66)
ax.set_xlabel("semantic separation:  cos(related pair) − cos(unrelated pair)", fontsize=9, color=MUTED)
ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
ax.legend(loc="upper right", bbox_to_anchor=(1.005, 1.30), fontsize=8.5, frameon=False)

fig.suptitle("Embeddings: raw 12B vectors vs a 146 MB sidecar",
             x=0.30, ha="left", fontsize=12.5, fontweight="bold", color=INK, y=0.95)
fig.text(0.30, 0.80, f"same texts, same host · left of zero = inverted similarity · {rep['date']}",
         fontsize=8.5, color=INK2)

out = sys.argv[2] if len(sys.argv) > 2 else path.replace(".json", ".png")
fig.savefig(out, facecolor=PAGE)
print("wrote", out)
