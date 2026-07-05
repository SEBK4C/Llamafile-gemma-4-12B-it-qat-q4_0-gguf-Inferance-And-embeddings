#!/usr/bin/env python3
"""Render harness-e2e results chart from bench/data/harness_e2e_*.json."""
import json, sys, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, PAGE = "#fcfcfb", "#f9f9f7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
GOOD, CRIT = "#0ca30c", "#d03b3b"

path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("bench/data/harness_e2e_*.json"))[-1]
rep = json.load(open(path))
rows = rep["results"]

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                     "axes.edgecolor": BASELINE, "xtick.color": MUTED, "ytick.color": INK2})
fig, ax = plt.subplots(figsize=(10, 3.6), dpi=150, facecolor=PAGE)
fig.subplots_adjust(left=0.34, right=0.96, top=0.72, bottom=0.16)
ax.set_facecolor(SURFACE)

rows = rows[::-1]
ys = range(len(rows))
vals = [r["wall_s"] for r in rows]
cols = [GOOD if r["verdict"] == "PASS" else CRIT for r in rows]
labels = [f"{r['id']} · {r['harness']}\n{r['task'][:46]}" for r in rows]

ax.barh(ys, vals, height=0.55, color=cols, zorder=3)
ax.set_yticks(list(ys), labels, fontsize=8)
ax.set_xlabel("end-to-end wall-clock seconds", fontsize=9, color=MUTED)
ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
ax.set_xlim(0, max(vals) * 1.35)

for y, r in zip(ys, rows):
    mark = "✓ PASS" if r["verdict"] == "PASS" else "✗ FAIL"
    ax.text(r["wall_s"] + max(vals) * 0.02, y,
            f"{r['wall_s']}s · {r['turns']} turns  {mark}",
            va="center", fontsize=8.5, color=INK2 if r["verdict"] == "PASS" else CRIT,
            fontweight="normal" if r["verdict"] == "PASS" else "bold", zorder=4)

fig.suptitle("Coding-harness e2e vs local Gemma 4 12B (/v1/messages)",
             x=0.34, ha="left", fontsize=12.5, fontweight="bold", color=INK, y=0.95)
fig.text(0.34, 0.80, f"{rep['model']} · {rep['server'].split('(')[0].strip()} · client: {rep['client_env'].split(':')[0]} · {rep['date']}",
         fontsize=8.5, color=INK2)

out = sys.argv[2] if len(sys.argv) > 2 else path.replace(".json", ".png")
fig.savefig(out, facecolor=PAGE)
print("wrote", out)
