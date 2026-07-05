#!/usr/bin/env python3
"""Render serving-baseline sub-score chart from the last N rows of serving-results.tsv."""
import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, PAGE = "#fcfcfb", "#f9f9f7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
S1, S2 = "#2a78d6", "#1baf7a"

tsv = sys.argv[1] if len(sys.argv) > 1 else "bench/serving-results.tsv"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
rows = list(csv.DictReader(open(tsv), delimiter="\t"))[-n:]

dims = [("acc", 1, "accuracy (0-1)"), ("hum", 5, "humanness (1-5)"),
        ("soph", 5, "sophistication (1-5)"), ("cal", 1, "calibration (0-1)"),
        ("tok_s", None, "tok/s")]

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                     "axes.edgecolor": BASELINE, "xtick.color": MUTED, "ytick.color": INK2})
fig, axes = plt.subplots(1, len(dims), figsize=(11, 3.4), dpi=150, facecolor=PAGE)
fig.subplots_adjust(left=0.06, right=0.985, top=0.68, bottom=0.16, wspace=0.5)

colors = [S1, S2]
for ax, (key, vmax, label) in zip(axes, dims):
    ax.set_facecolor(SURFACE)
    vals = [float(r[key]) for r in rows]
    xs = range(len(rows))
    ax.bar(xs, vals, width=0.6, color=[colors[i % 2] for i in xs], zorder=3)
    top = (vmax or max(vals)) * 1.14
    ax.set_ylim(0, top)
    for x, v in zip(xs, vals):
        ax.text(x, v + top * 0.03, f"{v:g}", ha="center", fontsize=8.5, color=INK2, zorder=4)
    ax.set_title(label, fontsize=8.5, color=INK2)
    names = ["bare" if "bare" in r["hypothesis"] else "Constitution" for r in rows]
    ax.set_xticks(list(xs), names, fontsize=8, rotation=0)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]

score_txt = "  ·  ".join(f"{('bare' if 'bare' in r['hypothesis'] else 'Constitution')}: serve_score {r['serve_score']}" for r in rows)
fig.suptitle("Serving-defaults baseline — full frozen battery (G1)", x=0.06, ha="left",
             fontsize=12.5, fontweight="bold", color=INK, y=0.97)
fig.text(0.06, 0.845, f"11 probes × 2 replicas, max_tokens 1600, GLM-5.2 judge  ·  {score_txt}",
         fontsize=8.5, color=INK2)

out = sys.argv[3] if len(sys.argv) > 3 else "bench/data/serving_baseline.png"
fig.savefig(out, facecolor=PAGE)
print("wrote", out)
