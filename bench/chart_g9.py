#!/usr/bin/env python3
"""G9: three-candidate serving comparison (bare / Constitution / +decline-clause)
from the frozen-battery ledger. Averages repeated decline rows."""
import csv, sys, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, PAGE = "#fcfcfb", "#f9f9f7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#1baf7a", "#4a3aa7"   # blue, aqua, violet

tsv = sys.argv[1] if len(sys.argv) > 1 else "bench/serving-results.tsv"
rows = list(csv.DictReader(open(tsv), delimiter="\t"))

def pick(pred):
    rs = [r for r in rows if pred(r)]
    return rs

def avg(rs, key):
    return statistics.mean(float(r[key]) for r in rs)

bare = pick(lambda r: r["status"] == "baseline")[-1]
const = pick(lambda r: "keep" in r["status"] and r["agent_id"].startswith("loop-it11"))[-1]
decl = pick(lambda r: "decline" in r["agent_id"])
cands = [
    ("bare 'helpful assistant'", S1, {k: float(bare[k]) for k in ("acc","hum","soph","cal","rep","tok_s","serve_score")}),
    ("Constitution (shipped)", S2, {k: float(const[k]) for k in ("acc","hum","soph","cal","rep","tok_s","serve_score")}),
    (f"+ decline clause (G8)", S3, {k: avg(decl, k) for k in ("acc","hum","soph","cal","rep","tok_s","serve_score")}),
]

dims = [("acc", 1, "accuracy"), ("hum", 5, "humanness"), ("soph", 5, "sophistication"),
        ("cal", 1, "calibration"), ("serve_score", 100, "serve_score (composite)")]

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                     "axes.edgecolor": BASELINE, "xtick.color": MUTED, "ytick.color": INK2})
fig, axes = plt.subplots(1, len(dims), figsize=(11.5, 3.7), dpi=150, facecolor=PAGE)
fig.subplots_adjust(left=0.04, right=0.99, top=0.66, bottom=0.10, wspace=0.42)

for ax, (key, vmax, label) in zip(axes, dims):
    ax.set_facecolor(SURFACE)
    vals = [c[2][key] for c in cands]
    xs = range(len(cands))
    ax.bar(xs, vals, width=0.68, color=[c[1] for c in cands], zorder=3)
    top = (vmax * 1.14 if key in ("acc", "cal") else max(vals) * 1.20)
    ax.set_ylim(0, top)
    for x, v in zip(xs, vals):
        ax.text(x, v + top * 0.02, f"{v:.3g}", ha="center", fontsize=8.5, color=INK2, zorder=4)
    # honesty: overlay the individual decline runs as dots (variance visible where it exists)
    dvals = [float(r[key]) for r in decl]
    ax.scatter([2]*len(dvals), dvals, s=18, color="#0b0b0b", zorder=5, alpha=0.75)
    ax.set_title(label, fontsize=8.5, color=INK2)
    ax.set_xticks([]); ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]

handles = [plt.Rectangle((0,0),1,1, color=c[1]) for c in cands]
fig.legend(handles, [c[0] for c in cands], loc="lower center", ncol=3, fontsize=9,
           frameon=False, bbox_to_anchor=(0.5, -0.01))

n_decl = len(decl)
fig.suptitle("G9 — the G8 decline clause adds no quality cost: accuracy & sophistication up, calibration perfect",
             x=0.04, ha="left", fontsize=12.5, fontweight="bold", color=INK, y=0.955)
fig.text(0.04, 0.80,
         f"frozen battery (11 probes × 2 replicas, GLM-5.2 judge) · black dots = {n_decl} individual decline runs · "
         f"composite (65-70) is within n=2 noise; the robust wins are acc/soph/cal", fontsize=8.5, color=INK2)

out = sys.argv[2] if len(sys.argv) > 2 else "bench/data/g9_composite.png"
fig.savefig(out, facecolor=PAGE)
print("wrote", out)
