#!/usr/bin/env python3
"""H11 two-panel: (L) decode tok/s + MTP acceptance vs depth; (R) predictable vs novel at fixed ctx."""
import json, sys, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, PAGE = "#fcfcfb", "#f9f9f7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
S1, S3, S8 = "#2a78d6", "#4a3aa7", "#eb6834"   # blue=decode, violet=accept, orange

path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("bench/data/h11_decode_*.json"))[-1]
d = json.load(open(path))
A, B = d["vs_depth"], d["vs_predictability"]

def kfmt(t):
    return f"{t/1000:.0f}K" if t >= 1000 else str(t)

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                     "axes.edgecolor": BASELINE, "xtick.color": MUTED, "ytick.color": INK2})
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3), dpi=150, facecolor=PAGE,
                               gridspec_kw={"width_ratios": [1.35, 1]})
fig.subplots_adjust(left=0.08, right=0.90, top=0.72, bottom=0.15, wspace=0.5)

# left: decode tok/s (bars) + acceptance (line, twin axis) vs depth
xs = list(range(len(A)))
axL.set_facecolor(SURFACE)
axL.bar(xs, [r["decode_tok_s"] for r in A], width=0.6, color=S1, zorder=3)
for x, r in zip(xs, A):
    axL.text(x, r["decode_tok_s"] + 4, f"{r['decode_tok_s']:.0f}", ha="center", fontsize=8.5, color=INK2, zorder=4)
axL.set_xticks(xs, [kfmt(r["actual_tokens"]) for r in A], fontsize=8.5)
axL.set_ylim(0, max(r["decode_tok_s"] for r in A) * 1.2)
axL.set_ylabel("decode tok/s (bars)", fontsize=9, color=S1)
axL.set_xlabel("context length (tokens)", fontsize=9, color=MUTED)
axL.grid(axis="y", color=GRID, lw=0.7, zorder=0)
axL.tick_params(length=0); [s.set_visible(False) for s in axL.spines.values()]
axLa = axL.twinx()
axLa.plot(xs, [r["accept"] for r in A], "-o", color=S3, lw=2, ms=7, zorder=5)
for x, r in zip(xs, A):
    axLa.annotate(f"{r['accept']:.2f}", (x, r["accept"]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color=S3)
axLa.set_ylim(0, 1.0); axLa.set_ylabel("MTP acceptance (line)", fontsize=9, color=S3)
axLa.tick_params(length=0); [s.set_visible(False) for s in axLa.spines.values()]
axL.set_title("Decode peaks mid-context, collapses when MTP acceptance dies at depth", fontsize=8.8, color=INK)

# right: predictable vs novel at fixed ctx (decode bars + accept labels)
xsB = list(range(len(B)))
axR.set_facecolor(SURFACE)
axR.bar(xsB, [r["decode_tok_s"] for r in B], width=0.55, color=[S1, S8], zorder=3)
for x, r in zip(xsB, B):
    axR.text(x, r["decode_tok_s"] + 2, f"{r['decode_tok_s']:.0f} tok/s\naccept {r['accept']:.2f}",
             ha="center", fontsize=8.5, color=INK2, zorder=4)
axR.set_xticks(xsB, [r["content"] for r in B], fontsize=9)
axR.set_ylim(0, max(r["decode_tok_s"] for r in B) * 1.28)
axR.set_ylabel("decode tok/s", fontsize=9, color=MUTED)
axR.grid(axis="y", color=GRID, lw=0.7, zorder=0)
axR.tick_params(length=0); [s.set_visible(False) for s in axR.spines.values()]
axR.set_title("Same context, different content:\nnovel decodes slower (lower acceptance)", fontsize=9, color=INK)

fig.suptitle("H11 — decode speed tracks MTP acceptance (peaks mid-context, collapses deep)",
             x=0.02, ha="left", fontsize=12.5, fontweight="bold", color=INK, y=0.955)
fig.text(0.02, 0.82,
         f"Gemma-4 12B + MTP speculative decode · {d['n_predict']} tok gen · {d['reps']} reps · {d['stamp']} · "
         "decode is NOT monotonic in depth — speculative acceptance peaks mid-context then collapses at depth",
         fontsize=8, color=INK2)

out = sys.argv[2] if len(sys.argv) > 2 else path.replace(".json", ".png")
fig.savefig(out, facecolor=PAGE)
print("wrote", out)
