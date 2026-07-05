#!/usr/bin/env python3
"""H10: prompt-cache — prefill cost per turn (cold once, warm ~free, new ctx cold again)."""
import json, sys, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, PAGE = "#fcfcfb", "#f9f9f7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
CRIT, GOOD = "#d03b3b", "#0ca30c"

path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("bench/data/h10_promptcache_*.json"))[-1]
d = json.load(open(path))
rows = d["rows"]

labels, vals, cols, cached = [], [], [], []
for r in rows:
    if r["kind"] == "COLD":
        labels.append("turn 0\n(cold)"); cols.append(CRIT)
    elif r["kind"] == "control-newctx":
        labels.append("new ctx\n(control)"); cols.append(CRIT)
    else:
        labels.append(f"turn {r['turn']}\n(warm)"); cols.append(GOOD)
    vals.append(r["prefill_ms"] or 0)
    cached.append(r["cached_tokens"] or 0)

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                     "axes.edgecolor": BASELINE, "xtick.color": MUTED, "ytick.color": INK2})
fig, ax = plt.subplots(figsize=(10, 4.6), dpi=150, facecolor=PAGE)
fig.subplots_adjust(left=0.10, right=0.965, top=0.72, bottom=0.16)
ax.set_facecolor(SURFACE)

xs = range(len(rows))
ax.bar(xs, vals, width=0.66, color=cols, zorder=3)
for x, v in zip(xs, vals):
    ax.text(x, v + max(vals) * 0.02, f"{round(v)} ms", ha="center", fontsize=8.5,
            color=INK2 if v > 100 else GOOD, fontweight="bold" if v < 100 else "normal", zorder=4)
ax.set_xticks(list(xs), labels, fontsize=8.5)
ax.set_ylim(0, max(vals) * 1.16)
ax.set_ylabel("prefill time (ms)", fontsize=9, color=MUTED)
ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]

cold = next(r for r in rows if r["kind"] == "COLD")
warm = next(r for r in rows if r["kind"] == "warm")
speedup = round(cold["prefill_ms"] / warm["prefill_ms"])
saved = round(100 * (1 - warm["prefill_ms"] / cold["prefill_ms"]))
fig.suptitle(f"H10 — prompt cache: deep-context prefill is paid ONCE ({speedup}× faster warm)",
             x=0.10, ha="left", fontsize=13, fontweight="bold", color=INK, y=0.95)
fig.text(0.10, 0.80,
         f"~{d['context_tokens_target']//1000}K-token fixed prefix, changing user turn · "
         f"warm reuses {warm['cached_tokens']}/{warm['prompt_tokens']} tokens from cache · {saved}% prefill saved · a new context is cold again",
         fontsize=8.5, color=INK2)

out = sys.argv[2] if len(sys.argv) > 2 else path.replace(".json", ".png")
fig.savefig(out, facecolor=PAGE)
print("wrote", out)
