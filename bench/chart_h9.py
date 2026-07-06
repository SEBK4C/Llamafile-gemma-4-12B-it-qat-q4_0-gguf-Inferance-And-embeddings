#!/usr/bin/env python3
"""H9 two-panel: needle-retrieval accuracy vs context depth + prefill latency vs depth."""
import json, sys, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, PAGE = "#fcfcfb", "#f9f9f7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
S1, S2, GOOD, CRIT = "#2a78d6", "#1baf7a", "#0ca30c", "#d03b3b"

path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("bench/data/h9_longctx_*.json"))[-1]
d = json.load(open(path))
rows = d["rows"]
xs = [r["actual_tokens"] for r in rows]

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                     "axes.edgecolor": BASELINE, "xtick.color": MUTED, "ytick.color": INK2})
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150, facecolor=PAGE)
fig.subplots_adjust(left=0.075, right=0.975, top=0.72, bottom=0.15, wspace=0.24)

def kfmt(t):
    return f"{t/1000:.0f}K" if t >= 1000 else str(t)

# left: retrieval accuracy vs depth
axL.set_facecolor(SURFACE)
accs = [r["accuracy"] for r in rows]
axL.plot(xs, accs, "-o", color=S2, lw=2, ms=8, zorder=3)
for x, r in zip(xs, rows):
    axL.annotate(f"{r['found']}/{r['n']}", (x, r["accuracy"]), textcoords="offset points",
                 xytext=(0, 9), ha="center", fontsize=8, color=INK2)
axL.set_ylim(0, 1.15); axL.set_xscale("log")
axL.set_xticks(xs, [kfmt(t) for t in xs], fontsize=8)
axL.set_xlabel("context length (actual tokens, log)", fontsize=9, color=MUTED)
axL.set_ylabel("needle-retrieval accuracy", fontsize=9, color=MUTED)
axL.axhline(1.0, color=GOOD, lw=1.1, ls="--", zorder=1)
axL.grid(color=GRID, lw=0.7, zorder=0)
axL.tick_params(length=0); [s.set_visible(False) for s in axL.spines.values()]
axL.set_title("Retrieval accuracy @ 50% depth", fontsize=9.5, color=INK)

# right: prefill latency vs depth
axR.set_facecolor(SURFACE)
lat = [r["prefill_s"] for r in rows]
axR.plot(xs, lat, "-o", color=S1, lw=2, ms=8, zorder=3)
for x, v in zip(xs, lat):
    if v is not None:
        axR.annotate(f"{v:.1f}s", (x, v), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8, color=INK2)
axR.set_xscale("log")
axR.set_xticks(xs, [kfmt(t) for t in xs], fontsize=8)
axR.set_ylim(0, max(v for v in lat if v) * 1.2)
axR.set_xlabel("context length (actual tokens, log)", fontsize=9, color=MUTED)
axR.set_ylabel("prefill time (seconds)", fontsize=9, color=MUTED)
axR.grid(color=GRID, lw=0.7, zorder=0)
axR.tick_params(length=0); [s.set_visible(False) for s in axR.spines.values()]
ptps = next((r["prefill_tok_s"] for r in rows if r.get("prefill_tok_s")), None)
ptps_lo=[r["prefill_tok_s"] for r in rows if r.get("prefill_tok_s")]
axR.set_title(f"Prefill cost — throughput fades {max(ptps_lo)}→{min(ptps_lo)} tok/s" if ptps_lo else "Prefill cost", fontsize=9.5, color=INK)

allfound = all(r["accuracy"] == 1.0 for r in rows)
verdict = "perfect retrieval through {} tokens".format(kfmt(max(xs))) if allfound else "retrieval degrades with depth"
fig.suptitle(f"H9 — long-context needle-in-haystack: {verdict}",
             x=0.02, ha="left", fontsize=12.5, fontweight="bold", color=INK, y=0.955)
fig.text(0.02, 0.82,
         f"Gemma-4 12B QAT-Q4, 128K ctx, RTX 3080 Ti · needle at 50% depth · enable_thinking=false · {d['reps']} reps · {d['stamp']}",
         fontsize=8.5, color=INK2)

out = sys.argv[2] if len(sys.argv) > 2 else path.replace(".json", ".png")
fig.savefig(out, facecolor=PAGE)
print("wrote", out)
