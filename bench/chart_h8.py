#!/usr/bin/env python3
"""H8: empty-content rate by prompt class (thinking ON vs OFF) + mean reasoning length."""
import json, sys, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, PAGE = "#fcfcfb", "#f9f9f7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
CRIT, GOOD, S1 = "#d03b3b", "#0ca30c", "#2a78d6"

path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("bench/data/h8_thinking_*.json"))[-1]
d = json.load(open(path))
C = d["classes"]
# order by think-ON empty rate desc so the footgun classes surface at top
order = sorted(C, key=lambda k: C[k]["think_on_empty_rate"], reverse=True)
LAB = {"creative_constrained": "creative (constrained)", "creative_open": "creative (open)",
       "factual_simple": "factual", "code": "code", "math": "math", "structured_list": "structured list"}

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                     "axes.edgecolor": BASELINE, "xtick.color": MUTED, "ytick.color": INK2})
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3), dpi=150, facecolor=PAGE,
                               gridspec_kw={"width_ratios": [1.25, 1]})
fig.subplots_adjust(left=0.16, right=0.965, top=0.72, bottom=0.14, wspace=0.36)

ys = range(len(order)); h = 0.38
axL.set_facecolor(SURFACE)
axL.barh([y + h/2 for y in ys], [C[k]["think_on_empty_rate"] for k in order], height=h,
         color=CRIT, zorder=3, label="thinking ON")
axL.barh([y - h/2 for y in ys], [C[k]["think_off_empty_rate"] for k in order], height=h,
         color=GOOD, zorder=3, label="thinking OFF (enable_thinking=false)")
for y, k in zip(ys, order):
    axL.text(C[k]["think_on_empty_rate"] + 0.02, y + h/2, f"{C[k]['think_on_empty_rate']:.2f}",
             va="center", fontsize=8, color=INK2)
axL.set_yticks(list(ys), [LAB.get(k, k) for k in order], fontsize=9)
axL.set_xlim(0, 1.15); axL.set_xlabel("empty-content rate (fraction returning no answer)", fontsize=8.5, color=MUTED)
axL.grid(axis="x", color=GRID, lw=0.7, zorder=0)
axL.tick_params(length=0); [s.set_visible(False) for s in axL.spines.values()]
axL.legend(loc="center right", fontsize=8, frameon=False)
axL.set_title("Empty answers by prompt class — the footgun is class-specific", fontsize=9.5, color=INK)

axR.set_facecolor(SURFACE)
vals = [C[k]["mean_reasoning_chars"] for k in order]
axR.barh(list(ys), vals, height=0.6, color=S1, zorder=3)
for y, v in zip(ys, vals):
    axR.text(v + max(vals)*0.02, y, f"{v}", va="center", fontsize=8, color=INK2)
axR.set_yticks(list(ys), ["" for _ in order])
axR.set_xlim(0, max(vals) * 1.2)
axR.set_xlabel("mean reasoning chars (thinking ON)", fontsize=8.5, color=MUTED)
axR.grid(axis="x", color=GRID, lw=0.7, zorder=0)
axR.tick_params(length=0); [s.set_visible(False) for s in axR.spines.values()]
axR.set_title("...and it tracks reasoning length", fontsize=9.5, color=INK)

fig.suptitle("H8 — Gemma-4 empty-content footgun: long reasoning starves the answer",
             x=0.02, ha="left", fontsize=12.5, fontweight="bold", color=INK, y=0.955)
fig.text(0.02, 0.82,
         f"max_tokens {d['budget']} · {d['reps']} reps/prompt × 2 prompts/class · "
         f"enable_thinking=false eliminates it everywhere (green = 0)", fontsize=8.5, color=INK2)

out = sys.argv[2] if len(sys.argv) > 2 else path.replace(".json", ".png")
fig.savefig(out, facecolor=PAGE)
print("wrote", out)
