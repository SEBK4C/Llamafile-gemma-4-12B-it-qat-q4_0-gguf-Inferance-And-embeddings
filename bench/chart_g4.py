#!/usr/bin/env python3
"""G4 two-panel: (L) DRY collateral pass-rates — none; (R) the thinking-starves-content footgun."""
import json, sys, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, PAGE = "#fcfcfb", "#f9f9f7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
S1, S2, S3, GOOD, CRIT = "#2a78d6", "#1baf7a", "#4a3aa7", "#0ca30c", "#d03b3b"

path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("bench/data/g4_dry_*.json"))[-1]
d = json.load(open(path))
B = d["test_B"]
prompts = ["refrain", "times_table", "count_list", "accumulator"]
LAB = {"refrain": "poem refrain (×4)", "times_table": "7× table (12 rows)",
       "count_list": "count 1–20", "accumulator": "accumulator loop"}
dry_levels = ["0.0", "0.8", "1.2"]
colors = {"0.0": S1, "0.8": S2, "1.2": S3}

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                     "axes.edgecolor": BASELINE, "xtick.color": MUTED, "ytick.color": INK2})
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150, facecolor=PAGE,
                               gridspec_kw={"width_ratios": [1.5, 1]})
fig.subplots_adjust(left=0.12, right=0.975, top=0.72, bottom=0.16, wspace=0.34)

# left: collateral pass-rate, grouped bars (prompt × dry)
axL.set_facecolor(SURFACE)
ys = range(len(prompts)); h = 0.26
for j, dl in enumerate(dry_levels):
    off = (1 - j) * h
    rates = [B[dl][p]["pass_rate"] for p in prompts]
    axL.barh([y + off for y in ys], rates, height=h, color=colors[dl], zorder=3,
             label=f"DRY {dl}" + (" (shipped)" if dl == "0.8" else ""))
axL.set_yticks(list(ys), [LAB[p] for p in prompts], fontsize=9)
axL.set_xlim(0, 1.15); axL.set_xlabel("structural pass-rate (3 reps)  — 1.0 = repetition preserved", fontsize=8.5, color=MUTED)
axL.axvline(1.0, color=GOOD, lw=1.2, ls="--", zorder=2)
axL.grid(axis="x", color=GRID, lw=0.7, zorder=0)
axL.tick_params(length=0); [s.set_visible(False) for s in axL.spines.values()]
axL.legend(loc="lower left", fontsize=8, frameon=False)
axL.set_title("Collateral damage from DRY: none — every case 3/3 through 1.2", fontsize=9.5, color=INK)

# right: thinking-starves-content footgun (measured this iteration)
axR.set_facecolor(SURFACE)
bars = [("think ON\n@700", 0, CRIT), ("think ON\n@1500", 0, CRIT), ("think ON\n@2400", 0, CRIT),
        ("think OFF", 619, GOOD)]
xs = range(len(bars))
axR.bar(xs, [b[1] for b in bars], width=0.7, color=[b[2] for b in bars], zorder=3)
for x, b in zip(xs, bars):
    axR.text(x, b[1] + 20, str(b[1]), ha="center", fontsize=8.5, color=INK2, zorder=4)
axR.set_xticks(list(xs), [b[0] for b in bars], fontsize=7.8)
axR.set_ylim(0, 720); axR.set_ylabel("answer (content) chars", fontsize=8.5, color=MUTED)
axR.grid(axis="y", color=GRID, lw=0.7, zorder=0)
axR.tick_params(length=0); [s.set_visible(False) for s in axR.spines.values()]
axR.set_title("Footgun: reasoning starves content\non the refrain prompt", fontsize=9.5, color=INK)

fig.suptitle("G4 — shipped DRY 0.8 is safe insurance; plus a thinking-control footgun",
             x=0.02, ha="left", fontsize=12.5, fontweight="bold", color=INK, y=0.955)
fig.text(0.02, 0.82,
         "left: DRY suppresses no legitimate repetition at 0.0/0.8/1.2  ·  "
         "right: reasoning_effort ignored — only enable_thinking=false frees content",
         fontsize=8, color=INK2)

out = sys.argv[2] if len(sys.argv) > 2 else path.replace(".json", ".png")
fig.savefig(out, facecolor=PAGE)
print("wrote", out)
