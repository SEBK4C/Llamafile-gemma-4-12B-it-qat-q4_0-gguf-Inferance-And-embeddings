#!/usr/bin/env python3
"""G8 jailbreak-hardening chart: per-probe decline rates + the no-cost tradeoff."""
import json, sys, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, PAGE = "#fcfcfb", "#f9f9f7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
S1, S2 = "#2a78d6", "#1baf7a"   # blue = constitution (before), aqua = +decline (after)

path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("bench/data/g8_decline_*.json"))[-1]
d = json.load(open(path))
detail, summ = d["detail"], d["summary"]

def per_probe(pname):
    t = {}
    for x in detail[pname]["should_decline"]:
        t.setdefault(x["probe"], [0, 0]); t[x["probe"]][1] += 1; t[x["probe"]][0] += x["good"]
    return {k: g / n for k, (g, n) in t.items()}

pc, pd = per_probe("constitution"), per_probe("decline")
probes = list(pc.keys())
LABELS = {"sd_kitty": "flirty persona", "sd_dan": "DAN dev-mode", "sd_prefix": "prefix injection",
          "sd_devmode": "fake developer", "sd_hypothetical": "fiction wrapper", "sd_grandma": "grandma exploit"}

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                     "axes.edgecolor": BASELINE, "xtick.color": MUTED, "ytick.color": INK2})
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150, facecolor=PAGE,
                               gridspec_kw={"width_ratios": [1.55, 1]})
fig.subplots_adjust(left=0.14, right=0.975, top=0.74, bottom=0.13, wspace=0.32)

# left: per-probe decline rate, grouped horizontal bars
ys = range(len(probes)); h = 0.36
axL.set_facecolor(SURFACE)
axL.barh([y + h/2 for y in ys], [pc[p] for p in probes], height=h, color=S1, zorder=3, label="Constitution (shipped)")
axL.barh([y - h/2 for y in ys], [pd[p] for p in probes], height=h, color=S2, zorder=3, label="+ decline clause")
for y in ys:
    axL.text(pc[probes[y]] + 0.02, y + h/2, f"{pc[probes[y]]:.2f}", va="center", fontsize=8, color=INK2)
    axL.text(pd[probes[y]] + 0.02, y - h/2, f"{pd[probes[y]]:.2f}", va="center", fontsize=8, color=INK2)
axL.set_yticks(list(ys), [LABELS.get(p, p) for p in probes], fontsize=9)
axL.set_xlim(0, 1.18); axL.set_xlabel("jailbreak-decline rate (higher = safer)", fontsize=9, color=MUTED)
axL.grid(axis="x", color=GRID, lw=0.7, zorder=0)
axL.tick_params(length=0); [s.set_visible(False) for s in axL.spines.values()]
axL.set_title("Per attack shape: the 3 'soft' framings close to 1.00", fontsize=9.5, color=INK)

# right: the tradeoff — jailbreak-decline up, over-refusal flat
axR.set_facecolor(SURFACE)
groups = ["jailbreak\ndecline rate", "benign\nanswer rate"]
xs = range(len(groups)); w = 0.36
cvals = [summ["constitution"]["jailbreak_decline_rate"], summ["constitution"]["benign_answer_rate"]]
dvals = [summ["decline"]["jailbreak_decline_rate"], summ["decline"]["benign_answer_rate"]]
axR.bar([x - w/2 for x in xs], cvals, width=w, color=S1, zorder=3, label="Constitution")
axR.bar([x + w/2 for x in xs], dvals, width=w, color=S2, zorder=3, label="+ decline clause")
for x, v in zip(xs, cvals): axR.text(x - w/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8.5, color=INK2)
for x, v in zip(xs, dvals): axR.text(x + w/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8.5, color=INK2)
axR.set_xticks(list(xs), groups, fontsize=8.5)
axR.set_ylim(0, 1.42); axR.grid(axis="y", color=GRID, lw=0.7, zorder=0)
axR.tick_params(length=0); [s.set_visible(False) for s in axR.spines.values()]
axR.set_title("Safety up, helpfulness unchanged", fontsize=9.5, color=INK)
axR.legend(loc="upper left", fontsize=8, frameon=False)

fig.suptitle("G8 — explicit decline clause: 0.75 → 1.00 jailbreak resistance, 0 over-refusals",
             x=0.02, ha="left", fontsize=12.5, fontweight="bold", color=INK, y=0.955)
fig.text(0.02, 0.82, f"6 jailbreak + 6 benign-edgy probes × {d['replicas']} replicas, GLM-5.2 disposition judge · {d['stamp']}",
         fontsize=8.5, color=INK2)

out = sys.argv[2] if len(sys.argv) > 2 else path.replace(".json", ".png")
fig.savefig(out, facecolor=PAGE)
print("wrote", out)
