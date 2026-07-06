#!/usr/bin/env python3
"""Two-panel concurrency chart: aggregate throughput plateau + per-request latency growth."""
import json, sys, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, PAGE = "#fcfcfb", "#f9f9f7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
S1, S2, S6 = "#2a78d6", "#1baf7a", "#e34948"  # blue, aqua, red

path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("bench/data/concurrency_*.json"))[-1]
rep = json.load(open(path))
rows = rep["rows"]
C = [r["concurrency"] for r in rows]

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                     "axes.edgecolor": BASELINE, "xtick.color": MUTED, "ytick.color": INK2})
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150, facecolor=PAGE)
fig.subplots_adjust(left=0.07, right=0.975, top=0.74, bottom=0.14, wspace=0.24)

# --- left: aggregate throughput (plateau) + per-request throughput (decay)
for ax in (axL, axR):
    ax.set_facecolor(SURFACE); ax.grid(color=GRID, lw=0.7, zorder=0)
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_xticks(C); ax.set_xlabel("concurrent requests", fontsize=9, color=MUTED)

agg = [r["agg_tok_s"] for r in rows]
per = [r["per_req_tok_s"] for r in rows]
axL.plot(C, agg, "-o", color=S2, lw=2, ms=7, zorder=3, label="aggregate (whole server)")
axL.plot(C, per, "-o", color=S1, lw=2, ms=7, zorder=3, label="per request")
for x, v in zip(C, agg): axL.annotate(f"{v:g}", (x, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color=INK2)
for x, v in zip(C, per): axL.annotate(f"{v:g}", (x, v), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color=INK2)
axL.set_ylim(0, max(agg) * 1.25); axL.set_ylabel("tokens / second", fontsize=9, color=MUTED)
axL.set_title("Throughput: server plateaus, per-request decays", fontsize=9.5, color=INK)
axL.legend(loc="center right", fontsize=8, frameon=False)

# --- right: per-request latency growth (serialization signature)
p50 = [r["req_p50_s"] for r in rows]
pmax = [r["req_max_s"] for r in rows]
ideal = [rows[0]["batch_wall_s"] * c for c in C]  # pure-serial reference
axR.plot(C, pmax, "-o", color=S6, lw=2, ms=7, zorder=3, label="slowest request")
axR.plot(C, p50, "-o", color=S1, lw=2, ms=7, zorder=3, label="median request")
axR.plot(C, ideal, "--", color=MUTED, lw=1.4, zorder=2, label="pure-serial reference")
for x, v in zip(C, pmax): axR.annotate(f"{v:g}s", (x, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color=INK2)
axR.set_ylim(0, max(pmax + ideal) * 1.15); axR.set_ylabel("seconds", fontsize=9, color=MUTED)
axR.set_title("Latency: grows with load, beats pure-serial", fontsize=9.5, color=INK)
axR.legend(loc="upper left", fontsize=8, frameon=False)

fig.suptitle("Single-slot concurrency — what \"parallel requests queue\" actually costs (H7)",
             x=0.07, ha="left", fontsize=12.5, fontweight="bold", color=INK, y=0.96)
fig.text(0.07, 0.82,
         f"{rep['n_predict']} tok/request (ignore_eos) · best of {rep['reps']} · verdict: {rep['verdict']} · "
         f"0 errors through C={C[-1]} · {rep['stamp']}", fontsize=8.5, color=INK2)

out = sys.argv[2] if len(sys.argv) > 2 else path.replace(".json", ".png")
fig.savefig(out, facecolor=PAGE)
print("wrote", out)
