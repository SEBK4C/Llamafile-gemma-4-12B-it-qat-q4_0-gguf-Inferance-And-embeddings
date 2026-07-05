#!/usr/bin/env python3
"""Render api_probe report chart (light mode, GitHub PNG)."""
import json, sys, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# palette (dataviz reference instance, light mode)
SURFACE, PAGE = "#fcfcfb", "#f9f9f7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
GOOD, CRIT = "#0ca30c", "#d03b3b"

path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("bench/data/api_probe_*.json"))[-1]
rep = json.load(open(path))
rows = rep["results"]

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                     "axes.edgecolor": BASELINE, "xtick.color": MUTED, "ytick.color": INK2})

fig = plt.figure(figsize=(11, 9.6), dpi=150, facecolor=PAGE)
gs = fig.add_gridspec(2, 1, height_ratios=[4.2, 1.0], hspace=0.30,
                      left=0.30, right=0.965, top=0.885, bottom=0.05)

# ---------------- panel 1: latency per test ----------------
ax = fig.add_subplot(gs[0]); ax.set_facecolor(SURFACE)
tests = rows[::-1]  # suite order, top-down
ys = range(len(tests))
vals = [max(r.get("wall_s") or 0.006, 0.006) for r in tests]
cols = [GOOD if r["status"] == "PASS" else CRIT for r in tests]
labels = [f"{r['test']}  ·  {r['endpoint'].replace('POST ','').replace('GET ','')}" for r in tests]

ax.barh(ys, vals, height=0.58, color=cols, zorder=3)
ax.set_xscale("log"); ax.set_xlim(0.005, 8)
ax.set_yticks(list(ys), labels, fontsize=8.5)
ax.set_xlabel("wall-clock seconds — log scale", fontsize=9, color=MUTED)
ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
ax.spines["left"].set_visible(True); ax.spines["left"].set_color(BASELINE)

for y, r, v in zip(ys, tests, vals):
    w = r.get("wall_s") or 0.0
    mark = "✓ PASS" if r["status"] == "PASS" else "✗ FAIL"
    txt = f"{'<0.01' if w < 0.01 else f'{w:.2f}'}s  {mark}"
    ax.text(v * 1.18, y, txt, va="center", fontsize=8,
            color=INK2 if r["status"] == "PASS" else CRIT,
            fontweight="normal" if r["status"] == "PASS" else "bold", zorder=4)

# ---------------- panel 2: speed stat tiles ----------------
ax2 = fig.add_subplot(gs[1]); ax2.set_facecolor(PAGE); ax2.axis("off")
def g(test, key):
    for r in rows:
        if r["test"] == test: return r.get(key)
    return None
tiles = [
    (f"{g('completion_native','tok_s_server') or '—'}", "tok/s generation\n(server-timed)"),
    (f"{(g('chat_stream','ttft_s') or 0)*1000:.0f} ms", "time to first token\n(chat SSE)"),
    (f"{g('vision_input','wall_s'):.1f} s", "vision Q&A\nround-trip"),
    (f"{g('audio_input','wall_s'):.1f} s", "spoken-audio Q&A\nround-trip"),
    (f"{g('tts_speech','rtf') or '—'}×", "TTS speed\n(× realtime)"),
    (f"{g('embeddings','dims') or '—'}", "embedding dims\n(semantics: FAIL)"),
]
n = len(tiles)
for i, (val, lab) in enumerate(tiles):
    x = i / n + 0.5 / n
    bad = "FAIL" in lab
    ax2.text(x, 0.72, val, ha="center", va="center", fontsize=17,
             color=CRIT if bad else INK, fontweight="bold", transform=ax2.transAxes)
    ax2.text(x, 0.22, lab, ha="center", va="center", fontsize=7.8, color=INK2,
             transform=ax2.transAxes)

fig.suptitle("gemma4-server.llamafile — full API-surface end-to-end probe", x=0.30, ha="left",
             fontsize=14, fontweight="bold", color=INK, y=0.975)
fig.text(0.30, 0.925,
         f"{rep['model']}  ·  RTX 3080 Ti, CT 118 via tailnet  ·  {rep['stamp']}  ·  "
         f"{rep['pass']} PASS / {rep['fail']} FAIL / {rep['skip']} SKIP in {rep['total_wall_s']}s",
         fontsize=9, color=INK2)
fig.text(0.30, 0.905, "modalities live: text · vision · audio-in · TTS   |   agent APIs: OpenAI chat+completions+responses · Anthropic messages   |   bench/api_probe.py",
         fontsize=8, color=MUTED)

out = sys.argv[2] if len(sys.argv) > 2 else "bench/data/api_probe_latest.png"
fig.savefig(out, facecolor=PAGE)
print("wrote", out)
