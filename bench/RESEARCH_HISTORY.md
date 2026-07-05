# Gemma 4 Serving-Defaults — Research History

Autonomous research log for tuning the DEFAULT serving behavior (system prompt +
sampler) of the CT 118 **Gemma 4 12B QAT-Q4_0** llamafile. Harness: `serve_bench.py`
(GLM-5.2/Fireworks judge). This file is the running record; each cron iteration
(every 30 min) appends findings, proposes goals, tests them end-to-end, and
records success or failure.

---

## Findings log

### F1 — CT 118 reboot recovery (2026-07-05)
After the 2026-07-04 host reboot the gemma service crash-looped (`--gpu nvidia
... wasn't available`, restart #7555). Cause: host `/dev/nvidia-uvm` absent →
the container's `optional` uvm bind-mount silently skipped. Fix: `nvidia-modprobe
-u -c 0` on host (majors came back 507/511, unchanged) → `pct reboot 118`.
Restored: 11020 MiB VRAM, ~88 tok/s chat. **Evidence:** journalctl, nvidia-smi.

### F2 — Bench architecture
Candidates = pure per-request API calls vs the running server (no restart per
candidate). GPU inference serialized behind a flock; judge calls (external GLM-5.2
on Fireworks) fan out. Sub-scores acc/hum/soph/cal/rep/tok_s → gated composite.
Key fetched at runtime from 1Password (in-memory only).

### F3 — reasoning/content split + empty-content trap
Gemma 4 splits output into `reasoning_content` (hidden scratchpad) and `content`
(served answer). Too-small `max_tokens` burns the whole budget on reasoning →
empty `content`. Harness judges quality on `content`, loop-detects over both,
penalizes empty content. **Implication for experiments: use ≥1500 max_tokens or
loops/answers are masked.**

### F4 — Official Gemma 4 sampler (published defaults)
Google official recipe (corroborated 6 ways: model card, unsloth, Ollama QAT):
**temp 1.0 / top_k 64 / top_p 0.95 / min_p ~0.01, repeat_penalty OFF (1.0)**.
Gemma 4 (June 2026 release) has a **NATIVE system role** (unlike Gemma 1/2/3).
DRY (0.8/1.75/2) is community anti-loop insurance, preferred over repeat_penalty
(which penalizes legit repeated names/numbers and misses line-level loops).

### F5 — System prompt = distilled Claude's Constitution
Cut-down from anthropic.com/constitution (CC0, Jan 2026): honest/no-fabrication,
calibrated, correct-false-premise, "diplomatically honest not dishonestly
diplomatic", non-sycophantic, follow-real-intent, capable-adult, not-over-cautious,
warm-not-obsequious. In `defaults.json`. Deploy target = WebUI default via
`--ui-config` `systemMessage` key (client-side, NOT the API path).

### F6 — Decline behavior under the new prompt
cal 0.875 on Kitty + DAN jailbreak probes (declines most, not bulletproof). The
loosened "don't refuse over unlikely harms" clause does not collapse safety.

---

## Experiments log

### E1 — Temperature × DRY sweep @ max_tokens 850 (FAILED to reproduce loop)
temp {0.0, 0.5, 1.0} × DRY {off, on} × 2 reps = 12 gens. **Result: 0/12 looped.**
Data: `data/loop_sweep.csv`. This CONTRADICTED the earlier greedy-loops demo →
hypothesis "temp 0 always loops" is wrong at this length. Root cause (see E2):
the loop manifests ~item 10 of the list, deep in the generation; 850 tokens was
too short to reach it (all budget went to reasoning before the loop point).
**Failure that produced the length insight (F3).**

### E2 — top_k × DRY reproduction @ temp 0, max_tokens 1500 (SUCCESS)
top_k {1, 64} × DRY {off, on} × 2 reps = 8 gens. Data: `data/topk_repro.csv`.

| condition | looped | tok/s |
|---|---|---|
| top_k 1, DRY off | **2/2** (`'10. Empathetic' x3`) | 144 |
| top_k 1, DRY on | 0/2 | 131 |
| top_k 64, DRY off | **2/2** (`'10. Empathetic' x3`) | 142 |
| top_k 64, DRY on | 0/2 | 130 |

**Decisive findings:**
1. Greedy (temp 0) loops **regardless of top_k** — my earlier "top_k 1 is the
   trigger" reasoning was WRONG; top_k 64 loops identically at temp 0.
2. **DRY reliably prevents the loop** (0/2 in both top_k conditions).
3. Looping runs decode ~10% FASTER (142-145 vs 130-131 tok/s) — repeated tokens
   are trivially predictable (high MTP acceptance). "Fast garbage."
4. Combined with E1: the loop needs **greedy decoding AND sufficient length
   (~1500 tokens)**; short generations mask it.

---

## Proposed research goals (next iterations)

- **G1 — Full frozen baseline.** Run the whole probe battery (acc/hum/soph/cal/
  rep/tok_s) at max_tokens ≥1500 to seed a trustworthy baseline row + confirm the
  decline rate at scale. (Pending; blocks the gate logic.)
- **G2 — Isolate temp vs length.** Sweep temp {0, 0.3, 0.7, 1.0} at max_tokens
  1500 (DRY off) to separate the temperature gate from the length gate. E1+E2
  leave this confounded.
- **G3 — A/B the Constitution prompt** vs bare/no-prompt on the full battery
  (hum/soph/cal deltas) — does the distilled prompt measurably improve serving
  quality without loosening the decline gate?
- **G4 — DRY parameter sensitivity.** dry_multiplier {0.4, 0.8, 1.2} × allowed_length
  {2, 4} — find the gentlest DRY that still kills loops without harming acc.
- **G5 — Publish cadence.** Charts + CSVs to GitHub + HF dataset repo each iteration.

---

## Publish log
- Data: `data/*.csv`  |  Charts: `data/*.png`
- GitHub: SEBK4C/Llamafile-gemma-4-12B-...  (main repo, SSH deploy key)
- HF dataset: SEBK4C/gemma4-serving-bench-data (created iteration 1)
