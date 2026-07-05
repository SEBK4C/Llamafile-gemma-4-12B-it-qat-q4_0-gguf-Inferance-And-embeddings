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

### E3 — Temperature isolation @ max_tokens 1500, DRY off (SUCCESS — resolves the E1/E2 confound)
temp {0.0, 0.3, 0.7, 1.0}, DRY off, top_k 64, max_tokens 1500, 2 reps.
Data: `data/temp_isolate.csv`.

| temperature | looped |
|---|---|
| 0.0 (greedy) | **2/2** |
| 0.3 | 0/2 |
| 0.7 | 0/2 |
| 1.0 | 0/2 |

**Decisive — the confound is resolved:**
1. The loop is STRICTLY a greedy (temp 0) phenomenon. Any temperature ≥ 0.3
   escapes it completely, even with DRY OFF. Sharp cliff (0 → 0.3).
2. This explains E1's null result: E1 included temp 0, but at max_tokens 850 the
   generation never reached the loop point (~10th list item). So **length gates
   whether the loop MANIFESTS (~1500 tokens); temperature gates whether it OCCURS
   AT ALL (only greedy)**.
3. TWO independent fixes for the greedy loop: (a) any temp ≥ 0.3, or (b) DRY
   (E2). The official Gemma recipe (temp 1.0) already avoids it; DRY is
   belt-and-suspenders. **The one dangerous config is temp 0 — never serve greedy.**
   (Exact threshold between 0 and 0.3 untested; practically irrelevant.)

### E4 — A/B: distilled Constitution prompt vs bare prompt (SURPRISE — no advantage, preliminary)
Same official sampler; 4 probes (1/category: qa_boil, fp_einstein, sa_chem,
sd_kitty), 2 replicas, max_tokens 1536, GLM-5.2 judge. Data: `data/ab_prompt.csv`.

| dimension | Constitution | bare "helpful assistant" |
|---|---|---|
| acc | 1.00 | 1.00 |
| hum (1-5) | 3.25 | 3.12 |
| soph (1-5) | 4.00 | **4.38** |
| cal (0-1) | 0.75 | **1.00** |
| serve_score | 74.0 | **76.6** |

**Finding (honest, preliminary):** the distilled Constitution prompt did NOT
outperform a bare prompt on this small battery — marginally WORSE on calibration
(0.75 vs 1.00) and sophistication (4.00 vs 4.38), tied on accuracy, slightly
ahead on humanness. Net serve_score favored the bare prompt (76.6 vs 74.0).
- **Caveat: tiny sample** (n=2 replicas × 1 probe/category = 4 points/dimension)
  + GLM-judge noise; gaps are within plausible noise. NOT conclusive.
- The cal gap is consistent with the flagged tradeoff: the "not over-cautious /
  capable adult" framing may decline jailbreaks slightly less reliably. Needs a
  per-probe breakdown to see if sd_kitty (jailbreak) or sa_chem (over-refusal)
  slipped.
- **Implication:** the Constitution prompt is NOT yet validated as an improvement.
  Do not deploy it as the WebUI default on this evidence — reinforces the earlier
  "present as candidate, don't ship" call.

---

## Proposed research goals (next iterations)

- **G1 — Full frozen baseline.** Run the whole probe battery (acc/hum/soph/cal/
  rep/tok_s) at max_tokens ≥1500 to seed a trustworthy baseline row + confirm the
  decline rate at scale. (Pending; blocks the gate logic.)
- **G2 — Isolate temp vs length.** ✅ RESOLVED (E3): temperature gates loop
  OCCURRENCE (only greedy/temp-0 loops; temp ≥ 0.3 never does); length gates
  MANIFESTATION (~1500 tokens to reach the loop point).
- **G3 — A/B the Constitution prompt** vs bare/no-prompt. ⚠️ ATTEMPTED (E4),
  INCONCLUSIVE: no advantage on n=2 (possibly slight cal/soph regression). Needs
  a higher-powered rerun before any verdict.
- **G6 — Higher-powered prompt A/B (from E4).** Repeat E4 with ≥5 replicas and
  ≥2 probes/category, and instrument per-probe disposition, to decide if the
  cal/soph gap is real signal or judge noise. This is the gate on whether the
  Constitution prompt ships.
- **G4 — DRY parameter sensitivity.** dry_multiplier {0.4, 0.8, 1.2} × allowed_length
  {2, 4} — find the gentlest DRY that still kills loops without harming acc.
- **G5 — Publish cadence.** Charts + CSVs to GitHub + HF dataset repo each iteration.

---

## Publish log
- Data: `data/*.csv`  |  Charts: `data/*.png`
- GitHub: SEBK4C/Llamafile-gemma-4-12B-...  (main repo, SSH deploy key)
- HF dataset: SEBK4C/gemma4-serving-bench-data (created iteration 1)
