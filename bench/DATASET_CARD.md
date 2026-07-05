---
license: cc-by-4.0
tags:
  - gemma
  - llama.cpp
  - llamafile
  - benchmark
  - serving
  - sampling
  - repetition-loop
  - dry-sampler
  - jailbreak
  - embeddings
  - agentic-harness
pretty_name: Gemma 4 12B Serving-Behavior Benchmark
---

# Gemma 4 12B (QAT-Q4_0) — Serving-Behavior Test Data

Test data, charts, and the running research log from an **autonomous research
loop** characterizing and tuning a **Gemma 4 12B QAT-Q4_0** model served via
llama.cpp/llamafile on a single **RTX 3080 Ti**. Every ~30 min the loop
summarizes findings, proposes a goal, tests it end-to-end, documents success or
failure, and publishes here + to GitHub.

- **Model under test:** `gemma-4-12b-it-qat-q4_0.gguf` (Google, June 2026), 128K
  ctx, f16 KV, MTP speculative decoding, on the [SEBK4C llamafile fork](https://github.com/SEBK4C/Llamafile-gemma-4-12B-it-qat-q4_0-gguf-Inferance-And-embeddings).
- **Judge (where used):** GLM-5.2 Fast on Fireworks — external to the model
  under test, temperature 0. Never Anthropic/Claude.
- **Full protocol + narrative:** `RESEARCH_HISTORY.md` in this repo (findings
  F1–F15, experiments E1–E16, goals G1–G9 / H1–H8).

> ⚠️ **Gemma 4 12B is a small local model, not a frontier coding model.** This
> data characterizes *serving behavior* — endpoints, modalities, sampler,
> safety, throughput — on consumer hardware. It is not a capability leaderboard.

## What's been established (headlines)

| Theme | Finding | Files |
|---|---|---|
| **Full API surface** | 18/19 endpoints pass e2e incl. native Anthropic `/v1/messages`, OpenAI Responses, vision, audio-in, TTS; 113 tok/s, 107 ms TTFT | `api_probe_*` |
| **Agentic harnesses** | Claude Code, OpenCode, OpenClaw all drive the model e2e (7–12 s tasks) via one server | `harness_e2e_*` |
| **Embeddings** | the chat model's `/v1/embeddings` are **not semantic** (unrelated > related); a 146 MB nomic sidecar fixes it (+0.4–0.5 separation) | `embeddings_compare_*` |
| **Concurrency** | "one slot" is near-serial: ~1.37× overlap, ~200 tok/s ceiling, 0 errors to C=8 | `concurrency_*` |
| **Repetition loop** | greedy (temp 0) only; DRY sampler prevents it; fragile/prompt-specific | `topk_repro.csv`, `loop_*`, `temp_*` |
| **DRY collateral** | shipped DRY 0.8 suppresses **no** legitimate repetition (refrains/tables/code) through 1.2 | `g4_dry_*` |
| **System prompt** | distilled-Constitution prompt ≥ bare on acc/hum/soph/cal at power | `ab_*`, `serving_baseline_*` |
| **Jailbreak hardening** | an explicit decline clause lifts jailbreak resistance 0.75→1.00 at **zero** over-refusal cost | `g8_decline_*` |
| **Quality de-risk** | the decline clause adds no quality regression on the full battery (acc/soph up) | `g9_composite_*` |

## Files

**Suites (JSON + chart PNG, one stamp each):**
- `api_probe_*` — full endpoint/modality probe (JSON report + TSV + chart). Reproduce: `bench/api_probe.py`.
- `harness_e2e_*` — Claude Code / OpenCode / OpenClaw end-to-end task runs.
- `embeddings_compare_*` — chat-model vs nomic-sidecar cosine separation.
- `concurrency_*` — throughput/latency vs concurrent requests. Reproduce: `bench/concurrency_probe.py`.
- `g4_dry_*` — DRY collateral-damage + thinking-control. Reproduce: `bench/ab_dry.py`.
- `g8_decline_*` — jailbreak-hardening A/B (6 jailbreak + 6 benign × 4 reps). Reproduce: `bench/ab_decline.py`.
- `h8_thinking_*` — empty-content-by-prompt-class footgun map. Reproduce: `bench/ab_thinking.py`.
- `serving_baseline_*`, `g9_composite_*` — frozen-battery composite scorecards.

**Ledger + early CSVs:**
- `serving-results.tsv` — the running composite ledger (`ts,agent_id,acc,hum,soph,cal,rep,tok_s,serve_score,gates,status,hypothesis`).
- `topk_repro.csv`, `loop_sweep.csv`, `temp_isolate.csv` — the iteration-1–3 loop experiments (columns: sampler, dry, rep, looped 0/1, tok_s, n_out, reason).
- `ab_prompt.csv`, `ab_power.csv`, `ab_soph.csv` — system-prompt A/B sub-scores.

**Narrative:** `RESEARCH_HISTORY.md` — the authoritative, chronological log.

## Reproducing

All harnesses are stdlib-only or reuse `serve_bench.py` primitives, in the
[GitHub repo](https://github.com/SEBK4C/Llamafile-gemma-4-12B-it-qat-q4_0-gguf-Inferance-And-embeddings)
under `bench/`. Point them at your own server with `--base`/`--server`. The
one-command hardware check:

```sh
python3 bench/api_probe.py --base http://127.0.0.1:8080
```

Code + integration guides: https://github.com/SEBK4C/Llamafile-gemma-4-12B-it-qat-q4_0-gguf-Inferance-And-embeddings
