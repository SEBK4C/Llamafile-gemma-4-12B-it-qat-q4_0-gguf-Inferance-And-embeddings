# v0.5.0 — Optimized Defaults

Bakes empirically-validated serving defaults into the Gemma 4 12B QAT-Q4_0
llamafile: an optimized sampler (Google's official recipe + DRY anti-loop
insurance) and a distilled Claude's-Constitution system prompt.

## What changed
- **Sampler (server default — applies to API *and* WebUI):** temperature 1.0,
  top_k 64, top_p 0.95, min_p 0.01, DRY (multiplier 0.8, base 1.75,
  allowed_length 2, penalty_last_n -1), repeat_penalty OFF (1.0). Google's
  official Gemma 4 recipe plus community DRY. Baked as server flags in `.args`,
  so `/v1` requests that omit sampler params get these defaults
  (`default_generation_settings.params`).
- **System prompt (WebUI default ONLY):** a distilled Claude's Constitution
  prompt (honest, calibrated, corrects false premises, non-sycophantic, follows
  real intent, capable-adult, not over-cautious, warm-not-obsequious). Delivered
  via `--ui-config-file /zip/ui-config.json`; the browser seeds it as the default
  system message per new conversation. **It is NOT injected into API `/v1`
  requests** — raw API callers get no system prompt unless they set one. (This
  asymmetry is deliberate: sampler is a server-wide default, the prompt is a WebUI
  convenience.)

## Why these defaults (evidence)
- **Anti-loop:** greedy decoding (temperature 0) reliably produces a degenerate
  single-line loop on this model — independent of top_k, once generation is long
  enough (~1500 tokens). Any temperature ≥ 0.3, or the DRY sampler, prevents it.
  **Never serve greedy.** (Experiments E1–E3.)
- **System prompt validated:** a higher-powered A/B vs a bare "helpful assistant"
  prompt shows the Constitution prompt wins or ties every judged dimension —
  accuracy 1.00=1.00, humanness 3.44>3.25, sophistication 4.75>4.38, calibration
  0.90>0.85 — and notably hardens against persona jailbreaks (declines the "Kitty"
  override 4/5 vs the bare prompt's 2/5) with no over-refusal on benign requests.
  (Experiments E4–E6.)

## Review the testing
- **Test data + charts:** https://huggingface.co/datasets/SEBK4C/gemma4-serving-bench-data
- **Protocol + full research log:** `bench/RESEARCH_HISTORY.md`, harness
  `bench/serve_bench.py`, spec `bench/program.md` (judge = GLM-5.2 on Fireworks,
  external to the model under test).

## Verification (this binary)
Verified standalone with no overriding args: `/props` `ui_settings.systemMessage`
= the 1087-char prompt (WebUI), `default_generation_settings.params` = the sampler
(API, confirmed behaviorally), server starts clean, and a raw API completion with
no system message is not prompt-shaped — confirming the prompt stays WebUI-only.

## Files & rollback
- `gemma4-server.llamafile` — APE binary with baked defaults. Multi-arch CUDA
  (sm_80 / 86 / 89 / 120 — Ampere through Blackwell).
- Rollback: the prior release retains the old defaults; `package/gemma4.args.bak`
  is the pre-optimization args file.
