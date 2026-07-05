# Serving Gemma 4 12B (QAT-Q4) well — a practical guide

Distilled from 20 iterations of end-to-end measurement on an RTX 3080 Ti
([full log](../bench/RESEARCH_HISTORY.md), [test data](https://huggingface.co/datasets/SEBK4C/gemma4-serving-bench-data)).
Every claim here links to the experiment that established it. Reproduce any of
it with the `bench/` harnesses.

> **What this model is.** A private, offline, genuinely multimodal endpoint
> (text + image + audio in, TTS out) that runs three real coding agents at
> interactive speed on one 12 GB GPU. **It is not a frontier coding model** —
> scope agentic tasks tightly and review the output. Within that scope it is
> fast and surprisingly capable.

## 1. Endpoints — what to use

The server (llama.cpp/llamafile fork) exposes far more than chat (E7, 18/19
probes pass):

| Use this | For |
|---|---|
| `POST /v1/chat/completions` | OpenAI chat — text, `image_url`, `input_audio`; SSE + function calling |
| `POST /v1/messages` (+`/count_tokens`) | **Anthropic Messages API, native** — Claude Code connects with env vars, no shim |
| `POST /v1/responses` | OpenAI Responses API (streaming) |
| `POST /tts/v1/audio/speech` | built-in Kokoro TTS → WAV |
| a **dedicated embedding sidecar** | retrieval / RAG — see §6 |

**Avoid:** the chat model's `/v1/embeddings` for anything semantic (§6), and
`/v1/completions` (raw completion on an instruct model degenerates).

One-command check that your build serves all of this at speed:
```sh
python3 bench/api_probe.py --base http://127.0.0.1:8080
```

## 2. Sampler & system prompt

**Sampler (shipped default, validated):** temp 1.0, top_k 64, top_p 0.95,
min_p 0.01, **DRY 0.8** (base 1.75, allowed_len 2), repeat_penalty **off**.
- Never serve greedy (temp 0): it's the one config that triggers a degenerate
  loop (E2/E3). Any temp ≥ 0.3 is safe; the shipped 1.0 is.
- DRY 0.8 is free insurance: it suppresses **zero** legitimate repetition
  (refrains, tables, code) even up to 1.2 (E16).

**System prompt (WebUI default):** a distilled Claude's-Constitution prompt beats
a bare "helpful assistant" on accuracy/humanness/sophistication/calibration at
power (E4–E6, E12). **Recommended upgrade:** add the explicit override-decline
clause — it lifts jailbreak resistance from 0.75 to **1.00 at zero over-refusal
cost** (E14) with no quality regression (E15). Candidate: `bench/candidates/decline.json`.

## 3. The empty-response footgun (important)

On **creative/constrained prompts** (a poem with a fixed refrain, an acrostic),
Gemma 4's hidden reasoning can run so long it never emits an answer within the
token budget — the API returns **empty content** even at 2400+ tokens (E17/F15).
Non-creative prompts (factual, code, math, lists) are unaffected.

- **Fix:** send `chat_template_kwargs: {"enable_thinking": false}`. It eliminates
  empty responses across every prompt class (0/24 in testing).
- `reasoning_effort: low/none` are **ignored** by this server — don't rely on them.
- If you keep thinking on, budget generous `max_tokens` (a small cap is the other
  cause of empty content).

## 4. Latency model — how to serve fast

Three independent effects; understand all three (E18/E19/E20):

| Phase | Behavior | Lever |
|---|---|---|
| **Prefill** (input) | ~2860 tok/s, fades to ~1770 at 74K ctx (O(n²)); ~42 s to first token at 74K | **prompt caching** |
| **Prompt cache** | warm turns reuse ~100% of a fixed prefix → **185× faster** prefill (4109 ms → 22 ms), stable across turns | send `cache_prompt: true` |
| **Decode** (output) | MTP-acceptance-bound, **non-monotonic**: ~137 tok/s at mid-ctx, collapses to ~36 at 74K as speculation stops helping | keep hot context moderate; expect ~40 tok/s deep |

**Practical:** for agentic multi-turn use (fixed system+context prefix, changing
turn), the big prefill is paid **once** — every later turn only prefills the new
tokens (E19). Keep `cache_prompt: true`. Deep-context *generation* (long outputs
at 47K+ ctx) is the one genuinely slow path (~40 tok/s); caching does not help
decode.

## 5. Concurrency

One slot by default: not strictly serial (2–4 concurrent requests overlap
~1.37×), but aggregate throughput **plateaus at ~200 tok/s** and per-request
latency degrades linearly — **zero errors/drops through 8 concurrent** (E13).
- 2 agents/tabs sharing one server: fine (~30% slowdown).
- 4+: throughput-capped and laggy. Budget ~200 tok/s **total**, not per client.
- Want real parallelism? Raise `-np N` + KV budget (untested; 12 GB is the limit).

## 6. Embeddings — use a sidecar

The chat model's `/v1/embeddings` returns valid-shaped vectors that **aren't
semantic** — unrelated pairs can score *higher* than related ones (E10). Run a
146 MB nomic-embed sidecar through the **same llamafile binary** (`-m` + CPU
flags; margins jump from ≈0 to +0.4–0.5). Full recipe: [docs/embeddings.md](embeddings.md).

## 7. Coding agents

Verified end-to-end (E8/E9/E11), each with a setup guide:
[Claude Code](integrations/claude-code.md) · [OpenCode](integrations/opencode.md)
· [OpenClaw](integrations/openclaw.md) · [Cline/Kilo](integrations/cline-kilo.md).
Small file-and-run tasks complete in 7–13 s. Long-context retrieval is reliable
(perfect needle recall through 74K tokens, E18) and efficient in multi-turn use
thanks to caching (§4).

## Reproduce everything

All harnesses live in `bench/` (stdlib-only or reusing `serve_bench.py`). Point
them at your server with `--base`/`--server`. Data + charts:
[SEBK4C/gemma4-serving-bench-data](https://huggingface.co/datasets/SEBK4C/gemma4-serving-bench-data).
