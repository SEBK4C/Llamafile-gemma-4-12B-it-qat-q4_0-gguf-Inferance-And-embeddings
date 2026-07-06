# Cline & Kilo Code × gemma4-server.llamafile

Point the [Cline](https://cline.bot) or [Kilo Code](https://kilocode.ai)
VS Code extensions at your **local** Gemma 4 12B llamafile through their
"OpenAI Compatible" provider.

> ## ⚠️ Temper your expectations
> **Gemma 4 12B QAT-Q4_0 is NOT a top coding model.** Cline-style agents lean
> on long system prompts and many tool round-trips; a 12B will handle small
> edits and questions, not large autonomous refactors. Keep requests scoped,
> review every diff.

## Honest verification status

**⚙️ Config verified at the API level — the GUI flow is not automated in our
test lab.** What we *did* verify end-to-end on this exact server (2026-07-05):

- the OpenAI-compatible surface these extensions use — `/v1/chat/completions`
  with SSE streaming and function calling (E9a), plus image input (E7) —
  all PASS via `bench/api_probe.py`;
- two other harnesses driving real agentic loops over the *same* surface
  (OpenCode, OpenClaw — see [the integrations table](../../README.md#agent-integrations-tested-end-to-end)).

If you run Cline/Kilo against this server, please report what you see — an
issue with your experience turns this ⚙️ into a ✅ (or a documented failure).

## Cline settings (VS Code)

| Setting | Value |
|---|---|
| API Provider | **OpenAI Compatible** |
| Base URL | `http://127.0.0.1:8080/v1` (your server + `/v1`) |
| API Key | any string (server ignores it unless started with `--api-key`) |
| Model ID | `gemma-4-12b-it-qat-q4_0.gguf` |
| Supports Images | **ON** (vision verified server-side) |
| Context window | **131072** (not 256000 — that's this model's real limit) |
| Input/output price | 0 (it's your GPU) |

## Kilo Code settings

Choose **"Bring my own Key"** → API Provider **OpenAI Compatible**, then the
same values as the Cline table (type the model ID and pick "Use custom" when
it appears).

## Notes

- These extensions send large system prompts every request — prefill runs on
  your GPU (~2.8k tok/s on a 3080 Ti, cached across turns). First response in
  a fresh workspace is the slow one.
- One request at a time: the default server has a single slot; parallel
  features (checkpoints diffing, background tasks) will queue.
- Don't enable anything embeddings/codebase-indexing against this model's
  `/v1/embeddings` — use the [embedding sidecar](../embeddings.md) instead.
