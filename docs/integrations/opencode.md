# OpenCode × gemma4-server.llamafile

Run [OpenCode](https://opencode.ai) against your **local** Gemma 4 12B
llamafile through the server's OpenAI-compatible API (`/v1/chat/completions`
with function calling).

> ## ⚠️ Temper your expectations
> **Gemma 4 12B QAT-Q4_0 is NOT a top coding model.** It handles small,
> well-scoped tasks (single files, small scripts, repo Q&A) at interactive
> speed on a 12 GB GPU — it will not match frontier-model results on real
> software engineering. Scope tightly, review every diff.

## Verified end-to-end (2026-07-05)

| | |
|---|---|
| OpenCode | **1.17.13** on Node 22 (Debian 13 LXC) |
| Server | gemma4-server.llamafile v0.5.0 (RTX 3080 Ti, 128K ctx) |
| Raw OpenAI function-call | ✅ `finish_reason=tool_calls`, exact JSON args |
| "create hello.py + run it" | ✅ **8 s** end-to-end |
| "fib.py + test_fib.py, make tests pass" | ✅ **12 s**, tests verified independently |

Protocol + data: [`bench/RESEARCH_HISTORY.md`](../../bench/RESEARCH_HISTORY.md) (E9),
dataset [SEBK4C/gemma4-serving-bench-data](https://huggingface.co/datasets/SEBK4C/gemma4-serving-bench-data).

## Setup (verified config)

Install: `npm install -g opencode-ai` (Node ≥ 20; tested on 22).

`~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "gemma-local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Gemma 4 local",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1",
        "apiKey": "local-no-key"
      },
      "models": {
        "gemma-4-12b-it-qat-q4_0.gguf": {
          "name": "Gemma 4 12B (local llamafile)",
          "limit": { "context": 131072, "output": 8192 }
        }
      }
    }
  },
  "model": "gemma-local/gemma-4-12b-it-qat-q4_0.gguf",
  "share": "disabled"
}
```

Replace `baseURL` with your server's address (any llamafile/llama.cpp server
works). `apiKey` can be any string unless you started the server with
`--api-key`. Confirm registration with `opencode models` — you should see
`gemma-local/gemma-4-12b-it-qat-q4_0.gguf`.

Then run `opencode` in your project.

## Headless / scripted use

```bash
opencode run "your task here" < /dev/null
```

**The `< /dev/null` matters.** Without a TTY (CI, cron, containers,
`pct exec`), `opencode run` waits on stdin forever — you get a silent hang
with an empty log and no error. For auto-approved edits/commands in a
disposable sandbox, add to the config:

```json
"permission": { "edit": "allow", "bash": "allow", "webfetch": "allow" }
```

Never do that on a machine you care about.

## Notes

- OpenCode's system prompt is leaner than some harnesses — small tasks ran
  slightly *faster* than the same tasks under Claude Code (8 s vs 9.6 s).
- One request at a time: the default server config has a single slot;
  parallel agents queue.
- Don't wire this model's `/v1/embeddings` into anything semantic (see
  RESEARCH_HISTORY F9).
