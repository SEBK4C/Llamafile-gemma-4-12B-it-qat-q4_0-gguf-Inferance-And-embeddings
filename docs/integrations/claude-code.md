# Claude Code × gemma4-server.llamafile

Run [Claude Code](https://claude.com/claude-code) — the Anthropic-compatible
agentic coding CLI — entirely against your **local** Gemma 4 12B llamafile. No
cloud API, no key, no adapter: this server implements the Anthropic Messages
API natively (`/v1/messages`, `/v1/messages/count_tokens`, SSE streaming, tool
use, thinking blocks).

> ## ⚠️ Temper your expectations
> **Gemma 4 12B QAT-Q4_0 is NOT a top coding model.** It will not match
> Claude/GPT-class results on real software engineering. What you get is a
> private, offline, surprisingly capable agent for **small, well-scoped tasks**
> — file edits, small scripts, quick refactors, repo Q&A. Keep tasks tight,
> cap turns, and review every diff. On a 12 GB RTX 3080 Ti it is *fast*
> (~110 tok/s, ~2.8k tok/s prefill), which makes the small-task loop pleasant.

## Verified end-to-end (2026-07-05)

| | |
|---|---|
| Claude Code | **2.1.201** on Node 22 (Debian 13 LXC) |
| Server | gemma4-server.llamafile v0.5.0 (RTX 3080 Ti, 128K ctx) |
| Raw tool round-trip | ✅ `tool_use` → `tool_result` → correct final answer |
| "create hello.py + run it" | ✅ 3 turns, **9.6 s** end-to-end |
| "fib.py + test_fib.py, make tests pass" | ✅ 4 turns, **12.5 s**, tests verified independently |

Test protocol + data: [`bench/RESEARCH_HISTORY.md`](../../bench/RESEARCH_HISTORY.md) (E8),
dataset [SEBK4C/gemma4-serving-bench-data](https://huggingface.co/datasets/SEBK4C/gemma4-serving-bench-data).

## Setup (env-var method — the verified path)

Start your server, then point Claude Code at it. **All five model overrides
matter** — they route subagent/background calls to your local model too:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080     # or your server's URL
export ANTHROPIC_API_KEY=local-no-key               # any value; server ignores it unless started with --api-key

M=gemma-4-12b-it-qat-q4_0.gguf
export ANTHROPIC_MODEL=$M \
       ANTHROPIC_SMALL_FAST_MODEL=$M \
       ANTHROPIC_DEFAULT_SONNET_MODEL=$M \
       ANTHROPIC_DEFAULT_HAIKU_MODEL=$M \
       ANTHROPIC_DEFAULT_OPUS_MODEL=$M

claude
```

Optional, recommended for offline use:

```bash
export DISABLE_TELEMETRY=1 DISABLE_ERROR_REPORTING=1 CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
```

### settings.json alternative

`~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8080",
    "ANTHROPIC_API_KEY": "local-no-key",
    "ANTHROPIC_MODEL": "gemma-4-12b-it-qat-q4_0.gguf",
    "ANTHROPIC_SMALL_FAST_MODEL": "gemma-4-12b-it-qat-q4_0.gguf",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "gemma-4-12b-it-qat-q4_0.gguf",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "gemma-4-12b-it-qat-q4_0.gguf",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "gemma-4-12b-it-qat-q4_0.gguf"
  }
}
```

## Headless / CI use

```bash
claude -p "your task here" --max-turns 12 --output-format json
```

Inside a disposable container you can add `--dangerously-skip-permissions`
(when running as root, also set `IS_SANDBOX=1`). Never do that on a machine
you care about.

## Caveats learned the hard way

- **Node ≥ 22.** On Debian 13's stock Node 20 the npm install "succeeds" with
  an EBADENGINE warning but installs no `claude` binary.
- **One slot.** The default server config serves one request at a time —
  concurrent subagents queue rather than parallelize.
- **Thinking tokens are real.** Gemma 4 emits a hidden reasoning block before
  the answer; tiny `max_tokens` budgets get eaten by reasoning. Leave the
  server default (unlimited) alone.
- **Keep CLAUDE.md / context small.** Every token is prefill on *your* GPU.
  Prefill is fast (~2.8k tok/s) and cached, but a 100k-token repo dump is not
  the small-task regime this model is good at.
- **Don't use `/v1/embeddings`** from this model for anything semantic
  (retrieval, memory). It returns valid-looking vectors that do not encode
  similarity (see RESEARCH_HISTORY F9).
- **Scope tasks like you'd brief a junior dev with amnesia**: one file, one
  behavior, explicit acceptance check ("run X and make it pass").
