# OpenClaw × gemma4-server.llamafile

Run [OpenClaw](https://openclaw.ai) — the self-hosted AI agent hub — with your
**local** Gemma 4 12B llamafile as its model provider, via the server's
OpenAI-compatible API.

> ## ⚠️ Temper your expectations
> **Gemma 4 12B QAT-Q4_0 is NOT a top coding model.** It runs OpenClaw's agent
> loop (chat + exec tools) reliably on small, well-scoped requests, fully
> offline. Don't expect frontier-model results; review what it does.

## Verified end-to-end (2026-07-05)

| | |
|---|---|
| OpenClaw | **2026.6.11 (e085fa1)**, Debian 13 LXC, Node 22 |
| Server | gemma4-server.llamafile v0.5.0 (RTX 3080 Ti) |
| Chat turn ("PONG") | ✅ **11 s** through the full agent stack |
| Exec-tool turn (create file + read back) | ✅ **7 s**, artifact independently verified |

Protocol + data: [`bench/RESEARCH_HISTORY.md`](../../bench/RESEARCH_HISTORY.md) (E11),
dataset [SEBK4C/gemma4-serving-bench-data](https://huggingface.co/datasets/SEBK4C/gemma4-serving-bench-data).

## Setup (verified config)

Install OpenClaw (`curl -fsSL https://openclaw.ai/install.sh | bash`), then
write `~/.openclaw/openclaw.json` **before** onboarding (skip the model/auth
wizard steps — they're already configured):

```json
{
  "agents": {
    "defaults": {
      "model": { "primary": "gemma-local/gemma-4-12b-it-qat-q4_0.gguf" },
      "models": {
        "gemma-local/gemma-4-12b-it-qat-q4_0.gguf": { "alias": "Gemma 4 12B local" }
      }
    }
  },
  "models": {
    "mode": "merge",
    "providers": {
      "gemma-local": {
        "baseUrl": "http://127.0.0.1:8080/v1",
        "apiKey": "local-no-key",
        "api": "openai-completions",
        "models": [
          {
            "id": "gemma-4-12b-it-qat-q4_0.gguf",
            "name": "Gemma 4 12B (local llamafile)",
            "reasoning": false,
            "input": ["text", "image"],
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
            "contextWindow": 131072,
            "maxTokens": 8192
          }
        ]
      }
    }
  }
}
```

Replace `baseUrl` with your server's address. Check the wiring:

```sh
openclaw config validate     # → Config valid
openclaw agents list         # → main (default) · Model: gemma-local/gemma-4-12b-it-qat-q4_0.gguf
```

Then either run the full daemon + dashboard
(`openclaw onboard --install-daemon`, chat UI at `http://127.0.0.1:18789`) or
use the embedded agent directly from the CLI — the path we verified:

```sh
openclaw agent --local --session-key mychat -m "your message" --json < /dev/null
```

## Gotchas (all hit during verification)

- **A session target is required** — `openclaw agent` errors without
  `--session-key`, `--agent`, or `--to`. Any string works for `--session-key`;
  turns with the same key share history.
- **Redirect stdin** (`< /dev/null`) in scripts/CI, or CLI agents can wait on
  a never-closing pipe (same trap as OpenCode, F11).
- `--local` runs the embedded agent without the gateway daemon — simplest for
  testing; the daemon/dashboard flow uses the same provider config.
- Set `"reasoning": false` in the model entry: the server already strips
  Gemma 4's thinking channel into `reasoning_content`, and OpenClaw consumes
  the clean `content` field.
- The dashboard/channel integrations (Telegram, Slack, …) are OpenClaw
  features beyond this guide — we verified the model-provider layer.
