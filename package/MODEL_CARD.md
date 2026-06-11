---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
base_model: google/gemma-4-12B-it-qat-q4_0-gguf
pipeline_tag: text-generation
tags:
- llamafile
- gguf
- gemma4
- embeddings
- speculative-decoding
---

# Gemma 4 12B IT (QAT q4_0) — multimodal dual-mode llamafile

A single self-contained executable that serves **chat completions (text +
image + audio input) and embeddings from one model instance** — one set of
weights in memory, one port. Built on a
[performance fork](https://github.com/SEBK4C/llamafile/tree/mtp-gemma4-drafter)
of [mozilla-ai/llamafile](https://github.com/mozilla-ai/llamafile) (v0.10.7)
with [google/gemma-4-12B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf)
weights, the multimodal projector, and Google's 423M MTP assistant drafter
baked in. Runs on macOS, Linux and BSD, on arm64 and x86_64, with no
installation.

**New in v0.10.7:** the KV cache now **survives restarts automatically** — slots autosave on graceful shutdown and autorestore at launch (`--no-slot-autosave` opts out), and a read-only launch directory no longer prevents startup (falls back to `~/.cache/llamafile/kv/`).

**New (2026-06-11): NVIDIA CUDA works out of the box** — a CUDA backend is
bundled (only the NVIDIA driver needed, no toolkit) and MTP speculative
decoding is validated on CUDA: **1.65× prose, up to 2.5× edit/copy** vs no
speculation on an RTX 4090 (165 / 239 tok/s with the recommended
`-sm none --spec-draft-n-max 4`). `--spec-type none` now genuinely disables
the baked-in drafter.

**New in v0.10.6:** true multi-token-prediction speculative
decoding on by default — **1.6× faster prose, 1.9× faster edit/copy tasks**
on Apple silicon vs no speculation — plus a Metal small-batch matmul
dispatch fix and removal of a hidden double-prefill (up to −43%
time-to-first-token).

## Usage

```sh
curl -LO https://huggingface.co/SEBK4C/gemma-4-12b-it-qat-q4_0-llamafile/resolve/main/gemma4-server.llamafile
chmod +x gemma4-server.llamafile
./gemma4-server.llamafile
```

Then:

```sh
# chat (OpenAI-compatible; thinking arrives in reasoning_content)
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Why is the sky blue?"}],"max_tokens":512}'

# embeddings (3840-dim, mean-pooled, L2-normalized) — same server instance
curl http://127.0.0.1:8080/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input":["the sky is blue","der Himmel ist blau"]}'
```

Images and audio go in as standard OpenAI content parts on the same chat
endpoint: `{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}`
and `{"type":"input_audio","input_audio":{"data":"<base64>","format":"wav"}}`
(wav or mp3; decoding, resizing and 16 kHz frame chunking happen server-side).

CLI flags override the baked-in defaults (`--port 9000`, `-c 16384`, …).

## Measured performance (Apple M4 Mac mini, 16 GB, Metal, default config)

| workload | tok/s | vs no speculation |
|---|---|---|
| freeform prose (greedy) | 21 | 1.6× |
| edit/copy task (greedy) | 25 | 1.9× |
| no-spec baseline (`--spec-type none`) | 13 | – |

CPU-only mode (`-ngl 0`) is roughly 4–5× slower than Metal on the same
machine; MTP still helps (~+14%).

## System requirements

| `-c` (context) | KV + buffers | + weights (mmap) | practical resident |
|---|---|---|---|
| 2048 | ~1.3 GB | 7.1 GB | ~8.4 GB |
| 8192 (default) | ~2.0 GB | 7.1 GB | ~9.1 GB |
| 32768 | ~2.4 GB | 7.1 GB | ~9.5 GB |

KV growth is sublinear (most of Gemma 4's layers use sliding-window
attention). 16 GB unified memory recommended on Apple silicon; 12 GB works
with `-c 4096`. Disk: 7.2 GB. Windows cannot run executables >4 GB — use a
llamafile release binary with external GGUF weights instead.

**NVIDIA/CUDA:** works out of the box — a CUDA backend (TinyBLAS, sm_75
through sm_90, only the NVIDIA driver required, no toolkit) is bundled and
extracted to `~/.llamafile` on first run. Validated with MTP on 2× RTX 4090
/ CUDA 12.8, flash attention on — the upstream CUDA+MTP crash cluster
(ggml-org/llama.cpp #24376, #24314, #24457) did not reproduce on sm_89; if
you hit it on other hardware, add `--flash-attn off` (MTP keeps most of its
gain). Measured (greedy, 400 tokens):

| config | prose tok/s | edit/copy tok/s |
|---|---|---|
| `--spec-type none` (no speculation) | 95 | 93 |
| default (MTP n=2, layers split across both GPUs) | 115 | 142 |
| `-sm none` (model fits on one GPU) | 155 | 189 |
| `-sm none --spec-draft-n-max 4` ← recommended | 165 | 239 |

When the 7 GB model fits on a single card, `-sm none` avoids the cross-GPU
pipeline hop — worth +35% by itself; on CUDA the batched verify is cheap
enough that draft length 4 beats the Metal-tuned default of 2.

## Details

- Defaults: ctx 8192, 2 parallel slots, ubatch 1024 (caps embedding input
  length), full GPU offload (`-ngl 999`; Metal on macOS — first run compiles
  the Metal module via Xcode CLT; use `-ngl 0` for CPU).
- Speculative decoding: `--spec-type draft-mtp` by default using the baked-in
  Gemma 4 MTP assistant (draft length 2, the measured optimum on M4 — longer
  drafts lose to Metal's batched-verify cost; on CUDA pass
  `--spec-draft-n-max 4`). `--spec-type` values combine: adding
  `--spec-type ngram-simple` runs ngram self-speculation alongside MTP
  (helps edit/copy-heavy output). `--spec-type none` disables speculation
  entirely; `--spec-type none --spec-type ngram-simple` replaces the default
  with pure ngram.
- Prefill checkpoints are disabled by default (`--ctx-checkpoints 0`): the
  upstream default silently runs every prompt through a second full forward
  pass to snapshot SWA KV state (~130 ms/request on M4). Restore with
  `--ctx-checkpoints 32` if your chat UI does frequent mid-history edits and
  you want cheap rollback.
- KV cache persistence is **automatic**: on graceful shutdown each
  text-only slot's state is written to a hidden `.gemma4-kv/` dir next to
  where you launched the file, and restored on the next start — a repeated
  long prompt skips straight to generation (measured: 365-token prompt,
  2.7 s cold, 87 ms after a restart). Manual named checkpoints still work
  via `POST /slots/{id}?action=save|restore` with `{"filename":"name.bin"}`.
- Includes llama.cpp patches: pooled embeddings for mixed-length batches
  under Gemma 4's sliding-window KV cache; Gemma4UV image preprocessing
  (budget-fill + F32 accumulation); Metal `ne11_mm_min` retune with a
  built-in per-op GPU profiler (`GGML_METAL_PROFILE_COMPUTE=N`).
- Embedding quality: this is a generative model, not a trained embedder —
  useful semantic signal, but a dedicated embedding model will rank better.
- Build pipeline, patches, measurements, client and tests:
  https://github.com/SEBK4C/Llamafile-gemma-4-12B-it-qat-q4_0-gguf-Inferance-And-embeddings

Weights are Google's Gemma 4 12B IT with quantization-aware training (q4_0),
redistributed unmodified under Apache 2.0 / the Gemma 4 license linked above.
