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
---

# Gemma 4 12B IT (QAT q4_0) — dual-mode llamafile

A single self-contained executable that serves **chat completions and
embeddings from one model instance** — one set of weights in memory, one
port. Built on [mozilla-ai/llamafile](https://github.com/mozilla-ai/llamafile)
v0.10.3 with [google/gemma-4-12B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf)
weights baked in. Runs on macOS, Linux and BSD, on arm64 and x86_64, with no
installation.

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

CLI flags override the baked-in defaults (`--port 9000`, `-c 16384`, …).

## Details

- Defaults: ctx 8192, 2 parallel slots, ubatch 2048 (caps embedding input
  length), full GPU offload (`-ngl 999`; Metal on macOS — first run compiles
  the Metal module via Xcode CLT; use `-ngl 0` for CPU).
- Speculative decoding on by default (`--spec-type ngram-simple`, model-free
  self-speculation): ~15% faster on outputs that echo the prompt (edits,
  RAG, code changes), neutral on freeform prose. `--spec-type none` disables.
- Includes a llama.cpp patch fixing pooled embeddings for mixed-length
  batches under Gemma 4's sliding-window KV cache (without it, batch
  composition silently changes embedding values).
- Embedding quality: this is a generative model, not a trained embedder —
  useful semantic signal, but a dedicated embedding model will rank better.
- Windows cannot run executables >4 GB; use a llamafile release binary with
  external GGUF weights instead.
- Build pipeline, patch, client and tests: https://github.com/SEBK4C/Llamafile-gemma-4-12B-it-qat-q4_0-gguf-Inferance-And-embeddings

Weights are Google's Gemma 4 12B IT with quantization-aware training (q4_0),
redistributed unmodified under Apache 2.0 / the Gemma 4 license linked above.
