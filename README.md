# Gemma 4 12B llamafile — inference *and* embeddings from one server instance

This repo bridges [mozilla-ai/llamafile](https://github.com/mozilla-ai/llamafile)
(v0.10.3, built from source with the Cosmopolitan toolchain) with
[google/gemma-4-12B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf)
and serves **chat completions and embeddings from a single model instance** —
one set of weights in memory, one KV cache, one port:

| Endpoint | What it does |
|---|---|
| `POST /v1/chat/completions` | OpenAI-style chat (Gemma 4 chat template, thinking channel in `reasoning_content`) |
| `POST /v1/embeddings` | OpenAI-style embeddings (3840-dim, mean-pooled, L2-normalized) |
| `GET /health`, `POST /tokenize`, … | usual llama-server extras |

## How the dual mode works

llama-server's `--embeddings` flag is documented as *"restrict to only support
embedding use case"*, and `/v1/embeddings` refuses to run without it. But the
scheduler calls `llama_set_embeddings(ctx, slot->need_embd())` per batch
(`llama.cpp/tools/server/server-context.cpp`), flipping the context between
logit and embedding output on the fly. So one instance started with
`--embeddings --pooling mean` serves both task types concurrently — generation
slots and embedding slots interleave in the same scheduler.

## Quickstart

```sh
make setup     # init submodules, apply patches, install cosmocc toolchain
make build     # build bin/llamafile + bin/zipalign from source (APE binaries)
make model     # download the GGUF weights (~7.1 GB, public repo)
make serve     # start the dual-mode server on 127.0.0.1:8080
make test      # smoke-test both APIs (run against the running server)
```

Or bake everything into one self-contained executable (runs on macOS/Linux/BSD,
arm64 + x86_64, no install — weights, args and server in a single file):

```sh
make package
./dist/gemma4-server.llamafile            # same dual-mode server
./dist/gemma4-server.llamafile --port 9000   # CLI args override baked-in ones
```

### Talking to it

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Why is the sky blue?"}],"max_tokens":512}'

curl http://127.0.0.1:8080/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input":["the sky is blue","der Himmel ist blau"]}'
```

Or with the zero-dependency Python client:

```python
from client.gemma4_client import Gemma4Client, cosine
c = Gemma4Client()
c.chat([{"role": "user", "content": "Why is the sky blue?"}])
v = c.embed(["the sky is blue", "der Himmel ist blau"])
cosine(v[0], v[1])
```

### Tunables (env vars for `make serve` / `scripts/serve.sh`)

| Var | Default | Meaning |
|---|---|---|
| `GEMMA4_HOST` / `GEMMA4_PORT` | `127.0.0.1` / `8080` | bind address |
| `GEMMA4_CTX` | `8192` | total context (model supports up to 262144) |
| `GEMMA4_SLOTS` | `2` | parallel slots; lets embeddings run beside an in-flight generation |
| `GEMMA4_UBATCH` | `2048` | physical batch; pooled embedding inputs can't split, so this caps embedding input length |
| `GEMMA4_POOLING` | `mean` | `mean`, `last`, `cls`, `none` |
| `GEMMA4_NGL` | `999` | GPU layers (Metal on macOS; see caveats) |
| `GEMMA4_VISION` | unset | `1` loads the mmproj for image input (more RAM) |

## Layout

```
vendor/llamafile/      mozilla-ai/llamafile @ v0.10.3 (submodule; nests llama.cpp)
patches/               our llama.cpp fixes, applied by `make setup` (see below)
scripts/               fetch-model / serve / package / apply-patches
package/gemma4.args    args baked into the single-file build
client/                zero-dependency Python client
tests/smoke_test.py    health + chat + embeddings + concurrent mixed load
models/, bin/, dist/   gitignored artifacts (weights, binaries, packaged file)
```

## The pooled-embedding bug we found (and patch)

`patches/0001-pooled-embeddings-one-seq-per-ubatch.patch` fixes a real
correctness bug in upstream llama.cpp (at pin `dbe9c0c`), found while testing
this setup:

When one request embeds **multiple texts of unequal length**, the iSWA
(sliding-window) KV cache that Gemma 4 uses splits the batch with
`split_equal()`, which can spread a sequence across several ubatches. Pooling
runs per ubatch and the last result wins, so the returned "mean" embedding of
the longest text covered **only its trailing tokens**. We reproduced it
exactly: embedding a 10-token and a 13-token text together returned, for the
longer one, `mean(tokens[10:13])` instead of `mean(tokens[0:13])` —
batch composition silently changed embedding values.

The patch makes both attention caches honor the `embd_all` flag (like the
recurrent/hybrid caches already do) by using `split_seq()`, which keeps each
sequence whole within a ubatch. `tests/smoke_test.py` carries a regression
check (batched vectors must equal solo vectors).

The patch is maintained here rather than upstreamed because llama.cpp does not
accept predominantly AI-generated contributions; if you want to report it
upstream, the analysis above plus the patch file is everything you need.

## Publishing the packaged llamafile

`dist/gemma4-server.llamafile` (6.5 GB) is too big for GitHub (100 MB file
cap; LFS free tier 2 GB), so this repo ships code only — weights and packaged
binaries are gitignored. Hugging Face model repos are the natural home for
the artifact (it's where mozilla-ai publishes their prebuilt llamafiles):

```sh
pip install -U huggingface_hub
hf auth login
hf repo create gemma-4-12b-it-qat-q4_0-llamafile --type model
hf upload <your-username>/gemma-4-12b-it-qat-q4_0-llamafile \
    dist/gemma4-server.llamafile gemma4-server.llamafile
```

Downstream users then need exactly two commands:

```sh
curl -LO https://huggingface.co/<your-username>/gemma-4-12b-it-qat-q4_0-llamafile/resolve/main/gemma4-server.llamafile
chmod +x gemma4-server.llamafile && ./gemma4-server.llamafile
```

If you publish, note the model weights are Apache 2.0 with Google's
[Gemma 4 license link](https://ai.google.dev/gemma/docs/gemma_4_license) on
the card — mirror that in your model card.

## Caveats

- **Embedding quality**: Gemma 4 12B IT is a generative model, not a
  contrastively-trained embedder. Mean-pooled embeddings are deterministic and
  carry real semantic signal (paraphrase 0.94 vs unrelated 0.86 in the smoke
  test) but won't match a dedicated embedding model for retrieval. Treat raw
  cosine values as anisotropic — compare rankings, not absolute numbers.
- **Embedding input length** is capped by `GEMMA4_UBATCH` (default 2048
  tokens) because pooled sequences can't split across physical batches.
- **Metal partial offload is broken** in llamafile 0.10.3 for this model:
  letting the auto-fit pick a partial layer split fails with
  `graph_compute` errors. Use full offload (`-ngl 999`, our default) or CPU
  (`-ngl 0`). First GPU run compiles the Metal module via Xcode CLT.
- **Windows** can't run executables >4 GB, so the packaged file won't work
  there — use `bin/llamafile` (or a release binary) with external weights.
- ~16 GB RAM recommended: weights are ~7 GB plus KV/compute buffers.
