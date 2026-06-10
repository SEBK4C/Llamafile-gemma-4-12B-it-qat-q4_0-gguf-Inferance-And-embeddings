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
| `GEMMA4_SPEC` | `ngram-simple` | speculative decoding type (`none`, `ngram-simple`, `ngram-cache`, …) |
| `GEMMA4_DRAFT` | unset | path to a draft GGUF (e.g. gemma-4-E2B) for classic draft-model speculation |

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

## Speculative decoding (and the MTP investigation)

The server runs with **`--spec-type ngram-simple`** by default: model-free
self-speculation that drafts continuations from n-gram matches over the
prompt and prior output, verified by the target model. Measured on the M4
(temperature 0, ngram-simple): freeform prose is unaffected (13.3 vs 13.4
tok/s), while outputs that echo parts of the prompt — edits, RAG answers
with quotes, code modification — run **~15-16%** faster (15.4 vs 13.3 tok/s).
`ngram-cache` accepts more drafts (+19% on edits) but costs ~8% on prose, so
it's opt-in: `GEMMA4_SPEC=ngram-cache make serve`, or `GEMMA4_SPEC=none` to
disable. Embeddings are unaffected (the smoke test verifies this).

**Why not true MTP?** Google ships an official drafter for this model
([gemma-4-12B-it-assistant](https://huggingface.co/google/gemma-4-12B-it-assistant),
423M params, "up to 3x"), and our llama.cpp vintage has a `draft-mtp` host
(`--spec-type mtp`, with `mtp-*.gguf` sibling auto-discovery and a
Qwen3.5-style split-MTP converter). They are not compatible: the Qwen-style
MTP head self-attends over its own KV cache and consumes *pre-norm* target
hidden states, while Gemma 4's `Gemma4UnifiedAssistantForCausalLM` has **no
K/V projections at all** — all 4 drafter layers cross-attend directly to the
*backbone's* K/V states (post-RoPE K, normed V from the 12B's last
non-shared sliding and full-attention layers, over the whole context), takes
*post-norm* hidden states concatenated with the target's scaled token
embeddings (2×3840→1024), and carries recurrence through a 1024→3840
`post_projection`. llama.cpp's memory model is strictly per-context — there
is no mechanism for a drafter context to read the target context's KV cache —
so a faithful port needs a new cross-context KV-sharing mechanism in core
llama.cpp plus a new arch, converter and host protocol. That's core-surgery
scale, not a patch; we documented the full analysis instead and use what
works today.

**Classic draft-model speculation** (for machines with >24 GB unified/GPU
memory): `gemma-4-E2B-it-qat-q4_0-gguf` (arch `gemma4`, same 262k vocab,
4.6 GB) works as a conventional drafter:

```sh
./scripts/fetch-model.sh --draft
GEMMA4_DRAFT=models/gemma-4-E2B_q4_0-it.gguf make serve
```

Not packaged into the single-file build: it would grow the file to ~11 GB,
and on 16 GB machines both models can't share the Metal budget (the drafter
would fall to CPU and draft slower than the target generates).

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
  when the auto-fit picks a partial layer split (typically because another
  instance is already holding GPU memory — e.g. two copies of this server),
  generation fails with `graph_compute` errors. Use full offload (`-ngl 999`,
  our default) or CPU (`-ngl 0`), and run one instance per GPU. First GPU run
  compiles the Metal module via Xcode CLT.
- **Windows** can't run executables >4 GB, so the packaged file won't work
  there — use `bin/llamafile` (or a release binary) with external weights.
- ~16 GB RAM recommended: weights are ~7 GB plus KV/compute buffers.
