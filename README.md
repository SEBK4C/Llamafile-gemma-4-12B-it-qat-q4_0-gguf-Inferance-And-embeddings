# Gemma 4 12B llamafile — inference *and* embeddings from one server instance

This repo bridges [mozilla-ai/llamafile](https://github.com/mozilla-ai/llamafile)
(v0.10.3, built from source with the Cosmopolitan toolchain) with
[google/gemma-4-12B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf)
and serves **chat completions and embeddings from a single model instance** —
one set of weights in memory, one KV cache, one port:

| Endpoint | What it does |
|---|---|
| `POST /v1/chat/completions` | OpenAI-style chat (Gemma 4 chat template, thinking channel in `reasoning_content`) — accepts text, **images** (`image_url` data URIs) and **audio** (`input_audio`, wav/mp3) |
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
| `GEMMA4_MM` | `1` | image + audio input via the mmproj; `0` for text-only |
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

## Image and audio input

Multimodal input is on by default (`GEMMA4_MM=0` disables). Gemma 4's
multimodality is *encoder-free*: the 175 MB
`mmproj-gemma-4-12b-it-qat-q4_0.gguf` contains no vision/audio towers
(`block_count = 0`), just projections — images become raw 224px/16px
patches and audio becomes raw 16 kHz waveform chopped into 640-sample
(40 ms) frames, all understood by the 12B backbone itself. Preprocessing
(image decode/resize, WAV/MP3 decode, frame chunking) happens inside the
server via the mtmd API, so clients just send standard OpenAI content parts:

```jsonc
// image
{"type": "image_url", "image_url": {"url": "data:image/png;base64,...."}}
// audio (wav or mp3)
{"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}}
```

Verified on this machine: exact color/position grounding on synthetic
images, and word-perfect transcription of spoken audio (`say`-generated,
16 kHz mono WAV). ~19-28 s per image/audio turn on the M4 (the projector
runs on CPU — `--no-mmproj-offload` — because this ggml vintage's Metal
conv kernels assert on the projector's op shapes; the 12B stays on Metal).

Support required backporting upstream commit `a731805ced` (the `gemma4uv`
/ `gemma4ua` unified projector types postdate our llama.cpp pin) — carried
as `patches/0005` + `0006`. Two more patches (`0007`, `0008`) make KV
persistence coexist with multimodal: slot save is now refused only for
slots whose state actually *contains* media (which can't be serialized),
instead of whenever multimodal is merely enabled.

### Cross-modal embeddings (and the modality gap)

`/v1/embeddings` also accepts media, using the server's tokenizer object
format (fetch the per-process marker from `/props`):

```json
{"input": {"prompt_string": "<__media_...__>", "multimodal_data": ["<base64 png/wav/mp3>"]}}
```

`tests/modality_gap.py` embeds the same sentences as text, rendered onto
images, and spoken aloud (macOS `say`). What we measured (3 topics): the
embedding space is dominated by **modality, not content** — same-modality
pairs score 0.55–0.94 while cross-modal pairs of the *same sentence* sit at
0.3–0.6, and raw cross-modal retrieval is barely above chance. The
modality drift is a large (|d| ≈ 1.0–1.15 on unit vectors), *partially*
consistent offset: image-drift directions agree at cos 0.69–0.81 across
topics, audio-drift only 0.46–0.53, and image vs audio drifts point in
different directions (cos 0.48). Subtracting the mean offset (the classic
modality-gap correction) helps only marginally (2/6 → 3/6 retrieval) —
unlike contrastively-trained encoders, this generative model's modality
gap is not a clean parallel translation, so cross-modal retrieval needs a
trained alignment, not a geometric fix. Caveat: render text ≥26px on
224px images — the vision input is 224²/16px patches and smaller text is
only partially legible to the model (verify with an OCR prompt first).

**Metal caveat**: media inputs on the *embeddings* endpoint currently
segfault the Metal backend (chat with media is fine; text embeddings are
fine). Run `GEMMA4_NGL=0 make serve` for cross-modal embedding work until
this is fixed.

## KV cache persistence (hidden dir next to the llamafile)

The server's in-RAM prompt cache dies with the process. We additionally
enable **on-disk KV state**: `scripts/serve.sh` runs with
`--slot-save-path .kvcache/` (repo root), and the packaged llamafile uses a
hidden `.gemma4-kv/` directory created next to wherever you launch it. A
slot's processed-prompt state (tokens + KV cache) can be saved, then
restored after a full server restart:

```sh
# after sending a request whose long prefix you want to keep (slot 0 or 1):
curl -X POST 'http://127.0.0.1:8080/slots/1?action=save' \
  -H 'Content-Type: application/json' -d '{"filename":"my-system-prompt.bin"}'

# ... restart the server, then:
curl -X POST 'http://127.0.0.1:8080/slots/1?action=restore' \
  -H 'Content-Type: application/json' -d '{"filename":"my-system-prompt.bin"}'
```

The Python client wraps these as `save_slot` / `restore_slot` / `erase_slot`.
Measured effect: an 870-token system prompt that took 6.2 s to process cold
is reused from the restored state with `prompt_n=1` in 0.14 s (~45× faster
time-to-first-token; the state file was 325 MB, restore took 91 ms).
State files scale with cached tokens (~0.3 MB/token at ctx 8192 defaults).

Making this work surfaced three more upstream/overlay bugs, fixed in
`patches/` (applied by `make setup`):

- `lf-0001`: the 0.10.x overlay routes every `llama_file` open through
  `llamafile_open_gguf`, which magic-checked all files — `"wb"` saves
  failed after creating an empty file, and restores of non-GGUF state
  files fell into the `.zip` branch and failed.
- `0003`: the COSMOCC `llama_file::write_raw` wrote through an
  uninitialized `FILE*` instead of the overlay's file handle.
- `0004`: for sliding-window models the server refused checkpoint-less
  partial reuse even when `seq_pos_min == 0` proves nothing was evicted —
  so restored states were always fully re-processed. (Checkpoints aren't
  serialized into state files, so this gate made restore useless for SWA
  models like Gemma 4.) Verified token-for-token identical greedy output
  with and without reuse, plus a regression check in the smoke test.
- `0002`: `--slot-save-path` now auto-creates the directory, so it can be
  baked into the packaged llamafile's defaults.

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
