# Kickoff prompt: multimodal embedding quality + Metal fix

> Use this file as the opening prompt for dev sessions on this branch
> (`mm-embedding-dev`, worktree `~/Projects/mm-embedding-gemma4`). It encodes
> the state of the 2026-06-10 modality-gap work (docs/mm-embedding.md,
> tests/modality_gap.py) so you start at full speed. Three workstreams, in
> rough order of cost: prompted embeddings (cheapest, zero C++), pooling
> sensitivity (mostly zero C++), and the Metal media-embeddings segfault
> (C++/ggml debugging).

## Context you can rely on

- One llamafile server instance serves chat + embeddings + image + audio
  (`make serve`; env knobs in `scripts/serve.sh`). Media embeddings work on
  **CPU only** for now: `GEMMA4_NGL=0 make serve` (Metal segfaults — WS3).
- `/v1/embeddings` accepts media:
  `{"input": {"prompt_string": "<marker>", "multimodal_data": ["<b64 png/wav/mp3>"]}}`
  where `<marker>` comes from `GET /props` → `media_marker` and is
  **randomized per server process** (re-fetch after every restart).
  `prompt_string` may contain arbitrary text around the marker — that's the
  whole basis of WS1's instruction wrapping.
- Harness: `tests/modality_gap.py` (3 topics × {text, 224px rendered-text
  image, `say` audio}; `--make-assets` regenerates). Findings to beat
  (docs/mm-embedding.md): embeddings cluster by modality (0.55–0.94) over
  content (0.30–0.61 cross-modal same-sentence); raw cross-modal retrieval
  2/6; mean-offset gap correction only 3/6; image drift semi-parallel
  across topics (cos 0.69–0.81), audio drift weakly parallel (0.46–0.53).
- Content-extraction is NOT the bottleneck: chat OCR reads the 26px
  renderings word-perfectly and audio transcribes word-perfectly. The gap
  is in what pooling exposes, not what the backbone knows.
- Server pooling is global (`--pooling mean|last|none`, env
  `GEMMA4_POOLING`), set at startup — pooling comparisons need restarts.
- Media token counts dominate sequences: a 224px image ≈ a couple hundred
  projected tokens, audio ≈ 25 tokens/sec (640-sample frames @16 kHz). A
  one-sentence wrapper is ~10–15 tokens — mean pooling is ~95% media rows.

## WS1 — Prompted embeddings (start here: zero C++, possibly a free win)

Hypothesis: wrapping media in an instruction pulls the pooled
representation toward the text manifold, because the backbone demonstrably
extracts the content when prompted in chat.

- Templates to sweep (same wrapper on BOTH sides of the comparison — also
  embed the text items wrapped, e.g. "this text says: {text}"):
  - `"transcribe this audio: {marker}"` / `"read the text in this image: {marker}"`
  - content-position variants: marker-first vs marker-last (causal attention
    means trailing text tokens can attend to all media rows — marker-FIRST
    plus trailing instruction likely matters for mean pooling, and is
    essential for last pooling)
  - PromptEOL-style compressor: `'this image says "{marker}" which means:'`
- Metric: the modality_gap.py battery (extend it with `--template-media` /
  `--template-text` args) — report same-modality block sims, cross-modal
  same-topic sims, retrieval n/6, drift magnitude + parallelism.
- Prior art note: PromptEOL alone did NOT fix text-text anisotropy in this
  model (tested 2026-06-10, ranking still failed) — but cross-modal manifold
  pulling is a different mechanism; don't dismiss from that result.
- Scale up beyond 3 topics before claiming anything: 30–50 sentences
  synthesized the same way (assets recipe in the harness). For audio at
  scale consider [`keithito/lj_speech`](https://huggingface.co/datasets/keithito/lj_speech)
  (13k clips + transcripts; resample to 16 kHz mono).

## WS2 — Pooling sensitivity (research phase needs no C++ either)

Key trick: you do NOT need server-side span pooling to research this.
Restart with `GEMMA4_POOLING=none` and use the **legacy** `/embedding`
endpoint (`{"content": ...}` — same object format works), which returns
**per-token embedding rows** when pooling is none. Then pool arbitrary
spans offline in Python:

- mean over all rows (replicates baseline), mean over media rows only,
  mean over text rows only, last-token, last-media-token, max-pool.
- Identify the media span by token count arithmetic: embed the wrapper text
  alone via `POST /tokenize` to count its tokens; media rows are the
  contiguous remainder (BOS at 0; marker expands to the media chunk).
- Caveat: `embd_normalize` is skipped for pooling none — normalize offline.
- Watch our patch 0001 territory: with pooling none the batch-splitting
  path differs; embed ONE input per request to stay safe, and re-verify
  batch-vs-solo consistency if you batch.
- If a clear winner emerges (e.g. trailing-instruction last-pooling), THEN
  consider a server-side patch (new pooling mode or per-request pooling
  override) so it works through `/v1/embeddings` — per-request pooling
  (`"pooling": "last"` in the request body) would be the nicest interface
  and is a contained server-side change (task params → cparams toggle is
  NOT per-request today; check how llama_set_embeddings is toggled per
  batch in server-context.cpp:3082 for the precedent).

## WS3 — Fix the Metal segfault (media + pooled-embeddings batches)

Repro (100%): server with Metal (`-ngl 999`, default) + multimodal; POST
`/v1/embeddings` with any media input → process dies (no assert text)
immediately after this log line from `llama-context`:
`"embeddings required but some input tokens were not marked as outputs -> overriding"`.
Matrix of what works: chat+media on Metal OK; text embeddings on Metal OK;
media embeddings on CPU OK. Failing combo = Metal × batch.embd input
(media chunk rows) × `cparams.embeddings` × pooling graph.

Step 0 — **check upstream first**: search ggml-org/llama.cpp issues/PRs for
media/mtmd + embeddings + Metal crashes fixed after our pin `dbe9c0c`
(2026-06; note: this branch may bump the pin anyway if the MTP branch's
integration lands — coordinate with `mtp-gemma4-drafter`, whose
MTP-prompt.md plans a bump to ≥ `04eb4c44`; a bump may fix or change this
bug — re-test the repro immediately after any bump).

Debug plan:
1. Narrow by pooling: repro with `GEMMA4_POOLING=last` and `none` on Metal.
   If last/none survive, the suspect is the `inp_mean` matmul path
   (F32 [n_tokens, n_seqs] matrix × media-row states) on Metal; if all
   pooling types crash, suspect the embd-input graph
   (`build_inp_embd` with raw F32 rows) interacting with `n_outputs = all`.
2. Get a stack: the crash prints no GGML_ASSERT, so it's a real segfault —
   run the server under `lldb` (the APE binary spawns via the ape loader;
   attach to the child by PID after startup: `lldb -p $(pgrep -f
   'bin/llamafile --server')`, then trigger the request).
3. Graph diff: `ggml_backend_sched_set_eval_callback` (or
   `GGML_SCHED_DEBUG=1`-style logging) comparing the CPU and Metal graph
   node lists for the same request; the divergent/last-executed node is
   the culprit op.
4. Fix lives in `vendor/llamafile/llama.cpp/ggml/src/ggml-metal/*` — **trap**:
   those sources are zip-embedded into the binary and compiled at runtime
   into `~/.llamafile/v/0.10.3/ggml-metal.dylib`. After editing them you
   must rebuild the llamafile AND delete `~/.llamafile/v/0.10.3/` so the
   dylib recompiles, or you'll test stale kernels.
5. If the root cause is deep, an acceptable interim ship: server-side
   fallback that routes media-embedding tasks to the CPU backend (or
   refuses with a clear error pointing at `GEMMA4_NGL=0`) instead of
   crashing. A crash is the only unacceptable state.
- When fixed: extract the change with the snapshot-diff workflow into
  `patches/` (mozilla's overlay also modifies many files — always diff
  against a pre-edit copy of the file, not `git diff`), renumber after
  0008, wire into `scripts/apply-patches.sh` ordering, and re-run the full
  `tests/smoke_test.py` + `tests/modality_gap.py` on Metal.

## Success criteria

1. WS1/WS2: a documented embedding recipe (template + pooling) measurably
   better than baseline on ≥30 paired items — target: cross-modal retrieval
   well above chance and within-topic cross-modal sim > cross-topic
   same-modality sim for at least one modality. Negative results are
   results: write them into docs/mm-embedding.md with numbers.
2. WS3: media embeddings run on Metal without crash (real fix or graceful
   CPU fallback), smoke suite green, README caveat removed/updated.
3. Everything lands as commits on `mm-embedding-dev` with
   docs/mm-embedding.md updated; patches extracted per repo convention.

## Traps already paid for (don't re-pay)

- `media_marker` is per-process — re-fetch from `/props` after every restart.
- Text on images: ≥26 px on the 224px canvas, verify with an OCR prompt
  before trusting any image-embedding numbers.
- Gemma 4 thinking channel eats `max_tokens` — budget ≥256 for any chat
  verification, or read `reasoning_content`.
- One server instance per GPU (two contending → broken partial offload).
- Projector must stay on CPU (`--no-mmproj-offload`, already default).
- Build with cosmocc's make: `vendor/llamafile/.cosmocc/4.0.2/bin/make -j10
  o//llamafile/llamafile` then `cp o/llamafile/llamafile ../../bin/`.
- The worktree shares git history with the main checkout but needs its own
  `make setup` (submodule + patches); `models/` is symlinked already.
- Background shells in the harness reset cwd between parallel calls — use
  absolute paths in scripts.
