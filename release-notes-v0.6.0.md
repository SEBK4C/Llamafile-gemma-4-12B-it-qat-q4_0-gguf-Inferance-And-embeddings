# v0.6.0 — embeddings and ingest, baked in

One file now serves **chat + retrieval-grade embeddings + document ingest**.

## New

- **Baked embedding sidecar** (`patches/lf-0002`, `0020`): the APE carries
  Qwen3-Embedding-0.6B Q8 (Apache-2.0, 1024-dim) and re-spawns itself as a
  supervised CPU embedding server, reverse-proxied at `/embed/v1/embeddings`,
  `/embed/health`, `/embed/tokenize`. Requires the fork's pooling-last fix
  (`patches/0019`). Opt-out `LLAMAFILE_NO_EMBED=1`. The supervisor is tied
  to the main server's lifetime — it self-reaps even after SIGKILL
  (validated by test).
- **`POST /v1/ingest`** (alias `/ingest`): text → one grammar-constrained
  enrichment call (title/summary/entities/task_domain/chunking hints,
  JSON valid by construction) → **deterministic fidelity gate** (entities
  must be grounded in the source: substring, strict date-tuple, strict
  digit-string, token-subset; ungrounded entities dropped + flagged) →
  token-budgeted chunking (~512/chunk via the sidecar tokenizer) →
  1024-dim doc + chunk embeddings → an `ingest.v1` envelope for hybrid
  BM25 + vector indexing.
- **`bench/api_probe.py`** grew `embed_baked` + `ingest` tests (17 PASS /
  1 expected-FAIL / 1 SKIP on this build; the expected FAIL is the main
  model's own `/v1/embeddings` anisotropy, documented since v0.4).

## Measured quality behind the pipeline (bench/, HF dataset)

- Enrichment: 6/6 schema-valid first-try, prompt-injection probe resisted.
- CORD-v2 labeled receipts: key-field recall 100% end-to-end, fidelity-gate
  false-drop 0%, hallucinated-number rate 6.7% (flag-only).
- Flickr30k people photos: retrieval by caption through enrichment text,
  hit@1 0.857 / MRR 0.917.
- Native STT: WER 0.030 on LibriSpeech test-clean (16 kHz mono; the audio
  encoder is speech-only — non-speech description needs a tagger sidecar).

## Known issues / scope

- File/PDF/audio ingest with OCR runs in the companion Python worker
  (`bench/ingest/ingest_worker.py`) until the OCR runtime is APE-portable
  (program I12). The in-APE gate covers substring/digits/date-tuples/
  token-subset; month-name date normalization is python-only.
- ~~Voice supervisor SIGKILL leak~~ — fixed before release: both the voice
  watchdog and the embed supervisor are now tied to the main server's
  lifetime and self-reap after any kill, including SIGKILL (validated:
  0 orphaned listeners/processes 26 s after `kill -9`).
- Windows: >4 GB APE limit unchanged — use `bin/llamafile` + external
  weights.

## CUDA e2e (RTX 3080 Ti, full offload — release gate PASSED 2026-07-06)

Healthy in 12 s, 11 GB VRAM (prod-equivalent footprint). Full api_probe:
**19 PASS** / 1 expected-FAIL (the 12B's own `/v1/embeddings` anisotropy,
documented) / 1 legacy-skip, in 15.7 s. Chat **105.5 tok/s** (prod band).
`/v1/ingest` 1.97 s end-to-end on GPU with fidelity 5/5; `/embed/v1/*`,
vision, TTS (1.79× realtime) all green; audio-in transcribes LibriSpeech
verbatim with `enable_thinking:false` (the probe's audio "FAIL" is its
thinking-budget artifact, documented as H8/F15). Production pause for the
test window: 2 min 07 s, restored and verified.
