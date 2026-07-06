# v0.7.0 — Mac Metal: full feature parity, voice, and a prewarmed first message

The previously CUDA-only stack now runs the complete v0.6.1 feature surface
on Apple Silicon, verified with the same 19-test probe the CUDA release used:
**21 PASS / 0 FAIL / 1 skip** on an M1 Pro 32 GB (`bench/api_probe.py`).

## One artifact, per-platform defaults

`gemma4-server-metal.llamafile` (HF) boots with the **Metal profile**
(`.args.xnu`, selected at runtime by `patches/lf-0002-platform-conditional-args.patch`):
MTP drafter, `-fa on`, `-c 8192 -np 2`, `-ub 1024`, CPU-side mmproj, f16 KV,
the v0.5.0 sampler + WebUI defaults. On every other OS it reads the standard
`.args` — behavior byte-identical to before. This fixes the ~6 tok/s crawl the
CUDA-tuned defaults produced on Macs: **19.9 tok/s bare-launch steady state**
(21.5–22.2 via `make serve` with external GGUFs).

**v0.7.0-universal is one file for every backend**: the CUDA TinyBLAS DSO
from the v0.6.1 build (byte-identical blob, which passed that release's
CUDA e2e) is baked alongside the Metal profile, prewarm state and embed
payload. One `gemma4-server.llamafile` on HF serves NVIDIA, Apple Silicon
and CPU. Handoff item (`docs/FEATURE-PARITY.md` row 19): one CUDA smoke
run of the universal file to confirm the transplanted DSO against this
build's binary.

## Voice on Mac, both directions

- **Speech in**: model-native audio through the CPU mmproj — word-perfect on
  clean speech; the 🎙 mic button records straight into it.
- **Read-aloud**: TTS.cpp Kokoro sidecar (CPU-only, RTF ≈ 0.5, pre-warmed,
  `-nt 4`) proxied at `/tts` via `LLAMAFILE_TTS_PORT` — the karaoke controls
  light up automatically. `make serve` spawns everything.
- **Pronunciation fixed**: the `no_espeak` GGUF's built-in G2P garbles real
  words ("specific" → "specificus"); default is now the espeak-ng build +
  `Kokoro_espeak_Q4.gguf` (`brew install espeak-ng`), roundtrip-verified by
  the new `tests/tts_roundtrip.py` (TTS → the model's own ears → compare).

## Prewarmed system prompt (Metal artifact)

The packaged file bakes a pre-computed slot state holding the WebUI system
prompt (patch 0021 + `scripts/make-prewarm-state.sh`). Verified on a clean
machine: the **very first message** reuses 310 cached tokens
(`cache_n=310 / prompt_n=11`) — no Constitution-prompt prefill tax, ever.

## Embeddings + ingest on Metal (v0.6.1 parity)

1024-dim `/v1/embeddings` (Qwen3 sidecar, self-respawn verified on APE
re-exec), `/embed/v1/*`, `/v1/ingest` (fidelity gate 5/5), and
`LLAMAFILE_EMBED_PORT` so the source-tree route reaches parity without the
packaged file. `serve.sh` runs the full stack: LLM + TTS + embeddings.

## Testing & process (new standing infrastructure)

- `tests/assets/` + `tests/upload_ingest_probe.py`: multimodal upload corpus
  (speech/images/PDFs with ground truth) — Mac run 9 PASS / 0 FAIL.
- `docs/FEATURE-PARITY.md`: the 24-row CUDA↔Metal matrix + cross-agent
  handoff protocol. `docs/RELEASE-CHECKLIST.md`: per-platform release gates.
- Mac serving-quality baseline in the autoresearch ledger (66.9 composite).

## Known issues

- **Mic button visibility (web UI)**: the 🎙 button can disappear from the
  composer unless an audio file is already attached, after which it appears
  again. The mic-replaces-Send interaction works; feature is usable. Injection
  DOM-anchoring bug — tracked for the next UI pass.
- Date/time numerals in audio transcription can spiral the reasoning channel
  into empty answers (`tests/assets/audio/meeting-date.wav` is the canary).
  Target of the queued reasoning-budget defaults candidate.
- Barge-in / spoken commands (v0.4-alpha) not yet manually verified on Mac.
- TTS `--use-metal` crashes this ggml vintage — sidecar stays CPU (by design
  it never contends with the LLM for the GPU).

## Verification (this release)

macOS/Metal column: `bench/api_probe.py` 21/0, `scripts/mac-full-test.sh`
13/13, upload corpus 9/0, pronunciation roundtrip 5/5×2, clean-room prewarm +
baked-embed respawn checks — all on M1 Pro 32 GB, 2026-07-06/07. CUDA column
unchanged from v0.6.1 (its artifact is untouched).
