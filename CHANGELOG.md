# Changelog

All notable changes to this project. Dates are 2026.

## [v0.7.1] — 2026-07-07 — hardware autotune restored; fork source recovered

The v0.3 autotune was never on the public fork — every artifact carried it
in the binary while its source lived only in CT 118's local checkout (the
same unpushed-tree drift as voice.c). Recovered: the CT tree is pushed as
fork branch `ct-prod-vendor`; the Mac platform work is rebased onto it as
`v0.7.x-universal` (autotune + per-OS baked args + sidecar env ports + UI
tag pin, composing cleanly — baked args and user flags always beat
autotune). The universal artifact now self-tunes bare on CUDA/Metal/CPU,
fixing the v0.7.0 bare-boot OOM on 12 GB NVIDIA cards. The retired
`patches/lf-*` files now live in-tree on the pushed fork branch; the
.gitmodules pin finally references public commits. Embedding sidecar keeps
`-np 2` (it spawns with its own args; main-model autotune cannot touch it).

## [v0.7.0] — 2026-07-07 — Mac Metal parity, voice, prewarmed first message

Full v0.6.1 feature parity on Apple Silicon (api_probe 21 PASS / 0 FAIL /
1 skip, M1 Pro): per-OS baked args (`.args.xnu`, lf-0002 — fixes the ~6 tok/s
CUDA-tuned-defaults crawl; 19.9 tok/s bare), voice in/out (espeak-backed
Kokoro sidecar via LLAMAFILE_TTS_PORT; pronunciation roundtrip-verified),
embeddings/ingest on Metal (+LLAMAFILE_EMBED_PORT for the source route),
shipped system-prompt prewarm (patch 0021; first message cache_n=310), the
multimodal upload corpus + probe, and the standing FEATURE-PARITY /
RELEASE-CHECKLIST process docs. Known: mic-button visibility quirk in the
web UI; date/time audio reasoning spiral (canary clip in tests/assets).
Full notes: release-notes-v0.7.0.md.

## [v0.4.0-alpha] — 2026-07-04 — the voice moves into the file (ALPHA)

**Read-aloud is now fully baked in.** The packaged llamafile bundles a
cosmocc-built TTS.cpp Kokoro server (8 MB APE) and the self-contained
`Kokoro_no_espeak_Q4.gguf` (198 MB, phonemizer inside the GGUF), spawns it at
startup and reverse-proxies it at `/tts` on the main port. The web UI's
read-aloud controls light up on any machine from the single file — no Python,
ONNX, espeak or sidecar. `LLAMAFILE_NO_VOICE=1` opts out; file grows ~206 MB.

Also in this release (see “Voice interface (ALPHA)” in the README for usage):

- **Talk-over barge-in**: speak while the model reads — playback pauses, your
  utterance records and auto-sends on a ~1 s pause (browser energy VAD,
  “+”-menu toggle).
- **Spoken UI commands** via Gemma 4's native function-calling from audio:
  stop / speed / read-again / new-chat / regenerate ride on barge-in sends.
- Karaoke player refinements: scroll-respecting follow, per-message speed
  controls, instant-start prefetch, re-render-proof word wrapping.
- TTS.cpp cosmocc port notes upstreamable in `voice/ttscpp/README.md`
  (C++23, static-init link order, generated header).

**Alpha caveats**: English-only voices, simple energy VAD (use headphones),
small literal command vocabulary, Apple-Silicon voice path untested,
estimated (not phoneme-exact) word timing.

## [v0.3.0] — 2026-07-04 — universal artifact: hardware auto-tuning

- **Hardware auto-tuning**: the packaged file now detects its platform at
  startup and applies validated defaults (CUDA / Apple-Silicon Metal / CPU)
  for every flag the user didn't set — one file, tuned everywhere. Override
  any flag individually, or `LLAMAFILE_NO_AUTOTUNE=1` to disable. The baked
  `.args` now carry only universal flags.
- **Duplicate-launch help**: starting the file while another copy is running
  no longer dies with a cryptic bind error — it explains in plain words,
  offers the fixes, and opens the existing web UI in your browser.
- **`--clear-all`**: wipe all on-disk state (KV slot-saves, extracted
  `~/.llamafile/v/*`) and start fresh.
- **Voice UI degrades gracefully**: read-aloud controls only render when the
  Kokoro sidecar is reachable (`/tts`); the 🎙 mic (audio *input*, which IS
  baked into the model) always works. Baking the voice into the APE itself
  (TTS.cpp: Kokoro on GGML/GGUF, no ONNX runtime) is scoped as the next
  milestone.
- README: zipalign guide for baking your own settings.

## [v0.2.0] — 2026-07-03 — CUDA GPU support, voice UI, karaoke read-aloud

Everything below was developed and verified on a Proxmox host with an RTX
3080 Ti (12 GB, sm_86) serving the packaged llamafile from an unprivileged
LXC container. See `docs/PLATFORM-NOTES.md` for the cross-platform
compatibility matrix that came out of this work.

### GPU / CUDA

- **The packaged llamafile now actually ships GPU support.** `cuda.sh` builds
  the TinyBLAS `ggml-cuda.so` DSO (driver-only at runtime) and `package.sh`
  bakes it into the APE. Measured on the RTX 3080 Ti: **9 → ~200 tok/s**
  raw completion, ~90–110 tok/s on chat/reasoning workloads.
- **Ported upstream CUDA fattn fixes as patches 0016/0017**
  (`e495d1e`/#25148, `0eca4d4`/#24945): Gemma 4's 512-dim attention heads hit
  a missing fattn tile config — a crash on some architectures and **silent
  garbage for small embedding batches on sm_86, which made audio input act
  deaf**. Mandatory for any CUDA build of this repo.
- **Enabled CUDA graphs** (`-DGGML_CUDA_USE_GRAPHS`): upstream CMake defaults
  this on; the hand-rolled nvcc invocation had it compiled out. +7–10%
  decode across workloads.
- **`GGML_CUDA_ARCHS` env override** in `cuda.sh` for single-arch SASS builds
  (5× faster DSO compile for a known GPU; do NOT publish single-arch builds).
- **glibc portability**: DSOs built on new-glibc hosts (2.38+) failed to
  dlopen on Debian 12-era systems (`__isoc23_*` symbols). Fixed via a
  forwarding shim (`llamafile/glibc-compat.c`) linked into the DSO, plus
  `-std=gnu17` for core C objects — same portability class as the existing
  static libstdc++ link.

### Voice & read-aloud UI (new)

Injected into the packaged web UI (`voice/tts-inject.html`, applied to
`tools/ui/dist/index.html` before `make build`); speech is synthesized by a
CPU-only Kokoro-82M sidecar (`voice/kokoro_server.py`, OpenAI-style
`POST /v1/audio/speech`) — zero VRAM cost:

- **Karaoke read-aloud**: one play/pause button per assistant message (placed
  under the reasoning dropdown, above the answer). Every word is highlighted
  as it is spoken; **click any word to jump playback there**, forward or
  back. Reasoning and the stats footer are never spoken.
- **Reading speed controls** (« 1.0× ») — pitch-corrected, applies instantly
  mid-sentence, persisted.
- **Instant start**: the first audio chunk is prefetched when a reply
  finishes streaming — press ▶ and it plays in ~0.13 s (was 2.5 s). During
  streaming, reading starts within ~1 s using golden-ratio-growing sentence
  chunks that keep pace with generation.
- **Auto-scroll that respects you**: the view follows the spoken word only
  until you scroll away; following resumes when you return or click.
- **🎙 microphone button** next to Send: records 16 kHz WAV in the browser
  (the server's decoder does not accept webm) and attaches it through the
  UI's own file pipeline. Pulses red while recording.
- Robust against the UI's re-renders: wrapped words re-attach automatically,
  audio is cached by spoken text, single global playback engine (starting
  any playback stops the previous one).

### Web UI defaults & toggles

- Server now passes `--ui-config '{"excludeReasoningFromContext":true,
  "preEncodeConversation":true}'` (note: systemd strips double quotes —
  single-quote the JSON).
- **“💭 Reason every turn”** toggle (in the composer's “+” menu, default on):
  works around the stock UI freezing `thinkingEnabled` per conversation,
  which made the model reason only on the first message. Forces
  `chat_template_kwargs.enable_thinking:true` per request.
- **“0° Temperature zero”** toggle (same menu): deterministic output and the
  best MTP speculative speedup.

### Performance findings (documented in docs/PLATFORM-NOTES.md)

- f16 KV cache decodes ~16% faster than q8_0 (per-step dequant cost); q8_0
  is only worth it when the extra context must fit. The deployed reference
  config is f16 KV @ 131072 ctx.
- MTP speculative decoding accelerates predictable raw text ~2× (acceptance
  0.75–0.95) but breaks even on diverse chat output (acceptance ~0.3) —
  ~90–110 tok/s is the honest chat speed on a 3080 Ti (77% of the memory-
  bandwidth ceiling).
- Audio input needs healthy levels: quiet recordings (≲ −35 dB mean) read as
  “no audio” — normalize with `ffmpeg -af loudnorm` first. Identification is
  reliable at category level (songbird / heavy machinery), not species level.

### Fixed

- CPU-mode + MTP could abort with `assert(sum > 0.0)` in the draft-probe
  path (upstream crash-cluster family) — documented; workaround
  `--spec-type none` for CPU runs.
- KV slot-autosave restoring stale states across config changes (purge the
  slot-save dir after correctness-affecting changes).
- Web UI missing entirely from source builds when the `ggml-org/llama-ui`
  asset download fails (the `latest` tag 404s): pick the tag by date — this
  repo's pin (04eb4c4, 2026-06-07) pairs with UI tag `b9578`.

## [v0.1.0] — 2026-06-11 — initial public release

Dual-mode (chat + embeddings) Gemma 4 12B llamafile: MTP speculative
decoding with the assistant drafter head, image + audio input, KV
persistence, Metal-tuned defaults, packaged single-file APE. See README for
the full feature tour.
