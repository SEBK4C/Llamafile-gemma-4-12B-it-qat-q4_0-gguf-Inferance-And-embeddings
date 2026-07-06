# CUDA ↔ Metal feature parity — the cross-agent handoff checklist

**Why this exists.** On 2026-07-06 two agents shipped in parallel: the CUDA
agent released v0.6.0/v0.6.1 on `main` (embeddings + ingest) while the Mac
agent built platform profiles and voice on a branch. Result: a patch-number
collision (two different `lf-0002-*.patch`), a patch that referenced a file
never pushed to the fork (`voice.c`), and features shipping CUDA-only with
"Apple-Silicon path untested" in the notes. This document is the contract
that prevents that: **a feature is DONE when both columns are green, and a
release is legal only when every row passes on the artifact being shipped**
(see `docs/RELEASE-CHECKLIST.md` for the release gates themselves).

## How to use this (agent handoff protocol)

1. **Before starting work**: read this file + `git log --oneline -15` on
   `main` AND any active platform branch. Your platform's column is your
   responsibility; the other column is your *interface*.
2. **Patch conventions** (the lf-0002 collision rule): patch numbers are
   allocated by NEXT-FREE across ALL branches — check `patches/` on `main`
   *and* active branches before numbering. `NNNN-*.patch` → nested
   `llama.cpp`; `lf-NNNN-*.patch` → the llamafile fork layer. Every patch
   must apply to the PUBLIC fork tip (`SEBK4C/llamafile@mtp-gemma4-drafter`)
   — if your local fork tree has extra files, either push them or inline
   them in the patch (a diff against a file the other agent doesn't have is
   a build break on their machine).
3. **Platform-conditional behavior** goes in per-OS args profiles
   (`package/gemma4.args` = CUDA/default, `package/gemma4.args.xnu` = Metal;
   selected at runtime by `patches/lf-0002-platform-conditional-args.patch`)
   or env-gated code (`LLAMAFILE_TTS_PORT`, `LLAMAFILE_EMBED_PORT`) — never
   in hardcoded defaults that are right on one backend and poison the other.
4. **On finishing a feature**: fill in YOUR column with the verify command
   output, mark the other column ⏳, and note what the other agent must run.
   Do not mark ✅ for a platform you did not run on.
5. **Sidecar pattern parity**: every baked sidecar (`/zip/<payload>` +
   respawn supervisor) must also honor an env-port override so the
   source-tree `make serve` route reaches parity without the 7 GB package
   (voice: `LLAMAFILE_TTS_PORT`; embeddings: `LLAMAFILE_EMBED_PORT`).

## Parity matrix — v0.6.1 feature surface

Legend: ✅ verified (with number/date) · ⏳ expected-works, unverified ·
❌ broken/missing · ➖ not applicable.

| # | Feature | Verify with | CUDA (CT 118, 3080 Ti) | Metal (M1 Pro 32 GB) |
|---|---|---|---|---|
| 1 | Chat completions + thinking channel | `bench/api_probe.py` | ✅ v0.6.0 e2e 19 PASS, 105.5 tok/s | ✅ 21.5–22.2 tok/s clean-state (07-06) |
| 2 | `/v1/embeddings` → 1024-dim sidecar vectors (v0.6.1) | api_probe / `docs/embeddings.md` recipe | ✅ 18 PASS probe suite | ✅ dims=1024, semantic sanity cos(cat,kitten) 0.861 > 0.548 (07-06) |
| 3 | `/embed/v1/*` proxy (health/tokenize/embeddings) | `curl /embed/health` | ✅ | ✅ api_probe embed_baked PASS (07-06) |
| 4 | Baked embed sidecar (self-respawn, `/zip/embed-model.gguf`) | packaged run, `embed:` log line | ✅ | ✅ isolated verify: APE re-exec child binds 8081, 1024-dim, reaped on parent death (07-06) |
| 5 | External embed sidecar (`LLAMAFILE_EMBED_PORT`) | `make serve` route | ➖ (CT uses systemd unit) | ✅ serve.sh auto-spawn verified (07-06) |
| 6 | `/v1/ingest` (enrichment + fidelity gate + chunks + vectors) | `INGEST_GUIDE.md` sample | ✅ ingest 1.97 s GPU; CORD recall 100% | ✅ enrich_ok, fidelity 5/5, 1024-dim, 15.4 s (07-06) |
| 7 | MTP speculative decoding | startup log `draft-mtp`, acceptance line | ✅ (n_max 4 optimum) | ✅ loads, acceptance 0.86–0.91 — but **break-even** on M1 Pro (batch-2 verify cost); n_max 2 |
| 8 | KV persistence (autosave + slot save/restore API) | smoke_test [5/5] | ✅ | ✅ 219/219 tokens reused (07-06) |
| 9 | Image input (mmproj) | smoke/mac-full-test | ✅ GPU projector | ✅ CPU projector (`--no-mmproj-offload` REQUIRED — Metal conv asserts) |
| 10 | Audio input (native STT) | `tests/tts_roundtrip.py` STT leg | ✅ WER 3.0% LibriSpeech | ✅ word-perfect on say-sample (07-06) |
| 11 | Web UI (b9578) + baked ui-config (Constitution + E15 decline + sampler) | `/props` ui_settings; headless Chrome | ✅ | ✅ rendered + screenshot-verified (07-06); browser-cache gotcha documented |
| 12 | Voice out: karaoke read-aloud + injected controls | UI click / `/tts/health` via proxy | ✅ kokoro-onnx sidecar (espeak G2P) | ✅ TTS.cpp sidecar RTF 0.48 (`-nt 4`, pre-warmed) |
| 13 | TTS pronunciation quality | `tests/tts_roundtrip.py` | ✅ espeak-ng G2P (kokoro-onnx) | ✅ espeak build + `Kokoro_espeak_Q4.gguf` default, roundtrip 5/5×2 (07-06); no_espeak garbles ("specificus") — fallback only |
| 14 | Voice in: mic button (records WAV → audio input) | UI; DOM check | ✅ | ✅ works; KNOWN BUG: button can hide until an audio file is attached (DOM-anchor quirk, v0.7.0 notes) |
| 15 | Barge-in + spoken UI commands (v0.4-alpha) | manual demo | ✅ (alpha, CT) | ⏳ browser-side code ships in injection; needs manual Mac demo pass |
| 16 | Baked voice APE (`/zip/tts-server.ape` spawn) | packaged bare run | ✅ artifact only — **spawner source never pushed** | ❌ blocked on missing `voice.c`; Mac equivalent = external sidecar (row 12); cosmocc TTS.cpp port = open work (`voice/BAKED-VOICE.md`) |
| 17 | Platform default args (one artifact, per-OS profiles) | bare packaged run, startup config lines | ✅ unchanged `.args` path | ✅ `.args.xnu` verified 19.9 tok/s bare (07-06) |
| 18 | Hardware autotune (v0.3, `LLAMAFILE_NO_AUTOTUNE`) | bare run on odd hardware | ✅ artifact | ❓ source location unknown to Mac agent — superseded on Mac by #17; **CUDA agent: confirm where v0.3 autotune lives** |
| 19 | CUDA DSO baked (TinyBLAS + 0016/0017) | packaged run on NVIDIA | ✅ | ➖ (not baked in Mac-built package — build `ggml-cuda.so` before publishing a universal artifact) |
| 20 | Duplicate-launch helper + `--clear-all` (v0.3) | launch twice / flag | ✅ artifact | ❓ same as #18 — source location unconfirmed |
| 21 | External ingest worker (OCR/PDF/audio → Python) | `bench/ingest/ingest_worker.py` | ✅ CT deployment | ➖ CT-only by design until OCR runtime is APE-portable |
| 22 | Serving-defaults ratchet (autoresearch ledger + gates) | `bench/serve_bench.py` / `mac_serve_bench.py` | ✅ baseline + E1–E15 | ✅ Mac baseline landed 07-06 (66.9 composite); candidate queued |
| 23 | Shipped system-prompt prewarm (baked slot-0 KV state) | clean-dir bare run: `extracted baked prewarm state` + first msg `cache_n≈310` | ➖ not shipped (CUDA agent: adopt via `scripts/make-prewarm-state.sh` + a `.prewarm-linux` profile if desired) | ✅ clean-room verified: autorestored 316 tok, first msg cache_n=310/prompt_n=11 (07-06) |
| 24 | Multimodal upload corpus + probe (`tests/assets/`, `tests/upload_ingest_probe.py`) | `python3 tests/upload_ingest_probe.py` | ⏳ run on CT | ✅ images 3/3, ingest 2/2, retrieval 2/2, audio 2/2 + 1 known-spiral canary (07-06) |

## Latest full-probe results

| date | platform | api_probe | notes |
|---|---|---|---|
| 2026-07-06 | CUDA (CT 118) | 19 PASS (v0.6.0 release run) | 105.5 tok/s; ingest 1.97 s |
| 2026-07-06 | Metal (M1 Pro) | **21 PASS / 0 FAIL / 1 skip** | 91.2 s wall; ingest 15.4 s; skip = optional `--embed-base` arg |

The one FAIL on the first Mac run (chat_completions, empty content) was a real
parity gap — serve.sh lacked the baked sampler defaults — fixed same day.

## Standing verification commands

```sh
# the one-command parity probe (19 tests, every endpoint/modality):
python3 bench/api_probe.py --base http://127.0.0.1:8080

# Mac E2E suite (13 tests incl. Metal-specific gates):
./scripts/mac-full-test.sh --start-server

# pronunciation regression (TTS→STT roundtrip, fully local):
python3 tests/tts_roundtrip.py

# serving-quality gates (candidates vs baseline):
python3 bench/mac_serve_bench.py --candidate bench/candidates/<c>.json   # Mac
python3 bench/serve_bench.py --candidate bench/candidates/<c>.json      # CT
```

## Known cross-platform footguns (short list — details in PLATFORM-NOTES.md)

- `-ub 2048` OOMs Metal command buffers → `.args.xnu` caps 1024.
- mmproj must stay CPU on Metal; GPU projector is CUDA-only.
- q8_0 KV: CUDA-only (and slower); f16 on Metal.
- MTP gains don't transfer M4→M1 Pro (break-even) — never gate Mac speed on it.
- TTS: `--use-metal` crashes; threads must be pinned (`-nt 4` on M1 Pro).
- temp=0 + small budgets ⇒ empty `content` (thinking channel) — test with the
  official sampler and ≥512-token budgets on BOTH platforms.
- Measure decode clean-state only; post-battery numbers read ~20% low.
- Date/time numerals in audio transcription trigger a reasoning-channel
  spiral (multilingual re-reading loop, empty content even at 4 k budget) —
  canary clip: `tests/assets/audio/meeting-date.wav`. Target of the queued
  reasoning-budget ratchet candidate; note the per-request
  `reasoning_budget_*` fields clip the sampler correctly but the forced
  end-tag currently trips Gemma 4's channel parser (500) — tag format needs
  iteration before the candidate runs.
