# Platform notes — what breaks where

The packaged `gemma4-server.llamafile` is one file that runs on macOS/Linux/BSD,
x86-64/arm64, CPU or GPU. But the *tuned configurations* are not universal.
This page lists every cross-platform footgun we have hit, so a flag that is
right on one box doesn't silently break another.

## Configuration compatibility matrix

| Flag / feature | NVIDIA GPU (CUDA) | CPU-only (any arch) | Apple Silicon (Metal) |
|---|---|---|---|
| `-fa` (flash attention) | ✅ on (needs DSO with patches 0016/0017, see below) | ⚠️ works, but see MTP assert below | ✅ per fork README |
| `-ctk/-ctv q8_0` (quantized KV) | ✅ requires FA on | ❌ **segfaults** — use f16 | untested; keep f16 |
| `-sm none` | ✅ single-GPU optimum | ❌ **fails to load**: "invalid value for main_gpu: 0" with zero devices — use `-sm layer` | n/a |
| `-c 262144` (max context) | ✅ fits 12 GB with q8 KV + `-ub 256` | ✅ RAM permitting (slow) | ❌ 16 GB Metal cap ≈ 12.1 GB working set; use `-c 8192`, drop to 4096 on 12 GB machines |
| `--spec-draft-n-max` | 4 (CUDA batched verify is cheap) | 2 | 2 (measured M4 optimum) |
| mmproj offload (no `--no-mmproj-offload`) | ✅ vision+audio projector on GPU | n/a | ❌ Metal conv kernels assert on projector shapes — keep `--no-mmproj-offload` |
| MTP (`--spec-type draft-mtp`) | ✅ | ⚠️ **known bug**: can abort `assert(sum > 0.0)` in `ggml-cpu/ops.cpp` (fully-masked rows in the draft-probe ubatch, upstream crash-cluster family #24376/#24457). Workaround: `--spec-type none` | ✅ per fork README (1.52× with d4c192d Metal fix) |
| `--host 0.0.0.0` | container/server use | ⚠️ binds all interfaces — use `127.0.0.1` on laptops | same |

**Rule of thumb for portable invocations** (any machine, CPU fallback):
`-ngl 0 --gpu disable -sm layer -ctk f16 -ctv f16 --flash-attn off --spec-type none`

## CUDA specifics

- The bundled `ggml-cuda.so` is TinyBLAS: needs only the NVIDIA **driver** at
  runtime. Rebuild: `llamafile/cuda.sh --minimize-size --output models/ggml-cuda.so`.
- `GGML_CUDA_ARCHS=86 cuda.sh ...` builds single-arch SASS (5× faster build,
  smaller DSO) — **only for the named GPU**; a DSO built with `GGML_CUDA_ARCHS=86`
  gives *no GPU at all* on sm_89/sm_120 cards (no PTX fallback). Publish
  multi-arch (default `--minimize-size` set) unless targeting one machine.
- **Patches 0016/0017 are mandatory in any CUDA DSO build**: without them,
  Gemma 4's 512-dim heads hit a missing fattn tile config — crash on some
  architectures, *silent garbage* for small embedding batches (audio acts
  deaf) on sm_86.
- glibc: DSOs built on new-glibc hosts need the `glibc-compat.c` shim
  (`__isoc23_*` forwarding, linked automatically by `build-functions.sh`) or
  they fail to dlopen on Debian 12-era systems with `GLIBC_2.38 not found`.
- The runtime extracts the DSO to `$HOME/.llamafile/v/<version>/` — under
  systemd with `WorkingDirectory=` and no `HOME`, that may resolve to the
  working directory (e.g. `/opt/.llamafile/`). Purge the right one after DSO
  updates.
- LXC passthrough: `/dev/nvidia-uvm` must exist on the host **before** the
  container starts (`optional` bind mounts skip silently; symptom:
  `nvidia-smi` works in-container but `cuInit` returns 999).

## Build-system gotchas

- **Web UI assets**: `make setup` downloads the Svelte bundle from the
  `ggml-org/llama-ui` HF bucket into `llama.cpp/tools/ui/dist/`. The `latest`
  fallback 404s and shallow clones have no tags — pick the tag by date
  (our pin 04eb4c4 = 2026-06-07 → tag `b9578`). **If dist/ is empty the build
  silently ships without the web UI** (`/` returns JSON 404).
- The voice/controls injection (`voice/tts-inject.html`) must be re-applied to
  `dist/index.html` after any asset re-fetch, **before** `make build`.
  Verify: `python3 -c "print(open('bin/llamafile','rb').read().count(b'g4v-btn'))"`
  (assets embed as hex arrays — `strings | grep` is unreliable).
- Baked `.args` end with `...` so CLI flags override; later duplicate flags win.
  systemd unit files strip double quotes — single-quote JSON arguments
  (`--ui-config '{...}'`).

## Server/runtime gotchas

- **KV slot autosave** (`--slot-save-path`, fork patch 0011) restores states
  across restarts *and config changes*. After any correctness-affecting
  change, purge the save dir (`rm -rf .gemma4-kv*`) or stale states mask the
  fix.
- `--embeddings` forces `n_batch = n_ubatch`.
- **Audio testing**: use *speech* samples (expect near-verbatim transcription).
  Short synthetic tones sit on a perception boundary and flip verdicts with
  harmless numerical noise — they are not a valid regression signal.
- **Audio input loudness matters**: quiet recordings (mean volume ≲ −35 dB) are
  perceived as "no audio". Normalize before sending
  (`ffmpeg -af loudnorm=I=-16:TP=-1.5`). Measured identification capability
  (12B QAT-Q4, audio marked experimental upstream): speech ≈ verbatim; bird
  song → correctly "songbird" + plausible-but-wrong species (House Sparrow vs
  European Robin); 1962 F1 V8 → "heavy machine/locomotive" (right category,
  wrong machine). Category-level ID is reliable; species/engine-level is not.
- Generation speed depends on sampling and context depth: temp 0.2 ≈ 190 tok/s
  vs temp 1.0 ≈ 154 tok/s (MTP acceptance), and ~60 tok/s at 7–8 K tokens of
  chat history on the 3080 Ti. Not a bug.
- UI reasoning: the stock UI freezes `thinkingEnabled` per conversation at
  creation; the injection's 💭 toggle forces `enable_thinking: true` per
  request (and drops `thinking_budget_tokens: 0`). Server-side, reasoning is
  per-turn and works when the flag is sent (verified).
