#!/bin/sh
# Serve Gemma 4 12B chat completions AND embeddings from ONE llamafile instance.
#
# How the dual mode works: llama-server normally refuses /v1/embeddings unless
# started with --embeddings, and that flag is documented as "embedding-only".
# But the scheduler calls llama_set_embeddings(ctx, slot->need_embd()) per
# batch, so a server started with --embeddings --pooling mean happily serves
# both task types from the same model weights, KV cache and context — pooled
# hidden-state embeddings for /v1/embeddings, logits for /v1/chat/completions.
#
# Tunables (env):
#   GEMMA4_HOST      bind address           (default 127.0.0.1)
#   GEMMA4_PORT      port                   (default 8080)
#   GEMMA4_CTX       total context size     (default 8192; model max is 262144)
#   GEMMA4_SLOTS     parallel slots         (default 2 — lets an embedding
#                    request run alongside an in-flight generation)
#   GEMMA4_UBATCH    physical batch         (default 2048 on Linux/CUDA, 1024
#                    on macOS; pooled embedding prompts cannot split, so this
#                    caps embedding input length. On Apple Silicon Metal a
#                    2048-wide ubatch exhausts command-buffer memory —
#                    kIOGPUCommandBufferCallbackErrorOutOfMemory on M1 Pro
#                    32 GB — while 1024 is verified stable.)
#   GEMMA4_NGL       GPU layers             (default 999 = all, Metal on macOS)
#   GEMMA4_MM=0      disable multimodal (image + audio) input; on by default.
#                    The projector runs on CPU (--no-mmproj-offload): it is a
#                    thin encoder-free projection, and the Metal conv kernels
#                    of this ggml vintage assert on its op shapes.
#   GEMMA4_SPEC      speculative decoding. Default: "draft-mtp" when the MTP
#                    drafter (models/mtp-*.gguf) is present — true MTP via
#                    Google's 423M assistant drafter, measured 1.6x baseline
#                    on M4 Metal (21.5 vs 13.1 tok/s prose, 25 tok/s on edit
#                    tasks) and +14% in CPU mode; see docs/mtp-status.md.
#                    Falls back to "ngram-simple" (model-free self-speculation,
#                    ~+15% on edit/RAG-style outputs) without the drafter.
#                    "none" disables; on machines with >24GB pass
#                    GEMMA4_DRAFT=path/to/gemma-4-E2B_q4_0-it.gguf to use
#                    classic draft-model speculation instead.
#   GEMMA4_SPEC_NMAX draft-mtp draft length (default 2 — measured optimum on
#                    M4 Metal; longer drafts lose to the batched-verify cost
#                    until the Metal matmul kernels are fixed)
#   GEMMA4_CKPT      context checkpoints per slot (default 0). The upstream
#                    default (32) splits every prefill into two full forward
#                    passes to snapshot SWA KV state — that costs ~130 ms per
#                    request on M4 Metal (b=16 prefill: 472 -> 267 ms with 0).
#                    Set GEMMA4_CKPT=32 to restore cheap mid-history rollback
#                    for chat UIs with frequent edits/regenerates.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${ROOT}/bin/llamafile"
MODEL="${ROOT}/models/gemma-4-12b-it-qat-q4_0.gguf"
MMPROJ="${ROOT}/models/mmproj-gemma-4-12b-it-qat-q4_0.gguf"

[ -x "$BIN" ]   || { echo "error: $BIN missing — run 'make build' first" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "error: $MODEL missing — run 'make model' first" >&2; exit 1; }

VISION_ARGS=""
[ "${GEMMA4_MM:-1}" = "1" ] && [ -f "$MMPROJ" ] && VISION_ARGS="--mmproj ${MMPROJ} --no-mmproj-offload"

DRAFTER="${ROOT}/models/mtp-gemma-4-12b-it-qat-q4_0.gguf"
SPEC_DEFAULT="ngram-simple"
[ -f "$DRAFTER" ] && SPEC_DEFAULT="draft-mtp"
SPEC="${GEMMA4_SPEC:-$SPEC_DEFAULT}"
SPEC_ARGS="--spec-type ${SPEC}"
if [ "$SPEC" = "draft-mtp" ]; then
    [ -f "$DRAFTER" ] || { echo "error: $DRAFTER missing — see docs/mtp-status.md" >&2; exit 1; }
    # --fit off: the fit probe cannot construct the MTP context (upstream quirk)
    SPEC_ARGS="--spec-type draft-mtp -md ${DRAFTER} --spec-draft-n-max ${GEMMA4_SPEC_NMAX:-2} --fit off"
fi
[ -n "${GEMMA4_DRAFT:-}" ] && SPEC_ARGS="--spec-type draft-simple -md ${GEMMA4_DRAFT}"

# Hidden on-disk KV store: POST /slots/{id}?action=save|restore persists a
# slot's KV cache here, surviving restarts (GEMMA4_KV_DIR overrides).
KV_DIR="${GEMMA4_KV_DIR:-${ROOT}/.kvcache}"

# Metal command buffers OOM above ~1024-wide ubatches (M1 Pro 32 GB measured).
UBATCH_DEFAULT=2048
[ "$(uname -s)" = "Darwin" ] && UBATCH_DEFAULT=1024
UBATCH="${GEMMA4_UBATCH:-$UBATCH_DEFAULT}"

# WebUI defaults (system prompt + sampler) — same file the packaged
# llamafile bakes; without it `make serve` gives a bare UI.
UI_CONFIG="${ROOT}/package/ui-config.json"
UI_ARGS=""
[ -f "$UI_CONFIG" ] && UI_ARGS="--ui-config-file $UI_CONFIG"

# Voice replies (read-aloud): spawn the TTS.cpp Kokoro sidecar when built
# (bin/tts-server + models/Kokoro_no_espeak_Q4.gguf, see voice/BAKED-VOICE.md)
# and point the server's /tts reverse proxy at it via LLAMAFILE_TTS_PORT
# (patches/lf-0003). The web UI probes /tts/health and shows the read-aloud
# controls only when it answers. GEMMA4_TTS=0 disables; GEMMA4_TTS_PORT
# overrides the port. The sidecar is CPU-only — it never contends with the
# LLM for the GPU. It outlives this script's exec as an orphan; use
# voice/voice-watchdog.sh for supervised deployments.
TTS_BIN="${ROOT}/bin/tts-server"
TTS_MODEL="${ROOT}/models/Kokoro_no_espeak_Q4.gguf"
if [ "${GEMMA4_TTS:-1}" = "1" ] && [ -x "$TTS_BIN" ] && [ -f "$TTS_MODEL" ]; then
    TTS_PORT="${GEMMA4_TTS_PORT:-8091}"
    if ! curl -s -o /dev/null "http://127.0.0.1:${TTS_PORT}/health" 2>/dev/null; then
        "$TTS_BIN" -mp "$TTS_MODEL" --port "$TTS_PORT" >/dev/null 2>&1 &
        echo "voice: started tts-server on 127.0.0.1:${TTS_PORT}" >&2
    fi
    LLAMAFILE_TTS_PORT="$TTS_PORT"
    export LLAMAFILE_TTS_PORT
fi

exec "$BIN" --server \
    $SPEC_ARGS \
    $UI_ARGS \
    --ctx-checkpoints "${GEMMA4_CKPT:-0}" \
    --slot-save-path "$KV_DIR" \
    -m "$MODEL" \
    $VISION_ARGS \
    --embeddings \
    --pooling "${GEMMA4_POOLING:-mean}" \
    -c "${GEMMA4_CTX:-8192}" \
    -np "${GEMMA4_SLOTS:-2}" \
    -b "$UBATCH" \
    -ub "$UBATCH" \
    -ngl "${GEMMA4_NGL:-999}" \
    --host "${GEMMA4_HOST:-127.0.0.1}" \
    --port "${GEMMA4_PORT:-8080}" \
    "$@"
