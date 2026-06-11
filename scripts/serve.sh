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
#   GEMMA4_UBATCH    physical batch         (default 2048; pooled embedding
#                    prompts cannot split, so this caps embedding input length)
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

exec "$BIN" --server \
    $SPEC_ARGS \
    --ctx-checkpoints "${GEMMA4_CKPT:-0}" \
    --slot-save-path "$KV_DIR" \
    -m "$MODEL" \
    $VISION_ARGS \
    --embeddings \
    --pooling "${GEMMA4_POOLING:-mean}" \
    -c "${GEMMA4_CTX:-8192}" \
    -np "${GEMMA4_SLOTS:-2}" \
    -b "${GEMMA4_UBATCH:-2048}" \
    -ub "${GEMMA4_UBATCH:-2048}" \
    -ngl "${GEMMA4_NGL:-999}" \
    --host "${GEMMA4_HOST:-127.0.0.1}" \
    --port "${GEMMA4_PORT:-8080}" \
    "$@"
