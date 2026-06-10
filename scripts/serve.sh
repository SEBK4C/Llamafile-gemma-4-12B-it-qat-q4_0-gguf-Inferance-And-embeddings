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
#   GEMMA4_VISION=1  also load the mmproj for image input (more RAM)
#   GEMMA4_SPEC      speculative decoding   (default ngram-simple: model-free
#                    self-speculation, ~+15% on edit/RAG-style outputs, neutral
#                    on freeform prose; "none" disables; on machines with >24GB
#                    pass GEMMA4_DRAFT=path/to/gemma-4-E2B_q4_0-it.gguf to use
#                    classic draft-model speculation instead)
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${ROOT}/bin/llamafile"
MODEL="${ROOT}/models/gemma-4-12b-it-qat-q4_0.gguf"
MMPROJ="${ROOT}/models/mmproj-gemma-4-12b-it-qat-q4_0.gguf"

[ -x "$BIN" ]   || { echo "error: $BIN missing — run 'make build' first" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "error: $MODEL missing — run 'make model' first" >&2; exit 1; }

VISION_ARGS=""
[ "${GEMMA4_VISION:-0}" = "1" ] && VISION_ARGS="--mmproj ${MMPROJ}"

SPEC_ARGS="--spec-type ${GEMMA4_SPEC:-ngram-simple}"
[ -n "${GEMMA4_DRAFT:-}" ] && SPEC_ARGS="--spec-type draft-simple -md ${GEMMA4_DRAFT}"

exec "$BIN" --server \
    $SPEC_ARGS \
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
