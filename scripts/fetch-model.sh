#!/bin/sh
# Download gemma-4-12b-it-qat-q4_0 GGUF weights from Hugging Face.
# The repo is public (Apache 2.0); set HF_TOKEN to raise rate limits if needed.
set -eu

REPO="google/gemma-4-12B-it-qat-q4_0-gguf"
BASE="https://huggingface.co/${REPO}/resolve/main"
DEST="$(dirname "$0")/../models"
mkdir -p "$DEST"

AUTH=""
[ -n "${HF_TOKEN:-}" ] && AUTH="--header Authorization: Bearer ${HF_TOKEN}"

fetch() {
    # -C - resumes partial downloads, so re-running is safe
    echo ">> $1"
    curl -L --fail --retry 3 -C - ${AUTH} -o "${DEST}/$1" "${BASE}/$1" \
        || { [ $? -eq 22 ] && echo "already complete"; }
}

fetch gemma-4-12b-it-qat-q4_0.gguf          # 6.98 GB — text model
fetch mmproj-gemma-4-12b-it-qat-q4_0.gguf   # 175 MB  — vision projector (optional)

# --draft: also fetch gemma-4-E2B (4.6 GB) for classic draft-model speculation.
# Only worth it on machines with >24 GB unified/GPU memory; see README.
if [ "${1:-}" = "--draft" ]; then
    BASE="https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/resolve/main"
    fetch gemma-4-E2B_q4_0-it.gguf
fi

echo "Done. Weights in ${DEST}/"
