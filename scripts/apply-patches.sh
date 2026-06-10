#!/bin/sh
# Apply this repo's llama.cpp patches on top of the vendored tree
# (after vendor/llamafile's own `make setup` has applied its overlay).
# Idempotent: already-applied patches are detected and skipped.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}/vendor/llamafile/llama.cpp"

for p in "${ROOT}"/patches/*.patch; do
    name="$(basename "$p")"
    if git apply --check "$p" 2>/dev/null; then
        git apply "$p"
        echo "applied: $name"
    elif git apply --reverse --check "$p" 2>/dev/null; then
        echo "already applied: $name"
    else
        echo "error: $name does not apply — vendored llama.cpp changed?" >&2
        exit 1
    fi
done
