#!/bin/sh
# Apply this repo's llama.cpp patches on top of the vendored tree
# (after vendor/llamafile's own `make setup` has applied its overlay).
# Idempotent: already-applied patches are detected and skipped.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# NNNN-*.patch apply to the nested llama.cpp tree; lf-*.patch apply to the
# llamafile repo itself (vendor/llamafile)
apply() {
    name="$(basename "$1")"
    if git apply --check "$1" 2>/dev/null; then
        git apply "$1"
        echo "applied: $name"
    elif git apply --reverse --check "$1" 2>/dev/null; then
        echo "already applied: $name"
    else
        echo "error: $name does not apply — vendored tree changed?" >&2
        exit 1
    fi
}

cd "${ROOT}/vendor/llamafile"
for p in "${ROOT}"/patches/lf-*.patch; do
    [ -e "$p" ] && apply "$p"
done

cd "${ROOT}/vendor/llamafile/llama.cpp"
for p in "${ROOT}"/patches/[0-9]*.patch; do
    [ -e "$p" ] && apply "$p"
done
