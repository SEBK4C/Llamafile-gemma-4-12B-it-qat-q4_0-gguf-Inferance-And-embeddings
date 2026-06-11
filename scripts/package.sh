#!/bin/sh
# Bake everything into a single self-contained executable:
#   dist/gemma4-server.llamafile = APE binary + 7 GB weights + default args.
#
# The result runs the dual-mode (chat + embeddings) server on any of
# macOS/Linux/BSD, arm64 or x86_64, with zero installation:
#   ./dist/gemma4-server.llamafile
# Trailing "..." in gemma4.args means CLI args are appended, so flags like
# --port 9000 still override the baked-in defaults.
#
# Note: >4 GB executables cannot run on Windows; use bin/llamafile with
# external weights there.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${ROOT}/models/gemma-4-12b-it-qat-q4_0.gguf"
MMPROJ="${ROOT}/models/mmproj-gemma-4-12b-it-qat-q4_0.gguf"
# MTP drafter (optional, +449 MB): baked in when present. Opt in at runtime:
#   ./gemma4-server.llamafile -ngl 0 --spec-type draft-mtp \
#     -md /zip/mtp-gemma-4-12b-it-qat-q4_0.gguf --fit off
# (+14% on CPU; slower than baseline on Metal — see docs/mtp-status.md)
DRAFTER="${ROOT}/models/mtp-gemma-4-12b-it-qat-q4_0.gguf"
OUT="${ROOT}/dist/gemma4-server.llamafile"

for f in "${ROOT}/bin/llamafile" "${ROOT}/bin/zipalign"; do
    [ -x "$f" ] || { echo "error: $f missing — run 'make build' first" >&2; exit 1; }
done
[ -f "$MODEL" ]  || { echo "error: $MODEL missing — run 'make model' first" >&2; exit 1; }
[ -f "$MMPROJ" ] || { echo "error: $MMPROJ missing — run 'make model' first" >&2; exit 1; }

mkdir -p "${ROOT}/dist"
cp "${ROOT}/bin/llamafile" "$OUT"

# Stage the .args under its in-zip name (zipalign stores the basename)
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cp "${ROOT}/package/gemma4.args" "${TMP}/.args"

# -j0: store aligned + uncompressed so weights mmap directly from the zip
EXTRA=""
[ -f "$DRAFTER" ] && EXTRA="$DRAFTER"
"${ROOT}/bin/zipalign" -j0 "$OUT" "$MODEL" "$MMPROJ" $EXTRA "${TMP}/.args"
chmod +x "$OUT"

echo "Built $(du -h "$OUT" | cut -f1) $(basename "$OUT")"
