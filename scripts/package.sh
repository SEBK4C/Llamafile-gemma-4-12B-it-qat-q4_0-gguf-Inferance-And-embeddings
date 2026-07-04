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
# MTP drafter (+449 MB): baked in and ENABLED BY DEFAULT via
# package/gemma4.args (--spec-type draft-mtp, n_max 2, --fit off):
# 1.52x baseline on Metal (needs the fork d4c192d Metal fix), +14% on CPU
# — see docs/mtp-status.md. Opt out at runtime:
#   ./gemma4-server.llamafile --spec-type none
DRAFTER="${ROOT}/models/mtp-gemma-4-12b-it-qat-q4_0.gguf"
# NVIDIA CUDA backend DSO: baked in when present so the packaged file works
# on NVIDIA GPUs out of the box (extracted to ~/.llamafile on first run; no
# CUDA toolkit needed — TinyBLAS build, only libcuda driver required).
# Build it with: vendor/llamafile/llamafile/cuda.sh --minimize-size \
#   --output models/ggml-cuda.so
CUDA_DSO="${ROOT}/models/ggml-cuda.so"
# Baked voice (optional): TTS.cpp Kokoro server APE + self-contained GGUF.
# When present the packaged server spawns it and proxies /tts, giving the
# web UI read-aloud with no sidecar. See voice/BAKED-VOICE.md.
VOICE_APE="${ROOT}/models/tts-server.ape"
VOICE_GGUF="${ROOT}/models/kokoro.gguf"
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
[ -f "$DRAFTER" ] || { echo "error: $DRAFTER missing — default args enable draft-mtp" >&2; exit 1; }
EXTRA="$DRAFTER"
if [ -f "$CUDA_DSO" ]; then
    EXTRA="$EXTRA $CUDA_DSO"
else
    echo "note: $CUDA_DSO missing — packaging without bundled NVIDIA GPU support"
fi
if [ -f "$VOICE_APE" ] && [ -f "$VOICE_GGUF" ]; then
    EXTRA="$EXTRA $VOICE_APE $VOICE_GGUF ${ROOT}/voice/voice-watchdog.sh"
else
    echo "note: voice payload missing — packaging without baked read-aloud"
fi
"${ROOT}/bin/zipalign" -j0 "$OUT" "$MODEL" "$MMPROJ" $EXTRA "${TMP}/.args"
chmod +x "$OUT"

echo "Built $(du -h "$OUT" | cut -f1) $(basename "$OUT")"
