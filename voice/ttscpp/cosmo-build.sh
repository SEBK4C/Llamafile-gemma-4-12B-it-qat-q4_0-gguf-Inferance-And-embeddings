#!/bin/bash
# Build TTS.cpp (Kokoro) as a Cosmopolitan APE: tts-server.ape
# CPU-only (voice synthesis is realtime on CPU), own ggml (mmwillet branch
# with STFT/ISTFT), fully independent from the llamafile binary.
set -u
COSMO=/root/gemma4-gpu-optim/repo/vendor/llamafile/.cosmocc/4.0.2/bin
CC="$COSMO/cosmocc"; CXX="$COSMO/cosmoc++"
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="$ROOT/cosmo-build"
mkdir -p "$OUT"

INC="-I$ROOT/ggml/include -I$ROOT/ggml/src -I$ROOT/ggml/src/ggml-cpu \
     -I$ROOT/ggml-patches -I$ROOT/include -I$ROOT/src -I$ROOT/examples/server -I$ROOT/build/examples/server"
DEFS="-DGGML_USE_CPU -DNDEBUG -D_GNU_SOURCE"
# x86 SIMD floor: AVX2+FMA+F16C (Haswell 2013+). ggml has no runtime dispatch
# in this standalone build, and generic SSE2 costs ~10x on the vocoder (RTF 2.2
# vs 0.25). cosmocc -Xx86_64- prefixed flags apply to the x86 half only; the
# aarch64 half keeps its NEON defaults.
SIMD="-Xx86_64-mavx -Xx86_64-mavx2 -Xx86_64-mfma -Xx86_64-mf16c"
CFLAGS="-O2 -fPIC $SIMD $INC $DEFS"
CXXFLAGS="-O2 -std=gnu++23 -fexceptions -frtti $SIMD $INC $DEFS"

C_SRCS=(
  ggml/src/ggml.c
  ggml/src/ggml-alloc.c
  ggml/src/ggml-quants.c
  ggml/src/ggml-cpu/ggml-cpu.c
  ggml/src/ggml-cpu/ggml-cpu-quants.c
  ggml/src/ggml-cpu/ggml-cpu-aarch64.c
  ggml/src/ggml-aarch64.c
  ggml/src/ggml-cpu/ggml-cpu-ffast-math.c
)
CXX_SRCS=(
  ggml/src/ggml-backend.cpp
  ggml/src/ggml-backend-reg.cpp
  ggml/src/ggml-threading.cpp
  ggml/src/ggml-opt.cpp
  ggml-patches/llama-mmap.cpp
)
# ggml-cpu extra c++ if present
for f in ggml/src/ggml-cpu/ggml-cpu.cpp ggml/src/ggml-cpu/ggml-cpu-traits.cpp; do
  [ -f "$f" ] && CXX_SRCS+=("$f")
done
# tts core + all model impls
while IFS= read -r f; do CXX_SRCS+=("$f"); done < <(find src -name '*.cpp' | sort)
CXX_SRCS+=(examples/server/server.cpp)

fail=0
for s in "${C_SRCS[@]}"; do
  [ -f "$s" ] || { echo "skip(missing): $s"; continue; }
  o="$OUT/$(echo "$s" | tr '/' '_').o"
  [ -f "$o" ] && [ "$o" -nt "$s" ] && continue
  echo "CC  $s"
  $CC $CFLAGS -c -o "$o" "$s" 2>>"$OUT/errors.log" || { echo "ERR $s"; fail=1; }
done
for s in "${CXX_SRCS[@]}"; do
  [ -f "$s" ] || { echo "skip(missing): $s"; continue; }
  o="$OUT/$(echo "$s" | tr '/' '_').o"
  [ -f "$o" ] && [ "$o" -nt "$s" ] && continue
  echo "CXX $s"
  $CXX $CXXFLAGS -c -o "$o" "$s" 2>>"$OUT/errors.log" || { echo "ERR $s"; fail=1; }
done
if [ "$fail" = 1 ]; then
  echo "---- first errors ----"
  grep -E 'error:' "$OUT/errors.log" | head -15
  exit 1
fi
echo "LINK tts-server.ape"
# loaders.cpp must be linked FIRST so its LOADERS map is constructed before
# the model loaders' static registrars run (static init order follows link order)
$CXX -o "$OUT/tts-server.ape" "$OUT/src_models_loaders.cpp.o" $(ls "$OUT"/*.o | grep -v src_models_loaders) -lpthread 2>>"$OUT/errors.log" || {
  grep -E 'error|undefined' "$OUT/errors.log" | tail -15; exit 1; }
echo "BUILT: $OUT/tts-server.ape"
