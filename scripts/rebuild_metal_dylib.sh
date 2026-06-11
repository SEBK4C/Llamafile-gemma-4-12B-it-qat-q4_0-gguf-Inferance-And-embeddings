#!/bin/sh
# Manual rebuild of ~/.llamafile/v/<ver>/ggml-metal.dylib from the extracted
# sources, mirroring llamafile/metal.c BuildMetal() flags exactly.
# Why this exists: editing the extracted sources and deleting the dylib does
# NOT work — extraction byte-compares against the zip member and clobbers
# local edits. Keep the dylib present; rebuild it with this (~30 s).
# See docs/metal-batch-kickoff.md "Tools & protocol".
set -e
VER="${1:-0.10.5}"
DIR="$HOME/.llamafile/v/$VER"
cd "$DIR"

[ -f ggml-metal.dylib.orig ] || cp ggml-metal.dylib ggml-metal.dylib.orig

SRCS="ggml.c ggml-alloc.c ggml-quants.c ggml-backend.cpp ggml-backend-meta.cpp ggml-threading.cpp ggml-metal.cpp ggml-metal-device.cpp ggml-metal-common.cpp ggml-metal-ops.cpp ggml-metal-device.m ggml-metal-context.m"

for s in $SRCS; do
  case "$s" in
    *.cpp) STD="-std=c++17" ;;
    *)     STD="" ;;
  esac
  cc -c "-I$DIR" $STD -O3 -fPIC -pthread -DNDEBUG -ffixed-x28 \
     -DTARGET_OS_OSX -DGGML_MULTIPLATFORM \
     -DGGML_VERSION='"manual"' -DGGML_COMMIT='"manual"' \
     -w -o "$s.o" "$s" &
done
wait

cc -shared -fPIC -pthread -ffixed-x28 -o ggml-metal.dylib.new \
   ggml.c.o ggml-alloc.c.o ggml-quants.c.o ggml-backend.cpp.o \
   ggml-backend-meta.cpp.o ggml-threading.cpp.o ggml-metal.cpp.o \
   ggml-metal-device.cpp.o ggml-metal-common.cpp.o ggml-metal-ops.cpp.o \
   ggml-metal-device.m.o ggml-metal-context.m.o \
   -framework Foundation -framework Metal -framework MetalKit -lc++

mv ggml-metal.dylib.new ggml-metal.dylib
echo "rebuilt $DIR/ggml-metal.dylib"
