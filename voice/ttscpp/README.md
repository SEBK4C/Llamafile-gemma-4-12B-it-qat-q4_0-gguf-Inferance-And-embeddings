# tts-server.ape — the voice as a Cosmopolitan binary

`cosmo-build.sh` (run from a TTS.cpp checkout) compiles mmwillet/TTS.cpp +
its ggml (`support-for-tts` branch: adds STFT/ISTFT for Kokoro's vocoder)
with the llamafile fork's cosmocc toolchain into a single portable APE
(~8 MB): macOS/Linux/BSD, arm64 + x86_64, CPU-only (Kokoro is realtime on CPU).

Verified 2026-07-04: serves POST /v1/audio/speech ({"input","voice"} →
audio/wav) from `Kokoro_no_espeak_Q4.gguf` (198 MB — phonemizer INSIDE the
GGUF; no espeak, no ONNX, no Python).

Port notes (upstreamable):
- C++23 needed (`starts_with`, `std::numbers`, ranges).
- `src/tokenizer.cpp` needs `#include <algorithm>`.
- `index.html.hpp` is CMake-generated: run one native configure first, or
  pre-generate it.
- **Static-init order**: `loaders.cpp` (the LOADERS map) must be linked FIRST
  or the model registrars fire into an unconstructed map and every
  architecture is "Unknown".

Remaining for full bake-in (next milestone): zipalign tts-server.ape +
Kokoro GGUF into gemma4-server.llamafile; gemma server extracts + spawns it
on 127.0.0.1 and proxies /tts; the web UI's /tts probe then lights up the
read-aloud controls automatically.
