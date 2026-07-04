# Baking the read-aloud voice into the file (in progress)

The read-aloud voice currently runs as a Python **Kokoro sidecar**
(`kokoro_server.py`, kokoro-onnx). This document tracks replacing it with a
**pure-C++ voice** that can be bundled into the llamafile APE itself — no
Python, no ONNX runtime, no external phonemizer.

## Proven (2026-07-04)

[mmwillet/TTS.cpp](https://github.com/mmwillet/TTS.cpp) synthesizes Kokoro on
GGML and is **API-compatible with our sidecar**:

- Model: `mmwillet2/Kokoro_GGUF` → **`Kokoro_no_espeak_Q4.gguf` (198 MB)**.
  The `no_espeak` variant carries its own phonemizer *inside the GGUF*
  (`phonemizer.rules.*`, `phonemizer.dictionary.*`, `phonemizer.graphemes`) —
  **zero external dependency** (no libespeak-ng).
- `tts-cli`: 6.2 s of audio synthesized in 2.6 s on host CPU (RTF ≈ 0.42).
- `tts-server`: exposes `POST /v1/audio/speech` with `{"input","voice"}` →
  `audio/wav`, plus `GET /v1/audio/voices` — the exact contract the injected
  web UI already calls at `/tts`. All Kokoro voices present (af_heart default).

### Native build recipe (host smoke test)

```sh
git clone https://github.com/mmwillet/TTS.cpp && cd TTS.cpp
git submodule update --init --recursive        # ggml + nested modules
# apt: pkg-config libsdl2-dev (SDL2 only for the playback CLI)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF -DGGML_CUDA=OFF -DTTS_BUILD_EXAMPLES=ON
make -j
./bin/tts-server -mp Kokoro_no_espeak_Q4.gguf --port 8095
```

## Remaining work to get it *inside the APE*

1. **cosmocc port**: TTS.cpp uses CMake + stock ggml; the llamafile build uses
   the Cosmopolitan `make`. Port the `tts` library + `tts-server` sources into
   the fork's build (like `whisperfile/`), compiling with cosmocc so the output
   is an APE. TTS.cpp is small (~1.5k LOC in `src/`) and depends only on ggml,
   which the fork already vendors.
2. **Bundle**: `zipalign` the `tts-server` APE + `Kokoro_no_espeak_Q4.gguf`
   into the main `gemma4-server.llamafile` zip.
3. **Spawn**: the gemma server launches the bundled tts-server on a loopback
   port at startup and `tailscale serve`/reverse-proxies it at `/tts` — the UI
   already probes `/tts/health` and only shows read-aloud when it answers, so
   this lights up automatically with no UI change.
4. Alternative interim (no APE change): ship `tts-server` + GGUF as a native
   binary replacing `kokoro_server.py` in the container — removes the
   Python/ONNX dependency now. NOTE: built on Debian 13 (glibc 2.41); needs a
   static build or the `glibc-compat` treatment to run on the Debian 12
   container (same skew we hit with `ggml-cuda.so`).

## Also on the speech-native roadmap

- **whisperfile is already in the fork's build tree** (`whisperfile/BUILD.mk`)
  — a first-class path to streaming STT for live transcription of barge-in
  utterances and more robust command recognition.
- Voice commands already work via Gemma 4's native function-calling from audio
  (`voice/tts-inject.html`: stop/speed/read-again/new-chat/regenerate).
