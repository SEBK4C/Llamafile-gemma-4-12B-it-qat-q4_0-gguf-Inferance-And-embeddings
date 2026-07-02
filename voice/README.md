# Voice replies (Kokoro sidecar + web UI injection)

- `kokoro_server.py` — CPU-only TTS sidecar for CT deployment (`/opt/kokoro/` +
  venv with kokoro-onnx + soundfile + espeak-ng; model files kokoro-v1.0.onnx +
  voices-v1.0.bin from thewh1teagle/kokoro-onnx releases). systemd unit
  `kokoro-tts.service` runs it on 127.0.0.1:8091; ExecStartPost re-asserts the
  tailscale serve path mount `--set-path=/tts` (the enrollment hookscript can
  reset serve config at container boot).
- `tts-inject.html` — script block appended to
  `vendor/llamafile/llama.cpp/tools/ui/dist/index.html` **before `make build`**
  (replaces the closing `</body></html>`). Intercepts the UI's own
  /v1/chat/completions stream (content deltas only, thinking never spoken) and
  plays replies through POST /tts/v1/audio/speech. Adds floating 🔊 replay and
  🅰 auto-speak buttons.

⚠ Re-running fetch-ui-assets.sh overwrites dist/index.html — re-apply the
injection before building. Verify with:
  strings bin/llamafile | grep -c 'gemma4 voice replies'

## v2 (composer integration + server-side UI defaults)
- `gemma.service` passes `--ui-config '{"excludeReasoningFromContext":true,"preEncodeConversation":true}'`
  (server-served defaults; users can still override in UI settings — note systemd
  needs the JSON single-quoted or it strips the double quotes).
- tts-inject v2 places buttons IN the composer, left of Send, styling inherited
  from the Send button: 🔊 auto-speak toggle (✓ badge = on, shift-click replays
  last reply) and 🎙 mic (records 16 kHz mono WAV via AudioContext — the server's
  audio decoder wants WAV, not MediaRecorder webm — and attaches it through the
  UI's own file input via DataTransfer + change event).
- Marker check after builds: python `open('bin/llamafile','rb').read().count(b'g4v-btn')`
  (assets embed as hex arrays; `strings | grep` is NOT reliable).
