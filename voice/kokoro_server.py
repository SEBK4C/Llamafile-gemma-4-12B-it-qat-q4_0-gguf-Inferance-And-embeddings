#!/usr/bin/env python3
"""Kokoro TTS sidecar for the gemma4 llamafile web UI.

OpenAI-compatible-ish speech endpoint, CPU-only, zero VRAM:
    POST /v1/audio/speech   {"input": "...", "voice": "af_heart", "speed": 1.0}
        -> audio/wav
    GET  /health            -> {"status":"ok"}

Single Kokoro session guarded by a lock (synthesis is fast enough to
serialize: measured RTF 0.15 on 8 cores).
"""
import io
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

MODEL = "/opt/kokoro/kokoro-v1.0.onnx"
VOICES = "/opt/kokoro/voices-v1.0.bin"
PORT = 8091
MAX_CHARS = 6000          # refuse absurd payloads
CHUNK_CHARS = 450         # split long text on sentence boundaries

kokoro = Kokoro(MODEL, VOICES)
lock = threading.Lock()

_md_junk = re.compile(
    r"```.*?```|`[^`]*`"          # code blocks / inline code
    r"|!\[[^\]]*\]\([^)]*\)"      # images
    r"|\[([^\]]*)\]\([^)]*\)"     # links -> keep label via sub below
    , re.S)


def sanitize(text: str) -> str:
    text = _md_junk.sub(lambda m: m.group(1) or " ", text)
    text = re.sub(r"[*_#>|~]+", " ", text)      # md decoration
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_sentences(text: str, limit: int) -> list[str]:
    parts, cur = [], ""
    for sent in re.split(r"(?<=[.!?;:])\s+", text):
        if len(cur) + len(sent) + 1 > limit and cur:
            parts.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        parts.append(cur)
    return parts or [text]


def synth(text: str, voice: str, speed: float) -> tuple[np.ndarray, int]:
    chunks = split_sentences(text, CHUNK_CHARS)
    waves, sr = [], 24000
    with lock:
        for c in chunks:
            samples, sr = kokoro.create(c, voice=voice, speed=speed, lang="en-us")
            waves.append(samples)
    return np.concatenate(waves), sr


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # journald-friendly one-liners
        print("%s - %s" % (self.address_string(), fmt % args), flush=True)

    def _json(self, code: int, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") in ("", "/health"):
            self._json(200, {"status": "ok", "model": "kokoro-82M-v1.0"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/audio/speech":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length))
            text = sanitize(str(req.get("input", "")))[:MAX_CHARS]
            if not text:
                self._json(400, {"error": "empty input"})
                return
            voice = str(req.get("voice", "af_heart"))
            speed = float(req.get("speed", 1.0))
            samples, sr = synth(text, voice, speed)
            buf = io.BytesIO()
            sf.write(buf, samples, sr, format="WAV")
            wav = buf.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(wav)))
            self.end_headers()
            self.wfile.write(wav)
        except BrokenPipeError:
            pass
        except Exception as e:  # noqa: BLE001 - report anything to the client
            self._json(500, {"error": str(e)})


if __name__ == "__main__":
    print(f"kokoro-tts listening on 127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
