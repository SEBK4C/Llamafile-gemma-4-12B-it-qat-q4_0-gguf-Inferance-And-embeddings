#!/usr/bin/env python3
"""TTS pronunciation roundtrip: synthesize words through the TTS sidecar,
then let the LLM's native speech recognition transcribe them back.

A word that survives the roundtrip is pronounced intelligibly; a word that
comes back mangled (e.g. 'specific' -> 'spekiffik') has a phonemizer bug.
Uses only local pieces: the /tts proxy and the model's audio input.

    python3 tests/tts_roundtrip.py                       # default word list
    python3 tests/tts_roundtrip.py specific espresso     # custom words
    python3 tests/tts_roundtrip.py --tts http://127.0.0.1:8093  # A/B a sidecar

Exit 0 if every word roundtrips (case-insensitive, punctuation-stripped),
1 otherwise. PLATFORM-NOTES: speech STT is near-verbatim on this model, so
failures point at the TTS side.
"""
import argparse
import base64
import json
import re
import sys
import urllib.request

# Words with non-trivial grapheme->phoneme mappings; 'specific' is the
# reported regression (no_espeak GGUF rule-based G2P).
DEFAULT_WORDS = [
    "specific", "specifically", "pacific", "species", "sufficient",
    "colonel", "choir", "epitome", "hyperbole", "infrastructure",
]


def tts(base, text, voice="af_heart"):
    body = json.dumps({"input": text, "voice": voice}).encode()
    req = urllib.request.Request(base + "/v1/audio/speech", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def stt(base, wav_bytes):
    b64 = base64.b64encode(wav_bytes).decode()
    body = json.dumps({
        "messages": [{
            "role": "user",
            "content": [
                {"type": "input_audio",
                 "input_audio": {"data": b64, "format": "wav"}},
                {"type": "text",
                 "text": "Transcribe this audio exactly, word for word, as "
                         "ordinary English words. No phonetic notation, no "
                         "IPA, no commentary — only the transcription."},
            ],
        }],
        "temperature": 1.0, "top_k": 64, "top_p": 0.95,
        "max_tokens": 1024, "stream": False,
    }).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.load(r)
    return resp["choices"][0]["message"].get("content") or ""


def norm(s):
    return re.sub(r"[^a-z]+", " ", s.lower()).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("words", nargs="*", default=None)
    ap.add_argument("--llm", default="http://127.0.0.1:8080")
    ap.add_argument("--tts", default=None,
                    help="TTS base URL (default: <llm>/tts, the proxy)")
    args = ap.parse_args()
    words = args.words or DEFAULT_WORDS
    tts_base = args.tts or (args.llm.rstrip("/") + "/tts")

    # One natural carrier sentence holding every word: isolated single-word
    # clips push the STT judge into dictionary mode (IPA / phonetic
    # respellings / empty content); connected speech transcribes reliably.
    carrier = "Please note that " + ", then ".join(
        f"I said {w}" for w in words) + ". Thank you."
    wav = tts(tts_base, carrier)
    heard = stt(args.llm, wav)
    print(f"  carrier: {carrier!r}")
    print(f"  heard:   {heard.strip()!r}\n")
    # Word-boundary match: 'specificus' must NOT vouch for 'specific'.
    heard_words = set(norm(heard).split())
    failures = [w for w in words if norm(w) not in heard_words]
    for w in words:
        print(f"  {'PASS' if w not in failures else 'FAIL'}  {w}")

    print(f"\n{len(words) - len(failures)}/{len(words)} words roundtrip clean")
    if failures:
        print("failed:", ", ".join(failures))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
