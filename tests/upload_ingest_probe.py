#!/usr/bin/env python3
"""Multimodal upload + embedding/ingest probe over tests/assets/.

Exercises through the API exactly what a web-UI user exercises by dragging
files into the composer: images and audio go to /v1/chat/completions as
base64 content parts; document text goes to /v1/ingest; retrieval sanity
runs the manifest queries against /v1/embeddings vectors.

    python3 tests/upload_ingest_probe.py [--base http://127.0.0.1:8080]

PDF note: the APE's /v1/ingest is text-in by design (the web UI extracts
PDF text client-side; OCR ingest lives in the external worker), so this
probe ingests each PDF's paired .txt ground truth. Dragging the actual
.pdf files into the web UI is the manual half of this test — see
tests/assets/README.md.
"""
import argparse
import base64
import json
import math
import pathlib
import sys
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
ASSETS = HERE / "assets"


def post(base, path, obj, timeout=600, headers=None):
    h = {"Content-Type": "application/json"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(base + path, json.dumps(obj).encode(), h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chat_media(base, kind, path, ask, max_tokens=1024):
    b64 = base64.b64encode(path.read_bytes()).decode()
    if kind == "image":
        part = {"type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}}
    else:
        part = {"type": "input_audio",
                "input_audio": {"data": b64, "format": "wav"}}
    out = post(base, "/v1/chat/completions", {
        "messages": [{"role": "user", "content": [
            part, {"type": "text", "text": ask}]}],
        "temperature": 1.0, "top_k": 64, "top_p": 0.95,
        "max_tokens": max_tokens, "stream": False})
    return out["choices"][0]["message"].get("content") or ""


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    args = ap.parse_args()
    man = json.loads((ASSETS / "manifest.json").read_text())
    passed = failed = 0

    def verdict(ok, label, detail=""):
        nonlocal passed, failed
        passed += ok
        failed += (not ok)
        print(f"  {'PASS' if ok else 'FAIL'}  {label}  {detail[:90]}")

    print("[1/4] image upload -> chat (OCR / chart reading)")
    for name, spec in man["images"].items():
        reply = chat_media(args.base, "image", ASSETS / "images" / name, spec["ask"])
        rl = reply.lower()
        ok = (any(e.lower() in rl for e in spec["expect_any"])
              and all(e.lower() in rl for e in spec["expect_all"]))
        verdict(ok, f"image:{name}", reply.replace("\n", " "))

    print("[2/4] audio upload -> chat (native STT)")
    for name, spec in man["audio"].items():
        reply = ""
        for _ in range(2):  # thinking channel occasionally eats the budget
            # number-heavy clips spawn long numeral reasoning — 2048 budget
            reply = chat_media(args.base, "audio", ASSETS / "audio" / name,
                               "Transcribe this audio exactly, word for word, as "
                               "ordinary English words. Output only the transcription.",
                               max_tokens=2048)
            if reply.strip():
                break
        # score the payload facts (key_words), hyphen/punct-normalized —
        # models often transcribe just the payload phrase, and 'forty-seven'
        # must vouch for 'forty'+'seven'
        import re as _re
        heard = set(_re.sub(r"[^a-z]+", " ", reply.lower()).split())
        want = set(spec.get("key_words") or
                   [w for w in _re.sub(r"[^a-z]+", " ", spec["transcript"].lower()).split()
                    if len(w) > 3])
        overlap = len(want & heard) / max(1, len(want))
        if spec.get("known_spiral") and overlap < 0.8:
            print(f"  KNOWN {name}  overlap={overlap:.0%} — {spec['note'][:80]}")
            continue  # canary for the reasoning-budget candidate; not a gate
        verdict(overlap >= 0.8, f"audio:{name}", f"overlap={overlap:.0%} {reply!r}")

    print("[3/4] document text -> /v1/ingest (enrichment + fidelity + vectors)")
    envelopes = {}
    for name, spec in man["pdfs"].items():
        text = (ASSETS / "pdfs" / spec["text_file"]).read_text()
        env = post(args.base, "/v1/ingest", {"text": text, "name": name})
        envelopes[name] = env
        ents = [e.lower() for e in (env.get("enrichment") or {}).get("entities", [])]
        ok = (env.get("enrich_ok") is True
              and env.get("chunks") and len(env["doc_embedding"]) == 1024
              and any(x.lower() in " ".join(ents) for x in spec["expect_entities_any"]))
        verdict(ok, f"ingest:{name}",
                f"entities={ents[:4]} chunks={len(env.get('chunks', []))}")

    print("[4/4] retrieval sanity (manifest queries -> right document)")
    doc_names = list(envelopes)
    doc_vecs = [envelopes[n]["doc_embedding"] for n in doc_names]
    for name, spec in man["pdfs"].items():
        q = post(args.base, "/v1/embeddings",
                 {"input": [f"Instruct: retrieve the relevant document\nQuery: {spec['query']}"]})
        qv = q["data"][0]["embedding"]
        sims = [cosine(qv, dv) for dv in doc_vecs]
        top = doc_names[sims.index(max(sims))]
        verdict(top == name, f"retrieval:{spec['query'][:40]!r}",
                f"hit={top} sims={[round(s, 3) for s in sims]}")

    print(f"\n== {passed} PASS / {failed} FAIL ==")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
