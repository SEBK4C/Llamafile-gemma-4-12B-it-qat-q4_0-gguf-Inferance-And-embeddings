#!/usr/bin/env python3
"""I5 e2e chain smoke — the whole phase-3 pipeline on one file:

  router (text-layer probe + rasterize) → PP-OCRv6 on scan pages →
  Gemma-4 grammar enrichment (prod GPU) → Qwen3 embedding (CT sidecar CPU)

This is the precursor of the I7 /v1/ingest worker. Prints a compact
envelope + per-stage latency; exits non-zero if any stage misbehaves.

Usage: chain_smoke.py FILE --base <gemma-url> --embed-base <embed-url>
"""
import argparse, json, os, sys, tempfile, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from router import route            # noqa: E402
from ocr import make_engine, extract  # noqa: E402
from enrich import enrich, semantic_valid  # noqa: E402


def embed(base, texts, timeout=120):
    body = json.dumps({"input": texts, "model": "embed"}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/v1/embeddings", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return [d["embedding"] for d in sorted(out["data"], key=lambda d: d["index"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--base", required=True)
    ap.add_argument("--embed-base", required=True)
    a = ap.parse_args()

    lat = {}
    t0 = time.time(); r = route(a.file); lat["route_ms"] = round(1000 * (time.time() - t0))

    ocr_texts = []
    scan_png = None
    if r["scan_pngs"]:
        engine = make_engine()
        t0 = time.time()
        for png in r["scan_pngs"]:
            scan_png = scan_png or png
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                f.write(png); tmp = f.name
            ocr_texts.append(extract(engine, tmp)["text"])
            os.unlink(tmp)
        lat["ocr_ms"] = round(1000 * (time.time() - t0))

    full_text = "\n\n".join(filter(None, [r["text"]] + ocr_texts))

    t0 = time.time()
    er = enrich(a.base, full_text, {**r["file"], "source_type": r["source_type"]}, scan_png)
    lat["enrich_ms"] = round(1000 * (time.time() - t0))
    e = er["enrichment"]
    ok_enrich = er["parse_error"] is None and semantic_valid(e)

    t0 = time.time()
    vecs = embed(a.embed_base, [e["summary"] if ok_enrich else full_text[:512], full_text[:2000]])
    lat["embed_ms"] = round(1000 * (time.time() - t0))

    envelope = {
        "file": r["file"], "source_type": r["source_type"],
        "pages": r["pages"], "text_chars": len(full_text),
        "enrichment": e, "embedding_dims": len(vecs[0]),
        "n_vectors": len(vecs), "latency": lat,
        "total_ms": sum(lat.values()),
    }
    print(json.dumps(envelope, indent=1, ensure_ascii=False))

    ok = (ok_enrich and len(vecs[0]) == 1024 and full_text and
          r["source_type"] is not None)
    print("CHAIN:", "PASS" if ok else "FAIL", "|",
          " ".join(f"{k}={v}" for k, v in lat.items()))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
