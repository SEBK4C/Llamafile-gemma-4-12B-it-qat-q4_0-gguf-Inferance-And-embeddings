#!/usr/bin/env python3
"""P1/I7 core: the ingest worker — chain_smoke as a reusable, PIPELINED
batch engine (bench/phase3-ingest-program.md P-SPRINT).

ingest_one(path) = route → OCR (scan pages) → [audio → native STT] →
grammar enrichment → hint-guided chunking → embeddings → ingest.v1
envelope written to OUT/<sha256>.json (sha256 = idempotency key: existing
envelopes are skipped unless --force).

Pipelining: per-document threads bounded by per-STAGE semaphores —
  ocr=2 (CPU), enrich=2 (GPU; H7: C=2 overlaps ~1.37×, C≥4 just queues),
  embed=2 (CPU sidecar, -np 2)
so a doc can be in OCR while another is in enrichment and a third in
embedding. Serial mode (--mode serial) is the baseline for the sprint
ledger. The whole batch takes the bench eval-lock (never race other
benchmarks for the GPU).

Usage:
  ingest_worker.py --batch LISTFILE --out DIR --base URL --embed-base URL
                   [--mode pipeline|serial] [--force]
  ingest_worker.py FILE --base URL --embed-base URL     # one doc, stdout
"""
import argparse, base64, concurrent.futures as cf, fcntl, json, os, sys
import tempfile, threading, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from router import route                     # noqa: E402
from ocr import make_engine, extract         # noqa: E402
from enrich import enrich, semantic_valid    # noqa: E402
from real_eval import chat_audio, to_16k_mono_wav  # noqa: E402
from fidelity import apply_fidelity          # noqa: E402
import chunker as chk                        # noqa: E402

LOCK = os.path.join(HERE, "..", ".eval.lock")

_ENGINE = None
_ENGINE_LOCK = threading.Lock()


def engine():
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = make_engine()
        return _ENGINE


def embed(base, texts, timeout=300):
    out = []
    for i in range(0, len(texts), 16):
        body = json.dumps({"input": texts[i:i+16], "model": "e"}).encode()
        rq = urllib.request.Request(base.rstrip("/") + "/v1/embeddings", data=body,
                                    headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(rq, timeout=timeout) as r:
            got = json.loads(r.read())
        out += [d["embedding"] for d in sorted(got["data"], key=lambda d: d["index"])]
    return out


class Stages:
    def __init__(self, ocr_c=2, enrich_c=2, embed_c=2):
        self.ocr = threading.Semaphore(ocr_c)
        self.enrich = threading.Semaphore(enrich_c)
        self.embed = threading.Semaphore(embed_c)
        self.t = {"ocr": 0.0, "stt": 0.0, "enrich": 0.0, "chunk": 0.0, "embed": 0.0}
        self.tl = threading.Lock()

    def add(self, k, dt):
        with self.tl:
            self.t[k] += dt


def ingest_one(path, base, embed_base, stages, keep_vectors=False):
    t_doc = time.time()
    r = route(path)
    ocr_texts, scan_png = [], None

    if r["scan_pngs"]:
        with stages.ocr:
            t0 = time.time()
            for png in r["scan_pngs"]:
                scan_png = scan_png or png
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    f.write(png); tmp = f.name
                ocr_texts.append(extract(engine(), tmp)["text"])
                os.unlink(tmp)
            stages.add("ocr", time.time() - t0)
    elif r["source_type"] == "image_photo":
        with stages.ocr:
            t0 = time.time()
            o = extract(engine(), path)
            if o["n_boxes"] >= 3:  # photographed document → text matters
                ocr_texts.append(o["text"])
            stages.add("ocr", time.time() - t0)
            if scan_png is None:
                scan_png = open(path, "rb").read()

    transcript = None
    if r["source_type"] == "audio":
        with stages.enrich:  # GPU stage
            t0 = time.time()
            wav = to_16k_mono_wav(open(path, "rb").read())
            transcript, _ = chat_audio(base, wav, "wav",
                "Transcribe this audio exactly. Output only the transcription.")
            stages.add("stt", time.time() - t0)

    full_text = "\n\n".join(filter(None, [r["text"], transcript] + ocr_texts))

    with stages.enrich:
        t0 = time.time()
        er = enrich(base, full_text,
                    {**{k: r["file"][k] for k in ("sha256", "name", "bytes")},
                     "source_type": r["source_type"]},
                    scan_png)
        stages.add("enrich", time.time() - t0)
    e = er["enrichment"]
    ok = er["parse_error"] is None and semantic_valid(e)
    fid = apply_fidelity(e, full_text) if ok else None  # Q1 gate (mutates e)

    t0 = time.time()
    chk._TOK_BASE = embed_base
    chunks = chk.chunk_text(full_text or (e["summary"] if ok else ""),
                            e if ok else None,
                            "csv" if r["source_type"] == "csv" else None)
    stages.add("chunk", time.time() - t0)

    with stages.embed:
        t0 = time.time()
        doc_text = chk.doc_summary_text(e) if ok else (full_text[:512] or "empty")
        vecs = embed(embed_base, [doc_text] + [c["text"] for c in chunks])
        stages.add("embed", time.time() - t0)

    env = {"schema_version": "ingest.v1", "pipeline_version": "p3.0",
           "file": r["file"], "source_type": r["source_type"],
           "pages": r["pages"], "text_chars": len(full_text),
           "transcript": transcript, "enrichment": e, "enrich_ok": ok,
           "fidelity": fid,
           "chunks": [dict(c, embedding=(vecs[1+i] if keep_vectors else None),
                            embedding_dims=len(vecs[1+i]))
                      for i, c in enumerate(chunks)],
           "doc_embedding": vecs[0] if keep_vectors else None,
           "doc_embedding_dims": len(vecs[0]),
           "wall_s": round(time.time() - t_doc, 2)}
    if not keep_vectors:
        for c in env["chunks"]:
            c.pop("embedding", None)
    return env


def ingest_text(text, base, embed_base, stages=None, name="api_text",
                keep_vectors=False):
    """Sebastian 2026-07-06: raw TEXT sent to the API gets the SAME JSON
    enrichment pass as files — this is the text route of the /v1/ingest
    surface (per F13, /v1/embeddings itself stays pure OpenAI shape; the
    enrichment-carrying contract is /v1/ingest)."""
    import hashlib
    stages = stages or Stages()
    t_doc = time.time()
    sha = hashlib.sha256(text.encode()).hexdigest()

    with stages.enrich:
        t0 = time.time()
        er = enrich(base, text, {"sha256": sha, "name": name,
                                  "bytes": len(text.encode()),
                                  "source_type": "text_api"}, None)
        stages.add("enrich", time.time() - t0)
    e = er["enrichment"]
    ok = er["parse_error"] is None and semantic_valid(e)
    fid = apply_fidelity(e, text) if ok else None  # Q1 gate (mutates e)

    t0 = time.time()
    chk._TOK_BASE = embed_base
    chunks = chk.chunk_text(text, e if ok else None, None)
    stages.add("chunk", time.time() - t0)

    with stages.embed:
        t0 = time.time()
        doc_text = chk.doc_summary_text(e) if ok else text[:512]
        vecs = embed(embed_base, [doc_text] + [c["text"] for c in chunks])
        stages.add("embed", time.time() - t0)

    env = {"schema_version": "ingest.v1", "pipeline_version": "p3.0",
           "file": {"sha256": sha, "name": name, "bytes": len(text.encode()),
                     "mtime": None, "exif": None},
           "source_type": "text_api", "pages": None,
           "text_chars": len(text), "transcript": None,
           "enrichment": e, "enrich_ok": ok, "fidelity": fid,
           "chunks": [dict(c, embedding=(vecs[1+i] if keep_vectors else None),
                            embedding_dims=len(vecs[1+i]))
                      for i, c in enumerate(chunks)],
           "doc_embedding": vecs[0] if keep_vectors else None,
           "doc_embedding_dims": len(vecs[0]),
           "wall_s": round(time.time() - t_doc, 2)}
    if not keep_vectors:
        for c in env["chunks"]:
            c.pop("embedding", None)
    return env


def run_batch(files, out_dir, base, embed_base, mode, force, keep_vectors):
    os.makedirs(out_dir, exist_ok=True)
    stages = Stages()
    done, skipped, failed = [], [], []

    def work(p):
        try:
            sha = route(p)["file"]["sha256"]  # cheap; route again inside worker
            dst = os.path.join(out_dir, sha + ".json")
            if os.path.exists(dst) and not force:
                skipped.append(p); return
            env = ingest_one(p, base, embed_base, stages, keep_vectors)
            json.dump(env, open(dst, "w"), ensure_ascii=False)
            done.append((p, env["wall_s"]))
            print(f"  ok {os.path.basename(p)} {env['wall_s']}s "
                  f"({env['source_type']}, {len(env['chunks'])} chunks)")
        except Exception as e:
            failed.append((p, f"{type(e).__name__}: {e}"))
            print(f"  FAIL {os.path.basename(p)}: {type(e).__name__}: {e}")

    t0 = time.time()
    with open(LOCK, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            engine()  # warm OCR session outside the timing of first doc
            if mode == "serial":
                for p in files:
                    work(p)
            else:
                with cf.ThreadPoolExecutor(max_workers=6) as ex:
                    list(ex.map(work, files))
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)
    wall = time.time() - t0
    n = len(done)
    return {"mode": mode, "files": len(files), "done": n, "skipped": len(skipped),
            "failed": [f"{os.path.basename(p)}: {m}" for p, m in failed],
            "wall_s": round(wall, 1),
            "docs_per_min": round(60 * n / wall, 2) if n else 0,
            "stage_busy_s": {k: round(v, 1) for k, v in stages.t.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?")
    ap.add_argument("--text", help="ingest a raw text string (the API text route)")
    ap.add_argument("--batch"); ap.add_argument("--out", default="envelopes")
    ap.add_argument("--base", required=True); ap.add_argument("--embed-base", required=True)
    ap.add_argument("--mode", choices=["pipeline", "serial"], default="pipeline")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep-vectors", action="store_true")
    ap.add_argument("--report")
    a = ap.parse_args()
    if a.text is not None:
        env = ingest_text(a.text, a.base, a.embed_base, keep_vectors=a.keep_vectors)
        print(json.dumps(env, indent=1, ensure_ascii=False))
        return 0
    if a.batch:
        files = [l.strip() for l in open(a.batch) if l.strip()]
        rep = run_batch(files, a.out, a.base, a.embed_base, a.mode, a.force, a.keep_vectors)
        print(json.dumps(rep, indent=1))
        if a.report:
            json.dump(rep, open(a.report, "w"), indent=1)
        return 0 if not rep["failed"] else 1
    stages = Stages()
    env = ingest_one(a.file, a.base, a.embed_base, stages, a.keep_vectors)
    print(json.dumps(env, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
