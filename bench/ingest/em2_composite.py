#!/usr/bin/env python3
"""EM2: which enrichment fields belong in the DOC-summary vector?

Variants recompose doc_text from the STORED 87 envelopes (no GPU
enrichment; ~30 s of sidecar embedding per variant), scored dense-only on
the frozen I9 query set. H-B's premise says people/scene matter for photo
retrieval; EM2 measures each field's marginal value.

  A  title+summary
  B  title+summary+scene+people
  C  title+summary+scene+entities+people   (shipped doc_text_of)
  D  summary only
  E  C + transcript[:500]                  (audio docs get their words)
  F  C + first-chunk text[:400]            (lead-chunk grounding)
"""
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from embed_real_bench import embed  # noqa: E402
from hybrid_store import frozen_queries, INSTR  # noqa: E402

ENVDIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/env_v100"
EMBED_BASE = sys.argv[2] if len(sys.argv) > 2 else None
assert EMBED_BASE, "usage: em2_composite.py <envdir> <embed-base>"

import numpy as np

envs = []
for n in sorted(os.listdir(ENVDIR)):
    if n.endswith(".json"):
        envs.append(json.load(open(os.path.join(ENVDIR, n))))
names = [e["file"]["name"] for e in envs]
name_to_idx = {}
for i, n in enumerate(names):
    name_to_idx.setdefault(n, i)


def compose(env, variant):
    e = env.get("enrichment") or {}
    title, summary = e.get("title") or "", e.get("summary") or ""
    scene = e.get("scene") or ""
    ents = " ".join(e.get("entities") or [])
    people = " ".join(p.get("doing", "") for p in e.get("people") or [])
    tr = (env.get("transcript") or "")[:500]
    chunk0 = ""
    if env.get("chunks"):
        chunk0 = (env["chunks"][0].get("text") or "")[:400]
    parts = {
        "A": [title, summary],
        "B": [title, summary, scene, people],
        "C": [title, summary, scene, ents, people],
        "D": [summary],
        "E": [title, summary, scene, ents, people, tr],
        "F": [title, summary, scene, ents, people, chunk0],
    }[variant]
    out = ". ".join(p for p in parts if p)
    return out if out else (env.get("transcript") or "empty")[:800]


queries = [(qt, tgt) for qt, tgt in frozen_queries() if tgt in name_to_idx]
qv = embed(EMBED_BASE, [INSTR + qt for qt, _ in queries])
QM = np.asarray(qv, dtype=np.float32)
QM /= np.linalg.norm(QM, axis=1, keepdims=True) + 1e-9
print(f"{len(queries)} queries, {len(envs)} docs")

results = {}
for v in "ABCDEF":
    dv = embed(EMBED_BASE, [compose(e, v) for e in envs])
    DM = np.asarray(dv, dtype=np.float32)
    DM /= np.linalg.norm(DM, axis=1, keepdims=True) + 1e-9
    S = QM @ DM.T
    hit1 = hit3 = 0
    mrr = 0.0
    for qi, (qt, tgt) in enumerate(queries):
        order = list(np.argsort(-S[qi]))
        rank = order.index(name_to_idx[tgt]) + 1
        hit1 += rank == 1
        hit3 += rank <= 3
        mrr += 1.0 / rank
    n = len(queries)
    results[v] = {"hit@1": round(hit1 / n, 3), "hit@3": round(hit3 / n, 3),
                   "mrr": round(mrr / n, 3)}
    print(f"{v}  hit@1={results[v]['hit@1']:.3f}  hit@3={results[v]['hit@3']:.3f}  mrr={results[v]['mrr']:.3f}")

out = os.path.join(HERE, "..", "data", "em2_composite_20260706.json")
json.dump(results, open(out, "w"), indent=1)
print("wrote", out)
