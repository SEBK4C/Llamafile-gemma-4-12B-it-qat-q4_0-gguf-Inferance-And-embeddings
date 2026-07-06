#!/usr/bin/env python3
"""EM3: Matryoshka (MRL) truncation on the NFCorpus baseline — how much
retrieval quality does a 2-8x smaller index cost?

Qwen3-Embedding is MRL-trained: truncate to the first k dims and
L2-renormalize. Doc vectors come from EM1's config-hash cache (no
re-embedding); queries re-embed once (~16 s). Output: nDCG@10 / Recall@100
at k ∈ {1024, 512, 256, 128} → the storage-dim recommendation for the I9
hybrid store.
"""
import hashlib, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from embed_real_bench import embed, INSTR, DOC_CHARS, D  # noqa: E402

EMBED_BASE = sys.argv[1] if len(sys.argv) > 1 else None
assert EMBED_BASE, "usage: em3_mrl.py <embed-base-url>"

import numpy as np
import pyarrow.parquet as pq

corpus = pq.read_table(os.path.join(D, "corpus.parquet"))
queries = pq.read_table(os.path.join(D, "queries.parquet"))
doc_ids = [str(x) for x in corpus["_id"]]
qmap = {str(i): str(t) for i, t in zip(queries["_id"], queries["text"])}
qrels = {}
with open(os.path.join(D, "qrels_test.tsv")) as f:
    next(f)
    for line in f:
        qid, did, score = line.rstrip("\n").split("\t")
        qrels.setdefault(qid, {})[did] = int(score)
test_qids = [q for q in qrels if q in qmap]

cfg_hash = hashlib.sha256(("qwen3-last-instr|" + str(DOC_CHARS)).encode()).hexdigest()[:12]
dv = json.load(open(os.path.join(D, f"docemb_{cfg_hash}.json")))
qv = embed(EMBED_BASE, [INSTR + qmap[q] for q in test_qids])
DM0 = np.asarray(dv, dtype=np.float32)
QM0 = np.asarray(qv, dtype=np.float32)

results = {}
for k in (1024, 512, 256, 128):
    DM = DM0[:, :k].copy(); QM = QM0[:, :k].copy()
    DM /= np.linalg.norm(DM, axis=1, keepdims=True) + 1e-9
    QM /= np.linalg.norm(QM, axis=1, keepdims=True) + 1e-9
    S = QM @ DM.T
    ndcg10, recall100 = [], []
    for qi, qid in enumerate(test_qids):
        rel = qrels[qid]
        order = np.argsort(-S[qi])
        gains = [(2 ** rel.get(doc_ids[d], 0) - 1) for d in order[:10]]
        dcg = sum(g / math.log2(r + 2) for r, g in enumerate(gains))
        ideal = sorted(rel.values(), reverse=True)[:10]
        idcg = sum((2 ** g - 1) / math.log2(r + 2) for r, g in enumerate(ideal))
        ndcg10.append(dcg / idcg if idcg > 0 else 0.0)
        got100 = {doc_ids[d] for d in order[:100]}
        relevant = {d for d, s in rel.items() if s > 0}
        recall100.append(len(got100 & relevant) / len(relevant) if relevant else 0.0)
    results[k] = {"ndcg@10": round(float(np.mean(ndcg10)), 4),
                   "recall@100": round(float(np.mean(recall100)), 4)}
    print(f"k={k:5d}  nDCG@10={results[k]['ndcg@10']:.4f}  R@100={results[k]['recall@100']:.4f}")

out = os.path.join(HERE, "..", "data", "em3_mrl_20260706.json")
json.dump({"config": "qwen3-last-instr", "dims": results}, open(out, "w"), indent=1)
print("wrote", out)
