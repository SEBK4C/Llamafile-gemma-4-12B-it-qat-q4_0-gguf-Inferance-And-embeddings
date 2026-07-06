#!/usr/bin/env python3
"""EM4: query-side instruction phrasing sweep on NFCorpus (docs cached from
EM1 — each phrasing costs one ~16 s query embed).

Tests the TASK-taxonomy hypothesis directly: NFCorpus is medical/nutrition,
so if domain instructions matter, TASK["med"] should beat the generic
web-search phrasing here. Also measures the value of having ANY instruction
(bare queries) — Qwen3 docs claim ~1-5%.
"""
import hashlib, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from embed_real_bench import embed, DOC_CHARS, D  # noqa: E402

EMBED_BASE = sys.argv[1] if len(sys.argv) > 1 else None
assert EMBED_BASE, "usage: em4_instr.py <embed-base-url>"

import numpy as np
import pyarrow.parquet as pq

PHRASINGS = {
    "generic-web (baseline)":
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
    "TASK.med":
        "Instruct: Given a clinical or medical question, retrieve relevant medical literature or notes\nQuery: ",
    "nfcorpus-tuned":
        "Instruct: Given a question about nutrition and health, retrieve scientific abstracts that answer it\nQuery: ",
    "short":
        "Instruct: Retrieve passages relevant to the query\nQuery: ",
    "bare (no instruction)": "",
}

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
DM = np.asarray(json.load(open(os.path.join(D, f"docemb_{cfg_hash}.json"))), dtype=np.float32)
DM /= np.linalg.norm(DM, axis=1, keepdims=True) + 1e-9

results = {}
for name, prefix in PHRASINGS.items():
    qv = embed(EMBED_BASE, [prefix + qmap[q] for q in test_qids])
    QM = np.asarray(qv, dtype=np.float32)
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
    results[name] = {"ndcg@10": round(float(np.mean(ndcg10)), 4),
                      "recall@100": round(float(np.mean(recall100)), 4)}
    print(f"{name:26s} nDCG@10={results[name]['ndcg@10']:.4f}  R@100={results[name]['recall@100']:.4f}")

out = os.path.join(HERE, "..", "data", "em4_instr_20260706.json")
json.dump(results, open(out, "w"), indent=1)
print("wrote", out)
