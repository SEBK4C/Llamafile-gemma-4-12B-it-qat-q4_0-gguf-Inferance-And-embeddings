#!/usr/bin/env python3
"""EM1 (H-C): BEIR NFCorpus retrieval benchmark against the LIVE embedding
sidecar — real nDCG@10 / Recall@100 comparable to published BEIR numbers.

Corpus: 3633 medical/nutrition abstracts; 323 test queries with graded
qrels (BeIR/nfcorpus + BeIR/nfcorpus-qrels, parquet/tsv under
datasets_real/nfcorpus/, LOCAL-ONLY).

Protocol (embed-research-program.md): docs embed BARE ("title. text",
truncated to ~480 tokens by chars — chunker-consistent); queries embed with
the generic web-search Instruct prefix (EM4 sweeps per-domain phrasings
later). Doc embeddings are CACHED per config hash — a candidate config pays
the ~10-min corpus embed once, re-runs are query-side only (~30 s).

Usage: embed_real_bench.py --embed-base URL [--config-tag qwen3-last-instr]
                            [--out J]
"""
import argparse, hashlib, json, math, os, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "datasets_real", "nfcorpus")

INSTR = ("Instruct: Given a web search query, retrieve relevant passages "
         "that answer the query\nQuery: ")
DOC_CHARS = 1900  # ≈ 480 tokens of abstract text


def embed(base, texts, timeout=600, batch=16, retries=3):
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        body = json.dumps({"input": chunk, "model": "e"}).encode()
        for a in range(retries):
            try:
                rq = urllib.request.Request(base.rstrip("/") + "/v1/embeddings", data=body,
                                            headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(rq, timeout=timeout) as r:
                    got = json.loads(r.read())
                out += [d["embedding"] for d in sorted(got["data"], key=lambda d: d["index"])]
                break
            except Exception as e:
                if a == retries - 1:
                    raise
                print(f"  embed batch {i} retry {a+1}: {type(e).__name__}", file=sys.stderr)
                time.sleep(3 * (a + 1))
    return out


def main():
    import pyarrow.parquet as pq
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-base", required=True)
    ap.add_argument("--config-tag", default="qwen3-last-instr")
    ap.add_argument("--out")
    a = ap.parse_args()

    corpus = pq.read_table(os.path.join(D, "corpus.parquet"))
    queries = pq.read_table(os.path.join(D, "queries.parquet"))
    doc_ids = [str(x) for x in corpus["_id"]]
    doc_txt = [(str(t) + ". " + str(x))[:DOC_CHARS]
               for t, x in zip(corpus["title"], corpus["text"])]
    qmap = {str(i): str(t) for i, t in zip(queries["_id"], queries["text"])}

    qrels = {}
    with open(os.path.join(D, "qrels_test.tsv")) as f:
        next(f)
        for line in f:
            qid, did, score = line.rstrip("\n").split("\t")
            qrels.setdefault(qid, {})[did] = int(score)
    test_qids = [q for q in qrels if q in qmap]
    print(f"corpus={len(doc_ids)} test_queries={len(test_qids)}")

    cfg_hash = hashlib.sha256((a.config_tag + "|" + str(DOC_CHARS)).encode()).hexdigest()[:12]
    cache = os.path.join(D, f"docemb_{cfg_hash}.json")
    if os.path.exists(cache):
        dv = json.load(open(cache))
        print(f"doc embeddings from cache {cache}")
    else:
        t0 = time.time()
        dv = embed(a.embed_base, doc_txt)
        print(f"embedded corpus in {time.time()-t0:.0f}s")
        json.dump(dv, open(cache, "w"))

    t0 = time.time()
    qv = embed(a.embed_base, [INSTR + qmap[q] for q in test_qids])
    print(f"embedded {len(test_qids)} queries in {time.time()-t0:.0f}s")

    # exact cosine ranking (3633 docs — brute force is fine)
    import numpy as np
    DM = np.asarray(dv, dtype=np.float32)
    DM /= np.linalg.norm(DM, axis=1, keepdims=True) + 1e-9
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

    res = {"config": a.config_tag, "n_docs": len(doc_ids), "n_queries": len(test_qids),
           "ndcg@10": round(float(np.mean(ndcg10)), 4),
           "recall@100": round(float(np.mean(recall100)), 4)}
    print("RESULT", json.dumps(res))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
