#!/usr/bin/env python3
"""I1 embedder A/B: nomic-embed-text-v1.5 vs Qwen3-Embedding-0.6B vs
embeddinggemma-300m, all served by the gemma4 llamafile (--embeddings, CPU).

Frozen fixture: 16 docs across the phase-3 task domains, 10 gold queries,
4 margin triplets. Each model runs in "raw" mode and in its canonical
prompt format (nomic: search_document:/search_query: prefixes; qwen3:
query-side Instruct; embeddinggemma: task:/title: templates). Metrics:
retrieval hit@1/hit@3/MRR, mean triplet margin, per-item embed latency.

Stdlib only. See bench/phase3-ingest-program.md (I1).

NOTE (F16): the fork crashes on --pooling last (ggml get_rows assert at
graph reserve), so Qwen3 runs MEAN-pooled here — non-canonical for that
model; treat its scores as a lower bound until the fork is patched.
"""
import argparse, json, math, time, urllib.request

DOCS = [
    ("code1", "code", "def binary_search(arr, target): performs O(log n) lookup on a sorted list by repeatedly halving the search interval."),
    ("code2", "code", "The systemd unit uses Restart=always and RestartSec=3 to supervise the embedding sidecar process."),
    ("code3", "code", "git rebase rewrites commit history by replaying commits onto a new base branch."),
    ("law1", "law", "The tenant may terminate the lease with 30 days written notice if the landlord fails to maintain habitable conditions."),
    ("law2", "law", "Under GDPR Article 17, a data subject has the right to erasure of personal data without undue delay."),
    ("law3", "law", "A non-compete clause restricts an employee from joining competitors for a defined period after leaving."),
    ("med1", "med", "Nabumetone is an NSAID used to treat pain and inflammation caused by osteoarthritis and rheumatoid arthritis."),
    ("med2", "med", "Type 2 diabetes management combines metformin, dietary changes, and regular HbA1c monitoring."),
    ("med3", "med", "Amoxicillin is a penicillin-class antibiotic prescribed for bacterial infections such as strep throat."),
    ("home1", "home_office", "Electricity invoice for March: 412 kWh consumed, total due 87.40 EUR, payment deadline April 15."),
    ("home2", "home_office", "Your car insurance policy renews on August 1; the annual premium is 640 EUR with a 300 EUR deductible."),
    ("home3", "home_office", "Boarding pass: flight LH1234 from Frankfurt to Lisbon departs 09:35, gate A22, seat 14C."),
    ("web1", "unstructured", "The Eiffel Tower was completed in 1889 as the entrance arch to the World's Fair in Paris."),
    ("web2", "unstructured", "Photosynthesis converts carbon dioxide and water into glucose and oxygen using light energy in chloroplasts."),
    ("photo1", "unstructured", "A family photo from a birthday party in the garden; three children smiling around a chocolate cake."),
    ("chart1", "unstructured", "Bar chart of quarterly revenue: Q1 1.2M, Q2 1.5M, Q3 1.4M, Q4 2.1M, showing a strong year-end peak."),
]

QUERIES = [
    ("how does binary search work", "code1", "code"),
    ("right to be forgotten under EU data protection law", "law2", "law"),
    ("what drug treats osteoarthritis inflammation", "med1", "med"),
    ("how much do I owe on the March electricity bill", "home1", "home_office"),
    ("when was the Eiffel Tower built", "web1", "unstructured"),
    ("how do plants make oxygen", "web2", "unstructured"),
    ("picture of kids at a birthday celebration", "photo1", "unstructured"),
    ("which quarter had the highest revenue", "chart1", "unstructured"),
    ("antibiotic for strep throat", "med3", "med"),
    ("can I quit my lease if the apartment is uninhabitable", "law1", "law"),
]

TRIPLETS = [  # margin = cos(a,b) - cos(a,c)
    ("cat", "kitten", "spreadsheet"),
    ("car", "automobile", "banana"),
    ("doctor", "physician", "guitar"),
    ("invoice", "bill", "sunset"),
]

TASK = {
    "code": "Given a natural language query, retrieve relevant code snippets or technical documentation",
    "law": "Given a legal question, retrieve relevant statutes, clauses, or case passages",
    "med": "Given a clinical or medical question, retrieve relevant medical literature or notes",
    "home_office": "Given a query about personal or administrative documents, retrieve relevant records",
    "unstructured": "Given a web search query, retrieve relevant passages that answer the query",
}

FMT = {  # mode -> (doc_fmt(text), query_fmt(q, domain))
    "raw": (lambda t: t, lambda q, d: q),
    "nomic": (lambda t: "search_document: " + t, lambda q, d: "search_query: " + q),
    "qwen3": (lambda t: t, lambda q, d: "Instruct: %s\nQuery: %s" % (TASK[d], q)),
    "egemma": (lambda t: "title: none | text: " + t, lambda q, d: "task: search result | query: " + q),
}


def embed(base, texts, timeout=120):
    body = json.dumps({"input": texts, "model": "embed"}).encode()
    t0 = time.time()
    req = urllib.request.Request(base + "/v1/embeddings", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    wall = time.time() - t0
    data = sorted(out["data"], key=lambda d: d["index"])
    return [d["embedding"] for d in data], wall


def cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def run(name, base, mode):
    dfmt, qfmt = FMT[mode]
    dvecs, dwall = embed(base, [dfmt(t) for _, _, t in DOCS])
    qvecs, qwall = embed(base, [qfmt(q, dom) for q, _, dom in QUERIES])
    hit1 = hit3 = 0; mrr = 0.0; ranks = []
    for qi, (q, gold, dom) in enumerate(QUERIES):
        sims = sorted(((cos(qvecs[qi], dvecs[di]), DOCS[di][0]) for di in range(len(DOCS))), reverse=True)
        rank = next(i + 1 for i, (_, did) in enumerate(sims) if did == gold)
        ranks.append((q, gold, rank, round(sims[0][0], 4), sims[0][1]))
        hit1 += rank == 1; hit3 += rank <= 3; mrr += 1.0 / rank
    tvecs, _ = embed(base, [dfmt(w) for tr in TRIPLETS for w in tr])
    margins = [cos(tvecs[i*3], tvecs[i*3+1]) - cos(tvecs[i*3], tvecs[i*3+2]) for i in range(len(TRIPLETS))]
    n = len(QUERIES)
    return {
        "config": name, "mode": mode, "dims": len(dvecs[0]),
        "hit1": hit1 / n, "hit3": hit3 / n, "mrr": round(mrr / n, 4),
        "margin_mean": round(sum(margins) / len(margins), 4),
        "margins": [round(m, 4) for m in margins],
        "ms_per_doc": round(1000 * dwall / len(DOCS), 1),
        "ms_per_query": round(1000 * qwall / n, 1),
        "ranks": ranks,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nomic"); ap.add_argument("--qwen"); ap.add_argument("--egemma")
    ap.add_argument("--out", default="embed_ab_results.json")
    a = ap.parse_args()
    plan = []
    if a.nomic: plan += [("nomic-raw", a.nomic, "raw"), ("nomic-prefixed", a.nomic, "nomic")]
    if a.qwen: plan += [("qwen3mean-raw", a.qwen, "raw"), ("qwen3mean-instr", a.qwen, "qwen3")]
    if a.egemma: plan += [("egemma-raw", a.egemma, "raw"), ("egemma-prompted", a.egemma, "egemma")]
    results = []
    print("config\tdims\thit@1\thit@3\tmrr\tmargin\tms/doc\tms/query")
    for name, base, mode in plan:
        try:
            r = run(name, base, mode)
        except Exception as e:
            r = {"config": name, "error": "%s: %s" % (type(e).__name__, e)}
            print("%s\tERROR: %s" % (name, r["error"]))
            results.append(r); continue
        print("%s\t%d\t%.2f\t%.2f\t%.3f\t%+.3f\t%.0f\t%.0f" % (
            r["config"], r["dims"], r["hit1"], r["hit3"], r["mrr"],
            r["margin_mean"], r["ms_per_doc"], r["ms_per_query"]))
        results.append(r)
    with open(a.out, "w") as f:
        json.dump({"fixture": {"docs": len(DOCS), "queries": len(QUERIES),
                               "triplets": len(TRIPLETS)}, "results": results}, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
