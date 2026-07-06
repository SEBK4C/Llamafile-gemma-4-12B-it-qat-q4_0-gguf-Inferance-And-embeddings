#!/usr/bin/env python3
"""I9 — the PHASE GATE: hybrid retrieval over ingest.v1 envelopes.

Store = SQLite FTS5 (real BM25, stdlib, zero new services) + a numpy dense
matrix (doc-summary + chunk vectors from the live sidecar). Query side =
BM25-only vs dense-only vs Reciprocal-Rank-Fusion hybrid, scored on a
FROZEN query set with known-correct targets drawn from the 87-envelope
VARIETY population:

  Flickr photos   query = held-out caption            → that photo
  CORD receipts   query = "receipt total <X> [item]"  → that receipt
  fixtures        hand-written fact queries           → that document
  speech          distinctive transcript phrase       → that audio doc

Deployment note: FTS5+numpy is the MEASUREMENT rig; swapping in Qdrant/
pgvector later changes ops, not the contract (envelopes carry everything).

Usage:
  hybrid_store.py build  --envelopes DIR --db store.sqlite --embed-base URL
  hybrid_store.py eval   --db store.sqlite --embed-base URL [--out J]
"""
import argparse, json, math, os, sqlite3, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from embed_real_bench import embed  # noqa: E402

INSTR = ("Instruct: Given a web search query, retrieve relevant passages "
         "that answer the query\nQuery: ")


def doc_text_of(env):
    e = env.get("enrichment") or {}
    bits = [e.get("title") or "", e.get("summary") or "",
            e.get("scene") or "", " ".join(e.get("entities") or []),
            " ".join(p.get("doing", "") for p in e.get("people") or [])]
    return ". ".join(b for b in bits if b)


def build(envdir, db_path, embed_base):
    import numpy as np
    envs = []
    for n in sorted(os.listdir(envdir)):
        if n.endswith(".json"):
            envs.append(json.load(open(os.path.join(envdir, n))))
    print(f"{len(envs)} envelopes")
    if os.path.exists(db_path):
        os.remove(db_path)
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE docs (rowid_ INTEGER PRIMARY KEY, name TEXT, sha TEXT, source_type TEXT, payload TEXT)")
    con.execute("CREATE VIRTUAL TABLE fts USING fts5(name, body)")
    rows, texts = [], []
    for i, env in enumerate(envs):
        name = env["file"]["name"]
        dtext = doc_text_of(env)
        chunk_txt = " ".join(c.get("text", "") for c in env.get("chunks") or [])
        body = (dtext + " " + (env.get("transcript") or "") + " " + chunk_txt).strip()
        con.execute("INSERT INTO docs VALUES (?,?,?,?,?)",
                    (i, name, env["file"].get("sha256") or env["file"].get("text_hash_fnv64"),
                     env["source_type"], json.dumps({"title": (env.get("enrichment") or {}).get("title")})))
        con.execute("INSERT INTO fts(rowid, name, body) VALUES (?,?,?)", (i, name, body))
        texts.append(dtext if dtext else body[:1500])
        rows.append(name)
    con.commit()
    vecs = embed(embed_base, texts)
    M = np.asarray(vecs, dtype=np.float32)
    M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    np.save(db_path + ".npy", M)
    json.dump(rows, open(db_path + ".names.json", "w"))
    print(f"indexed {len(rows)} docs → {db_path} (+ .npy dense)")
    con.close()


def frozen_queries():
    q = []
    ppl = json.load(open(os.path.join(HERE, "datasets_real", "people", "manifest.json")))
    for name, m in sorted(ppl.items()):
        q.append((m["captions"][0], name))
    cord = json.load(open(os.path.join(HERE, "datasets_real", "cord", "manifest.json")))
    for name, m in sorted(cord.items())[:10]:
        extra = (" " + m["menu_names"][0]) if m.get("menu_names") else ""
        q.append((f"receipt with total {m['total']}{extra}", name))
    q += [
        ("GridPower invoice paid by SEPA transfer", "pdf_text.pdf"),
        ("move the backup job and verify ZFS snapshots", "notes.md"),
        ("python function that builds a health check url", "script.py"),
        ("PharmaCity purchase amount", "table.csv"),
        ("pipeline that converts archive files into enriched text", "doc_page.png"),
        ("homelab supply receipt with USB NIC and NVMe", "receipt.png"),
        ("bar chart of quarterly revenue with year end peak", "chart_bars.png"),
        ("they were absorbed in his theology", "utt_00.wav"),
    ]
    return q


def evaluate(db_path, embed_base, out_path):
    import numpy as np
    con = sqlite3.connect(db_path)
    names = json.load(open(db_path + ".names.json"))
    M = np.load(db_path + ".npy")
    name_to_idx = {}
    for i, n in enumerate(names):
        name_to_idx.setdefault(n, i)
    queries = [(qt, tgt) for qt, tgt in frozen_queries() if tgt in name_to_idx]
    print(f"{len(queries)} frozen queries with targets present")

    qv = embed(embed_base, [INSTR + qt for qt, _ in queries])
    QM = np.asarray(qv, dtype=np.float32)
    QM /= np.linalg.norm(QM, axis=1, keepdims=True) + 1e-9
    S = QM @ M.T

    def fts_rank(qtext, k=50):
        toks = [t for t in "".join(c if c.isalnum() else " " for c in qtext).split() if len(t) > 1]
        if not toks:
            return []
        match = " OR ".join(toks)
        try:
            cur = con.execute("SELECT rowid FROM fts WHERE fts MATCH ? ORDER BY bm25(fts) LIMIT ?",
                              (match, k))
            return [r[0] for r in cur.fetchall()]
        except sqlite3.OperationalError:
            return []

    def has_exact_tokens(qtext):
        # digits, ids, amounts — the class BM25 is kept for (EM6 router)
        return any(ch.isdigit() for ch in qtext)

    def ranked(mode, qi, qt):
        bm = fts_rank(qt)
        dn = list(np.argsort(-S[qi]))
        if mode == "bm25":
            return bm or dn
        if mode == "dense":
            return dn
        if mode.startswith("rrf"):  # rrf[:wd:wb:k]
            _, wd, wb, k = mode.split(":")
            wd, wb, k = float(wd), float(wb), int(k)
            sc = {}
            for r, d in enumerate(bm):
                sc[d] = sc.get(d, 0) + wb / (k + r + 1)
            for r, d in enumerate(dn[:100]):
                sc[d] = sc.get(d, 0) + wd / (k + r + 1)
            return sorted(sc, key=sc.get, reverse=True)
        if mode == "router":  # EM6: dense-first; fuse BM25 only on exact-token queries
            if not has_exact_tokens(qt):
                return dn
            sc = {}
            for r, d in enumerate(bm):
                sc[d] = sc.get(d, 0) + 1.0 / (60 + r + 1)
            for r, d in enumerate(dn[:100]):
                sc[d] = sc.get(d, 0) + 3.0 / (60 + r + 1)
            return sorted(sc, key=sc.get, reverse=True)
        raise ValueError(mode)

    res = {}
    for mode in ("bm25", "dense", "rrf:1:1:60", "rrf:2:1:60", "rrf:3:1:60",
                 "rrf:5:1:60", "rrf:3:1:20", "router"):
        hit1 = hit3 = 0
        mrr = 0.0
        for qi, (qt, tgt) in enumerate(queries):
            order = ranked(mode, qi, qt)
            t = name_to_idx[tgt]
            rank = order.index(t) + 1 if t in order else len(names) + 1
            hit1 += rank == 1
            hit3 += rank <= 3
            mrr += 1.0 / rank
        n = len(queries)
        res[mode] = {"hit@1": round(hit1 / n, 3), "hit@3": round(hit3 / n, 3),
                      "mrr": round(mrr / n, 3)}
        print(f"{mode:12s} hit@1={res[mode]['hit@1']:.3f} hit@3={res[mode]['hit@3']:.3f} mrr={res[mode]['mrr']:.3f}")
    if out_path:
        json.dump({"n_queries": len(queries), "n_docs": len(names), "modes": res},
                  open(out_path, "w"), indent=1)
        print("wrote", out_path)
    con.close()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "eval"])
    ap.add_argument("--envelopes"); ap.add_argument("--db", required=True)
    ap.add_argument("--embed-base", required=True); ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.envelopes, a.db, a.embed_base)
    else:
        evaluate(a.db, a.embed_base, a.out)


if __name__ == "__main__":
    main()
