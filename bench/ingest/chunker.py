#!/usr/bin/env python3
"""I6 chunker: enrichment-hint-guided logical chunks for the ingest.v1
envelope (bench/phase3-ingest-program.md).

Splitting hierarchy (never cut mid-sentence or mid-line):
  1. atomic units = markdown headers / blank-line paragraphs / OCR line
     groups; oversized units split at sentence ends only
  2. enrichment chunking_hints whose labels literally occur in the text
     become hard section boundaries (labels are advisory otherwise —
     they seed chunk labels)
  3. greedy packing into TARGET tokens (default 512, hard max 1024,
     runts < MIN merged forward), optional 1-unit overlap
  4. CSV: header row is prepended to every chunk so each chunk is
     self-describing; rows are never split

Token counts use the LIVE embedder's own /tokenize (exact Qwen3 counts;
chars/4 fallback offline). Doc-level summary text = title + summary +
entities (the hierarchical doc vector input).

Usage:
  chunker.py FILE.txt [--enrichment E.json]     chunk one text
  chunker.py --bench --embed-base URL           fixture battery + retrieval
"""
import argparse, json, os, re, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

TARGET, HARD_MAX, MIN_CHUNK = 512, 1024, 128

_TOK_BASE = None
_TOK_WARNED = [False]


def n_tokens(text):
    """Exact count via the live embedder's /tokenize; chars/4 fallback with a
    ONE-TIME warning (silent fallback under-counts and defeats HARD_MAX —
    the I6 bug). Callers tokenize each unit ONCE and pack by summed counts:
    deterministic, O(n) HTTP calls, and summing over-estimates joins (BPE
    merges across boundaries only shrink), which is the safe direction."""
    if _TOK_BASE:
        try:
            body = json.dumps({"content": text}).encode()
            req = urllib.request.Request(_TOK_BASE.rstrip("/") + "/tokenize", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return len(json.loads(r.read())["tokens"])
        except Exception as e:
            if not _TOK_WARNED[0]:
                _TOK_WARNED[0] = True
                print(f"WARN n_tokens fallback (chars/4): {type(e).__name__}: {e}",
                      file=sys.stderr)
    return max(1, len(text) // 4)


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split_sentences(text):
    return [s for s in _SENT.split(text) if s.strip()]


def atomic_units(text):
    """Markdown-header/paragraph units; oversized ones split at sentences."""
    units = []
    block = []
    for line in text.splitlines():
        if re.match(r"^#{1,6} ", line) and block:
            units.append("\n".join(block)); block = [line]
        elif line.strip() == "":
            if block:
                units.append("\n".join(block)); block = []
        else:
            block.append(line)
    if block:
        units.append("\n".join(block))
    out = []  # list of (text, tok) — every unit tokenized exactly once
    for u in units:
        t = n_tokens(u)
        if t <= HARD_MAX:
            out.append((u, t))
            continue
        # oversized block: split at sentences; bullet lists may have no
        # regex-visible sentence boundaries ('\n- …') → pack whole LINES
        sents = split_sentences(u)
        joiner = " "
        if len(sents) < 2:
            sents, joiner = u.splitlines(), "\n"
        cur, cur_t = [], 0
        for s in sents:
            st = n_tokens(s)
            if cur and cur_t + st > TARGET:
                out.append((joiner.join(cur), cur_t)); cur, cur_t = [], 0
            cur.append(s); cur_t += st
        if cur:
            out.append((joiner.join(cur), cur_t))
    return out


def chunk_text(text, enrichment=None, source_type=None, overlap=False):
    if source_type == "csv":
        lines = [l for l in text.splitlines() if l.strip()]
        header, rows = lines[0], lines[1:]
        chunks, cur = [], []
        for row in rows:
            cur.append(row)
            if n_tokens("\n".join([header] + cur)) >= TARGET:
                chunks.append({"text": "\n".join([header] + cur), "label": "rows"})
                cur = []
        if cur:
            chunks.append({"text": "\n".join([header] + cur), "label": "rows"})
    else:
        hints = [(h.get("label") or "").strip()
                 for h in (enrichment or {}).get("chunking_hints", [])]
        hint_rx = [re.compile(r"^\s*#{0,6}\s*" + re.escape(h), re.I) for h in hints if h]
        units = atomic_units(text)  # [(text, tok)] — packing sums counts
        chunks, cur, cur_t, cur_label = [], [], 0, None
        for u, ut in units:
            hinted = next((hints[i] for i, rx in enumerate(hint_rx)
                           if rx.match(u.splitlines()[0])), None)
            if cur and (hinted or cur_t + ut > TARGET):
                chunks.append({"text": "\n\n".join(x for x, _ in cur),
                                "tok": cur_t, "label": cur_label})
                cur = [cur[-1]] if overlap and not hinted else []
                cur_t = sum(t for _, t in cur)
                cur_label = None
            if hinted:
                cur_label = hinted
            cur.append((u, ut)); cur_t += ut
        if cur:
            chunks.append({"text": "\n\n".join(x for x, _ in cur),
                            "tok": cur_t, "label": cur_label})
        # merge runts forward (summed counts; ≤ HARD_MAX guaranteed)
        merged = []
        for c in chunks:
            if merged and c["tok"] < MIN_CHUNK and \
                    merged[-1]["tok"] + c["tok"] <= HARD_MAX:
                merged[-1]["text"] += "\n\n" + c["text"]
                merged[-1]["tok"] += c["tok"]
                merged[-1]["label"] = merged[-1]["label"] or c["label"]
            else:
                merged.append(c)
        chunks = merged

    out = []
    for i, c in enumerate(chunks):
        # exact re-tokenize once per FINAL chunk (cheap: n_chunks calls)
        t = n_tokens(c["text"])
        out.append({"id": f"c{i}", "label": c.get("label") or f"part {i+1}",
                    "tokens": t, "text": c["text"]})
    return out


def doc_summary_text(enrichment):
    e = enrichment or {}
    bits = [e.get("title") or "", e.get("summary") or "",
            " ".join(e.get("entities") or [])]
    return ". ".join(b for b in bits if b)


def mid_sentence_cuts(chunks):
    """Real boundary metric: a cut is bad iff the sentence CONTINUES across
    the chunk border — previous chunk ends without terminal punctuation AND
    the next chunk starts lowercase. (Cuts only ever happen at unit/
    sentence/line boundaries by construction, so ends-without-period on a
    final bullet line are fine.)"""
    bad = 0
    for a, b in zip(chunks, chunks[1:]):
        prev = a["text"].rstrip()
        nxt = b["text"].lstrip()
        if prev and nxt and prev[-1] not in ".!?:;\"')]" and not prev[-1].isdigit() \
                and nxt[0].islower():
            bad += 1
    return bad


def bench(embed_base):
    global _TOK_BASE
    _TOK_BASE = embed_base
    import math
    from ocr import make_engine, extract

    docs = {}
    docs["research_history.md"] = (open(os.path.join(HERE, "..", "RESEARCH_HISTORY.md")).read(), None, None)
    docs["doc_page"] = (extract(make_engine(), os.path.join(HERE, "fixtures", "doc_page.png"))["text"], None, None)
    docs["table.csv"] = (open(os.path.join(HERE, "fixtures_router", "table.csv")).read(), None, "csv")
    enr = {"chunking_hints": [{"label": "Findings log", "reason": ""},
                               {"label": "Experiments log", "reason": ""}]}
    docs["history_hinted"] = (docs["research_history.md"][0], enr, None)

    all_stats, viol, oob = {}, 0, 0
    total_chunks = 0
    for name, (text, e, st) in docs.items():
        chunks = chunk_text(text, e, st)
        total_chunks += len(chunks)
        v = mid_sentence_cuts(chunks)
        o = sum(1 for c in chunks if not (MIN_CHUNK <= c["tokens"] <= HARD_MAX)) - \
            (1 if chunks and chunks[-1]["tokens"] < MIN_CHUNK else 0)  # last runt allowed
        viol += v; oob += max(0, o)
        hdr_ok = all(c["text"].startswith("date,") for c in chunks) if st == "csv" else None
        all_stats[name] = {"n_chunks": len(chunks), "boundary_violations": v,
                            "out_of_bounds": max(0, o), "csv_header_ok": hdr_ok,
                            "tokens": [c["tokens"] for c in chunks],
                            "labels": [c["label"] for c in chunks]}
        print(f"{name}: {len(chunks)} chunks, tok={all_stats[name]['tokens']}, "
              f"viol={v}, oob={max(0,o)}, labels={all_stats[name]['labels'][:4]}")

    # e2e retrieval sanity vs the LIVE sidecar: fact queries must hit the
    # chunk that contains them (self-retrieval over history chunks)
    chunks = chunk_text(docs["research_history.md"][0])
    texts = [c["text"] for c in chunks]

    def embed(base, xs):
        got = []
        for i in range(0, len(xs), 16):
            body = json.dumps({"input": xs[i:i+16], "model": "e"}).encode()
            rq = urllib.request.Request(base.rstrip("/") + "/v1/embeddings", data=body,
                                        headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(rq, timeout=300) as r:
                out = json.loads(r.read())
            got += [d["embedding"] for d in sorted(out["data"], key=lambda d: d["index"])]
        return got

    def cos(a, b):
        return sum(x*y for x, y in zip(a, b)) / (math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(x*x for x in b)) or 1)

    queries = [
        ("what fixed the greedy repetition loop", "DRY"),
        ("which system prompt was validated as the WebUI default", "Constitution"),
        ("how fast is the prompt cache warm hit", "185"),
    ]
    dv = embed(embed_base, texts)
    hits = 0
    qres = []
    for q, marker in queries:
        qv = embed(embed_base, ["Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: " + q])[0]
        order = sorted(range(len(texts)), key=lambda i: -cos(qv, dv[i]))
        hit = any(marker.lower() in texts[i].lower() for i in order[:3])
        hits += hit
        qres.append({"q": q, "marker": marker, "hit_top3": hit, "top_chunk": order[0]})
        print(f"retrieval: {q!r} -> top3-contains-{marker}: {hit}")

    summary = {"docs": all_stats, "total_chunks": total_chunks,
               "boundary_violations": viol, "out_of_bounds": oob,
               "self_retrieval_hits": hits, "self_retrieval_n": len(queries),
               "retrieval": qres}
    print("summary", json.dumps({k: summary[k] for k in
          ("total_chunks", "boundary_violations", "out_of_bounds", "self_retrieval_hits")}))
    return summary


def main():
    global _TOK_BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?")
    ap.add_argument("--enrichment"); ap.add_argument("--source-type")
    ap.add_argument("--embed-base"); ap.add_argument("--bench", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.bench:
        s = bench(a.embed_base)
        if a.out:
            json.dump(s, open(a.out, "w"), indent=1)
            print("wrote", a.out)
        return 0
    _TOK_BASE = a.embed_base
    e = json.load(open(a.enrichment)) if a.enrichment else None
    for c in chunk_text(open(a.file).read(), e, a.source_type):
        print(json.dumps({**c, "text": c["text"][:80] + "…"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
