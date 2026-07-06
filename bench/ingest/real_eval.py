#!/usr/bin/env python3
"""Real-data evals over public benchmark subsets (fetch_datasets.py):

  people  Flickr30k people photos → enrichment: people[] detection rate,
          caption-overlap rate, AND the phase-premise retrieval test —
          embed each photo's enrichment text (docs, bare) and query with a
          held-out caption (Instruct form) via the live sidecar: hit@1/MRR.
  funsd   real scanned forms → PP-OCRv6: unordered word-bag precision/
          recall/F1 vs FUNSD word GT (order-free because FUNSD GT has no
          canonical reading order).
  stt     LibriSpeech test-clean → Gemma-4 native audio transcription →
          jiwer WER (both sides normalized: lower, no punctuation).
  sounds  ESC-50 clips → "describe this sound" → category-synonym hit rate.

Only aggregate metrics + model TEXT outputs are published; the media stays
in datasets_real/ (local-only). GPU calls serialized via bench/.eval.lock.

Usage: real_eval.py all|people|funsd|stt|sounds --base URL --embed-base URL [--out J]
"""
import argparse, base64, fcntl, json, math, os, re, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from enrich import enrich, semantic_valid  # noqa: E402

DATA = os.path.join(HERE, "datasets_real")
LOCK = os.path.join(HERE, "..", ".eval.lock")

STOP = set("a an the is are was were with of in on at and or to for his her "
           "their its this that there two three some very man's it's".split())

SOUND_SYNONYMS = {
    "dog": ["dog", "bark", "barking", "puppy"],
    "rain": ["rain", "drizzle", "raining"],
    "siren": ["siren", "alarm", "ambulance", "police", "emergency"],
    "chainsaw": ["chainsaw", "saw", "engine", "motor"],
    "crying_baby": ["cry", "crying", "baby", "infant", "wail"],
    "rooster": ["rooster", "crow", "cock", "chicken"],
    "sea_waves": ["wave", "ocean", "sea", "surf", "shore", "water"],
    "clock_alarm": ["alarm", "clock", "beep", "ring"],
    "helicopter": ["helicopter", "rotor", "chopper", "propeller"],
    "thunderstorm": ["thunder", "storm", "lightning"],
    "church_bells": ["bell", "bells", "church", "chime", "toll"],
    "laughing": ["laugh", "laughing", "giggle", "chuckle", "laughter"],
}


def to_16k_mono_wav(raw):
    """F22: the server's audio path does NOT resample — anything but 16 kHz
    mono aliases into 'high-pitched electronic beep' garbage. Normalize
    client-side (linear interp; good enough for eval)."""
    import io as _io
    import numpy as np
    import soundfile as sf
    data, sr = sf.read(_io.BytesIO(raw), always_2d=True)
    mono = data.mean(axis=1)
    if sr != 16000:
        n_out = int(len(mono) * 16000 / sr)
        x_old = np.linspace(0, 1, num=len(mono), endpoint=False)
        x_new = np.linspace(0, 1, num=n_out, endpoint=False)
        mono = np.interp(x_new, x_old, mono)
    buf = _io.BytesIO()
    sf.write(buf, mono, 16000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def chat_audio(base, audio_bytes, fmt, prompt, max_tokens=512, timeout=600):
    body = json.dumps({
        "max_tokens": max_tokens, "temperature": 0, "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "input_audio", "input_audio": {
                "data": base64.b64encode(audio_bytes).decode(), "format": fmt}}]}],
    }).encode()
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return ((out.get("choices") or [{}])[0].get("message", {}).get("content") or "",
            round(time.time() - t0, 2))


def embed(base, texts, timeout=300):
    body = json.dumps({"input": texts, "model": "embed"}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/v1/embeddings", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return [d["embedding"] for d in sorted(out["data"], key=lambda d: d["index"])]


def cos(a, b):
    d = sum(x * y for x, y in zip(a, b))
    return d / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b)) or 1)


def norm_words(s):
    return [w for w in re.sub(r"[^a-z0-9 ]", " ", s.lower()).split() if w]


def eval_people(base, embed_base):
    man = json.load(open(os.path.join(DATA, "people", "manifest.json")))
    items, det, overlap = [], 0, 0
    for name, m in sorted(man.items()):
        png = open(os.path.join(DATA, "people", name), "rb").read()
        r = enrich(base, None, {"name": name, "mime": "image/jpeg"}, png)
        e = r["enrichment"] if r["parse_error"] is None else None
        ok = e is not None and semantic_valid(e)
        etext = json.dumps(e, ensure_ascii=False).lower() if ok else ""
        cap_words = set(w for c in m["captions"] for w in norm_words(c)) - STOP
        hit_words = [w for w in cap_words if w in etext]
        has_people = ok and len(e["people"]) > 0
        det += has_people
        overlap += len(hit_words) >= 3
        items.append({"img": name, "valid": ok, "people_n": len(e["people"]) if ok else 0,
                      "overlap_words": len(hit_words), "wall_s": r["wall_s"],
                      "doc_text": (e["title"] + ". " + e["summary"] + " " +
                                    (e["scene"] or "") + " " +
                                    " ".join(p["doing"] for p in e["people"])) if ok else "",
                      "query": m["captions"][0]})
        print(f"people {name} valid={ok} people_n={items[-1]['people_n']} "
              f"overlap={len(hit_words)} {r['wall_s']}s")
    # phase-premise retrieval: held-out caption -> enrichment text
    docs = [it["doc_text"] for it in items]
    qs = ["Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: " + it["query"] for it in items]
    dv = embed(embed_base, docs)
    qv = embed(embed_base, qs)
    hit1, mrr = 0, 0.0
    for qi in range(len(qs)):
        order = sorted(range(len(docs)), key=lambda di: -cos(qv[qi], dv[di]))
        rank = order.index(qi) + 1
        hit1 += rank == 1
        mrr += 1 / rank
        items[qi]["retrieval_rank"] = rank
    n = len(items)
    s = {"n": n, "people_detected_rate": round(det / n, 3),
         "caption_overlap_rate": round(overlap / n, 3),
         "retrieval_hit1": round(hit1 / n, 3), "retrieval_mrr": round(mrr / n, 3)}
    print("people summary", json.dumps(s))
    return {"summary": s, "items": items}


def eval_funsd():
    from ocr import make_engine, extract
    engine = make_engine()
    man = json.load(open(os.path.join(DATA, "funsd", "manifest.json")))
    items, f1s = [], []
    for name, m in sorted(man.items()):
        r = extract(engine, os.path.join(DATA, "funsd", name))
        got = norm_words(" ".join(r["lines"]))
        want = norm_words(" ".join(m["gt_words"]))
        from collections import Counter
        cg, cw = Counter(got), Counter(want)
        inter = sum((cg & cw).values())
        p = inter / max(1, sum(cg.values()))
        rec = inter / max(1, sum(cw.values()))
        f1 = 2 * p * rec / max(1e-9, p + rec)
        f1s.append(f1)
        items.append({"img": name, "precision": round(p, 3), "recall": round(rec, 3),
                      "f1": round(f1, 3), "ms": r["ms"], "gt_words": len(want)})
        print(f"funsd {name} P={p:.3f} R={rec:.3f} F1={f1:.3f} {r['ms']:.0f}ms")
    s = {"n": len(items), "mean_f1": round(sum(f1s) / len(f1s), 3)}
    print("funsd summary", json.dumps(s))
    return {"summary": s, "items": items}


def eval_stt(base):
    import jiwer
    man = json.load(open(os.path.join(DATA, "speech", "manifest.json")))
    items, wers = [], []
    tr = lambda s: " ".join(norm_words(s))
    for name, m in sorted(man.items()):
        raw = open(os.path.join(DATA, "speech", name), "rb").read()
        fmt = "wav" if name.endswith(".wav") else "mp3"
        got, wall = chat_audio(base, raw, fmt,
                               "Transcribe this audio exactly. Output only the transcription.")
        w = jiwer.wer(tr(m["text"]), tr(got))
        wers.append(w)
        items.append({"utt": name, "wer": round(w, 3), "wall_s": wall,
                      "ref": m["text"], "hyp": got})
        print(f"stt {name} wer={w:.3f} {wall}s")
    s = {"n": len(items), "mean_wer": round(sum(wers) / len(wers), 3),
         "median_wer": round(sorted(wers)[len(wers) // 2], 3)}
    print("stt summary", json.dumps(s))
    return {"summary": s, "items": items}


def eval_sounds(base):
    man = json.load(open(os.path.join(DATA, "sounds", "manifest.json")))
    items, hits = [], 0
    for name, m in sorted(man.items()):
        raw = to_16k_mono_wav(open(os.path.join(DATA, "sounds", name), "rb").read())
        got, wall = chat_audio(base, raw, "wav",
                               "Describe the sound in this audio clip in one short sentence.")
        syn = SOUND_SYNONYMS[m["category"]]
        hit = any(s in got.lower() for s in syn)
        hits += hit
        items.append({"clip": name, "category": m["category"], "hit": hit,
                      "wall_s": wall, "description": got})
        print(f"sounds {name} hit={hit} :: {got[:70]}")
    s = {"n": len(items), "category_hit_rate": round(hits / len(items), 3)}
    print("sounds summary", json.dumps(s))
    return {"summary": s, "items": items}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["all", "people", "funsd", "stt", "sounds"])
    ap.add_argument("--base"); ap.add_argument("--embed-base")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = {}
    with open(LOCK, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            if a.which in ("all", "people"):
                out["people"] = eval_people(a.base, a.embed_base)
            if a.which in ("all", "funsd"):
                out["funsd"] = eval_funsd()
            if a.which in ("all", "stt"):
                out["stt"] = eval_stt(a.base)
            if a.which in ("all", "sounds"):
                out["sounds"] = eval_sounds(a.base)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
