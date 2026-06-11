"""WS2: span-restricted pooling, researched OFFLINE — no server-side pooling
patch needed. Restart the server with pooling disabled:

    GEMMA4_NGL=0 GEMMA4_POOLING=none make serve

then the legacy /embedding endpoint ({"content": ...}, same object format as
/v1/embeddings input) returns per-token embedding rows, and we pool arbitrary
spans here in Python:

    python3 tests/span_pooling.py --config instr-trail --config prompteol

IMPORTANT LIMITATION discovered 2026-06-11: with pooling none the server
only returns rows for TEXT tokens — mtmd-helper marks every media-chunk
token logits=false (mtmd-helper.cpp:161/179/196), and the all-outputs
override in llama-batch.cpp only fires for pooled embeddings. So media-row
pooling (mean-over-media-rows, last-media-token) is NOT offline-testable;
what IS testable is pooling over the wrapper-text rows, which under causal
attention attend to all media rows — i.e. exactly the trailing-instruction
hypotheses. Variants, applied symmetrically to both sides:

    mean_all   mean over every returned row (media side: BOS+wrapper rows)
    mean_trail mean over rows AFTER the content (media/text) — needs a
               template with a trailing instruction
    last       last row

Caveats honoured (from MM-prompt.md):
- embd_normalize is skipped when pooling is none → we L2-normalize offline.
- patch-0001 territory: ONE input per request, no batching.
"""

import argparse
import json
import urllib.request

import modality_gap as mg
from template_sweep import CONFIGS

ASSETS = mg.ASSETS
TOPICS = mg.TOPICS


def post(url, path, obj, timeout=600):
    req = urllib.request.Request(url + path, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def n_tok(url, s):
    if not s:
        return 0
    return len(post(url, "/tokenize", {"content": s})["tokens"])


def rows_for(url, payload):
    r = post(url, "/embedding", {"content": payload})
    if isinstance(r, list):
        r = r[0]
    rows = r["embedding"]
    assert isinstance(rows[0], list), "got a pooled vector — server not running with GEMMA4_POOLING=none?"
    return rows


POOLERS = ("mean_all", "mean_trail", "last")


def pool(rows, n_trail, how):
    if how == "mean_all":
        v = mg.mean(rows)
    elif how == "mean_trail":
        if not n_trail:
            return None
        v = mg.mean(rows[-n_trail:])
    elif how == "last":
        v = rows[-1]
    return mg.norm(v)


def fetch_all(url, cfg, bos, items=None):
    """Returns {(mod, topic): (rows, n_trail)} — n_trail = trailing-wrapper rows.

    items: {key: (words, img_path, wav_path)}; defaults to the 3-topic battery.
    """
    if items is None:
        items = {t: (w, f"{ASSETS}/{t}_224.png", f"{ASSETS}/{t}.wav")
                 for t, w in TOPICS.items()}
    marker = json.loads(urllib.request.urlopen(url + "/props", timeout=10).read())["media_marker"]
    out = {}
    for t, (words, img_path, wav_path) in items.items():
        tmpl = cfg["t_text"]
        pre, _, post_s = tmpl.partition("{text}")
        rows = rows_for(url, tmpl.format(text=words))
        n_pre, n_c, n_post = n_tok(url, pre), n_tok(url, words), n_tok(url, post_s)
        if bos + n_pre + n_c + n_post != len(rows):
            print(f"  note: text/{t} row count {len(rows)} != "
                  f"{bos}+{n_pre}+{n_c}+{n_post} (tokenizer boundary merge)")
        out[("text", t)] = (rows, min(n_post, len(rows) - 1))
        for mod, tk, fpath in (("image", "t_image", img_path),
                               ("audio", "t_audio", wav_path)):
            tmpl = cfg[tk]
            pre, _, post_s = tmpl.partition("{marker}")
            rows = rows_for(url, {"prompt_string": tmpl.format(marker=marker),
                                  "multimodal_data": [mg.b64(fpath)]})
            n_pre, n_post = n_tok(url, pre), n_tok(url, post_s)
            # media rows are NOT returned (logits=false in mtmd-helper):
            # expected rows = BOS + leading wrapper + trailing wrapper
            if bos + n_pre + n_post != len(rows):
                print(f"  note: {mod}/{t} row count {len(rows)} != "
                      f"{bos}+{n_pre}+{n_post} (boundary merge)")
            out[(mod, t)] = (rows, min(n_post, len(rows) - 1))
        print(f"fetched rows: {t} (text={len(out[('text', t)][0])}, "
              f"image={len(out[('image', t)][0])}, audio={len(out[('audio', t)][0])})", flush=True)
    return out


def detect_bos(url):
    s = "hello world"
    return len(rows_for(url, s)) - n_tok(url, s)


def scale_metrics(E, keys):
    """scale_eval-style retrieval over the full corpus per modality."""
    n = len(keys)
    out = {}
    for mod in ("image", "audio"):
        sims = [[mg.cos(E[(mod, a)], E[("text", b)]) for b in keys] for a in keys]
        at1 = sum(max(range(n), key=lambda j: sims[i][j]) == i for i in range(n))
        at5 = sum(i in sorted(range(n), key=lambda j: -sims[i][j])[:5] for i in range(n))
        block = [mg.cos(E[(mod, keys[i])], E[(mod, keys[j])])
                 for i in range(n) for j in range(i + 1, n)]
        same = [sims[i][i] for i in range(n)]
        out[mod] = {"r@1": at1, "r@5": at5, "n": n,
                    "mean_xmodal_same": round(sum(same) / n, 4),
                    "mean_block": round(sum(block) / len(block), 4),
                    "margin": round(min(same) - max(block), 4)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--config", action="append", default=None)
    ap.add_argument("--scale", action="store_true",
                    help="run on the 32-item corpus (tests/scale_corpus.py)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    names = args.config or ["instr-trail", "prompteol"]

    bos = detect_bos(args.url)
    print(f"BOS rows detected: {bos}")

    items = None
    if args.scale:
        from scale_corpus import ASSETS as SCALE_ASSETS, SENTENCES
        items = {f"{i:02d}": (w, f"{SCALE_ASSETS}/{i:02d}.png", f"{SCALE_ASSETS}/{i:02d}.wav")
                 for i, w in enumerate(SENTENCES)}

    results = {}
    for name in names:
        print(f"== {name} ==", flush=True)
        data = fetch_all(args.url, CONFIGS[name], bos, items)
        for how in POOLERS:
            E = {k: pool(rows, n_trail, how) for k, (rows, n_trail) in data.items()}
            if any(v is None for v in E.values()):
                print(f"  (skipping {how}: template has no trailing wrapper)")
                continue
            if args.scale:
                r = scale_metrics(E, sorted(items))
                results[f"{name}/{how}"] = r
                for mod in ("image", "audio"):
                    d = r[mod]
                    print(f"  {how:<12} {mod}: r@1={d['r@1']}/{d['n']} r@5={d['r@5']}/{d['n']} "
                          f"xmod={d['mean_xmodal_same']:.3f} block={d['mean_block']:.3f} "
                          f"margin={d['margin']:+.3f}", flush=True)
            else:
                results[f"{name}/{how}"] = mg.metrics(E)

    if args.scale:
        if args.json:
            with open(args.json, "w") as f:
                json.dump(results, f, indent=1)
            print("wrote", args.json)
        return

    def rng(xs):
        return f"{min(xs):.2f}-{max(xs):.2f}"
    print(f"\n{'config/pooling':<24} {'xmod-img':>10} {'xmod-aud':>10} "
          f"{'blk-img':>10} {'blk-aud':>10} {'ret-raw':>8} {'ret-cor':>8} "
          f"{'mrg-img':>8} {'mrg-aud':>8}")
    for name, m in results.items():
        print(f"{name:<24} {rng(m['xmodal']['image']):>10} {rng(m['xmodal']['audio']):>10} "
              f"{rng(m['block']['image']):>10} {rng(m['block']['audio']):>10} "
              f"{m['retrieval_raw']:>6}/{m['retrieval_total']} "
              f"{m['retrieval_corrected']:>6}/{m['retrieval_total']} "
              f"{m['margin']['image']:>+8.3f} {m['margin']['audio']:>+8.3f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
