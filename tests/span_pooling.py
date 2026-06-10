"""WS2: span-restricted pooling, researched OFFLINE — no server-side pooling
patch needed. Restart the server with pooling disabled:

    GEMMA4_NGL=0 GEMMA4_POOLING=none make serve

then the legacy /embedding endpoint ({"content": ...}, same object format as
/v1/embeddings input) returns PER-TOKEN embedding rows, and we pool arbitrary
spans here in Python:

    python3 tests/span_pooling.py [--config baseline] [--config instr-trail]

Span identification is token-count arithmetic: tokenize the wrapper pieces
around the marker via POST /tokenize; the media rows are the contiguous
remainder (BOS first; the marker expands to the media chunk). The same span
logic applies to the text side (content tokens vs wrapper tokens) so every
pooling variant is symmetric across the comparison.

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
    """Per-token rows from the legacy /embedding endpoint (pooling none)."""
    r = post(url, "/embedding", {"content": payload})
    # tolerate both {"embedding": rows} and [{"embedding": rows}] shapes
    if isinstance(r, list):
        r = r[0]
    rows = r["embedding"]
    assert isinstance(rows[0], list), "got a pooled vector — server not running with GEMMA4_POOLING=none?"
    return rows


POOLERS = ("mean_all", "mean_content", "mean_wrapper", "last", "last_content", "max")


def pool(rows, span, how):
    """span = (start, end) of the content rows (media chunk / raw text)."""
    s, e = span
    content = rows[s:e]
    wrapper = rows[:s] + rows[e:]
    if how == "mean_all":
        v = mg.mean(rows)
    elif how == "mean_content":
        v = mg.mean(content)
    elif how == "mean_wrapper":
        v = mg.mean(wrapper) if wrapper else mg.mean(rows)
    elif how == "last":
        v = rows[-1]
    elif how == "last_content":
        v = content[-1]
    elif how == "max":
        v = [max(c) for c in zip(*rows)]
    return mg.norm(v)


def fetch_all(url, cfg, bos):
    """Returns {(mod, topic): (rows, content_span)}."""
    marker = json.loads(urllib.request.urlopen(url + "/props", timeout=10).read())["media_marker"]
    out = {}
    for t, words in TOPICS.items():
        # text: content span = the {text} tokens inside the wrapper
        tmpl = cfg["t_text"]
        pre, _, post_s = tmpl.partition("{text}")
        rows = rows_for(url, tmpl.format(text=words))
        n_pre, n_c, n_post = n_tok(url, pre), n_tok(url, words), n_tok(url, post_s)
        if bos + n_pre + n_c + n_post != len(rows):  # boundary-merge drift: trust ends
            n_c = len(rows) - bos - n_pre - n_post
        out[("text", t)] = (rows, (bos + n_pre, bos + n_pre + n_c))
        # media: content span = the marker expansion (the media chunk rows)
        for mod, tk, fname in (("image", "t_image", f"{t}_224.png"),
                               ("audio", "t_audio", f"{t}.wav")):
            tmpl = cfg[tk]
            pre, _, post_s = tmpl.partition("{marker}")
            rows = rows_for(url, {"prompt_string": tmpl.format(marker=marker),
                                  "multimodal_data": [mg.b64(f"{ASSETS}/{fname}")]})
            n_pre, n_post = n_tok(url, pre), n_tok(url, post_s)
            s, e = bos + n_pre, len(rows) - n_post
            assert e - s > 1, f"media span arithmetic broke for {mod}/{t}: {s}..{e} of {len(rows)}"
            out[(mod, t)] = (rows, (s, e))
        print(f"fetched rows: {t} "
              f"(text={len(out[('text', t)][0])}, image={len(out[('image', t)][0])}, "
              f"audio={len(out[('audio', t)][0])} rows)", flush=True)
    return out


def detect_bos(url):
    """Row count minus token count of a bare string reveals BOS handling."""
    s = "hello world"
    return len(rows_for(url, s)) - n_tok(url, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--config", action="append", default=None,
                    help="template config name(s) from template_sweep.CONFIGS")
    ap.add_argument("--json", default=None, help="write raw metrics JSON here")
    args = ap.parse_args()
    names = args.config or ["baseline", "instr-trail"]

    bos = detect_bos(args.url)
    print(f"BOS rows detected: {bos}")

    results = {}
    for name in names:
        print(f"== {name} ==", flush=True)
        data = fetch_all(args.url, CONFIGS[name], bos)
        for how in POOLERS:
            E = {k: pool(rows, span, how) for k, (rows, span) in data.items()}
            results[f"{name}/{how}"] = mg.metrics(E)

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
