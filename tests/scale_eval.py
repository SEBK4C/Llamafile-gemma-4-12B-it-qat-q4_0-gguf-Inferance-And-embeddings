"""Evaluate an embedding recipe (template config × server pooling) on the
32-item scale corpus (tests/scale_corpus.py — run that first).

    python3 tests/scale_eval.py --config instr-trail [--mods image,audio]
    python3 tests/scale_eval.py --ocr-check          # legibility spot-check

Reports per modality: retrieval@1 and @5 (media query → 32 text candidates,
both directions), mean same-topic cross-modal sim, mean cross-topic
same-modality sim, and the success-criterion margin. Success per
MM-prompt.md: retrieval well above chance (chance@1 = 1/32) and
within-topic cross-modal sim > cross-topic same-modality sim.
"""

import argparse
import json
import urllib.request

import modality_gap as mg
from scale_corpus import ASSETS, SENTENCES
from template_sweep import CONFIGS


def post(url, path, obj, timeout=600):
    req = urllib.request.Request(url + path, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def embed(url, payload):
    return post(url, "/v1/embeddings", {"input": payload})["data"][0]["embedding"]


def ocr_check(url, n=5):
    for i in range(n):
        r = post(url, "/v1/chat/completions", {
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Read the text in this image. Reply with ONLY the text."},
                {"type": "image_url", "image_url": {"url":
                    "data:image/png;base64," + mg.b64(f"{ASSETS}/{i:02d}.png")}},
            ]}]})
        msg = r["choices"][0]["message"]
        # Gemma 4's thinking channel can swallow the whole budget — accept the
        # transcription from either field (it usually appears inside the thought)
        got = " ".join(((msg.get("content") or "") + " " +
                        (msg.get("reasoning_content") or "")).split())
        want = SENTENCES[i].lower()
        ok = want in got.lower() or all(w in got.lower() for w in want.split())
        print(f"[{i:02d}] {'OK ' if ok else 'FAIL'} expect: {SENTENCES[i]}\n          got: {got[:160]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--config", default="baseline")
    ap.add_argument("--mods", default="image,audio")
    ap.add_argument("--img-dir", default=None,
                    help="directory with NN.png renders (default: the legacy "
                         "224px assets dir); audio always comes from ASSETS")
    ap.add_argument("--ocr-check", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    img_dir = args.img_dir or ASSETS
    if args.ocr_check:
        ocr_check(args.url)
        return

    cfg = CONFIGS[args.config]
    marker = json.loads(urllib.request.urlopen(args.url + "/props", timeout=10).read())["media_marker"]
    mods = args.mods.split(",")
    n = len(SENTENCES)

    T = [mg.norm(embed(args.url, cfg["t_text"].format(text=s))) for s in SENTENCES]
    print(f"embedded {n} texts", flush=True)
    M = {}
    for mod in mods:
        tmpl = cfg["t_image" if mod == "image" else "t_audio"]
        M[mod] = []
        for i in range(n):
            fpath = (f"{img_dir}/{i:02d}.png" if mod == "image"
                     else f"{ASSETS}/{i:02d}.wav")
            M[mod].append(mg.norm(embed(args.url, {
                "prompt_string": tmpl.format(marker=marker),
                "multimodal_data": [mg.b64(fpath)]})))
            if (i + 1) % 8 == 0:
                print(f"embedded {i + 1}/{n} {mod}", flush=True)

    out = {"config": args.config, "n": n}
    for mod in mods:
        sims = [[mg.cos(M[mod][i], T[j]) for j in range(n)] for i in range(n)]
        at1 = sum(max(range(n), key=lambda j: sims[i][j]) == i for i in range(n))
        at5 = sum(i in sorted(range(n), key=lambda j: -sims[i][j])[:5] for i in range(n))
        # reverse direction: text query → media candidates
        r_at1 = sum(max(range(n), key=lambda j: sims[j][i]) == i for i in range(n))
        same_topic = sum(sims[i][i] for i in range(n)) / n
        # cross-topic same-modality
        block = [mg.cos(M[mod][i], M[mod][j]) for i in range(n) for j in range(i + 1, n)]
        x_block = sum(block) / len(block)
        margin = min(sims[i][i] for i in range(n)) - max(block)
        out[mod] = {"r@1": at1, "r@5": at5, "rev_r@1": r_at1,
                    "mean_xmodal_same": round(same_topic, 4),
                    "mean_block": round(x_block, 4),
                    "max_block": round(max(block), 4),
                    "min_xmodal_same": round(min(sims[i][i] for i in range(n)), 4),
                    "margin": round(margin, 4)}
        print(f"{mod}: media→text r@1={at1}/{n} r@5={at5}/{n} text→media r@1={r_at1}/{n} "
              f"same-topic-xmod={same_topic:.3f} cross-topic-block={x_block:.3f} "
              f"margin={margin:+.3f}", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
