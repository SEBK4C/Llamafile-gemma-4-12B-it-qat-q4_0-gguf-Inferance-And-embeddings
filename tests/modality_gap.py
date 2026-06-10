"""Cross-modal embedding probe: the same words as text, rendered on an
image, and spoken aloud — how far apart do the three embeddings sit, and
is the modality offset consistent across topics?

Asset generation (macOS):
    uv run --with pillow python3 tests/modality_gap.py --make-assets
    (renders 224px text images; synthesizes speech via `say` + `afconvert`)

Run (server must run with multimodal enabled and, currently, -ngl 0 —
media embeddings crash the Metal backend, see README):
    python3 tests/modality_gap.py
"""

import argparse
import base64
import json
import math
import os
import subprocess
import urllib.request

ASSETS = "/tmp/modality"
TOPICS = {
    "fox":   "The quick brown fox jumps over the lazy dog",
    "pasta": "My favourite pasta recipe uses guanciale and pecorino",
    "money": "Quarterly revenue grew nine percent in the third fiscal quarter",
}


def make_assets():
    from PIL import Image, ImageDraw, ImageFont
    import textwrap
    os.makedirs(ASSETS, exist_ok=True)
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 26)
    for name, words in TOPICS.items():
        img = Image.new("RGB", (224, 224), "white")
        d = ImageDraw.Draw(img)
        y = 20
        for line in textwrap.wrap(words, width=15):
            d.text((8, y), line, fill="black", font=font)
            y += 32
        img.save(f"{ASSETS}/{name}_224.png")
        aiff = f"{ASSETS}/{name}.aiff"
        subprocess.run(["say", words, "-o", aiff], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                        aiff, f"{ASSETS}/{name}.wav"], check=True)
    print("assets written to", ASSETS)


def main(url):
    def get(path):
        return json.loads(urllib.request.urlopen(url + path, timeout=10).read())

    def embed(payload):
        req = urllib.request.Request(url + "/v1/embeddings",
            data=json.dumps({"input": payload}).encode(),
            headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=600).read())["data"][0]["embedding"]

    def b64(path):
        return base64.b64encode(open(path, "rb").read()).decode()

    marker = get("/props")["media_marker"]
    tops = list(TOPICS)
    E = {}
    for t, words in TOPICS.items():
        E[("text", t)] = embed(words)
        E[("image", t)] = embed({"prompt_string": marker,
                                 "multimodal_data": [b64(f"{ASSETS}/{t}_224.png")]})
        E[("audio", t)] = embed({"prompt_string": marker,
                                 "multimodal_data": [b64(f"{ASSETS}/{t}.wav")]})
        print("embedded", t, flush=True)

    def norm(v):
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v]

    def cos(a, b):
        return sum(x * y for x, y in zip(a, b))

    def sub(a, b):
        return [x - y for x, y in zip(a, b)]

    def mean(vs):
        return [sum(c) / len(c) for c in zip(*vs)]

    E = {k: norm(v) for k, v in E.items()}

    print("\n-- same-modality topic similarity --")
    for mod in ("text", "image", "audio"):
        pairs = [(a, b) for i, a in enumerate(tops) for b in tops[i + 1:]]
        print(f"  {mod}: " + "  ".join(f"{a}-{b}={cos(E[(mod,a)],E[(mod,b)]):.3f}" for a, b in pairs))

    print("\n-- drift d(topic) = e_modality - e_text --")
    drift = {}
    for mod in ("image", "audio"):
        d = {t: sub(E[(mod, t)], E[("text", t)]) for t in tops}
        drift[mod] = mean(list(d.values()))
        mag = [round(math.sqrt(sum(x * x for x in d[t])), 2) for t in tops]
        agree = [round(cos(norm(d[a]), norm(d[b])), 3)
                 for i, a in enumerate(tops) for b in tops[i + 1:]]
        print(f"  {mod}: |d|={mag} direction-agreement={agree}")
    print(f"  image-drift vs audio-drift: cos = "
          f"{cos(norm(drift['image']), norm(drift['audio'])):.3f}")

    print("\n-- cross-modal retrieval (nearest text) --")
    for label, correct in (("raw", False), ("gap-corrected", True)):
        hits = 0
        for mod in ("image", "audio"):
            for t in tops:
                v = norm(sub(E[(mod, t)], drift[mod])) if correct else E[(mod, t)]
                scores = {t2: cos(v, E[("text", t2)]) for t2 in tops}
                hits += max(scores, key=scores.get) == t
        print(f"  {label}: {hits}/6")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-assets", action="store_true")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    args = ap.parse_args()
    if args.make_assets:
        make_assets()
    else:
        main(args.url)
