"""Cross-modal embedding probe: the same words as text, rendered on an
image, and spoken aloud — how far apart do the three embeddings sit, and
is the modality offset consistent across topics?

Asset generation (macOS):
    uv run --with pillow python3 tests/modality_gap.py --make-assets
    (renders 224px text images; synthesizes speech via `say` + `afconvert`)

Run (server must run with multimodal enabled and, currently, -ngl 0 —
media embeddings crash the Metal backend, see README):
    python3 tests/modality_gap.py

Prompted-embedding templates (WS1): wrap the media marker / text in an
instruction before pooling. Placeholders: {marker} for media, {text} for
text. The SAME kind of wrapper should be applied on both sides of the
comparison, e.g.:
    python3 tests/modality_gap.py \
        --template-image 'read the text in this image: {marker}' \
        --template-audio 'transcribe this audio: {marker}' \
        --template-text  'read this text: {text}'
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


def norm(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def cos(a, b):
    return sum(x * y for x, y in zip(a, b))


def sub(a, b):
    return [x - y for x, y in zip(a, b)]


def mean(vs):
    return [sum(c) / len(c) for c in zip(*vs)]


def b64(path):
    return base64.b64encode(open(path, "rb").read()).decode()


def embed_battery(url, t_text="{text}", t_image="{marker}", t_audio="{marker}",
                  topics=TOPICS, verbose=True):
    """Embed every topic through all three modalities; returns {(mod, topic): unit vec}."""
    def get(path):
        return json.loads(urllib.request.urlopen(url + path, timeout=10).read())

    def embed(payload):
        req = urllib.request.Request(url + "/v1/embeddings",
            data=json.dumps({"input": payload}).encode(),
            headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=600).read())["data"][0]["embedding"]

    marker = get("/props")["media_marker"]
    E = {}
    for t, words in topics.items():
        E[("text", t)] = embed(t_text.format(text=words))
        E[("image", t)] = embed({"prompt_string": t_image.format(marker=marker),
                                 "multimodal_data": [b64(f"{ASSETS}/{t}_224.png")]})
        E[("audio", t)] = embed({"prompt_string": t_audio.format(marker=marker),
                                 "multimodal_data": [b64(f"{ASSETS}/{t}.wav")]})
        if verbose:
            print("embedded", t, flush=True)
    return {k: norm(v) for k, v in E.items()}


def metrics(E, topics=None):
    """Compute the full battery on unit-norm embeddings {(mod, topic): vec}.

    Returns a plain dict so sweeps can tabulate:
      block[mod]      same-modality cross-topic sims (list)
      xmodal[mod]     cross-modal same-topic sims vs text (list)
      drift_mag[mod], drift_agree[mod], drift_xmod
      retrieval_raw / retrieval_corrected (hits out of 2*n_topics)
      margin[mod]     min(same-topic cross-modal sim) - max(cross-topic same-modality sim)
    """
    tops = topics or sorted({t for _, t in E})
    pairs = [(a, b) for i, a in enumerate(tops) for b in tops[i + 1:]]
    m = {"block": {}, "xmodal": {}, "drift_mag": {}, "drift_agree": {}, "margin": {}}

    for mod in ("text", "image", "audio"):
        m["block"][mod] = [cos(E[(mod, a)], E[(mod, b)]) for a, b in pairs]
    for mod in ("image", "audio"):
        m["xmodal"][mod] = [cos(E[(mod, t)], E[("text", t)]) for t in tops]

    drift = {}
    for mod in ("image", "audio"):
        d = {t: sub(E[(mod, t)], E[("text", t)]) for t in tops}
        drift[mod] = mean(list(d.values()))
        m["drift_mag"][mod] = [math.sqrt(sum(x * x for x in d[t])) for t in tops]
        m["drift_agree"][mod] = [cos(norm(d[a]), norm(d[b])) for a, b in pairs]
    m["drift_xmod"] = cos(norm(drift["image"]), norm(drift["audio"]))

    for label, correct in (("retrieval_raw", False), ("retrieval_corrected", True)):
        hits = 0
        for mod in ("image", "audio"):
            for t in tops:
                v = norm(sub(E[(mod, t)], drift[mod])) if correct else E[(mod, t)]
                scores = {t2: cos(v, E[("text", t2)]) for t2 in tops}
                hits += max(scores, key=scores.get) == t
        m[label] = hits
    m["retrieval_total"] = 2 * len(tops)

    # success-criterion margin: within-topic cross-modal vs cross-topic same-modality
    for mod in ("image", "audio"):
        m["margin"][mod] = min(m["xmodal"][mod]) - max(m["block"][mod])
    return m


def report(m):
    f3 = lambda xs: "  ".join(f"{x:.3f}" for x in xs)
    print("\n-- same-modality topic similarity (lower = less anisotropic) --")
    for mod in ("text", "image", "audio"):
        print(f"  {mod}: {f3(m['block'][mod])}")
    print("\n-- cross-modal same-topic similarity vs text (higher = smaller gap) --")
    for mod in ("image", "audio"):
        print(f"  {mod}: {f3(m['xmodal'][mod])}   margin={m['margin'][mod]:+.3f}")
    print("\n-- drift d(topic) = e_modality - e_text --")
    for mod in ("image", "audio"):
        print(f"  {mod}: |d|={[round(x, 2) for x in m['drift_mag'][mod]]} "
              f"direction-agreement={[round(x, 3) for x in m['drift_agree'][mod]]}")
    print(f"  image-drift vs audio-drift: cos = {m['drift_xmod']:.3f}")
    print("\n-- cross-modal retrieval (nearest text) --")
    print(f"  raw: {m['retrieval_raw']}/{m['retrieval_total']}")
    print(f"  gap-corrected: {m['retrieval_corrected']}/{m['retrieval_total']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-assets", action="store_true")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--template-text", default="{text}")
    ap.add_argument("--template-media", default="{marker}",
                    help="default for both image and audio")
    ap.add_argument("--template-image", default=None)
    ap.add_argument("--template-audio", default=None)
    ap.add_argument("--json", action="store_true", help="dump metrics as JSON")
    args = ap.parse_args()
    if args.make_assets:
        make_assets()
    else:
        E = embed_battery(args.url,
                          t_text=args.template_text,
                          t_image=args.template_image or args.template_media,
                          t_audio=args.template_audio or args.template_media,
                          verbose=not args.json)
        m = metrics(E)
        if args.json:
            print(json.dumps(m))
        else:
            report(m)
