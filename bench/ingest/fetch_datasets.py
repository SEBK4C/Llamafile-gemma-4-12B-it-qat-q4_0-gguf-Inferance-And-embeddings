#!/usr/bin/env python3
"""Fetch SMALL subsets of public benchmark datasets for the phase-3 real-data
evals (Sebastian's directive 2026-07-05: public sets instead of private
photos). Everything lands in bench/ingest/datasets_real/ which is
LOCAL-ONLY: never committed, never uploaded — only aggregate metrics are
published. Licenses recorded in sources.json.

Subsets:
  people/   14 Flickr30k-test images whose captions mention people
            (nlphuji/flickr_1k_test_image_text_retrieval; CC BY 4.0 research)
  funsd/    8 scanned forms + word GT (FUNSD, research use)
  speech/   10 LibriSpeech test-clean utterances + transcripts (CC BY 4.0)
  sounds/   12 ESC-50 clips across categories + labels (CC BY-NC 3.0)

Idempotent: skips anything already present.
"""
import io, json, os, sys, urllib.request, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "datasets_real")
PERSON_WORDS = ("man", "woman", "boy", "girl", "people", "person", "child",
                "children", "men", "women", "kid", "guy", "lady")

SOURCES = {}


def done(tag, n, license_, src):
    SOURCES[tag] = {"n": n, "license": license_, "source": src}
    print(f"[{tag}] ready: {n} items")


def _tok():
    p = os.path.expanduser("~/.cache/huggingface/token")
    return open(p).read().strip() if os.path.exists(p) else None


def hf_get(url):
    req = urllib.request.Request(url)
    t = _tok()
    if t:
        req.add_header("Authorization", "Bearer " + t)
    return urllib.request.urlopen(req, timeout=120).read()


def hf_rows(dataset, config, split, offset=0, length=100):
    """datasets-server rows API — works for parquet-native repos without the
    datasets library (5.0 dropped script datasets)."""
    from urllib.parse import quote
    u = ("https://datasets-server.huggingface.co/rows?dataset=" + quote(dataset, safe="") +
         f"&config={config}&split={split}&offset={offset}&length={length}")
    return json.loads(hf_get(u))["rows"]


def fetch_people():
    d = os.path.join(ROOT, "people")
    os.makedirs(d, exist_ok=True)
    man_p = os.path.join(d, "manifest.json")
    if os.path.exists(man_p):
        done("people", len(json.load(open(man_p))), "Flickr30k research/CC BY 4.0",
             "hf:nlphuji/flickr_1k_test_image_text_retrieval"); return
    rows = hf_rows("nlphuji/flickr_1k_test_image_text_retrieval", "TEST", "test", 0, 100)
    man, n = {}, 0
    for r in rows:
        row = r["row"]
        caps = row.get("caption") or []
        if not any(w in " ".join(caps).lower().split() for w in PERSON_WORDS):
            continue
        src = (row.get("image") or {}).get("src")
        if not src:
            continue
        name = f"img_{n:02d}.jpg"
        open(os.path.join(d, name), "wb").write(hf_get(src))
        man[name] = {"captions": caps}
        n += 1
        if n >= 14:
            break
    json.dump(man, open(man_p, "w"), indent=1)
    done("people", n, "Flickr30k research/CC BY 4.0",
         "hf:nlphuji/flickr_1k_test_image_text_retrieval")


def fetch_funsd():
    d = os.path.join(ROOT, "funsd")
    os.makedirs(d, exist_ok=True)
    man_p = os.path.join(d, "manifest.json")
    if os.path.exists(man_p):
        done("funsd", len(json.load(open(man_p))), "FUNSD research license",
             "guillaumejaume.github.io/FUNSD"); return
    buf = urllib.request.urlopen(
        "https://guillaumejaume.github.io/FUNSD/dataset.zip", timeout=120).read()
    z = zipfile.ZipFile(io.BytesIO(buf))
    imgs = sorted(n for n in z.namelist()
                  if "testing_data/images/" in n and n.endswith(".png")
                  and "__MACOSX" not in n and not os.path.basename(n).startswith("._"))[:8]
    man = {}
    for ip in imgs:
        base = os.path.basename(ip)[:-4]
        ap = ip.replace("/images/", "/annotations/").replace(".png", ".json")
        open(os.path.join(d, base + ".png"), "wb").write(z.read(ip))
        ann = json.loads(z.read(ap).decode("utf-8-sig", errors="replace"))
        words = []
        for block in ann["form"]:
            for w in block["words"]:
                if w["text"].strip():
                    words.append(w["text"].strip())
        man[base + ".png"] = {"gt_words": words}
    json.dump(man, open(man_p, "w"), indent=1)
    done("funsd", len(man), "FUNSD research license", "guillaumejaume.github.io/FUNSD")


def fetch_speech():
    d = os.path.join(ROOT, "speech")
    os.makedirs(d, exist_ok=True)
    man_p = os.path.join(d, "manifest.json")
    if os.path.exists(man_p):
        done("speech", len(json.load(open(man_p))), "CC BY 4.0",
             "hf:openslr/librispeech_asr test-clean"); return
    # parquet mirror (script-based openslr repo unusable with datasets 5.0)
    rows = hf_rows("fixie-ai/librispeech_asr", "clean", "test", 0, 12)
    man, n = {}, 0
    for r in rows:
        row = r["row"]
        auds = row.get("audio") or []
        src = auds[0].get("src") if auds else None
        text = (row.get("text") or "").lower()
        if not src or not text:
            continue
        raw = hf_get(src)
        name = f"utt_{n:02d}" + (".wav" if b"WAVE" in raw[:16] else ".mp3")
        open(os.path.join(d, name), "wb").write(raw)
        man[name] = {"text": text}
        n += 1
        if n >= 10:
            break
    json.dump(man, open(man_p, "w"), indent=1)
    done("speech", n, "CC BY 4.0", "hf:fixie-ai/librispeech_asr test-clean (parquet mirror)")


def fetch_sounds():
    d = os.path.join(ROOT, "sounds")
    os.makedirs(d, exist_ok=True)
    man_p = os.path.join(d, "manifest.json")
    if os.path.exists(man_p):
        done("sounds", len(json.load(open(man_p))), "CC BY-NC 3.0",
             "github:karolpiczak/ESC-50"); return
    raw = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/"
    meta = urllib.request.urlopen(raw + "meta/esc50.csv", timeout=60).read().decode()
    rows = [l.split(",") for l in meta.strip().splitlines()[1:]]
    want = ["dog", "rain", "siren", "chainsaw", "crying_baby", "rooster",
            "sea_waves", "clock_alarm", "helicopter", "thunderstorm",
            "church_bells", "laughing"]
    man = {}
    for cat in want:
        fn = next(r[0] for r in rows if r[3] == cat)
        data = urllib.request.urlopen(raw + "audio/" + fn, timeout=120).read()
        name = f"{cat}.wav"
        open(os.path.join(d, name), "wb").write(data)
        man[name] = {"category": cat}
    json.dump(man, open(man_p, "w"), indent=1)
    done("sounds", len(man), "CC BY-NC 3.0", "github:karolpiczak/ESC-50")


def main():
    os.makedirs(ROOT, exist_ok=True)
    for fn in (fetch_people, fetch_funsd, fetch_speech, fetch_sounds):
        try:
            fn()
        except Exception as e:
            print(f"[{fn.__name__}] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
    json.dump(SOURCES, open(os.path.join(ROOT, "sources.json"), "w"), indent=1)
    print("sources.json written")


if __name__ == "__main__":
    sys.exit(main())
