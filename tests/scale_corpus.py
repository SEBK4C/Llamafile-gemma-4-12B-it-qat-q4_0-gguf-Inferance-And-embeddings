"""Scale-up corpus for the modality-gap work: 32 single-sentence items,
rendered/synthesized with exactly the modality_gap.py recipe (224px canvas,
26px Helvetica, `say` → 16kHz mono WAV).

    uv run --with pillow python3 tests/scale_corpus.py            # legacy 224px
    uv run --with pillow python3 tests/scale_corpus.py --native   # 1584px

Writes /tmp/modality_scale/{NN}.png, {NN}.wav and manifest.json.

LEGACY 224px renders (kept in datasets/legacy-224 as a worked example):
26px glyphs are far below the model's working resolution — the gemma4uv
runtime upscales every image to fill the soft-token budget (7.07x at the
1120-token OCR budget), and interpolation cannot restore detail the
source never had. OCR on these is ~40% reliable. See
docs/mm-embedding.md "Image pipeline: resolution research".

NATIVE 1584px renders (--native, datasets/native-1584): sized exactly to
the 1120-token budget (33x33 patches of 48px -> budget-fill resize is a
no-op), 80px Helvetica, text lines on a 96px pitch starting at y=48 so
every line occupies whole patch rows (horizontal glyph cuts across patch
rows are the damaging case — arXiv 2402.07384). Serve with
`--image-max-tokens 1120` or the runtime will DOWNSCALE these to 768px.
"""

import argparse
import json
import os
import subprocess

ASSETS = "/tmp/modality_scale"
ASSETS_NATIVE = "/tmp/modality_native"

SENTENCES = [
    "The quick brown fox jumps over the lazy dog",
    "My favourite pasta recipe uses guanciale and pecorino",
    "Quarterly revenue grew nine percent in the third fiscal quarter",
    "The hurricane made landfall near the gulf coast at dawn",
    "Her violin solo silenced the entire concert hall",
    "Fresh snow blanketed the mountain village overnight",
    "The startup raised forty million dollars in series B funding",
    "Grandma's sourdough starter is older than my father",
    "The telescope captured a spiral galaxy in stunning detail",
    "Union workers voted to end the six week strike",
    "A stray cat adopted our office as its home",
    "The marathon route passes seven historic bridges",
    "Solar panels now power the entire school district",
    "The chef caramelized onions for nearly an hour",
    "Archaeologists unearthed a bronze age burial site",
    "The goalkeeper saved two penalties in the final",
    "Heavy traffic delayed the morning commute by an hour",
    "The jury deliberated for three days before the verdict",
    "Wild salmon swim upstream to spawn each autumn",
    "The library extended its hours during exam season",
    "Engineers tested the bridge cables for metal fatigue",
    "The orchard yielded a record crop of honeycrisp apples",
    "Volunteers planted two thousand trees along the river",
    "The museum acquired a rare impressionist painting",
    "Lightning struck the old oak tree behind the barn",
    "The senator proposed a bill to fund rural hospitals",
    "Divers discovered a shipwreck loaded with silver coins",
    "The bakery sells out of croissants by nine each morning",
    "Astronauts completed a six hour spacewalk to fix the antenna",
    "The vineyard harvest started early after a hot summer",
    "Children built sandcastles while gulls circled overhead",
    "The factory recall affected twelve thousand vehicles",
]


def make_audio(dest):
    for i, words in enumerate(SENTENCES):
        aiff = f"{dest}/{i:02d}.aiff"
        subprocess.run(["say", words, "-o", aiff], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                        aiff, f"{dest}/{i:02d}.wav"], check=True)
        os.remove(aiff)


def make_assets(audio=True):
    from PIL import Image, ImageDraw, ImageFont
    import textwrap
    os.makedirs(ASSETS, exist_ok=True)
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 26)
    for i, words in enumerate(SENTENCES):
        img = Image.new("RGB", (224, 224), "white")
        d = ImageDraw.Draw(img)
        y = 20
        lines = textwrap.wrap(words, width=15)
        assert len(lines) <= 6, f"sentence {i} too long for canvas: {words!r}"
        for line in lines:
            d.text((8, y), line, fill="black", font=font)
            y += 32
        img.save(f"{ASSETS}/{i:02d}.png")
    if audio:
        make_audio(ASSETS)
    with open(f"{ASSETS}/manifest.json", "w") as f:
        json.dump(SENTENCES, f, indent=1)
    print(f"{len(SENTENCES)} items written to {ASSETS}")


def make_native_assets(audio=False):
    """1584x1584 = the exact gemma4uv input at the 1120-token OCR budget.

    80px Helvetica on a 96px line pitch starting at y=48: every text line
    occupies exactly two whole 48px patch rows (no horizontal glyph cuts
    across rows). Audio is identical to the legacy corpus; regenerate only
    if needed.
    """
    from PIL import Image, ImageDraw, ImageFont
    import textwrap
    os.makedirs(ASSETS_NATIVE, exist_ok=True)
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 80)
    for i, words in enumerate(SENTENCES):
        img = Image.new("RGB", (1584, 1584), "white")
        d = ImageDraw.Draw(img)
        y = 48
        lines = textwrap.wrap(words, width=30)
        assert len(lines) <= 15, f"sentence {i} too long: {words!r}"
        for line in lines:
            d.text((48, y), line, fill="black", font=font)
            y += 96
        img.save(f"{ASSETS_NATIVE}/{i:02d}.png")
    if audio:
        make_audio(ASSETS_NATIVE)
    with open(f"{ASSETS_NATIVE}/manifest.json", "w") as f:
        json.dump(SENTENCES, f, indent=1)
    print(f"{len(SENTENCES)} native items written to {ASSETS_NATIVE}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", action="store_true",
                    help="render the 1584px patch-aligned corpus instead")
    ap.add_argument("--audio", action="store_true",
                    help="also synthesize WAVs (say + afconvert)")
    args = ap.parse_args()
    if args.native:
        make_native_assets(audio=args.audio)
    else:
        make_assets(audio=True)
