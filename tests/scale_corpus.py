"""Scale-up corpus for the modality-gap work: 32 single-sentence items,
rendered/synthesized with exactly the modality_gap.py recipe (224px canvas,
26px Helvetica, `say` → 16kHz mono WAV).

    uv run --with pillow python3 tests/scale_corpus.py --make-assets

Writes /tmp/modality_scale/{NN}.png, {NN}.wav and manifest.json.
Sentences are kept ≤ 80 chars so they fit the 224px canvas at 26px
(6 lines × ~15 chars); verify legibility with an OCR spot-check before
trusting image numbers (tests/scale_eval.py --ocr-check).
"""

import json
import os
import subprocess

ASSETS = "/tmp/modality_scale"

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


def make_assets():
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
        aiff = f"{ASSETS}/{i:02d}.aiff"
        subprocess.run(["say", words, "-o", aiff], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                        aiff, f"{ASSETS}/{i:02d}.wav"], check=True)
        os.remove(aiff)
    with open(f"{ASSETS}/manifest.json", "w") as f:
        json.dump(SENTENCES, f, indent=1)
    print(f"{len(SENTENCES)} items written to {ASSETS}")


if __name__ == "__main__":
    make_assets()
