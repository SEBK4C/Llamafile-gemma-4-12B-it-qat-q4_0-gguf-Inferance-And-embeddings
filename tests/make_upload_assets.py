#!/usr/bin/env python3
"""Generate the multimodal upload/ingest test corpus (macOS host tools).

Creates tests/assets/{audio,images,pdfs}/ with KNOWN ground truth in
manifest.json, for (a) the automated API probe (tests/upload_ingest_probe.py)
and (b) manual web-UI upload testing (drag each file into the composer).

Generators: `say` + `afconvert` (speech), Pillow (text/invoice/chart images —
run under `uv run --with pillow` if Pillow isn't installed), `cupsfilter`
(text → PDF; ships with macOS).

    uv run --with pillow python3 tests/make_upload_assets.py
"""
import json
import pathlib
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
ASSETS = HERE / "assets"

MANIFEST = {"audio": {}, "images": {}, "pdfs": {}}

# ── audio: spoken facts (speech-only encoder; normalize loudness) ────────────
SPEECH = {
    "receipt-total.wav": "The total amount on the receipt is forty seven dollars and twenty cents.",
    "meeting-date.wav": "The project review meeting is scheduled for March twelfth at ten thirty.",
    "fox-pangram.wav": "The quick brown fox jumps over the lazy dog.",
}

# ── images: rendered text with extractable facts ────────────────────────────
INVOICE_LINES = [
    "INVOICE  #INV-2047",
    "Date: 2026-06-15",
    "Bill to: Truecarbon Labs GmbH",
    "",
    "  Metal GPU hours        12    EUR 340.00",
    "  Storage (TB-month)      3    EUR  45.00",
    "",
    "TOTAL DUE: EUR 385.00",
    "Payment within 30 days.",
]
SIGN_TEXT = ["CAUTION", "Server room 42B —", "authorized personnel only"]
CHART = {  # simple bar chart, values are the ground truth
    "title": "Decode speed by backend (tok/s)",
    "bars": {"CUDA": 105, "Metal": 22, "CPU": 5},
}

# ── pdfs: text documents (cupsfilter renders; same text is API ground truth) ─
PDF_DOCS = {
    "receipt.pdf": (
        "ACME HARDWARE — RECEIPT #R-88231\n"
        "Date: 2026-05-02\n\n"
        "1x Torque wrench          EUR 89.90\n"
        "4x M6 titanium bolts      EUR 12.40\n"
        "1x Thread locker          EUR  6.10\n\n"
        "TOTAL: EUR 108.40\n"
        "Paid by card ending 4421\n"
    ),
    "policy.pdf": (
        "REMOTE WORK POLICY (v3)\n\n"
        "Effective 2026-01-01, employees may work remotely up to three days\n"
        "per week. Equipment requests above EUR 500 require written approval\n"
        "from the department head. Security training must be renewed every\n"
        "twelve months. Contact: it-security@example.com\n"
    ),
}


def sh(*cmd):
    subprocess.run(cmd, check=True, capture_output=True)


def gen_audio():
    d = ASSETS / "audio"
    d.mkdir(parents=True, exist_ok=True)
    for name, text in SPEECH.items():
        aiff = d / (name + ".aiff")
        sh("say", "-o", str(aiff), text)
        sh("afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
           str(aiff), str(d / name))
        aiff.unlink()
        MANIFEST["audio"][name] = {"transcript": text}
        print(f"  audio/{name}")


def gen_images():
    from PIL import Image, ImageDraw
    d = ASSETS / "images"
    d.mkdir(parents=True, exist_ok=True)

    def text_img(name, lines, size=(760, 400), fontsize=28):
        img = Image.new("RGB", size, "white")
        dr = ImageDraw.Draw(img)
        y = 30
        for ln in lines:
            dr.text((40, y), ln, fill="black", font_size=fontsize)
            y += int(fontsize * 1.5)
        img.save(d / name)
        print(f"  images/{name}")

    text_img("invoice.png", INVOICE_LINES, size=(760, 480))
    MANIFEST["images"]["invoice.png"] = {
        "facts": ["INV-2047", "385.00", "2026-06-15", "Truecarbon"],
        "ask": "Read this document. What is the invoice number and the total due?",
        "expect_any": ["INV-2047"], "expect_all": ["385"]}

    text_img("sign.png", SIGN_TEXT, size=(640, 260), fontsize=36)
    MANIFEST["images"]["sign.png"] = {
        "facts": ["CAUTION", "42B"],
        "ask": "What does this sign say?",
        "expect_any": ["caution"], "expect_all": ["42B"]}

    # bar chart with labeled values
    img = Image.new("RGB", (760, 460), "white")
    dr = ImageDraw.Draw(img)
    dr.text((40, 20), CHART["title"], fill="black", font_size=30)
    x = 90
    for label, v in CHART["bars"].items():
        h = int(v * 2.8)
        dr.rectangle([x, 400 - h, x + 130, 400], fill="steelblue")
        dr.text((x + 30, 400 - h - 34), str(v), fill="black", font_size=26)
        dr.text((x + 25, 408), label, fill="black", font_size=26)
        x += 210
    img.save(d / "chart.png")
    print("  images/chart.png")
    MANIFEST["images"]["chart.png"] = {
        "facts": ["CUDA 105", "Metal 22", "CPU 5"],
        "ask": "This is a chart. Which backend is fastest and what is its value?",
        "expect_any": ["cuda"], "expect_all": ["105"]}


def gen_pdfs():
    d = ASSETS / "pdfs"
    d.mkdir(parents=True, exist_ok=True)
    for name, text in PDF_DOCS.items():
        txt = d / (name + ".txt")
        txt.write_text(text)
        pdf = d / name
        with open(pdf, "wb") as f:
            subprocess.run(["/usr/sbin/cupsfilter", str(txt)],
                           check=True, stdout=f, stderr=subprocess.DEVNULL)
        # keep the .txt: it is the API-side ground truth for /v1/ingest
        # (the APE ingests text; PDF→text extraction happens in the web UI
        # via pdf.js or in the external worker)
        MANIFEST["pdfs"][name] = {"text_file": name + ".txt"}
        print(f"  pdfs/{name} (+ .txt ground truth)")
    MANIFEST["pdfs"]["receipt.pdf"].update({
        "expect_entities_any": ["108.40", "R-88231"],
        "query": "how much was the hardware store total"})
    MANIFEST["pdfs"]["policy.pdf"].update({
        "expect_entities_any": ["2026-01-01", "500"],
        "query": "how many remote days per week are allowed"})


def main():
    ASSETS.mkdir(exist_ok=True)
    print("generating audio ...");  gen_audio()
    print("generating images ..."); gen_images()
    print("generating pdfs ...");   gen_pdfs()
    (ASSETS / "manifest.json").write_text(json.dumps(MANIFEST, indent=1))
    print(f"\nwrote {ASSETS}/manifest.json")


if __name__ == "__main__":
    main()
