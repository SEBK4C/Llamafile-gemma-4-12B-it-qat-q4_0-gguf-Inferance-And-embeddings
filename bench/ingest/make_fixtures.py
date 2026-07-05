#!/usr/bin/env python3
"""Deterministic OCR golden fixtures (I2). Renders known text with DejaVu
fonts into PNGs + a manifest.json holding ground truth. Re-runnable: same
output every time (no randomness, fixed fonts/sizes)."""
import json, os, sys
from PIL import Image, ImageDraw, ImageFont

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_M = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

CLEAN = "The quick brown fox jumps over 13 lazy dogs on 2026-07-05."

PAGE_LINES = [
    "Phase 3 converts every file in the archive into enriched text before",
    "embedding. Scanned pages pass through PP-OCRv6 detection and",
    "recognition, while digital PDFs keep their native text layer intact.",
    "The Gemma 4 model then answers one structured question battery per",
    "document: does the image contain text, is it a chart, who is present,",
    "and where are the logical chunk boundaries for retrieval?",
    "",
    "Each chunk of 256 to 1024 tokens receives its own vector from the",
    "embeddinggemma-300m sidecar running on CPU port 8081. The document",
    "summary gets one additional vector for hierarchical search. Hybrid",
    "retrieval fuses BM25 over the enrichment JSON with dense cosine",
    "similarity, using reciprocal rank fusion at query time.",
    "",
    "Throughput is bounded by the enrichment call, not the OCR stage.",
    "The server ceiling near 200 tokens per second implies roughly two to",
    "six documents per minute during bulk backfill of the family archive.",
]

RECEIPT_LINES = [
    "PVE HOMELAB SUPPLY",
    "2026-07-05  14:32",
    "----------------------",
    "USB 10G NIC     89.00",
    "NVMe 2TB       119.90",
    "Cat6a 10m        8.45",
    "----------------------",
    "TOTAL  EUR     217.35",
    "VAT 19%         34.71",
    "Card **** 4821",
]


def text_img(lines, font_path, size, pad, fg=0, bg=255, width=None, spacing=None):
    font = ImageFont.truetype(font_path, size)
    spacing = spacing if spacing is not None else int(size * 0.55)
    d0 = ImageDraw.Draw(Image.new("L", (8, 8), bg))
    widths = [d0.textlength(l, font=font) for l in lines]
    w = width or int(max(widths) + 2 * pad)
    line_h = size + spacing
    h = int(len(lines) * line_h + 2 * pad)
    img = Image.new("L", (w, h), bg)
    d = ImageDraw.Draw(img)
    for i, l in enumerate(lines):
        d.text((pad, pad + i * line_h), l, font=font, fill=fg)
    return img


def main():
    os.makedirs(OUT, exist_ok=True)
    man = {}

    text_img([CLEAN], FONT, 36, 22).save(f"{OUT}/clean_line.png")
    man["clean_line.png"] = [CLEAN]

    # ~A4 at 200 DPI
    page = text_img(PAGE_LINES, FONT, 40, 120, width=1654)
    page = page.crop((0, 0, 1654, max(page.height, 2339))) if page.height < 2339 else page
    canvas = Image.new("L", (1654, 2339), 255)
    canvas.paste(page, (0, 0))
    canvas.save(f"{OUT}/doc_page.png")
    man["doc_page.png"] = [l for l in PAGE_LINES if l]

    text_img(RECEIPT_LINES, FONT_M, 24, 30).save(f"{OUT}/receipt.png")
    man["receipt.png"] = RECEIPT_LINES

    rot = text_img([CLEAN], FONT, 36, 60).rotate(15, expand=True, fillcolor=255)
    rot.save(f"{OUT}/rotated15.png")
    man["rotated15.png"] = [CLEAN]

    text_img([CLEAN], FONT_B, 36, 22, fg=136, bg=204).save(f"{OUT}/low_contrast.png")
    man["low_contrast.png"] = [CLEAN]

    with open(f"{OUT}/manifest.json", "w") as f:
        json.dump(man, f, indent=1, ensure_ascii=False)
    print("wrote", OUT, "fixtures:", list(man))


if __name__ == "__main__":
    sys.exit(main())
