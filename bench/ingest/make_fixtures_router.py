#!/usr/bin/env python3
"""I5 router fixtures: one file per source_type the router must classify.
Deterministic content; expectations in manifest_router.json.

- pdf_text.pdf   — 2 pages of real text layer (PyMuPDF insert_text)
- pdf_scan.pdf   — doc_page.png embedded as a full-page image, NO text layer
- pdf_mixed.pdf  — page 1 text layer, page 2 image-only
- table.csv      — small transactions table
- notes.md       — markdown text
- script.py      — python source
- photo_exif.jpg — gray JPEG with EXIF (camera, DateTimeOriginal, GPS)
- fake.dng       — minimal TIFF header with .dng name (RAW detection only)
- tone.wav       — 0.5 s of silence, 16 kHz mono PCM
"""
import json, os, struct, sys, wave

import fitz  # pymupdf
import piexif
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fixtures_router")

PDF_TEXT_P1 = "Invoice 2026-104 from GridPower: 412 kWh for March, total 87.40 EUR."
PDF_TEXT_P2 = "Payment is due by April 15 via SEPA transfer to DE89 3704 0044 0532 0130 00."


def make_pdfs():
    doc = fitz.open()
    for text in (PDF_TEXT_P1, PDF_TEXT_P2):
        page = doc.new_page()
        page.insert_text((72, 100), text, fontsize=12)
    doc.save(f"{OUT}/pdf_text.pdf")
    doc.close()

    doc = fitz.open()
    page = doc.new_page()
    page.insert_image(page.rect, filename=os.path.join(HERE, "fixtures", "doc_page.png"))
    doc.save(f"{OUT}/pdf_scan.pdf")
    doc.close()

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), PDF_TEXT_P1, fontsize=12)
    page = doc.new_page()
    page.insert_image(page.rect, filename=os.path.join(HERE, "fixtures", "receipt.png"))
    doc.save(f"{OUT}/pdf_mixed.pdf")
    doc.close()


def make_jpeg_exif():
    img = Image.new("RGB", (320, 240), (140, 150, 160))
    gps = {
        piexif.GPSIFD.GPSLatitudeRef: b"N",
        piexif.GPSIFD.GPSLatitude: ((52, 1), (31, 1), (1200, 100)),
        piexif.GPSIFD.GPSLongitudeRef: b"E",
        piexif.GPSIFD.GPSLongitude: ((13, 1), (24, 1), (3600, 100)),
    }
    exif = {"0th": {piexif.ImageIFD.Make: b"Canon", piexif.ImageIFD.Model: b"EOS R5"},
            "Exif": {piexif.ExifIFD.DateTimeOriginal: b"2026:06:28 14:22:31"},
            "GPS": gps}
    img.save(f"{OUT}/photo_exif.jpg", exif=piexif.dump(exif), quality=90)


def main():
    os.makedirs(OUT, exist_ok=True)
    make_pdfs()
    make_jpeg_exif()

    open(f"{OUT}/table.csv", "w").write(
        "date,merchant,amount_eur\n2026-06-02,GridPower,87.40\n2026-06-11,PharmaCity,12.90\n")
    open(f"{OUT}/notes.md", "w").write(
        "# Homelab notes\n\nMove the backup job to 03:00 and verify the ZFS snapshot count.\n")
    open(f"{OUT}/script.py", "w").write(
        "def health(url):\n    return url.rstrip('/') + '/health'\n")

    # minimal little-endian TIFF header (II*\0 + IFD offset 8, 0 entries)
    open(f"{OUT}/fake.dng", "wb").write(struct.pack("<2sHI H I", b"II", 42, 8, 0, 0))

    w = wave.open(f"{OUT}/tone.wav", "wb")
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes(b"\x00\x00" * 8000)
    w.close()

    man = {
        "pdf_text.pdf": {"source_type": "pdf_text", "pages": [{"kind": "text"}, {"kind": "text"}],
                          "text_must_contain": "GridPower"},
        "pdf_scan.pdf": {"source_type": "pdf_scan", "pages": [{"kind": "scan"}]},
        "pdf_mixed.pdf": {"source_type": "pdf_mixed", "pages": [{"kind": "text"}, {"kind": "scan"}],
                           "text_must_contain": "Invoice 2026-104"},
        "table.csv": {"source_type": "csv", "text_must_contain": "GridPower"},
        "notes.md": {"source_type": "text", "text_must_contain": "ZFS"},
        "script.py": {"source_type": "code", "text_must_contain": "health"},
        "photo_exif.jpg": {"source_type": "image_photo",
                            "exif": {"camera": "Canon EOS R5",
                                     "datetime": "2026:06:28 14:22:31",
                                     "gps_prefix": "52.5"}},
        "fake.dng": {"source_type": "image_raw"},
        "tone.wav": {"source_type": "audio", "duration_s": 0.5},
    }
    json.dump(man, open(f"{OUT}/manifest_router.json", "w"), indent=1)
    print("wrote", OUT, list(man))


if __name__ == "__main__":
    sys.exit(main())
