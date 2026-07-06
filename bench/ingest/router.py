#!/usr/bin/env python3
"""I5 ingest router + deterministic extractors (bench/phase3-ingest-program.md).

route(path) classifies a file by magic bytes (extension as tie-break) and
runs the DETERMINISTIC extraction stage — no model calls here:

  pdf_text / pdf_scan / pdf_mixed  PyMuPDF text-layer probe per page; scan
                                    pages rasterized to ~200 DPI PNG bytes
                                    for the OCR stage (Tier 0: digital PDFs
                                    never touch OCR)
  csv                               parsed, normalized text preview
  code / text                       read as text (language by extension)
  image_photo / image_document      EXIF via piexif (camera, datetime, GPS)
  image_raw                         detected only (TIFF-family magic + ext)
  audio                             WAV duration via header; STT stage later

Output is the ingest.v1 `file` block + `source_type` + `extraction` inputs.
EXIF/GPS/dates are extracted here deterministically and only *given to* the
enrichment model as context (F20 discipline: models never invent metadata).

Usage:
  router.py FILE [FILE...]      JSON per file
  router.py --bench DIR         expectations bench vs manifest_router.json
"""
import argparse, csv, hashlib, io, json, os, sys, wave

CODE_EXT = {".py", ".sh", ".js", ".ts", ".c", ".cpp", ".h", ".go", ".rs", ".yaml", ".yml", ".json", ".toml", ".ini", ".service"}
TEXT_EXT = {".md", ".txt", ".rst", ".org", ".log"}
RAW_EXT = {".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".orf", ".rw2"}
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sniff(path):
    ext = os.path.splitext(path)[1].lower()
    head = open(path, "rb").read(16)
    if head.startswith(b"%PDF"):
        return "pdf", ext
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg", ext
    if head.startswith(b"\x89PNG"):
        return "png", ext
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff", ext  # TIFF family: classic TIFF or RAW (DNG/CR2/NEF…)
    if head.startswith(b"RIFF") and head[8:12] == b"WAVE":
        return "wav", ext
    if head.startswith(b"ID3") or (len(head) > 1 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):
        return "mp3", ext
    if head[4:8] == b"ftyp":
        return "mp4", ext
    return "unknown", ext


def exif_block(path):
    out = {"datetime": None, "gps": None, "camera": None, "is_raw": False}
    try:
        import piexif
        ex = piexif.load(path)
        z = ex.get("0th", {})
        make = (z.get(piexif.ImageIFD.Make) or b"").decode(errors="ignore").strip("\x00 ")
        model = (z.get(piexif.ImageIFD.Model) or b"").decode(errors="ignore").strip("\x00 ")
        if make or model:
            out["camera"] = (make + " " + model).strip()
        dt = ex.get("Exif", {}).get(piexif.ExifIFD.DateTimeOriginal)
        if dt:
            out["datetime"] = dt.decode(errors="ignore")
        g = ex.get("GPS", {})
        if piexif.GPSIFD.GPSLatitude in g:
            def dms(v, ref):
                d = v[0][0] / v[0][1] + v[1][0] / v[1][1] / 60 + v[2][0] / v[2][1] / 3600
                return -d if ref in (b"S", b"W", "S", "W") else d
            out["gps"] = {
                "lat": round(dms(g[piexif.GPSIFD.GPSLatitude], g.get(piexif.GPSIFD.GPSLatitudeRef, b"N")), 6),
                "lon": round(dms(g[piexif.GPSIFD.GPSLongitude], g.get(piexif.GPSIFD.GPSLongitudeRef, b"E")), 6),
            }
    except Exception:
        pass
    return out


def route(path, raster_dpi=200):
    kind, ext = sniff(path)
    st = os.stat(path)
    meta = {"sha256": sha256(path), "name": os.path.basename(path),
            "bytes": st.st_size, "mtime": int(st.st_mtime),
            "exif": {"datetime": None, "gps": None, "camera": None, "is_raw": False}}
    res = {"file": meta, "source_type": None, "text": None, "pages": None,
           "scan_pngs": None, "duration_s": None}

    if kind == "pdf":
        import fitz
        doc = fitz.open(path)
        pages, scans, texts = [], [], []
        for page in doc:
            t = page.get_text().strip()
            if len(t) >= 20:  # real text layer, not stray artifacts
                pages.append({"kind": "text"}); texts.append(t)
            else:
                pix = page.get_pixmap(dpi=raster_dpi)
                pages.append({"kind": "scan"}); scans.append(pix.tobytes("png"))
        doc.close()
        kinds = {p["kind"] for p in pages}
        res["source_type"] = ("pdf_mixed" if kinds == {"text", "scan"}
                               else "pdf_scan" if kinds == {"scan"} else "pdf_text")
        res["pages"], res["text"] = pages, "\n\n".join(texts) or None
        res["scan_pngs"] = scans or None
    elif kind == "tiff" and ext in RAW_EXT:
        meta["exif"]["is_raw"] = True
        res["source_type"] = "image_raw"
    elif kind in ("jpeg", "png", "tiff"):
        meta["exif"].update(exif_block(path))
        # documents photographed/scanned vs photos: decided later by OCR
        # density; router only splits raw vs renderable
        res["source_type"] = "image_photo"
    elif kind == "wav" or (kind in ("mp3", "mp4", "unknown") and ext in AUDIO_EXT):
        res["source_type"] = "audio"
        if kind == "wav":
            try:
                w = wave.open(path, "rb")
                res["duration_s"] = round(w.getnframes() / w.getframerate(), 3)
                w.close()
            except Exception:
                pass
    elif ext == ".csv" or ext == ".tsv":
        res["source_type"] = "csv"
        delim = "\t" if ext == ".tsv" else ","
        with open(path, newline="", errors="replace") as f:
            rows = list(csv.reader(f, delimiter=delim))
        res["text"] = "\n".join(delim.join(r) for r in rows[:200])
        res["rows"] = len(rows)
    elif ext in CODE_EXT:
        res["source_type"] = "code"
        res["text"] = open(path, errors="replace").read()
    else:
        res["source_type"] = "text"
        res["text"] = open(path, errors="replace").read()
    return res


def bench(fixdir):
    man = json.load(open(os.path.join(fixdir, "manifest_router.json")))
    n_ok = 0
    print("fixture\tsource_type\tok\tnotes")
    for name, exp in man.items():
        r = route(os.path.join(fixdir, name))
        ok, notes = True, []
        if r["source_type"] != exp["source_type"]:
            ok = False; notes.append(f"type={r['source_type']}!={exp['source_type']}")
        if "pages" in exp:
            got = [p["kind"] for p in (r["pages"] or [])]
            want = [p["kind"] for p in exp["pages"]]
            if got != want:
                ok = False; notes.append(f"pages={got}!={want}")
        if "text_must_contain" in exp and exp["text_must_contain"] not in (r["text"] or ""):
            ok = False; notes.append("text-miss")
        if "exif" in exp:
            e = r["file"]["exif"]
            if exp["exif"].get("camera") and e["camera"] != exp["exif"]["camera"]:
                ok = False; notes.append(f"camera={e['camera']}")
            if exp["exif"].get("datetime") and e["datetime"] != exp["exif"]["datetime"]:
                ok = False; notes.append(f"dt={e['datetime']}")
            if exp["exif"].get("gps_prefix") and not str((e["gps"] or {}).get("lat", "")).startswith(exp["exif"]["gps_prefix"]):
                ok = False; notes.append(f"gps={e['gps']}")
        if "duration_s" in exp and r["duration_s"] != exp["duration_s"]:
            ok = False; notes.append(f"dur={r['duration_s']}")
        n_ok += ok
        print("%s\t%s\t%s\t%s" % (name, r["source_type"], ok, ";".join(notes)))
    print("summary\t%d/%d ok" % (n_ok, len(man)))
    return n_ok, len(man)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--bench", metavar="DIR")
    a = ap.parse_args()
    if a.bench:
        ok, n = bench(a.bench)
        return 0 if ok == n else 1
    for p in a.files:
        r = route(p)
        r["scan_pngs"] = f"<{len(r['scan_pngs'])} page png(s)>" if r["scan_pngs"] else None
        print(json.dumps(r, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
