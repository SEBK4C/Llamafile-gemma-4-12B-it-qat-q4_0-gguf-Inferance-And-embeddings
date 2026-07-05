#!/usr/bin/env python3
"""I2 OCR extractor: PP-OCRv6 medium det+rec (ONNX) via rapidocr_onnxruntime,
CPU-only. This is the phase-3 ingest pipeline's text extractor for scanned
pages and photos (bench/phase3-ingest-program.md).

The HF `PaddlePaddle/PP-OCRv6_medium_det_onnx` model is DETECTION-only
(polygons); text comes from the paired `PP-OCRv6_medium_rec_onnx`. The rec
char dict (18708 chars, 50 languages) is extracted from the rec model's
inference.yml by tools in this repo — rapidocr prepends CTC blank and appends
space, matching the ONNX head's 18710 classes exactly.

Usage:
  ocr.py IMAGE [IMAGE...]        -> JSON per image on stdout
  ocr.py --bench FIXTURES_DIR    -> CER + speed vs manifest.json (I2 metrics)

Env overrides: PPOCR_DET, PPOCR_REC, PPOCR_DICT (model file paths).
"""
import argparse, json, os, sys, time

MODELS = "/root/gemma4-gpu-optim/models-dl"
DET = os.environ.get("PPOCR_DET", f"{MODELS}/ppocrv6_det/inference.onnx")
REC = os.environ.get("PPOCR_REC", f"{MODELS}/ppocrv6_rec/inference.onnx")
DICT = os.environ.get("PPOCR_DICT", f"{MODELS}/ppocrv6_rec/ppocrv6_dict.txt")


def make_engine():
    from rapidocr_onnxruntime import RapidOCR
    kw = dict(det_model_path=DET, rec_model_path=REC, rec_keys_path=DICT,
              det_db_thresh=0.2, det_db_box_thresh=0.45, det_db_unclip_ratio=1.4)
    try:
        return RapidOCR(**kw)
    except TypeError:  # older/newer arg names — fall back to model paths only
        return RapidOCR(det_model_path=DET, rec_model_path=REC, rec_keys_path=DICT)


def group_lines(result):
    """Reconstruct reading order: cluster boxes whose y-centers overlap into
    physical lines, sort clusters top-to-bottom and members left-to-right.
    Wide gaps (receipt columns, tables) otherwise arrive as separate boxes in
    detector order, which scrambles text for chunking/embedding."""
    items = []
    for box, text, _ in result:
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        items.append({"x": min(xs), "yc": (min(ys) + max(ys)) / 2,
                      "h": max(ys) - min(ys), "text": text})
    items.sort(key=lambda it: it["yc"])
    rows = []
    for it in items:
        if rows and abs(it["yc"] - rows[-1]["yc"]) < 0.6 * max(it["h"], rows[-1]["h"]):
            rows[-1]["members"].append(it)
            n = len(rows[-1]["members"])
            rows[-1]["yc"] += (it["yc"] - rows[-1]["yc"]) / n
            rows[-1]["h"] = max(rows[-1]["h"], it["h"])
        else:
            rows.append({"yc": it["yc"], "h": it["h"], "members": [it]})
    return [" ".join(m["text"] for m in sorted(r["members"], key=lambda m: m["x"]))
            for r in rows]


def extract(engine, path):
    t0 = time.time()
    result, _ = engine(path)
    ms = 1000 * (time.time() - t0)
    result = result or []
    lines = group_lines(result)
    return {"file": os.path.basename(path), "ms": round(ms, 1),
            "n_boxes": len(result), "text": "\n".join(lines),
            "lines": lines,
            "raw_lines": [r[1] for r in result],
            "boxes": [[[round(x, 1) for x in pt] for pt in r[0]] for r in result],
            "scores": [round(float(r[2]), 4) for r in result]}


def lev(a, b):
    if a == b: return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def norm(s):
    return " ".join(s.split())


def bench(engine, fixdir):
    man = json.load(open(os.path.join(fixdir, "manifest.json")))
    rows, details = [], {}
    print("fixture\tcer\tboxes\tms")
    for name, gt_lines in man.items():
        path = os.path.join(fixdir, name)
        r = extract(engine, path)
        # decorative separator lines (no alphanumerics) are legitimately not
        # text detections — exclude them from the CER reference
        gt = norm(" ".join(l for l in gt_lines if any(c.isalnum() for c in l)))
        pred = norm(" ".join(r["lines"]))
        cer = lev(pred, gt) / max(1, len(gt))
        rows.append((name, cer, r["n_boxes"], r["ms"]))
        details[name] = {"cer": round(cer, 4), "gt_chars": len(gt), **r}
        print("%s\t%.4f\t%d\t%.0f" % (name, cer, r["n_boxes"], r["ms"]))
    # throughput: doc_page is the representative full page
    reps = [extract(engine, os.path.join(fixdir, "doc_page.png"))["ms"] for _ in range(3)]
    pps = 1000.0 / (sorted(reps)[1])
    mean_cer = sum(r[1] for r in rows) / len(rows)
    print("mean_cer\t%.4f" % mean_cer)
    print("doc_page median ms\t%.0f\tpages/sec\t%.2f" % (sorted(reps)[1], pps))
    return {"mean_cer": round(mean_cer, 4), "doc_page_ms_median": sorted(reps)[1],
            "pages_per_sec": round(pps, 3), "reps_ms": reps, "fixtures": details,
            "models": {"det": DET, "rec": REC, "dict": DICT}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*")
    ap.add_argument("--bench", metavar="FIXTURES_DIR")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    engine = make_engine()
    if a.bench:
        res = bench(engine, a.bench)
        if a.out:
            json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
            print("wrote", a.out)
        return 0
    if not a.images:
        print(__doc__); return 2
    for p in a.images:
        print(json.dumps(extract(engine, p), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
