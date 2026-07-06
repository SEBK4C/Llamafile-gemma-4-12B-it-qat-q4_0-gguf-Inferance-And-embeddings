#!/usr/bin/env python3
"""I3 vision-legibility V-probe (GATE for I4 enrichment design).

Question: can PROD Gemma-4 actually READ document text through the vision
path, or is the image pipeline still text-illegible (docs/mm-embedding.md,
2026-06-11 patch-geometry bug)? F8 only ever proved color recognition.

Method: temp-0 transcription of the I2 golden fixtures via
/v1/chat/completions (image_url data URI, enable_thinking=false,
cache_prompt=false), scored as CER against the same manifest ground truth
PP-OCRv6 scored 0.0000 on. Plus a binary has_text probe on a blank image
and on the doc page. Inference serialised behind bench/.eval.lock like
serve_bench. Run with the ocrenv python (needs PIL for the crop condition
and rapidocr for the OCR-side comparison on derived images).

Usage: vprobe.py --base https://<node>/  [--out results.json]
"""
import argparse, base64, fcntl, io, json, os, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ocr import lev, norm, make_engine, extract  # noqa: E402

FIX = os.path.join(HERE, "fixtures")
LOCK = os.path.join(HERE, "..", ".eval.lock")

PROMPT = ("Transcribe all text visible in this image exactly as written, "
          "preserving reading order. Output ONLY the transcribed text with "
          "no commentary or formatting.")
HAS_TEXT_PROMPT = ("Does this image contain any readable text? "
                   "Answer with exactly one word: yes or no.")


def chat(base, png_bytes, prompt, max_tokens=2048, timeout=600):
    img = base64.b64encode(png_bytes).decode()
    body = json.dumps({
        "max_tokens": max_tokens, "temperature": 0, "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}}]}],
    }).encode()
    t0 = time.time()
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    wall = time.time() - t0
    msg = (out.get("choices") or [{}])[0].get("message", {})
    return {"content": msg.get("content") or "", "wall_s": round(wall, 2),
            "usage": out.get("usage", {}),
            "finish": (out.get("choices") or [{}])[0].get("finish_reason")}


def content_lines(text):
    """Drop lines with no alphanumerics (dash rulers etc.) BEFORE CER.
    Applied symmetrically to prediction and ground truth: detectors
    legitimately skip decorative separators, VLMs legitimately transcribe
    them — neither should count (F19)."""
    return " ".join(l for l in text.splitlines() if any(c.isalnum() for c in l))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from PIL import Image
    man = json.load(open(os.path.join(FIX, "manifest.json")))

    # derived images: top-6-lines crop of the doc page + a blank control
    page = Image.open(os.path.join(FIX, "doc_page.png"))
    crop = page.crop((0, 0, 1654, 550))
    buf = io.BytesIO(); crop.save(buf, format="PNG"); crop_png = buf.getvalue()
    blank = Image.new("L", (800, 600), 207)
    buf = io.BytesIO(); blank.save(buf, format="PNG"); blank_png = buf.getvalue()

    conds = [
        ("clean_line", open(os.path.join(FIX, "clean_line.png"), "rb").read(),
         " ".join(man["clean_line.png"])),
        ("receipt", open(os.path.join(FIX, "receipt.png"), "rb").read(),
         " ".join(l for l in man["receipt.png"] if any(c.isalnum() for c in l))),
        ("doc_page_top6", crop_png, " ".join(man["doc_page.png"][:6])),
        ("doc_page_full", open(os.path.join(FIX, "doc_page.png"), "rb").read(),
         " ".join(man["doc_page.png"])),
    ]

    engine = make_engine()  # PP-OCRv6 comparison on the derived crop
    res = {"conditions": [], "has_text": {}}

    with open(LOCK, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            print("cond\tvlm_cer\tocr_cer\tvlm_s\timg_tok\tfinish")
            for name, png, gt in conds:
                r = chat(a.base, png, PROMPT)
                cer = lev(norm(content_lines(r["content"])), norm(gt)) / max(1, len(norm(gt)))
                if name == "doc_page_top6":
                    tmp = "/tmp/vprobe_crop.png"
                    open(tmp, "wb").write(png)
                    o = extract(engine, tmp)
                    ocr_cer = lev(norm(content_lines("\n".join(o["lines"]))), norm(gt)) / max(1, len(norm(gt)))
                else:  # full fixtures already benched at 0.0000 in I2
                    ocr_cer = 0.0
                row = {"cond": name, "vlm_cer": round(cer, 4), "ocr_cer": round(ocr_cer, 4),
                       "wall_s": r["wall_s"], "prompt_tokens": r["usage"].get("prompt_tokens"),
                       "finish": r["finish"], "gt_chars": len(norm(gt)),
                       "vlm_text": r["content"][:2000]}
                res["conditions"].append(row)
                print("%s\t%.4f\t%.4f\t%.1f\t%s\t%s" % (
                    name, cer, ocr_cer, r["wall_s"], r["usage"].get("prompt_tokens"), r["finish"]))

            for name, png, expect in [("blank", blank_png, "no"),
                                      ("doc_page", open(os.path.join(FIX, "doc_page.png"), "rb").read(), "yes")]:
                r = chat(a.base, png, HAS_TEXT_PROMPT, max_tokens=16)
                got = r["content"].strip().lower().rstrip(".")
                res["has_text"][name] = {"expect": expect, "got": got, "ok": got.startswith(expect)}
                print("has_text[%s]\texpect=%s got=%r ok=%s" % (name, expect, got, got.startswith(expect)))
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)

    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
