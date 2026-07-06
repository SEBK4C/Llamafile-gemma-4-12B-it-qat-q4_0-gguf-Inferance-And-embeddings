#!/usr/bin/env python3
"""Q2: labeled hallucination/fidelity rates on CORD-v2 receipts.

Labels resolve Q1's ambiguity three ways for the KEY field (total price):
  total_in_ocr      OCR transcribed it        (OCR coverage)
  total_pre_gate    enrichment produced it    (model recall, pre-gate)
  total_post_gate   survives the Q1 gate      (what the index actually gets)
  gate_false_drop   pre_gate ∧ dropped ∧ in GT  (gate error on a TRUE value)
  halluc_numbers    prose numbers ∉ OCR ∧ ∉ GT  (true-hallucination estimate)

Menu-name recall is soft/secondary (token-subset vs enrichment text).
Media stays local; only metrics publish. Usage:
  q2_labeled.py --base URL --embed-base URL [--out J]
"""
import argparse, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ingest_worker import ingest_one, Stages  # noqa: E402
from fidelity import _digitstrings, _tokens   # noqa: E402

CORD = os.path.join(HERE, "datasets_real", "cord")


def digits(s):
    return re.sub(r"[^0-9]", "", s or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--embed-base", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    man = json.load(open(os.path.join(CORD, "manifest.json")))
    stages = Stages()
    rows, agg = [], {"n": 0, "total_in_ocr": 0, "total_pre_gate": 0,
                      "total_post_gate": 0, "gate_false_drop": 0,
                      "halluc_numbers": 0, "prose_numbers": 0,
                      "menu_recall_sum": 0.0}
    print("receipt\tin_ocr\tpre\tpost\tfalse_drop\thalluc/prose\tmenu_rec\twall")
    for name, gt in sorted(man.items()):
        env = ingest_one(os.path.join(CORD, name), a.base, a.embed_base, stages)
        e, fid = env["enrichment"] or {}, env["fidelity"] or {}
        gt_total = digits(gt["total"])
        # source OCR text is what grounding saw: reconstruct from envelope
        # text_chars>0 ⇒ OCR ran; we need the digits of the OCR text — the
        # envelope doesn't carry full text, so re-derive from chunks
        src_text = "\n".join(c.get("text", "") for c in env["chunks"]) if env["chunks"] else ""
        # chunks text was dropped from envelope? chunks carry no text field
        # in stored form — use fidelity's view instead: grounded==present in
        # OCR. For OCR coverage use a direct check on the doc:
        from ocr import make_engine, extract
        src_text = extract(make_engine(), os.path.join(CORD, name))["text"]
        s_dig = _digitstrings(src_text)
        kept = [str(x) for x in (e.get("entities") or [])]
        dropped = [str(x) for x in (fid.get("entities_dropped") or [])]
        prose = " ".join(filter(None, [e.get("summary"), e.get("chart_reading")]))
        prose_nums = re.findall(r"\d[\d.,]*\d|\d", prose)

        def has_total(strs):
            return any(gt_total and gt_total == digits(x) for x in strs) or \
                   any(gt_total in digits(x) and gt_total for x in strs)

        in_ocr = gt_total in {re.sub(r"[^0-9]", "", d) for d in s_dig} or \
                 any(gt_total == d for d in s_dig)
        pre = has_total(kept + dropped + prose_nums)
        post = has_total(kept + prose_nums)
        fdrop = has_total(dropped) and not has_total(kept)
        halluc = sum(1 for nstr in prose_nums
                     if re.sub(r"[.,]", "", nstr) not in s_dig
                     and digits(nstr) != gt_total)
        etext = json.dumps(e, ensure_ascii=False).lower()
        mrec = 0.0
        if gt["menu_names"]:
            hits = sum(1 for m in gt["menu_names"]
                       if _tokens(m) and _tokens(m) <= _tokens(etext))
            mrec = hits / len(gt["menu_names"])
        agg["n"] += 1; agg["total_in_ocr"] += in_ocr; agg["total_pre_gate"] += pre
        agg["total_post_gate"] += post; agg["gate_false_drop"] += fdrop
        agg["halluc_numbers"] += halluc; agg["prose_numbers"] += len(prose_nums)
        agg["menu_recall_sum"] += mrec
        rows.append({"receipt": name, "gt_total": gt["total"], "in_ocr": in_ocr,
                     "pre_gate": pre, "post_gate": post, "false_drop": fdrop,
                     "halluc": halluc, "prose_n": len(prose_nums),
                     "menu_recall": round(mrec, 2), "wall_s": env["wall_s"]})
        print(f"{name}\t{in_ocr}\t{pre}\t{post}\t{fdrop}\t{halluc}/{len(prose_nums)}\t{mrec:.2f}\t{env['wall_s']}")
    n = agg["n"]
    summary = {"n": n,
               "total_in_ocr_rate": round(agg["total_in_ocr"] / n, 3),
               "total_pre_gate_rate": round(agg["total_pre_gate"] / n, 3),
               "total_post_gate_rate": round(agg["total_post_gate"] / n, 3),
               "gate_false_drop_rate": round(agg["gate_false_drop"] / n, 3),
               "halluc_number_rate": round(agg["halluc_numbers"] / max(1, agg["prose_numbers"]), 3),
               "menu_recall_mean": round(agg["menu_recall_sum"] / n, 3)}
    print("summary", json.dumps(summary))
    if a.out:
        json.dump({"summary": summary, "rows": rows}, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
