#!/usr/bin/env python3
"""Q3: seeded-fault calibration of the Q1 fidelity gate — the catch-rate
measurement that makes the gate a calibrated instrument instead of
decoration (bench/phase3-ingest-program.md Q-GOALS).

For each real source text (CORD receipt OCR, FUNSD form OCR, router/enrich
fixtures, the lease/CSV texts) we synthesize entities of known status:

  TRUE controls
    substr_true        verbatim span from the source            must PASS
    reformat_date      true date, other format (15.06.2026)     must PASS
    reorder_name       true multiword name, tokens reordered    must PASS
  SEEDED faults
    digit_mutation     one digit changed in a true number       must CATCH
    composed_date      real Y-M fused with a different day (T1) must CATCH
    fake_name          name with one token not in source        must CATCH
    fabricated_amount  amount whose digits nowhere in source    must CATCH
  KNOWN BLIND SPOT (documented, expected MISS)
    unit_swap          true digits + true-but-elsewhere unit
                       ("412 kWh" -> "412 EUR"): digits ground and the
                       unit token grounds elsewhere -> token/digit logic
                       cannot see the mismatch. Measured, not hidden.

No model calls — the gate is deterministic; this runs in milliseconds.
Output: catch-rate per class + overall gate precision on this seed set.
"""
import json, os, random, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fidelity import grounded, _dates, _digitstrings, _tokens  # noqa: E402

rng = random.Random(20260706)

LEASE = ("Lease addendum: the tenant may sublet the apartment at Hauptstrasse 12 "
         "only with prior written consent of the landlord. The security deposit "
         "of 1800 EUR is returned within 30 days of termination, less documented "
         "damages. Signed 2026-06-15 by both parties.")
CSVTXT = ("date,merchant,amount_eur\n2026-06-02,GridPower,87.40\n"
          "2026-06-11,PharmaCity,12.90\n2026-06-27,HetznerCloud,41.10")
FAKE_TOKENS = ["nebelwald", "quorvex", "brimshaw", "taldorf", "yestrel", "kupfermann"]
FAKE_UNITS = ["EUR", "USD", "kWh", "GB", "km"]


def sources():
    out = [("lease", LEASE), ("csv", CSVTXT)]
    from ocr import make_engine, extract
    eng = make_engine()
    cord = os.path.join(HERE, "datasets_real", "cord")
    if os.path.exists(cord):
        for n in sorted(os.listdir(cord))[:5]:
            if n.endswith(".jpg") and not n.startswith("._"):
                out.append(("cord:" + n, extract(eng, os.path.join(cord, n))["text"]))
    funsd = os.path.join(HERE, "datasets_real", "funsd")
    if os.path.exists(funsd):
        for n in sorted(os.listdir(funsd))[:3]:
            if n.endswith(".png") and not n.startswith("._"):
                out.append(("funsd:" + n, extract(eng, os.path.join(funsd, n))["text"]))
    out.append(("docpage", open(os.path.join(HERE, "fixtures", "doc_page.png") + ".txt").read()
                if os.path.exists(os.path.join(HERE, "fixtures", "doc_page.png") + ".txt")
                else "Phase 3 converts every file into enriched text before embedding. "
                     "Scanned pages pass through PP-OCRv6 detection on 2026-07-05, total 217.35 EUR."))
    return out


def spans(text):
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", text)]
    nums = re.findall(r"\d[\d.,]*\d|\d", text)
    dates = re.findall(r"\d{4}[-/:.]\d{1,2}[-/:.]\d{1,2}|\d{1,2}[./]\d{1,2}[./]\d{4}", text)
    multiword = re.findall(r"[A-Z][A-Za-z0-9'-]+(?: [A-Z0-9][A-Za-z0-9'-]*)+", text)
    return words, nums, dates, multiword


def seed_cases(name, text):
    cases = []  # (class, entity, expect_pass)
    words, nums, dates, multi = spans(text)
    s_tok = _tokens(text)

    if multi:
        m = rng.choice(multi)
        cases.append(("substr_true", m, True))
        parts = m.split()
        if len(parts) >= 2:
            cases.append(("reorder_name", " ".join(reversed(parts)), True))
    elif words:
        cases.append(("substr_true", rng.choice(words), True))

    for d in dates[:1]:
        m = re.match(r"(\d{4})[-/:.](\d{1,2})[-/:.](\d{1,2})", d)
        if m:
            y, mo, dd = m.groups()
            cases.append(("reformat_date", f"{int(dd):02d}.{int(mo):02d}.{y}", True))
            other = next((n for n in nums if n.isdigit() and 1 <= int(n) <= 28
                          and int(n) != int(dd)), None)
            if other:
                cases.append(("composed_date", f"{y}-{int(mo):02d}-{int(other):02d}", False))

    plain = [n for n in nums if len(re.sub(r"[.,]", "", n)) >= 3]
    if plain:
        n0 = rng.choice(plain)
        digs = re.sub(r"[.,]", "", n0)
        pos = rng.randrange(len(digs))
        mutated = digs[:pos] + str((int(digs[pos]) + 3) % 10) + digs[pos + 1:]
        if mutated not in _digitstrings(text):
            cases.append(("digit_mutation", n0.replace(re.sub(r"[.,]", "", n0)[:len(digs)], mutated)
                          if False else mutated, False))
        unit = next((u for u in FAKE_UNITS if u.lower() in s_tok), None)
        wrong_units = [u for u in FAKE_UNITS if u.lower() not in s_tok]
        if unit is None and wrong_units:
            pass
        if unit:  # blind spot: true digits + true unit from ELSEWHERE
            cases.append(("unit_swap", f"{n0} {unit}", None))  # expected MISS

    fake = next(t for t in FAKE_TOKENS if t not in s_tok)
    anchor = rng.choice(words) if words else "office"
    cases.append(("fake_name", f"{anchor.capitalize()} {fake.capitalize()}", False))

    fab = "9" + "".join(rng.choice("0123456789") for _ in range(4))
    while fab in _digitstrings(text):
        fab = "9" + "".join(rng.choice("0123456789") for _ in range(4))
    cases.append(("fabricated_amount", f"{fab[:2]}.{fab[2:]} EUR"
                  if "eur" in s_tok else fab, False))
    return cases


def main():
    stats = {}
    details = []
    for name, text in sources():
        if not text or len(text) < 40:
            continue
        src = text.lower()
        s_dates, s_digits, s_tokens = _dates(text), _digitstrings(text), _tokens(text)
        for cls, ent, expect_pass in seed_cases(name, text):
            got_pass = grounded(ent, src, s_dates, s_digits, s_tokens)
            if expect_pass is True:
                ok = got_pass          # control must pass (else false-drop)
            elif expect_pass is False:
                ok = not got_pass      # fault must be caught
            else:
                ok = None              # blind-spot: record outcome only
            st = stats.setdefault(cls, {"n": 0, "ok": 0, "passed_gate": 0})
            st["n"] += 1
            st["passed_gate"] += bool(got_pass)
            if ok is not None:
                st["ok"] += bool(ok)
            details.append({"src": name, "class": cls, "entity": ent,
                            "gate_passed": got_pass, "expected_pass": expect_pass,
                            "correct": ok})
    print(f"{'class':<18} {'n':>3} {'gate-correct':>12} {'notes'}")
    summary = {}
    for cls, st in sorted(stats.items()):
        if cls == "unit_swap":
            note = f"BLIND SPOT: {st['passed_gate']}/{st['n']} slipped (expected; needs unit-context check)"
            rate = None
            shown = "-"
        else:
            rate = st["ok"] / st["n"]
            shown = f"{st['ok']}/{st['n']} = {rate:.2f}"
            note = ""
        print(f"{cls:<18} {st['n']:>3} {shown:>12} {note}")
        summary[cls] = {"n": st["n"], "correct": st["ok"] if rate is not None else None,
                        "slipped": st["passed_gate"] if cls == "unit_swap" else None,
                        "rate": None if rate is None else round(rate, 3)}
    out = {"summary": summary, "details": details}
    path = os.path.join(HERE, "..", "data", "q3_seeded_20260706.json")
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
    print("wrote", path)


if __name__ == "__main__":
    main()
