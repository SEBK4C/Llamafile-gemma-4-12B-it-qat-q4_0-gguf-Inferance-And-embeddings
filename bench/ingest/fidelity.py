#!/usr/bin/env python3
"""Q1 deterministic fidelity gate (bench/phase3-ingest-program.md Q-GOALS).

Grounds enrichment ENTITIES and prose NUMBERS against the source text —
no model calls, per-document guarantees for the F20/T1 risk class
(plausible composed values: names, codes, dates, amounts).

Matching rules (order matters):
  1. case/whitespace-insensitive substring of source
  2. value classes get STRICT matching:
       dates   → parse to (y,m,d) tuples; entity date must equal a source
                 date. Token-subset would WRONGLY pass composed dates
                 (T1's "2026-06-30": '2026','06','30' all occur separately
                 in the source) — hence strict-by-value.
       amounts → digit-string (separators stripped) must occur among the
                 source's digit-strings.
  3. NAME-like entities (no digits) may pass by token-subset (all alnum
     tokens present in source) — handles reordering like "EUR 87.40" vs
     "87.40 EUR" and casing.

Policy (Q5 defaults, adjustable): ungrounded ENTITIES are DROPPED +
flagged 'ungrounded_entity'; ungrounded NUMBERS in summary/chart_reading
are FLAGGED only (prose is not rewritten). Pure-visual docs (no source
text, e.g. photos) skip grounding — vision-derived entities are not
text-checkable; fidelity.source = "none".

Cross-field rules the grammar cannot express:
  is_chart=false  → chart_reading forced to null (flag 'chart_reading_on_nonchart')
  has_text=false + entities present → flag 'entities_without_text' (kept:
  visual entities are legitimate)

--selftest injects known-fake entities (mini-Q3): the gate must catch all.
"""
import json, re, sys

MIN_SOURCE_CHARS = 20

_DATE_PATTERNS = [
    (re.compile(r"\b(\d{4})[-/:.](\d{1,2})[-/:.](\d{1,2})\b"), (1, 2, 3)),
    (re.compile(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b"), (3, 2, 1)),
]
_MONTHS = {m: i + 1 for i, m in enumerate(
    "january february march april may june july august september october november december".split())}
_MONTH_RX = re.compile(r"\b(" + "|".join(_MONTHS) + r")\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})?\b|"
                       r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(" + "|".join(_MONTHS) + r")\s*(\d{4})?\b|"
                       r"\b(" + "|".join(_MONTHS) + r")\s+(\d{4})\b", re.I)
_NUM_RX = re.compile(r"\d[\d.,]*\d|\d")


def _dates(text):
    out = set()
    for rx, order in _DATE_PATTERNS:
        for m in rx.finditer(text):
            y, mo, d = (int(m.group(order[0])), int(m.group(order[1])), int(m.group(order[2])))
            if y < 100:
                continue
            out.add((y, mo, d))
    for m in _MONTH_RX.finditer(text.lower()):
        g = m.groups()
        if g[0]:
            out.add((int(g[2]) if g[2] else None, _MONTHS[g[0]], int(g[1])))
        elif g[3]:
            out.add((int(g[5]) if g[5] else None, _MONTHS[g[4]], int(g[3])))
        elif g[6]:
            out.add((int(g[7]), _MONTHS[g[6]], None))
    return out


def _digitstrings(text):
    return {re.sub(r"[.,]", "", n) for n in _NUM_RX.findall(text)}


def _tokens(text):
    return set(t for t in re.sub(r"[^a-z0-9]+", " ", text.lower()).split() if len(t) > 1)


def grounded(entity, source, s_dates, s_digits, s_tokens):
    e = entity.strip()
    if not e:
        return False
    if e.lower() in source:
        return True
    e_dates = _dates(e)
    if e_dates:
        # STRICT: every date mentioned by the entity must exist in source
        # (None fields match any value — partial dates like "March 2026")
        def match(ed):
            return any(all(a is None or b is None or a == b for a, b in zip(ed, sd))
                       for sd in s_dates)
        return all(match(ed) for ed in e_dates)
    e_digits = _digitstrings(e)
    alpha_tokens = {t for t in _tokens(e) if not t.isdigit()}
    if e_digits:
        # STRICT digits — and mixed entities ("Nebenstrasse 12") must ALSO
        # ground their name part, else a shared number vouches for a fake name
        return e_digits <= s_digits and alpha_tokens <= s_tokens
    return alpha_tokens <= s_tokens  # name-like: token subset


def apply_fidelity(enrichment, source_text):
    """Mutates enrichment in place; returns the fidelity block."""
    fid = {"source": "none", "entities_total": 0, "entities_grounded": 0,
           "entities_dropped": [], "numbers_total": 0, "numbers_grounded": 0,
           "flags": []}
    e = enrichment
    if not isinstance(e, dict):
        return fid

    # cross-field rules (always applicable)
    if e.get("is_chart") is False and e.get("chart_reading"):
        e["chart_reading"] = None
        fid["flags"].append("chart_reading_on_nonchart")
    if e.get("has_text") is False and e.get("entities"):
        fid["flags"].append("entities_without_text")

    if not source_text or len(source_text.strip()) < MIN_SOURCE_CHARS:
        return fid  # pure-visual doc: vision entities not text-checkable

    src = source_text.lower()
    s_dates, s_digits, s_tokens = _dates(source_text), _digitstrings(source_text), _tokens(source_text)

    fid["source"] = "text"
    ents = e.get("entities") or []
    fid["entities_total"] = len(ents)
    keep = []
    for ent in ents:
        if grounded(str(ent), src, s_dates, s_digits, s_tokens):
            keep.append(ent)
        else:
            fid["entities_dropped"].append(ent)
    fid["entities_grounded"] = len(keep)
    if fid["entities_dropped"]:
        e["entities"] = keep
        flags = e.get("quality_flags") or []
        if "ungrounded_entity" not in flags:
            flags.append("ungrounded_entity")
        e["quality_flags"] = flags

    # prose numbers (summary + chart_reading): flag-only
    prose = " ".join(filter(None, [e.get("summary"), e.get("chart_reading")]))
    nums = _NUM_RX.findall(prose)
    fid["numbers_total"] = len(nums)
    ok = sum(1 for n in nums if re.sub(r"[.,]", "", n) in s_digits)
    fid["numbers_grounded"] = ok
    if ok < len(nums):
        flags = e.get("quality_flags") or []
        if "ungrounded_number" not in flags:
            flags.append("ungrounded_number")
        e["quality_flags"] = flags
    return fid


def selftest():
    """Mini-Q3: seeded fakes MUST be caught; truths MUST pass."""
    src = ("Lease addendum: the tenant may sublet the apartment at "
           "Hauptstrasse 12 only with prior written consent of the landlord. "
           "The security deposit of 1800 EUR is returned within 30 days of "
           "termination, less documented damages. Signed 2026-06-15 by both parties.")
    truths = ["Hauptstrasse 12", "1800 EUR", "EUR 1800", "2026-06-15",
              "15.06.2026", "security deposit"]
    fakes = ["2026-06-30", "2400 EUR", "Nebenstrasse 12", "GridPower",
             "2026-07-15", "deposit of 1900"]
    s_dates, s_digits, s_tokens = _dates(src), _digitstrings(src), _tokens(src)
    t_pass = [t for t in truths if grounded(t, src.lower(), s_dates, s_digits, s_tokens)]
    f_caught = [f for f in fakes if not grounded(f, src.lower(), s_dates, s_digits, s_tokens)]
    print(f"truths passed: {len(t_pass)}/{len(truths)}  {t_pass}")
    print(f"fakes caught:  {len(f_caught)}/{len(fakes)}  {f_caught}")
    ok = len(t_pass) == len(truths) and len(f_caught) == len(fakes)
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    enr = json.load(open(sys.argv[1]))
    src = open(sys.argv[2]).read()
    fid = apply_fidelity(enr, src)
    print(json.dumps({"fidelity": fid, "enrichment": enr}, indent=1, ensure_ascii=False))
