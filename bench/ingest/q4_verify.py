#!/usr/bin/env python3
"""Q4: model verify-pass A/B on the Q3 seed set (same rng → same 48 cases).

The deterministic gate is calibrated (Q3): 100% on its claimed classes,
0% on unit_swap by construction. Q4 asks whether ONE cheap verify call —
"is CLAIM supported by SOURCE, true/false" (grammar bool, thinking off,
temp 0, DRY) — can cover the blind spot, and what it costs:

  catch-rate on unit_swap        (the gate's known 0%)
  catch-rate on gate classes     (redundancy check / verifier power)
  false-reject rate on controls  (the price of a second opinion)
  wall per call                  (GPU cost per entity)

Known risk to measure, not assume: the verifier shares the generator's
priors (F20) — plausible composed values may verify as 'supported'.
"""
import fcntl, json, os, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import q3_seeded  # noqa: E402  (rng seeded at import → same cases)

LOCK = os.path.join(HERE, "..", ".eval.lock")

SCHEMA = {"type": "object", "properties": {"supported": {"type": "boolean"}},
          "required": ["supported"], "additionalProperties": False}


def verify(base, claim, source, timeout=300):
    body = json.dumps({
        "max_tokens": 64, "temperature": 0,
        "dry_multiplier": 0.8, "dry_base": 1.75,
        "dry_allowed_length": 2, "dry_penalty_last_n": -1,
        "cache_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object", "schema": SCHEMA},
        "messages": [
            {"role": "system", "content":
             "You verify extracted entities against a source document. "
             "supported=true ONLY if the entity appears in the source with the "
             "SAME value, unit and meaning (formatting may differ). A number "
             "with the wrong unit, a date not stated, or a name not present "
             "is supported=false. Answer with the JSON object only."},
            {"role": "user", "content":
             "SOURCE:\n<<<\n" + source + "\n>>>\n\nENTITY: \"" + claim + "\"\n"
             "Is this entity supported by the source?"}
        ],
    }).encode()
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    wall = time.time() - t0
    try:
        sup = json.loads(out["choices"][0]["message"]["content"])["supported"]
    except Exception:
        sup = None
    return sup, wall


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else None
    assert base, "usage: q4_verify.py <gemma-base-url>"
    stats, details, walls = {}, [], []
    with open(LOCK, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            for name, text in q3_seeded.sources():
                if not text or len(text) < 40:
                    continue
                for cls, ent, expect_pass in q3_seeded.seed_cases(name, text):
                    sup, wall = verify(base, ent, text)
                    walls.append(wall)
                    if expect_pass is True:
                        ok = sup is True            # control must be supported
                    elif expect_pass is False:
                        ok = sup is False           # fault must be rejected
                    else:  # unit_swap blind spot: rejection = catch
                        ok = sup is False
                    st = stats.setdefault(cls, {"n": 0, "ok": 0})
                    st["n"] += 1
                    st["ok"] += bool(ok)
                    details.append({"src": name, "class": cls, "entity": ent,
                                    "supported": sup, "expected_pass": expect_pass,
                                    "verifier_correct": ok, "wall_s": round(wall, 2)})
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)
    print(f"{'class':<18} {'verifier-correct':>16}")
    summary = {}
    for cls, st in sorted(stats.items()):
        print(f"{cls:<18} {st['ok']}/{st['n']} = {st['ok']/st['n']:.2f}")
        summary[cls] = {"n": st["n"], "correct": st["ok"],
                         "rate": round(st["ok"] / st["n"], 3)}
    summary["_mean_wall_s"] = round(sum(walls) / len(walls), 2)
    print(f"mean wall/call: {summary['_mean_wall_s']}s")
    out = os.path.join(HERE, "..", "data", "q4_verify_20260706.json")
    json.dump({"summary": summary, "details": details}, open(out, "w"),
              indent=1, ensure_ascii=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
