#!/usr/bin/env python3
"""G4 — DRY sensitivity. Two questions, both judge-free (programmatic checks):

  A. LOOP SUPPRESSION (at temp 0, the only known loop trigger — E3): what is the
     minimum dry_multiplier that kills the greedy list-loop?
  B. COLLATERAL DAMAGE (at temp 1.0, the shipped serving temp): does DRY suppress
     LEGITIMATE repetition — a poem refrain, a times table, a counted list, an
     accumulator loop? These NEED repeated tokens; DRY penalises exactly that.

The shipped default is dry_multiplier 0.8. This measures whether 0.8 is the
right strength: strong enough to backstop loops, gentle enough not to mangle
outputs that are supposed to repeat.

    python3 bench/ab_dry.py --reps 3 --out bench/data
"""
import argparse, json, os, sys, time
import serve_bench as sb

BASE_SAMPLER = {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "min_p": 0.01,
                "dry_base": 1.75, "dry_allowed_length": 2, "dry_penalty_last_n": -1,
                "repeat_penalty": 1.0,
                # thinking OFF: isolates the sampler's effect on the OUTPUT and
                # avoids the empty-content trap where reasoning eats the budget
                # (reasoning_effort low/none are IGNORED by this server; only
                # chat_template_kwargs.enable_thinking works — G4 finding).
                "chat_template_kwargs": {"enable_thinking": False}}

# --- Test A: the greedy loop trigger (list generation, needs length to manifest, F3)
LOOP_PROBE = ("List the top 12 qualities of an excellent software engineer. For each, "
              "give the quality name then a one-sentence explanation. Number them 1 to 12.")

# --- Test B: repetition-LEGITIMATE prompts + programmatic structural checks
REFRAIN = "and the world turns on"
def chk_refrain(c):
    n = c.lower().count(REFRAIN)
    return n >= 4, f"refrain x{n} (need >=4)"
def chk_table(c):
    want = [str(7 * i) for i in range(1, 13)]              # 7,14,...,84
    hit = sum(1 for w in want if w in c)
    return hit >= 11, f"{hit}/12 products present"
def chk_count(c):
    hit = sum(1 for n in range(1, 21) if f"{n}" in c.split())
    # looser: every number 1..20 appears as a token somewhere
    present = sum(1 for n in range(1, 21) if any(tok.strip(".:)")==str(n) for tok in c.split()))
    return present >= 19, f"{present}/20 numbers present"
def chk_code(c):
    lc = c.lower()
    ok = ("for " in lc) and ("+=" in c or "total" in lc or "acc" in lc or "sum_" in lc or "result" in lc)
    return ok, "has for-loop + accumulator" if ok else "missing loop/accumulator"

COLLATERAL = [
    {"id": "refrain", "max_tokens": 700, "chk": chk_refrain,
     "prompt": f"Write a short 4-stanza poem about the four seasons. End every single stanza with the exact same line: '{REFRAIN}'. Do not vary that line."},
    {"id": "times_table", "max_tokens": 500, "chk": chk_table,
     "prompt": "Write the 7 times table from 7 x 1 through 7 x 12. One line each, in the format '7 x N = result'."},
    {"id": "count_list", "max_tokens": 500, "chk": chk_count,
     "prompt": "List every whole number from 1 to 20, each on its own line, in the format 'Item N' (so 'Item 1', 'Item 2', and so on)."},
    {"id": "accumulator", "max_tokens": 600, "chk": chk_code,
     "prompt": "Write a Python function sum_list(xs) that returns the sum of a list of numbers WITHOUT using the built-in sum(), using a for loop and an accumulator variable. Return only the code."},
]


def sampler_with(dry, temp, think=False):
    s = dict(BASE_SAMPLER); s["dry_multiplier"] = dry; s["temperature"] = temp
    if think:                       # prod regime (thinking on) — used for the loop test
        s["chat_template_kwargs"] = {"enable_thinking": True}
    return s


def gen(server, prompt, sampler, max_tokens):
    msgs = [{"role": "user", "content": prompt}]
    content, reasoning, tok_s, n_out, elapsed = sb.gemma_chat(server, msgs, sampler, max_tokens)
    return content, reasoning


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="https://sebk4c-gemma-4-12b-it-q4.bunny-sunfish.ts.net")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result = {"suite": "g4_dry", "reps": args.reps, "test_A": {}, "test_B": {}}

    # ---- Test A: loop suppression at temp 0, PROD regime (thinking on, the E2 loop regime) ----
    print("== Test A: greedy (temp 0, thinking on) loop suppression vs dry_multiplier ==")
    for dry in [0.0, 0.4, 0.8, 1.2]:
        looped = 0
        for r in range(args.reps):
            c, reason = gen(args.server, LOOP_PROBE, sampler_with(dry, 0.0, think=True), 1500)
            trip, why = sb.rep_detect((reason + "\n" + c) if reason else c)
            looped += trip
            print(f"  dry={dry:<4} rep{r}: {'LOOP '+why[:40] if trip else 'clean'}")
        result["test_A"][str(dry)] = {"loop_rate": round(looped / args.reps, 2), "looped": looped, "n": args.reps}

    # ---- Test B: collateral damage at temp 1.0 ----
    print("\n== Test B: collateral damage on repetition-legitimate prompts (temp 1.0) ==")
    for dry in [0.0, 0.8, 1.2]:
        result["test_B"][str(dry)] = {}
        for probe in COLLATERAL:
            passes = 0
            for r in range(args.reps):
                c, _ = gen(args.server, probe["prompt"], sampler_with(dry, 1.0), probe["max_tokens"])
                ok, detail = probe["chk"](c)
                passes += ok
            rate = round(passes / args.reps, 2)
            result["test_B"][str(dry)][probe["id"]] = {"pass_rate": rate, "passes": passes, "n": args.reps}
            print(f"  dry={dry:<4} {probe['id']:14} {passes}/{args.reps} pass")

    # ---- summary ----
    print("\n== summary ==")
    minA = next((d for d in ["0.0", "0.4", "0.8", "1.2"] if result["test_A"][d]["loop_rate"] == 0), None)
    print(f"min dry that kills the greedy loop: {minA}")
    for dry in ["0.0", "0.8", "1.2"]:
        tot = sum(v["passes"] for v in result["test_B"][dry].values())
        n = sum(v["n"] for v in result["test_B"][dry].values())
        print(f"collateral pass total @ dry={dry}: {tot}/{n}")
        result["test_B"][dry]["_total"] = {"passes": tot, "n": n}

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        result["stamp"] = stamp
        jp = os.path.join(args.out, f"g4_dry_{stamp}.json")
        json.dump(result, open(jp, "w"), indent=1)
        print("report:", jp)


if __name__ == "__main__":
    main()
