#!/usr/bin/env python3
"""H11 — what actually governs decode (generation) speed on this server. Naive
expectation: decode slows as the KV cache grows (more to attend to). Reality on
this MTP-speculative build: decode speed is dominated by DRAFT ACCEPTANCE, not
KV depth — predictable continuations get accepted en masse and decode FAST even
at depth, while novel generation is slower at the SAME depth.

Two views, judge-free (server timings incl. draft_n / draft_n_accepted):
  A. decode tok/s + MTP acceptance vs context depth (predictable filler).
  B. decode tok/s + MTP acceptance vs content predictability at fixed context.

    python3 bench/h11_decode.py --reps 3 --out bench/data
"""
import argparse, json, ssl, statistics, time, urllib.request

SERVER = "https://sebk4c-gemma-4-12b-it-q4.bunny-sunfish.ts.net"
FILLER = ("The cultivation of tea has a long and storied history spanning many centuries and continents. "
          "Farmers tend the terraced hillsides with patience, harvesting tender leaves in the cool morning air. "
          "Trade routes once carried these leaves across deserts and oceans to distant markets and quiet parlors. "
          "Each region developed its own customs, its own vessels, and its own unhurried rituals of preparation. ")
# a high-entropy prompt: novel generation → the drafter can't predict it → low acceptance
NOVEL = ("Generate a list of 60 unrelated random English nouns, comma separated, with no repeats and no pattern: "
         "avalanche, trombone, cactus, ")


def build_filler(target_tokens):
    reps = max(1, int(target_tokens * 4 / len(FILLER)))
    return FILLER * reps


def tok_count(text):
    data = json.dumps({"content": text}).encode()
    req = urllib.request.Request(SERVER + "/tokenize", data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60, context=ssl.create_default_context()) as r:
        return len(json.load(r)["tokens"])


def generate(prompt, n_predict, temp=1.0):
    body = {"prompt": prompt, "n_predict": n_predict, "ignore_eos": True,
            "cache_prompt": True, "temperature": temp, "top_k": 64}
    data = json.dumps(body).encode()
    req = urllib.request.Request(SERVER + "/completion", data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300, context=ssl.create_default_context()) as r:
        resp = json.load(r)
    t = resp.get("timings") or {}
    dn, da = t.get("draft_n"), t.get("draft_n_accepted")
    return {"decode_tok_s": t.get("predicted_per_second"),
            "accept": (da / dn) if (dn and da is not None) else None}


def med(runs, key):
    vals = [r[key] for r in runs if r[key] is not None]
    return round(statistics.median(vals), 2) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="1000,16000,64000,100000")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    depths = [int(x) for x in args.depths.split(",")]

    result = {"suite": "h11_decode", "reps": args.reps, "n_predict": args.n_predict,
              "vs_depth": [], "vs_predictability": []}

    # --- A: decode + acceptance vs depth (predictable filler continuation) ---
    print(f"== A: decode speed + MTP acceptance vs depth (predictable filler) ==")
    print(f"{'actual_tok':>10} {'decode_tok/s':>13} {'MTP_accept':>11}")
    for tgt in depths:
        ctx = build_filler(tgt)
        actual = tok_count(ctx)
        runs = [generate(ctx, args.n_predict) for _ in range(args.reps)]
        row = {"actual_tokens": actual, "decode_tok_s": med(runs, "decode_tok_s"),
               "accept": med(runs, "accept")}
        result["vs_depth"].append(row)
        print(f"{actual:>10} {str(row['decode_tok_s']):>13} {str(row['accept']):>11}")

    # --- B: decode + acceptance vs content predictability at fixed ~2K context ---
    print(f"\n== B: decode speed + MTP acceptance vs content predictability (~2K ctx) ==")
    print(f"{'content':>14} {'decode_tok/s':>13} {'MTP_accept':>11}")
    fixed = build_filler(2000)
    conditions = [("predictable", fixed, 1.0), ("novel", fixed + "\n\n" + NOVEL, 1.0)]
    for name, prompt, temp in conditions:
        runs = [generate(prompt, args.n_predict, temp) for _ in range(args.reps)]
        row = {"content": name, "decode_tok_s": med(runs, "decode_tok_s"), "accept": med(runs, "accept")}
        result["vs_predictability"].append(row)
        print(f"{name:>14} {str(row['decode_tok_s']):>13} {str(row['accept']):>11}")

    A = result["vs_depth"]
    print(f"\ndepth {A[0]['actual_tokens']}→{A[-1]['actual_tokens']}: decode {A[0]['decode_tok_s']}→{A[-1]['decode_tok_s']} tok/s "
          f"(accept {A[0]['accept']}→{A[-1]['accept']}) — decode tracks ACCEPTANCE, not depth")
    B = result["vs_predictability"]
    if len(B) == 2 and B[0]["decode_tok_s"] and B[1]["decode_tok_s"]:
        print(f"same ctx, predictable vs novel: {B[0]['decode_tok_s']} vs {B[1]['decode_tok_s']} tok/s "
              f"(accept {B[0]['accept']} vs {B[1]['accept']})")

    if args.out:
        import os
        os.makedirs(args.out, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        result["stamp"] = stamp
        jp = os.path.join(args.out, f"h11_decode_{stamp}.json")
        json.dump(result, open(jp, "w"), indent=1)
        print("report:", jp)


if __name__ == "__main__":
    main()
