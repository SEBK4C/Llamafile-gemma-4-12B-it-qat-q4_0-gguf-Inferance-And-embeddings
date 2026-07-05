#!/usr/bin/env python3
"""H9 — long-context needle-in-haystack + prefill latency. The model advertises
128K context and drives agentic harnesses; this measures whether retrieval
accuracy and latency actually hold as context grows.

Judge-free (exact passcode match). Uses enable_thinking:false (H8) so the answer
isn't reasoning-starved. One command:
    python3 bench/h9_longctx.py --depths 1000,4000,16000,64000 --reps 3 --out bench/data
"""
import argparse, json, ssl, time, urllib.request

SERVER = "https://sebk4c-gemma-4-12b-it-q4.bunny-sunfish.ts.net"
NEEDLE = "IMPORTANT FACT: The secret passcode hidden in this document is MELON-7431. Remember it exactly."
ANSWER = "MELON-7431"
QUESTION = "What is the secret passcode hidden somewhere in the document above? Reply with just the passcode, nothing else."
FILLER = ("The cultivation of tea has a long and storied history spanning many centuries and continents. "
          "Farmers tend the terraced hillsides with patience, harvesting tender leaves in the cool morning air. "
          "Trade routes once carried these leaves across deserts and oceans to distant markets and quiet parlors. "
          "Each region developed its own customs, its own vessels, and its own unhurried rituals of preparation. ")


def tok_count(text):
    data = json.dumps({"content": text}).encode()
    req = urllib.request.Request(SERVER + "/tokenize", data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60, context=ssl.create_default_context()) as r:
        return len(json.load(r)["tokens"])


def build_context(target_tokens):
    """Filler padded to ~target tokens with the needle inserted at ~50% depth."""
    approx_reps = max(1, int(target_tokens * 4 / len(FILLER)))   # ~4 chars/token
    body = FILLER * approx_reps
    mid = len(body) // 2
    doc = body[:mid] + "\n\n" + NEEDLE + "\n\n" + body[mid:]
    return doc


def ask(doc):
    prompt = doc + "\n\n" + QUESTION
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 32,
            "temperature": 1.0, "top_k": 64, "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    data = json.dumps(body).encode()
    req = urllib.request.Request(SERVER + "/v1/chat/completions", data, {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300, context=ssl.create_default_context()) as r:
        resp = json.load(r)
    wall = time.time() - t0
    ch = resp["choices"][0]
    content = ch["message"].get("content") or ""
    usage = resp.get("usage") or {}
    tim = resp.get("timings") or {}
    return {"content": content, "found": ANSWER in content, "wall": wall,
            "prompt_tokens": usage.get("prompt_tokens"),
            "prompt_ms": tim.get("prompt_ms"), "prompt_per_s": tim.get("prompt_per_second")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="1000,4000,16000,64000")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    depths = [int(x) for x in args.depths.split(",")]

    result = {"suite": "h9_longctx", "reps": args.reps, "rows": []}
    print(f"== H9: needle-in-haystack @ 50% depth · enable_thinking=false · {args.reps} reps ==\n")
    print(f"{'target':>8} {'actual_tok':>10} {'found':>7} {'acc':>5} {'prefill_s':>10} {'prefill_tok/s':>13}")

    doc_cache = {}
    for tgt in depths:
        doc = build_context(tgt)
        actual = tok_count(doc)
        doc_cache[tgt] = actual
        found = 0
        walls, prefill_s, prefill_tps = [], [], []
        for _ in range(args.reps):
            r = ask(doc)
            found += r["found"]
            walls.append(r["wall"])
            if r["prompt_ms"]:
                prefill_s.append(r["prompt_ms"] / 1000.0)
                prefill_tps.append(r["prompt_per_s"])
        acc = round(found / args.reps, 2)
        ps = round(sum(prefill_s) / len(prefill_s), 2) if prefill_s else None
        ptps = round(sum(prefill_tps) / len(prefill_tps)) if prefill_tps else None
        row = {"target": tgt, "actual_tokens": actual, "found": found, "n": args.reps,
               "accuracy": acc, "prefill_s": ps, "prefill_tok_s": ptps,
               "wall_s": round(sum(walls) / len(walls), 2)}
        result["rows"].append(row)
        print(f"{tgt:>8} {actual:>10} {found:>4}/{args.reps} {acc:>5} {str(ps):>10} {str(ptps):>13}")

    if args.out:
        import os
        os.makedirs(args.out, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        result["stamp"] = stamp
        jp = os.path.join(args.out, f"h9_longctx_{stamp}.json")
        json.dump(result, open(jp, "w"), indent=1)
        print("report:", jp)


if __name__ == "__main__":
    main()
