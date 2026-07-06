#!/usr/bin/env python3
"""H10 — prompt-cache effectiveness. H9 recommended prompt-caching to amortize
the deep-context prefill cost; this MEASURES it. Simulates the agentic multi-turn
pattern: a fixed long context/system prefix, a changing short user turn each
call. If caching works, only the changing suffix is re-prefilled.

Judge-free (server-reported cached_tokens + prefill_ms). One command:
    python3 bench/h10_promptcache.py --context-tokens 16000 --turns 4 --out bench/data
"""
import argparse, json, ssl, time, urllib.request

SERVER = "https://sebk4c-gemma-4-12b-it-q4.bunny-sunfish.ts.net"
FILLER = ("A distributed system coordinates many independent computers so they appear as one coherent service. "
          "Nodes exchange messages over unreliable networks, and consensus protocols keep their state agreed. "
          "Failures are the normal case, not the exception, so designs assume partitions, delays, and crashes. "
          "Idempotency, retries with backoff, and careful ordering keep the whole fabric correct under stress. ")


def build_context(target_tokens, nonce):
    reps = max(1, int(target_tokens * 4 / len(FILLER)))
    return f"[session {nonce}] Reference material follows.\n\n" + FILLER * reps


def chat(prefix, question, cache):
    body = {"messages": [{"role": "user", "content": prefix + "\n\nQuestion: " + question}],
            "max_tokens": 24, "temperature": 1.0, "top_k": 64, "cache_prompt": cache,
            "chat_template_kwargs": {"enable_thinking": False}}
    data = json.dumps(body).encode()
    req = urllib.request.Request(SERVER + "/v1/chat/completions", data, {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300, context=ssl.create_default_context()) as r:
        resp = json.load(r)
    wall = time.time() - t0
    usage = resp.get("usage") or {}
    tim = resp.get("timings") or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    return {"prompt_tokens": usage.get("prompt_tokens"), "cached_tokens": cached,
            "prefill_ms": tim.get("prompt_ms"), "wall": round(wall, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context-tokens", type=int, default=16000)
    ap.add_argument("--turns", type=int, default=4, help="warm follow-up turns after the cold one")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # a unique nonce guarantees the COLD call is genuinely uncached
    nonce = str(int(time.time()))[-6:]
    ctx = build_context(args.context_tokens, nonce)
    questions = ["Summarize the reference material in one sentence.",
                 "What kind of system is described?",
                 "Name one failure mode mentioned.",
                 "What keeps the fabric correct under stress?",
                 "Is consensus discussed? Yes or no.",
                 "Give one design assumption stated."]

    print(f"== H10: prompt-cache effectiveness · ~{args.context_tokens} tok context · same prefix, changing turn ==\n")
    print(f"{'turn':>6} {'prompt_tok':>10} {'cached_tok':>10} {'prefill_ms':>10} {'note':>8}")
    rows = []
    for i in range(args.turns + 1):
        r = chat(ctx, questions[i % len(questions)], cache=True)
        kind = "COLD" if i == 0 else "warm"
        r["turn"] = i; r["kind"] = kind
        rows.append(r)
        print(f"{i:>6} {str(r['prompt_tokens']):>10} {str(r['cached_tokens']):>10} "
              f"{str(round(r['prefill_ms']) if r['prefill_ms'] else '-'):>10} {kind:>8}")

    # control: a DIFFERENT unique context should be cold again
    ctx2 = build_context(args.context_tokens, nonce + "X")
    ctrl = chat(ctx2, questions[0], cache=True)
    ctrl["turn"] = "ctrl"; ctrl["kind"] = "control-newctx"
    rows.append(ctrl)
    print(f"{'ctrl':>6} {str(ctrl['prompt_tokens']):>10} {str(ctrl['cached_tokens']):>10} "
          f"{str(round(ctrl['prefill_ms']) if ctrl['prefill_ms'] else '-'):>10} {'newctx':>8}")

    cold = rows[0]; warm = rows[1]
    if cold["prefill_ms"] and warm["prefill_ms"]:
        speedup = round(cold["prefill_ms"] / warm["prefill_ms"], 1)
        saved = round(100 * (1 - warm["prefill_ms"] / cold["prefill_ms"]))
        print(f"\ncold prefill {round(cold['prefill_ms'])} ms → warm {round(warm['prefill_ms'])} ms  "
              f"= {speedup}× faster, {saved}% saved; warm cached_tokens={warm['cached_tokens']}/{warm['prompt_tokens']}")

    if args.out:
        import os
        os.makedirs(args.out, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        result = {"suite": "h10_promptcache", "stamp": stamp,
                  "context_tokens_target": args.context_tokens, "rows": rows}
        jp = os.path.join(args.out, f"h10_promptcache_{stamp}.json")
        json.dump(result, open(jp, "w"), indent=1)
        print("report:", jp)


if __name__ == "__main__":
    main()
