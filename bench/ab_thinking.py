#!/usr/bin/env python3
"""H8 — characterize the empty-content footgun (F15). On some prompts Gemma-4's
reasoning channel runs so long it never emits `content` within a normal token
budget → the API returns an empty answer. This maps WHICH prompt classes trigger
it at a realistic serving budget, and confirms `enable_thinking:false` is a
universal fix.

Judge-free (pure structural: is content empty?). One command:
    python3 bench/ab_thinking.py --reps 2 --budget 1024 --out bench/data
"""
import argparse, json, os, ssl, time, urllib.request

SERVER = "https://sebk4c-gemma-4-12b-it-q4.bunny-sunfish.ts.net"

# 6 prompt classes × 2 prompts each
BATTERY = {
    "creative_constrained": [
        "Write a 4-stanza poem about the seasons. End every stanza with the exact line: 'and the world turns on.'",
        "Write an acrostic poem where the first letters of the lines spell HELLO.",
    ],
    "creative_open": [
        "Write a haiku about autumn leaves.",
        "Write a two-sentence horror story.",
    ],
    "factual_simple": [
        "What is the capital of France?",
        "Who wrote the play Hamlet?",
    ],
    "code": [
        "Write a Python function that prints FizzBuzz for numbers 1 to 15.",
        "Write a Python one-liner that reverses a string.",
    ],
    "math": [
        "What is 17 multiplied by 23? Give just the number.",
        "Is 91 a prime number? Answer yes or no with a one-line reason.",
    ],
    "structured_list": [
        "List 12 qualities of an excellent software engineer, numbered, one line each.",
        "Give 5 concise tips for better sleep, as a numbered list.",
    ],
}


def call(prompt, budget, think):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": budget,
            "temperature": 1.0, "top_k": 64, "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": think}}
    data = json.dumps(body).encode()
    req = urllib.request.Request(SERVER + "/v1/chat/completions", data,
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180, context=ssl.create_default_context()) as r:
        resp = json.load(r)
    ch = resp["choices"][0]
    msg = ch["message"]
    return {"content_len": len(msg.get("content") or ""),
            "reasoning_len": len(msg.get("reasoning_content") or ""),
            "finish": ch.get("finish_reason"),
            "wall": round(time.time() - t0, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--budget", type=int, default=1024, help="realistic serving max_tokens")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result = {"suite": "h8_thinking", "budget": args.budget, "reps": args.reps, "classes": {}}
    print(f"== H8: empty-content by prompt class @ max_tokens {args.budget} ==\n")
    print(f"{'class':22} {'think-ON empty':>14} {'mean reason':>12} {'think-OFF empty':>16}")

    for cls, prompts in BATTERY.items():
        on_empty = off_empty = 0
        on_reason = []
        n = 0
        for p in prompts:
            for _ in range(args.reps):
                n += 1
                on = call(p, args.budget, True)
                off = call(p, args.budget, False)
                # empty == NO answer at all (content_len 0, the reasoning-starved
                # case). A valid terse answer like "391" is NOT empty — an earlier
                # `< 5` threshold wrongly flagged it (H8 self-correction).
                on_empty += (on["content_len"] == 0)
                off_empty += (off["content_len"] == 0)
                on_reason.append(on["reasoning_len"])
        result["classes"][cls] = {
            "n": n,
            "think_on_empty_rate": round(on_empty / n, 2),
            "think_off_empty_rate": round(off_empty / n, 2),
            "mean_reasoning_chars": round(sum(on_reason) / len(on_reason)),
        }
        r = result["classes"][cls]
        print(f"{cls:22} {on_empty}/{n} ({r['think_on_empty_rate']:.2f})   "
              f"{r['mean_reasoning_chars']:>8}   {off_empty}/{n} ({r['think_off_empty_rate']:.2f})")

    tot_on = sum(c["think_on_empty_rate"] * c["n"] for c in result["classes"].values())
    tot_off = sum(c["think_off_empty_rate"] * c["n"] for c in result["classes"].values())
    ntot = sum(c["n"] for c in result["classes"].values())
    print(f"\noverall empty: think-ON {tot_on:.0f}/{ntot}  vs  think-OFF {tot_off:.0f}/{ntot}")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        result["stamp"] = stamp
        jp = os.path.join(args.out, f"h8_thinking_{stamp}.json")
        json.dump(result, open(jp, "w"), indent=1)
        print("report:", jp)


if __name__ == "__main__":
    main()
