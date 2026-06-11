#!/usr/bin/env python3
"""Measure Metal batched-decode cost vs batch width via uncached prefill.

Sends /completion requests with cache_prompt=false and n_predict=1 so each
request re-prefills exactly `b` tokens in one ubatch; prompt_ms then isolates
the cost of a width-b decode step. Run against a WARM server (one request
already served) or the first row absorbs ~500 ms of pipeline warmup.

Context: docs/metal-batch-kickoff.md. On a healthy memory-bound setup the
total should stay near-flat in b until compute saturates; a linear slope
means the matmul kernels re-read weights per column.

Usage:
    python3 tests/probe_batch_cost.py [--port 8090] [--batches 1,2,4,8,...]
"""
import argparse
import json
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument(
        "--batches",
        default="1,2,3,5,7,9,12,16,24,48,96",
        help="comma-separated word counts (token count is words+1 for BOS)",
    )
    args = ap.parse_args()
    url = f"http://{args.host}:{args.port}/completion"

    # warmup
    _post(url, {"prompt": "warmup run please", "n_predict": 4, "cache_prompt": False})

    print(f"{'b':>4} {'total_ms':>10} {'ms/token':>9}")
    for words in (int(w) for w in args.batches.split(",")):
        body = {
            "prompt": " ".join(["apple"] * words),
            "n_predict": 1,
            "cache_prompt": False,
            "temperature": 0,
        }
        t = _post(url, body)["timings"]
        print(f"{t['prompt_n']:>4} {t['prompt_ms']:>10.1f} {t['prompt_per_token_ms']:>9.1f}")


def _post(url, body):
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


if __name__ == "__main__":
    main()
