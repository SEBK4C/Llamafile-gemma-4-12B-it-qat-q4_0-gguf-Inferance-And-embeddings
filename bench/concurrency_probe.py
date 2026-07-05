#!/usr/bin/env python3
"""concurrency_probe.py — measure how a single-slot llama.cpp/llamafile server
behaves under concurrent load. Every integration doc says "one slot, parallel
requests queue" — this quantifies exactly what that costs.

    python3 bench/concurrency_probe.py --base http://127.0.0.1:8080

Fires batches of C identical fixed-cost requests (n_predict tokens each with
ignore_eos, so every request decodes the SAME amount regardless of sampling)
and reports per-request latency and aggregate throughput as C grows. Pure
stdlib. Writes JSON/TSV to --out.

Reading the result: on a single-slot server, batch wall-time grows ~linearly
with C (requests serialize), per-request p50 latency grows ~linearly, and
aggregate tok/s stays roughly FLAT (no parallel speedup) — that's the queue.
"""
import argparse, concurrent.futures as cf, json, os, ssl, statistics, sys, time, urllib.request, urllib.error


def one_request(base, n_predict, idx):
    body = {"prompt": "Write a long detailed numbered list about software testing.",
            "n_predict": n_predict, "ignore_eos": True, "cache_prompt": False,
            "temperature": 1.0, "top_k": 64}
    data = json.dumps(body).encode()
    req = urllib.request.Request(base.rstrip("/") + "/completion", data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=300, context=ssl.create_default_context())
        raw = resp.read()
        wall = time.time() - t0
        b = json.loads(raw)
        tim = b.get("timings", {})
        return {"ok": True, "wall": wall, "start": t0,
                "gen": (tim.get("predicted_n") or n_predict),
                "server_tok_s": tim.get("predicted_per_second"),
                "prompt_n": tim.get("prompt_n")}
    except Exception as e:
        return {"ok": False, "wall": time.time() - t0, "start": t0, "err": str(e)[:120]}


def run_batch(base, C, n_predict):
    """Launch C requests as simultaneously as possible; return timing summary."""
    with cf.ThreadPoolExecutor(max_workers=C) as ex:
        t_launch = time.time()
        futs = [ex.submit(one_request, base, n_predict, i) for i in range(C)]
        results = [f.result() for f in futs]
    t_done = time.time()
    ok = [r for r in results if r["ok"]]
    walls = sorted(r["wall"] for r in ok)
    starts = [r["start"] for r in ok]
    ends = [r["start"] + r["wall"] for r in ok]
    batch_wall = (max(ends) - min(starts)) if ok else 0.0
    total_tok = sum(r["gen"] for r in ok)
    return {
        "concurrency": C,
        "n_ok": len(ok), "n_err": len(results) - len(ok),
        "batch_wall_s": round(batch_wall, 3),
        "req_p50_s": round(statistics.median(walls), 3) if walls else None,
        "req_min_s": round(walls[0], 3) if walls else None,
        "req_max_s": round(walls[-1], 3) if walls else None,
        "agg_tok_s": round(total_tok / batch_wall, 1) if batch_wall else None,
        "per_req_tok_s": round(sum(r["gen"] for r in ok) / sum(r["wall"] for r in ok), 1) if ok else None,
        "errors": [r.get("err") for r in results if not r["ok"]][:3],
    }


def main():
    ap = argparse.ArgumentParser(description="Concurrency / queue characterization for a single-slot server")
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--levels", default="1,2,4,8", help="comma list of concurrency levels")
    ap.add_argument("--n-predict", type=int, default=128, help="fixed tokens per request (ignore_eos)")
    ap.add_argument("--reps", type=int, default=2, help="repeat each level, keep the best batch_wall")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    levels = [int(x) for x in args.levels.split(",")]
    print(f"# concurrency_probe — {args.base}  ({args.n_predict} tok/req, ignore_eos, {args.reps} reps)\n")
    print(f"{'C':>3} {'ok':>3} {'err':>3} {'batch_s':>8} {'req_p50':>8} {'req_max':>8} "
          f"{'agg_tok/s':>10} {'per_req_t/s':>11}  speedup")

    rows, single = [], None
    for C in levels:
        best = None
        for _ in range(args.reps):
            b = run_batch(args.base, C, args.n_predict)
            if best is None or (b["batch_wall_s"] and b["batch_wall_s"] < best["batch_wall_s"]):
                best = b
            time.sleep(1)
        if C == 1:
            single = best["batch_wall_s"]
        # ideal parallel speedup vs measured: how much faster than C serial single-reqs?
        best["speedup"] = round((single * C) / best["batch_wall_s"], 2) if (single and best["batch_wall_s"]) else None
        rows.append(best)
        print(f"{C:>3} {best['n_ok']:>3} {best['n_err']:>3} {best['batch_wall_s']:>8} "
              f"{best['req_p50_s']:>8} {best['req_max_s']:>8} {best['agg_tok_s']:>10} "
              f"{best['per_req_tok_s']:>11}  {best['speedup']}x")

    verdict = "SERIALIZED (single slot)" if rows[-1]["speedup"] and rows[-1]["speedup"] < 1.4 else "PARALLEL"
    print(f"\n== queue behavior: {verdict} "
          f"— aggregate tok/s stayed ~{rows[0]['agg_tok_s']}→{rows[-1]['agg_tok_s']} as C {levels[0]}→{levels[-1]} ==")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        meta = {"suite": "concurrency", "stamp": stamp, "base": args.base,
                "n_predict": args.n_predict, "reps": args.reps, "levels": levels,
                "verdict": verdict, "rows": rows}
        jp = os.path.join(args.out, f"concurrency_{stamp}.json")
        json.dump(meta, open(jp, "w"), indent=1)
        keys = ["concurrency", "n_ok", "n_err", "batch_wall_s", "req_p50_s", "req_max_s",
                "agg_tok_s", "per_req_tok_s", "speedup"]
        tp = os.path.join(args.out, f"concurrency_{stamp}.tsv")
        with open(tp, "w") as f:
            f.write("\t".join(keys) + "\n")
            for r in rows:
                f.write("\t".join(str(r.get(k, "")) for k in keys) + "\n")
        print(f"report: {jp}\n        {tp}")


if __name__ == "__main__":
    main()
