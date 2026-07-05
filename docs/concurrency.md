# Concurrency: what "one slot" actually costs

Every integration guide here says *"one request at a time — parallel agents
queue."* That's directionally right but imprecise. Measured on the live server
(`bench/concurrency_probe.py`, 128 fixed tokens/request via `ignore_eos`,
best of 2, RTX 3080 Ti):

| concurrent | batch wall | median req | slowest req | aggregate tok/s | per-req tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.88 s | 0.88 s | 0.88 s | 145 | 145 |
| 2 | 1.29 s | 1.00 s | 1.29 s | 199 | 128 |
| 4 | 2.56 s | 1.54 s | 2.56 s | 200 | 82 |
| 8 | 5.53 s | 3.15 s | 5.53 s | 185 | 41 |

**It is not strictly serial, but it does not scale either.** Two concurrent
requests finish ~1.37× faster than running them back-to-back (real prefill/
decode overlap), so the aggregate rate climbs from 145 to ~200 tok/s and then
**plateaus** — ~200 tok/s is the whole server's ceiling no matter how many
clients hit it. Meanwhile each individual request gets steadily slower
(per-request throughput 145 → 41 tok/s from C=1 to C=8) and the slowest
request's latency grows almost linearly with load.

**Zero errors, zero drops** through C=8 — extra requests queue and wait, they
don't fail.

## Practical guidance

- **2 agents/tabs against one server: fine.** ~30% per-request slowdown, no
  errors — comfortably interactive.
- **4+: throughput is capped and latency degrades.** A busy Cline workspace
  plus a Claude Code session plus a chat tab will feel sluggish, not broken.
- **The server's total output is ~200 tok/s** regardless of client count —
  budget for that, not `200 × clients`.
- Need genuine parallelism? Raise the slot count (`-np N` / `--parallel N`)
  and the KV cache to match — but on a 12 GB card the KV budget is the limit,
  and more slots means less context per slot. Untested here (would require a
  prod restart); the single-slot numbers above are the shipped default.

Reproduce on your hardware:

```sh
python3 bench/concurrency_probe.py --base http://127.0.0.1:8080 --levels 1,2,4,8
```

Data + chart: `bench/data/concurrency_*` ·
[test-data repo](https://huggingface.co/datasets/SEBK4C/gemma4-serving-bench-data).
