# program-mac.md — autoresearch: Mac Metal E2E test + serving-defaults optimization

Mac-local adaptation of `program.md`.  Same objective, same eval harness
(`mac_serve_bench.py` wraps the frozen `serve_bench.py`), same mutable
artifact (`bench/defaults.json`).  Three things differ from the CUDA run:

| | CUDA (CT 118) | Mac Metal (this machine) |
|---|---|---|
| Server | `https://sebk4c-gemma-4-12b-it-q4.bunny-sunfish.ts.net` | `http://127.0.0.1:8080` |
| KV purge | `pct exec 118 -- rm -rf /opt/.gemma4-kv*` | `rm -rf $ROOT/.kvcache` |
| LAT_NORM | 120 tok/s (f16/128K CUDA ceiling) | 25 tok/s (MTP+Metal ceiling on M4) |

## Quickstart

```sh
# Terminal 1 — start the server (Metal, MTP, all layers to GPU)
make serve

# Terminal 2 — seed a baseline row (first run only)
python3 bench/mac_serve_bench.py --baseline

# Subsequent iterations — run a candidate
python3 bench/mac_serve_bench.py --candidate bench/candidates/my-cand.json
```

## /loop 30m
One experiment per 30 min (single Metal GPU; inference is serialised).
Self-improve protocol is identical to program.md — read that file for
the full spec.  Key constraints specific to Metal:

- **Concurrency**: only ONE instance of mac_serve_bench.py should run
  inference at a time (same GPU lock via bench/.eval.lock as on CUDA).
- **KV purge**: mac_serve_bench.py handles this locally — `rm -rf .kvcache`.
  Still send `cache_prompt: false` in every API call.
- **Speed expectations (M1 Pro 32 GB, measured 2026-07-06)**: 21.5–22.2 tok/s
  clean-state on the WAL probe, `-fa on`, np1/c4096 and np2/c8192 identical.
  MTP is BREAK-EVEN on M1 Pro (0.91 acceptance but batch-2 verify ≈ 1.75×
  single decode — probe_batch_cost.py: 39-40 ms/tok at b=2 vs ~46 single).
  The M4's 1.5× MTP gain does not transfer.  `lat_norm` anchored to 25.0.
- **Measure speed on a FRESH server only**: after a test battery (multimodal,
  smoke, KV restore) the same probe reads ~17 tok/s — residual slot state,
  not a regression.  mac-full-test.sh gates the clean-state number only.
- **ubatch cap**: `-ub 2048` OOMs Metal command buffers on M1 Pro
  (`kIOGPUCommandBufferCallbackErrorOutOfMemory`); serve.sh caps to 1024 on
  Darwin (verified with ~800-token embeddings + concurrent chat).
- **Thinking channel**: temp=0 + small max_tokens ⇒ EMPTY `content` (the
  reasoning channel eats the budget).  Probe with the Gemma-4 official
  sampler (temp 1.0 / top_k 64 / top_p 0.95) and ≥512-token budgets.
- **Bench budget**: the harness's 768-token default is TOO SMALL even at
  temp 1.0 — on hard probes reasoning alone runs ~700 tokens and content
  comes back empty (iteration-1 smoke: acc 0.2, hum/soph pinned at the
  1.0 empty-content floor).  mac_serve_bench.py forces --max-tokens 2048
  unless you pass your own.  Battery tok/s at temp 1.0 ≈ 14-15 (not the
  21.5 greedy-prose number — MTP acceptance drops on diverse sampling).
- **Context cap**: `-c 8192 -np 2` verified on 32 GB (Metal window 25.5 GB).
  Do NOT test `-c 262144` on Metal.
- **mmproj**: loaded on CPU (`--no-mmproj-offload`).  Metal conv kernels assert
  on the projector's op shapes in this ggml vintage.
- **KV dtype**: keep f16 on Metal (q8_0 KV is untested, risks corruption).

## Objective
Same as program.md: factually accurate, calibrated, warm/human, technically
sophisticated when warranted, no degenerate repetition.  Candidate space =
(system_prompt × sampler).  Mutable artifact = `bench/defaults.json`.

## Mac-specific open items (to fill before trusting cal scores)
Same as program.md: replace REPLACE_ME should_answer/should_decline probes in
`bench/probes.json` with Sebastian's two actual bad examples.

## E2E feature tests (run before the first self-improve iteration)

```sh
# Full Mac E2E — verifies all features on Metal
./scripts/mac-full-test.sh
```

Feature coverage:
1. Chat completions (inference via Metal)
2. Embeddings from the same server instance
3. Concurrent mixed load (chat + embeddings simultaneously)
4. KV slot save / restore / erase
5. Multimodal: image input with mmproj on CPU
6. Speculative decoding (MTP draft-model on Metal)
7. Metal batch decode cost profile (probe_batch_cost.py)
8. Embedding consistency — batch vs solo (regression: SWA iSWA cache bug)

Results are written to `bench/mac-test-results.log`.

## Improvement loop command

```sh
# From the repo root, with the server running in another terminal:
/loop 30m python3 bench/mac_serve_bench.py --candidate bench/defaults.json
```

See program.md §PROTOCOL for the full per-iteration steps.
