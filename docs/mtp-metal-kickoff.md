# Kickoff: Option B — make draft-mtp actually fast on Metal

> Use this as the opening prompt for the Metal-placement debugging sessions.
> It encodes everything learned on 2026-06-11 (branch `mtp-gemma4-drafter`)
> so nothing has to be rediscovered. Companions: `docs/mtp-status.md`
> (integration state, traps), `docs/mtp-upstream-recon.md` (upstream wiring,
> file:line), `MTP-prompt.md` (drafter architecture reference).

## Mission

Gemma4 draft-mtp on the M4/16GB currently runs at 9.15 tok/s on Metal vs
13.23 baseline (−31%) despite a healthy 51% acceptance rate. Root-cause the
slowdown — working hypothesis: the MTP graph is scheduled on CPU — and fix
it in our fork (SEBK4C/llamafile, vendored at `vendor/llamafile`). Success:
draft-mtp ≥ 1.5× Metal baseline (≈20 tok/s) on greedy prose, no smoke-test
regressions. Stretch: numbers worth reporting upstream (by a human —
llama.cpp rejects AI-authored content).

## The evidence (all measured 2026-06-11, greedy, 400-token prose, temp 0)

| Config | gen tok/s | note |
|---|---|---|
| Metal no-spec | 13.23 | healthy baseline, 49/49 layers offloaded |
| Metal draft-mtp n=4 | 9.15 | 522 drafted / 268 accepted (51.3%) |
| Metal draft-mtp n=8 | 11.39 | 936/281 (30%) — longer drafts amortize the per-round cost |
| Metal ngram-simple | 13.18 | spec scaffolding itself is NOT the problem |
| CPU no-spec | 7.95 | |
| CPU draft-mtp n=4 | 9.08 | **+14% — MTP works fine on CPU** |

The smoking gun: **Metal+MTP ≡ CPU+MTP** (9.15 vs 9.08, and it was 9.12 and
9.16 in two more runs — invariant under everything we changed, including a
completely broken-Metal build). Enabling `-md` collapses Metal to CPU-class
throughput. ngram-simple shares the batched-verify scaffolding and is
unaffected, so the regression is specific to the MTP context/graph.

Per-round arithmetic (n=4): 131 rounds for 400 tokens; the drafter's own
generate cost is only 3.1 s of 43.9 s total (server prints a
`statistics draft-mtp: ... dur(b,g,a)` line at shutdown — use it). That
leaves ~311 ms per verify round where a single baseline decode step costs
75.6 ms. The unexplained ~200 ms/round is the target.

## Hypotheses, ranked

1. **H1 — drafter-context tensors can't be placed on Metal.** `ctx_dft` is
   a separate `llama_context` created with `LLAMA_CONTEXT_TYPE_MTP` and
   `cparams.ctx_other = ctx_tgt` (upstream server-context.cpp:913-960 at
   pin 04eb4c44). Its KV is *literal ggml tensor aliasing* into the target
   context's cache buffers (the GEMMA4_ASSISTANT `share` callback,
   llama-model.cpp:2151-2179, resolved via `map_layer_ids`,
   llama-kv-cache.cpp:177-192; drafter swa layers → target layer n-2, full
   layer → n-1). If the drafter's scheduler doesn't recognize those
   cross-context views as Metal-resident, it places the drafter graph (and
   possibly KV roundtrip copies) on CPU. Consistent with CPU≡Metal
   throughput. NOTE: drafter generate is only 3.1 s though — so if H1 is
   the whole story, the copies/sync it induces must be billed to the
   verify side (e.g. target-side flush/sync per round), not the drafter.
2. **H2 — MTP-enabled target decode is degraded.** With `-md`, the target's
   verify decode must also export the post-final-norm hidden state
   (n_embd_out=3840) for the drafter's recurrence each round. If that
   output path (or `store_full_length_kv`-style cache handling for the two
   shared layers) forces a CPU sync or a full-context copy per round, the
   ~200 ms lands exactly where we measured it. Test: run the target with
   `-md` but acceptance forced to zero vs without `-md` at all.
3. **H3 — per-round Metal command/pipeline overhead** from alternating
   target-verify and drafter graphs. Less likely alone (ngram-simple also
   alternates batch shapes and is fine), but may stack on H1/H2.

## Debug plan

1. **See the splits.** `GGML_SCHED_DEBUG=2` (and `GGML_METAL_*` as needed)
   with `-lv 5` — llamafile only surfaces GGML debug env output at
   verbosity 5. Run the Metal draft-mtp config, one short request, and
   read which ops/graph segments are assigned to CPU vs MTL0. This likely
   decides H1 vs H2 in one run. Expect HUGE output — redirect to a file.
2. **Isolate the target side.** Measure target-only MTP overhead: server
   with `-md` + `--spec-draft-n-max 0` (issue #23752 reports even n_max=0
   loses 11% — reproduce on our build; that isolates per-round fixed cost
   with zero drafting).
3. **Instrument.** lldb CANNOT attach to APE binaries (hangs; SIP blocks
   the loader; no crash reports). The loop is printf + rebuild:
   `vendor/llamafile/.cosmocc/4.0.2/bin/make -j10 o//llamafile/llamafile`
   (~10 min). After editing any HEADER, `rm` the dependent `.o` files —
   cosmocc make has no header dependency tracking.
4. **Fix directions, by hypothesis.**
   - H1: teach the drafter context to reuse the target's backend
     instances/scheduler (they share a device anyway — `ctx_other` is
     already plumbed through cparams), or register the aliased views with
     the Metal backend; worst case, copy the two shared layers' K/V into
     drafter-local Metal buffers per round (2 layers × ≤8k ctx — bounded,
     and upstream #24086 shows strided-copy consolidation is the accepted
     idiom).
   - H2: batch or defer the hidden-state export; check whether the 3840-d
     output triggers `ggml_backend_sched_synchronize` per round.
5. **Re-verify.** The full check matrix lives in docs/mtp-status.md: greedy
   parity (522/268 on the sky-is-blue prompt is the fingerprint),
   tests/smoke_test.py in the CPU prod config, then the 6-config bench
   table above.

## Environment / process traps (every one of these cost us time already)

- **Version-keyed dylib cache**: the runtime-compiled Metal backend caches
  per llamafile version (`~/.llamafile/v/X.Y.Z/`). ANY ggml/llama.cpp
  change that should reach the Metal dylib needs either a version bump
  (we're at 0.10.4) or `--recompile`. `--recompile` clobbers the version's
  shared cache — fine now (we own 0.10.4), and the right tool for the
  printf loop ONLY if the printf is in Metal-dylib sources. Symptom of a
  stale dylib: CPU speeds + no `ggml_metal_init` + empty `MTL :` feature
  list, while "offloaded 49/49 layers" still prints.
- The `--fit` probe cannot construct the MTP context (needs ctx_other);
  it degrades placement. Always pass `--fit off` with `-md`.
- Metal: `-ngl 999` or `-ngl 0`, never partial; `--no-mmproj-offload`
  always; media embeddings only in CPU runs.
- `/health` returns 503 (curl exit 0!) while loading — wait for `"ok"`.
- Chat needs `max_tokens ≥ 256` (thinking channel) and `temperature: 0`
  for the reproducible 522/268 fingerprint.
- One 12B server per GPU; kill by PID, never `pkill -f` patterns (shared
  machine etiquette; a sibling session was collateral damage once).
- zsh in this harness: words starting with `=` (e.g. `echo ===`) trigger
  equals-expansion and abort the whole command line.
- Working reference invocation:
  `bin/llamafile --server -m models/gemma-4-12b-it-qat-q4_0.gguf -md
  models/mtp-gemma-4-12b-it-qat-q4_0.gguf --spec-type draft-mtp
  --spec-draft-n-max 4 --fit off -ngl 999 -c 8192 --port 8090`
  (drop `--mmproj` while perf-debugging: irrelevant and slows loads).
  Timings come back in the response `timings` object (`draft_n`,
  `draft_n_accepted`, `predicted_per_second`).

## Upstream state (checked 2026-06-11)

- No fix exists. The only post-pin Metal MTP perf commit, e95dae1
  (PR #24086), targets `ggml_gated_delta_net` (Qwen3.6 recurrent MTP) —
  not Gemma4's path; +4% on CUDA.
- ggerganov in #23011: "It's not optimized in general. It's low on my todo
  list because my macs have a lot of memory." Both Metal-MTP issues
  (#23011, #23752) are CLOSED without a fix; #23752 was answered "not a
  bug". Nobody upstream has documented the CPU-placement equivalence we
  measured — that finding is the leverage if Sebastian reports it.
- Our pin: llama.cpp 04eb4c446d (#23398 merge). Fork with vendor commits:
  https://github.com/SEBK4C/llamafile branch `mtp-gemma4-drafter`. Keep
  fixes as fork commits (push fork BEFORE committing gitlink bumps in the
  parent repo, or fresh clones break).
