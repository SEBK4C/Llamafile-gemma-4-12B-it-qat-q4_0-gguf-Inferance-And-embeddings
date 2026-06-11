# Kickoff: Metal small/mid-batch matmul — the remaining ~30% (and 4× prefill)

> Opening prompt for the follow-up Metal-kernel sessions. Encodes everything
> measured on 2026-06-11 while fixing the draft-mtp regression (fork commit
> `d4c192d`, parent `04d22d4`) so nothing has to be rediscovered. Companions:
> `docs/mtp-status.md` ("Metal MTP slowdown: root cause & fix" + traps),
> `docs/mtp-metal-kickoff.md` (the previous, resolved mission).

## Mission

After disabling ggml-metal's `mul_mv_ext` kernels, draft-mtp n=2 reaches
20.2 tok/s (1.52× baseline) on the M4/16GB — but Metal batched decode is
*still* far from roofline at every width that matters:

- A full single-token decode is ~75 ms (≈ one 5.8 GB weight pass at
  ~110 GB/s — i.e. b=1 is already memory-bandwidth-bound and healthy).
- An *ideal* batched step for b ≤ ~32 should cost nearly the same ~80-100 ms
  (one weight pass, amortized across columns). Measured today (post-fix,
  plain mv path for b≤8, mm for b>8): b=4 → 119 ms, b=8 → 229 ms,
  b=13-25 → flat ~360 ms, b=22 prefill → 581 ms total.
- So: **prefill runs ~4× slower than the memory roofline**, and spec-verify
  batches (b = n_draft+1) pay 1.6-3× what they should.

Success: (a) understand *why* the ext and/or mm kernels underperform on this
stack, (b) make verify widths b=3..9 cost ≤ ~1.3× a single decode — that
moves the optimal `--spec-draft-n-max` from 2 up to ~4-6 and is worth
roughly **+20-30% on top of today's 20.2 tok/s** (51% acceptance × 5-7 token
verify windows at ~110-130 ms/round ⇒ ~23-27 tok/s), (c) ideally fix mm so
prefill drops from ~26 ms/token toward ~5-8 ms/token. Stretch: upstream
report material (filed by a human — llama.cpp rejects AI-authored content).

## The evidence (2026-06-11, M4 Mac mini 16GB, gemma4-12b q4_0, -ngl 999)

Uncached-prefill probe (`tests/probe_batch_cost.py`, n_predict=1,
cache_prompt=false, warm server). Total ms for the batch:

| b | ext kernels (stock) | plain mv (shipped fix) | mm forced (mm_min=1) |
|---|---|---|---|
| 2 | 117.7 | **76.8** | 227.4 |
| 4 | 200.4 | **119.4** | 232.6 |
| 8 | 387.3 | **229.3** | 452.8 |
| 10 | 469.8 | 283.1 | 453.8 |
| 13 | 420.1 (mm) | 358.4 (mm) | 466.7 |
| 25 | 416.4 (mm) | 359.3 (mm) | 464.0 |
| 49 | 621.0 (mm) | — | — |
| 97 | 836.5 (mm) | — | — |

Reference points: b=1 ≈ 75 ms; one full weight read ≈ 55-66 ms.

Readings:

1. **ext** (`kernel_mul_mv_ext_*`, dispatched for q4_0 at ne11 ∈ [2,8]) is
   ~1.7× *worse* than plain mv everywhere in its window, even though its
   whole design goal is multi-column amortization (r1ptg groups columns per
   weight read — it "should" be near-flat in b). `nsg=4/8` instead of 2:
   no change (tested). Now disabled by default in our fork; re-enable with
   `GGML_METAL_MV_EXT=1`.
2. **mv** beats ext but still scales ~28 ms per extra token at b=2..10 —
   ~0.4 weight-passes per added column instead of ~0.
3. **mm** (`mul_mm`, simdgroup matrix kernel, used at ne11 > 8) is flat
   ~360-460 ms from b=13 to b=25 — flat is the right *shape*, but the level
   is ~6× one weight pass. This caps prefill (b=22 → 581 ms) and makes
   raising `ne11_mm_min` pointless until fixed.
4. Crossover today: mv wins up to b≈12, mm beyond. The dispatch threshold
   `ne11_mm_min = 8` is therefore slightly wrong post-fix (b=10 mv 283 <
   mm 453) but it's a second-order detail.

MTP context (why this is +30%): per verify round, cost ≈ verify(b) + ~6 ms ×
n_draft (drafter) + ~35 ms fixed. With verify(b) near-flat, longer drafts
become free coverage; with today's mv slope they aren't — n=2 (20.23 tok/s)
beats n=4 (15.74) despite n=4 accepting more tokens per round.

## Hypotheses, ranked

1. **H-compile — llamafile's runtime shader compile produces worse code than
   upstream's build.** llamafile compiles `ggml-metal.metal` at startup via
   `newLibraryWithSource:` (ggml-metal-device.m ~line 228) after a homegrown
   preprocessing step (`PreprocessMetalShader` in llamafile/metal.c inlines
   ggml-common.h/ggml-metal-impl.h includes). Upstream builds a `.metallib`
   offline with `xcrun metal` and specific flags. Missing/different compile
   options, missing preprocessor defines (only `GGML_METAL_HAS_TENSOR` and
   `GGML_METAL_EMBED_LIBRARY` are conditionally set; fast-math is default-on,
   the explicit disable is commented out), or the source-concat step could
   degrade exactly the simdgroup-heavy kernels (ext, mm) while leaving the
   simple mv kernels fine. **Discriminating test (do this first):** build
   stock upstream llama.cpp (cmake, Metal) at or near our pin 04eb4c446d on
   this machine, run `llama-batched-bench` / equivalent probe with the same
   gguf, compare ext/mm numbers. If stock is fast → it's our compile path;
   diff the kernel dispatch params and compile options, or ship a prebuilt
   metallib. If stock is slow too → M4-specific upstream issue (report).
2. **H-kernel — the kernels are genuinely weak on M4.** ggerganov benchmarked
   ext kernels as wins on M1/M2/M3 in late 2024; M4's memory subsystem or
   simdgroup scheduling may differ, and the hardcoded tuning (`nsg=2`,
   `nxpsg=8/16`, `r1ptg≤5`; mm uses fixed 32×32-ish tiles) was never
   re-tuned. The mm kernel's 6× gap at small N smells like dequant-bound
   tiles with too little parallel work per weight byte (each threadgroup
   dequantizes its own tile slice; at N=1 tile column the GPU is mostly
   idle). Fix direction: sweep tile/simdgroup params; or a fused
   "mv-with-N-accumulators" kernel that streams weights once and keeps b
   accumulator registers (what ext was supposed to be).
3. **H-dispatch — secondary wins in the dispatch logic** (ggml-metal-ops.cpp
   `ggml_metal_op_mul_mat`, ~line 2057): re-tune `ne11_mm_min` after any
   kernel fix; consider per-shape choice using measured tables rather than
   one threshold; note the lm_head/vocab matmul (262k × 3840) follows the
   same paths.

## Tools & protocol (hard-won — read before touching anything)

- **Fast Metal iteration loop (~30 s, no cosmocc rebuild)**: edit the
  extracted sources in `~/.llamafile/v/0.10.4/`, then compile the dylib
  MANUALLY mirroring the cc flags in `llamafile/metal.c` `BuildMetal()`
  (session script: compile the 12 sources with
  `cc -c -I$DIR -std=c++17 -O3 -fPIC -pthread -DNDEBUG -ffixed-x28
  -DTARGET_OS_OSX -DGGML_MULTIPLATFORM -w`, link with
  `cc -shared ... -framework Foundation -framework Metal -framework
  MetalKit -lc++` into `ggml-metal.dylib`), then start the server. An
  existing dylib suppresses re-extraction. You CANNOT edit sources and just
  `rm` the dylib: extraction byte-compares against the zip member and
  clobbers local edits. The .metal shader source is also in that dir —
  same rule applies.
- Debug prints inside the dylib: `fprintf(stderr, ...)`. `GGML_LOG_*` from
  the dylib never reaches llamafile's log (which is also why
  `ggml_metal_init` lines are absent even when Metal works — check
  `MTL0_Mapped model buffer size` instead).
- Probe: `python3 tests/probe_batch_cost.py --port 8090` against a warm
  server (send one warmup request first; the first request after load pays
  ~500 ms of pipeline warmup). Uses `cache_prompt:false` so every request
  re-prefills. Per-token MTP truth comes from the end-to-end chat request
  (`timings` object) + the `statistics draft-mtp: ... dur(b,g,a)` line the
  server prints at shutdown.
- `GGML_SCHED_DEBUG=2` works but needs `-lv 5` and prints only on graph
  topology changes (graph-reuse makes steady-state silent). Placement is
  NOT the problem — already verified all-MTL0.
- One 12B server at a time (16 GB box). The prod server on 8080 may be
  taken down for tests (dev machine, owner approved 2026-06-11); its argv
  for restore:
  `bin/llamafile --server --spec-type ngram-simple --slot-save-path
  <repo>/.kvcache -m <repo>/models/gemma-4-12b-it-qat-q4_0.gguf --mmproj
  <repo>/models/mmproj-gemma-4-12b-it-qat-q4_0.gguf --no-mmproj-offload
  --embeddings --pooling mean -c 8192 -np 2 -b 2048 -ub 2048 -ngl 0
  --host 127.0.0.1 --port 8080` (run from
  `~/Projects/Llamafile-gemma-4-12B-…`).
- Kill servers by PID (`lsof -ti :8090`), never `pkill -f`. zsh here
  explodes on words starting with `=`. `--recompile` is rejected in
  `--server` mode — delete `~/.llamafile/v/0.10.4` instead (then the next
  launch re-extracts + rebuilds, ~10 s).
- After ANY change that must reach end users' dylibs: the binary
  early-returns "using cached" whenever the dylib file exists, so either
  bump the llamafile version or document `rm -rf ~/.llamafile/v/<ver>`.
- Verification gates for any kernel change: `tests/probe_batch_cost.py`
  table, greedy-parity (spec output byte-identical to no-spec at temp 0,
  400 tokens), `tests/smoke_test.py` on the CPU prod config, and the
  draft-mtp n-sweep (n=2 must not regress below 20 tok/s).

## Where the code is

- Dispatch: `~/.llamafile/v/0.10.4/ggml-metal-ops.cpp`
  `ggml_metal_op_mul_mat` (~line 2057): ext gate (now
  `GGML_METAL_MV_EXT`-guarded), `ne11_mm_min`, mm branch, mv fallback.
  Canonical copy: `vendor/llamafile/llama.cpp/ggml/src/ggml-metal/` +
  our patch `llama.cpp.patches/patches/ggml_src_ggml-metal_ggml-metal-ops.cpp.patch`.
- Kernels: `ggml-metal.metal` — `kernel_mul_mv_ext_q4_0_f32_r1_{2..5}`
  (templated `mul_mv_ext_q4_f32_disp`), `kernel_mul_mm_*`, plain
  `kernel_mul_mv_q4_0_f32`.
- Pipeline lookup: `ggml-metal-device.cpp`
  `ggml_metal_library_get_pipeline_mul_mv_ext` (~line 653).
- Shader compile: `ggml-metal-device.m` (~line 200-250) + llamafile-side
  preprocessing/build in `vendor/llamafile/llamafile/metal.c`
  (`PreprocessMetalShader`, `BuildMetal`).

## Current state (post d4c192d)

| Config (Metal, 256-token greedy) | tok/s |
|---|---|
| no-spec baseline | 13.3 |
| draft-mtp n=1 | 20.0 |
| draft-mtp n=2 (shipped default) | **20.2** |
| draft-mtp n=3 | 18.2 |
| draft-mtp n=4 | 15.7 |
| ngram-simple | 13.3 |

CPU draft-mtp n=4 remains +14% over CPU baseline. Greedy parity and full
smoke verified on the fixed build. If verify(b) goes near-flat, re-sweep n
— the optimum should move to n≈4-6 and ~23-27 tok/s.
