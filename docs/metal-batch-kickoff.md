# Kickoff: Metal small/mid-batch matmul — the remaining ~30% (and 4× prefill)

> Opening prompt for the follow-up Metal-kernel sessions. Encodes everything
> measured on 2026-06-11 while fixing the draft-mtp regression (fork commit
> `d4c192d`, parent `04d22d4`) so nothing has to be rediscovered. Companions:
> `docs/mtp-status.md` ("Metal MTP slowdown: root cause & fix" + traps),
> `docs/mtp-metal-kickoff.md` (the previous, resolved mission).

> **PROGRESS 2026-06-11 (same day, later session):** H-compile is REFUTED
> and H-dispatch is partially done. Stock upstream llama.cpp built at our
> exact pin on the same M4 shows the SAME slow numbers (instrument-matched
> server probe: b=2/4/8/10 → 108/194/381/468 ms, ext kernels active), and
> brew's prebuilt (offline-metallib, ggml master d2462f8f7 with
> byte-identical kernels and dispatch) benches identically — so neither
> llamafile's runtime shader compile nor a missing upstream fix is the
> cause. Our fork is now uniformly ≥ stock at every width. Shipped in
> 0.10.5: `ne11_mm_min` 8 → 12 (mv covers b≤12: 259-337 ms vs mm's
> 453-468 ms) and dylib error-log forwarding. What remains of this mission
> is REAL KERNEL WORK (H-kernel): the mv slope (~28 ms/extra token) and
> the mm base cost (~360-420 ms flat) — likely needs a fused
> multi-column mv kernel done right for M4, and profiling requires full
> Xcode (no `xcrun metal` on CLT-only machines; GPU captures unavailable).
> Note `xcrun metal` absence also means offline metallib can't be built
> or shipped from this machine. The +30% estimate stands.

> **SESSION 2 PLAN (Xcode kernel patching — Sebastian is installing full
> Xcode).** With Xcode in place, the blockers fall: `xcrun metal` exists
> and, more importantly, Metal GPU captures / Instruments profiling work.
> Order of attack:
> 1. `sudo xcode-select -s /Applications/Xcode.app` + accept license;
>    verify `xcrun -sdk macosx metal --version`.
> 2. Profile ONE slow case first — a b=4 decode (mv path) and a b=13
>    decode (mm path) — via Instruments "Metal System Trace" or
>    programmatic capture (`MTLCaptureManager`, triggerable from the
>    dylib with a small patch; remember the manual-dylib-build loop
>    below, ~30 s/iteration). Get per-kernel GPU times: how much of the
>    420 ms mm cost is the matmul kernels vs everything else; whether mv's
>    28 ms/token is bandwidth (expected re-reads) or occupancy.
> 3. Only then patch kernels: candidates are (a) a fused multi-column mv
>    (stream weights once, b accumulators — what ext should have been;
>    check why ext loses: occupancy? nxpsg layout? dequant duplication?),
>    (b) mm tile/simdgroup params for small N, (c) re-tune dispatch
>    thresholds per measured tables.
> 4. Verification gates and tooling are unchanged (probe script, n-sweep,
>    parity, smoke; bench against baseline 13.3 and n=2 20.1).
> Upstream watch (2026-06-11 recon): #24267+#24277 (shared-cells fixes)
> are ALREADY BACKPORTED (fork 669ed81; --fit works with -md now);
> #24086 D2D-copy removal is Qwen-gated-delta-net only (not our path);
> #24282 added Gemma-4 E2B/E4B assistants (small-Mac drafter option);
> #24480 (open) fixes Gemma4 MTP llama-server on Windows; DFlash
> block-drafting is in vLLM + draft llama.cpp PR #22105 — re-check all
> before starting kernel work in case upstream moved.

> **PROGRESS 2026-06-11 (session 2, Xcode installed).** Toolchain live:
> Metal compiler needed `xcodebuild -downloadComponent MetalToolchain`
> (done) and works via `DEVELOPER_DIR=/Applications/Xcode.app` (global
> `xcode-select -s` still pending, not required). Three deliverables:
>
> 1. **Per-op GPU profiler (new tool).** `GGML_METAL_PROFILE_COMPUTE=N`
>    profiles the Nth graph compute: one cmd_buf per op, GPUStart/EndTime,
>    aggregated table to stderr. Also `GGML_METAL_COUNT_COMPUTE=1` (numbers
>    every compute for calibration) and `GGML_METAL_MM_MIN=k` (dispatch
>    threshold override, no rebuild). All in the extracted 0.10.5 sources;
>    diff saved at `llama.cpp.patches/session-metal-profiler/`. Rebuild
>    loop is now `scripts/rebuild_metal_dylib.sh` (~3 s, parallel cc).
>    Calibration on this server config: load+model warmup = computes #1-2,
>    a `n_predict=4` warmup request = #3-7, first probe request = #8.
>    Xcode gputraces also work (`GGML_METAL_CAPTURE_COMPUTE=8` +
>    `MTL_CAPTURE_ENABLED=1`): /tmp/cap-b4.gputrace, /tmp/cap-b13.gputrace.
> 2. **Hidden prefill split found — every probe number in the table below
>    is contaminated.** The server splits every prompt of b≥5 into TWO full
>    forward passes (b-4, then 4): upstream checkpoint logic
>    (`server-context.cpp:3082`, `checkpoint_offsets[] = {4+n_ubatch, 4}`,
>    PR #20288, for SWA-model rollback — Gemma4 is SWA). On M4 Metal the
>    trailing-4 pass costs a full weight read (~130 ms). `--ctx-checkpoints
>    0` removes it: b=13 392→263 ms, b=16 472→267, b=25 359→250. Safe with
>    draft-mtp: spec rollback uses `slot.spec_ckpt`, created independently
>    of `n_ctx_checkpoints` (verified in code, lines 2543/3485). Trade-off:
>    no cheap mid-history reprocess on divergent prompts. Upstream idea:
>    skip the split when the prompt is short. NOTE: decode-time spec verify
>    batches do NOT go through this path — only prefills were affected.
> 3. **Corrected single-graph curves (ckpt off, this M4):** mv: 90/104/131/
>    160/188/216/245/274 ms at b=2..9 (slope 28.5 ms/col); mm: FLAT ~240-258
>    for b=3..16 (not 360-460 — those were 2-graph sums; mm(49)=466).
>    Crossover is b=8 ⇒ `ne11_mm_min` should be **7**, not 12 (b=9..12 drop
>    274/308/338/360 → ~245). Per-op profile of b=1 decode (98 ms
>    serialized): FFN mv kernels 60 ms, lm_head q6_K 262k-vocab single op
>    9.9 ms (≈bandwidth-bound, fine), attn matmuls ~16 ms, FA 1.4 ms. At
>    b=4 the same mv kernels are +31-38% — the slope is IN the kernels, not
>    graph overhead. H-kernel mission unchanged: fused multi-col mv would
>    put verify(b≤8) near ~100 ms; mm small-N at ~245 ms is still ~3.5
>    weight passes. Upstream recon: no movement on these kernels/thresholds
>    since our pin (which IS the Gemma4-MTP merge #23398); #24086 confirmed
>    Qwen-gated-delta-net only (`ggml_gated_delta_net` signature change),
>    no backport needed.
>
> Next: (a) end-to-end draft-mtp n-sweep with `--ctx-checkpoints 0` +
> mm_min=7, re-check optimal n; (b) ship mm_min=7 + the env overrides into
> the canonical patches; (c) H-kernel work with the profiler as the
> measurement loop. Verification gates unchanged.

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
