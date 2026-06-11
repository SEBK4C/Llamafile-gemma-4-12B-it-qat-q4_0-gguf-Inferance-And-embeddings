# MTP integration — status & handoff (2026-06-11)

> Context-collapse document: everything needed to continue the Gemma4 MTP
> integration without the original conversation. Read together with
> `MTP-prompt.md` (plan + architecture reference) and
> `docs/mtp-upstream-recon.md` (upstream wiring, file:line refs).

## Where we are

Upstream llama.cpp PR #23398 (Gemma4 MTP, merge commit `04eb4c446d`) was
integrated on branch `mtp-gemma4-drafter`. **Build and conversion are DONE
and verified; runtime verification/bench/package are NOT done** (a
verification agent hit a session limit after ~67 steps and persisted
nothing — no commits, no logs; redo from scratch).

| Phase | Status |
|---|---|
| 1. Pin bump to 04eb4c446d | ✅ commit `17118db` |
| 2. Patch-stack rebase | ✅ commit `3d423f6` |
| 3. cosmocc build green | ✅ `bin/llamafile` (50 MB) has `draft-mtp` + `gemma4-assistant` |
| 4. Upstream recon | ✅ `docs/mtp-upstream-recon.md` (commit `5b045bc`) |
| 5. Drafter GGUF | ✅ `models/mtp-gemma-4-12b-it-qat-q4_0.gguf` (q8_0, 449 MB, all tensor checks pass) |
| 6. Verify on M4 | ✅ 2026-06-11: loads, drafts, 51% acceptance; full smoke PASS |
| 7. Bench + package | ✅ measured (see below); 7.1G artifact with drafter baked in |

## Verification results (2026-06-11, M4 Mac mini 16GB, greedy, 400-token prose)

> **2026-06-11 (later): the Metal loss is FIXED** — root cause was the ggml
> Metal `mul_mv_ext` small-batch kernels, not CPU placement. See "Metal MTP
> slowdown: root cause & fix" below. Post-fix Metal numbers: n=2 **20.2
> tok/s (1.52× baseline)**, n=4 15.5-15.7, baseline/ngram unchanged.

Pre-fix numbers (kept for history):

| Config | gen tok/s | vs baseline |
|---|---|---|
| CPU no-spec | 7.95 | — |
| CPU draft-mtp n=4 | 9.08 | **+14%** |
| Metal no-spec | 13.23 | — |
| Metal draft-mtp n=4 | 9.15 | **−31%** |
| Metal draft-mtp n=8 | 11.39 | −14% (acceptance halves to 30%) |
| Metal ngram-simple | 13.18 | ±0 |

~~**Conclusion: draft-mtp is a CPU-mode win and a Metal loss.**~~ The
"scheduler places the MTP graph on CPU" hypothesis was investigated with
`GGML_SCHED_DEBUG=2` and is **refuted**: both the target and drafter graphs
schedule fully on MTL0 (only the standard token-embd GET_ROWS is on CPU,
same as baseline), graph reuse works, and the drafter costs a flat ~6 ms
per drafted token. The real cause is below.

## Metal MTP slowdown: root cause & fix (2026-06-11, fixed in fork d4c192d)

**Root cause:** ggml-metal's `mul_mv_ext` "small-batch" kernels (dispatched
for q4_0 weight matmuls at ne11 ∈ [2,8]) are ~1.7× slower than the plain
mat-vec kernels on the M4 with llamafile's runtime-compiled dylib. Batched
decode cost was near-linear in batch width (~47 ms/token at b=2..10 vs
75 ms for a whole single-token decode — almost zero amortization), and
speculative *verify* batches (width n_draft+1) live exactly in that window.
That made each verify round cost ≈ b × a full decode, sinking draft-mtp
below baseline. It also explains every prior observation:

- Metal+MTP ≡ CPU+MTP was coincidence, not CPU placement.
- ngram-simple was "unaffected" only because it almost never drafts on
  neutral prose, so its verify batches stay width 1.
- Issue #23752's "n_max=0 loses 11%" does NOT reproduce here: our build
  clamps n_max=0 to 1-token drafts, which WON (+15% pre-fix) — the
  per-round fixed MTP cost (hidden-state export etc.) is small (~35 ms).
- nsg=4/8 tuning of the ext kernels changed nothing; forcing the mm
  matrix-matrix kernel down to b≥2 was even worse (~230-460 ms flat).

**Fix:** disable the ext path so small batches use the plain mv kernels
(`GGML_METAL_MV_EXT=1` re-enables for experiments). Carried as
`llama.cpp.patches/patches/ggml_src_ggml-metal_ggml-metal-ops.cpp.patch`,
fork commit `d4c192d`.

**Post-fix bench (same protocol, 256-token greedy, quiet machine):**

| Config | pre-fix tok/s | post-fix tok/s |
|---|---|---|
| Metal no-spec | 13.31 | 13.25-13.35 (unchanged) |
| Metal draft-mtp n=1 | 15.16 | 20.00 |
| Metal draft-mtp n=2 | 14.39 | **20.23 (1.52× baseline)** |
| Metal draft-mtp n=3 | 12.37 | 18.18 |
| Metal draft-mtp n=4 | 8.95 | 15.74 |
| Metal ngram-simple | 13.18 | 13.27 (unchanged) |

**Best Metal config is now `--spec-draft-n-max 2`** (n=1 is statistically
tied; n≥3 loses because verify batches re-enter the still-imperfect b≥3
zone and acceptance decays). Greedy parity verified: 400-token draft-mtp
n=4 output is byte-identical to no-spec baseline. Full smoke test PASS in
the CPU prod config + draft-mtp flags. Note: drafted/accepted counts
shifted slightly post-fix (540/263 vs 522/268 at 400 tokens) — mv vs ext
numerics flip an occasional greedy tie; output equality is the parity
check that matters.

Upstream-report material (by a human — llama.cpp rejects AI content): the
ext-kernel slowdown measurement, plus the still-odd mm-kernel cost
(~420 ms flat for b=13..25 where ~100 ms is expected — prefill is ~4×
slower than it could be, independent of MTP).

Shipped configuration: ngram-simple stays the default; draft-mtp is opt-in
via `GEMMA4_SPEC=draft-mtp` (serve.sh, auto-adds `-md … --fit off`) or for
the packaged artifact:
`./gemma4-server.llamafile -ngl 0 --fit off --spec-type draft-mtp -md /zip/mtp-gemma-4-12b-it-qat-q4_0.gguf`.
draft-mtp is the best prod spec config on CPU (+14% on neutral prose where
ngram gives ~0) — and now also the best Metal config (1.52× at n=2).

### Traps found during verification

1. **Version-keyed dylib cache**: llamafile compiles its Metal backend at
   runtime and caches per version (`~/.llamafile/v/X.Y.Z/`). After ANY
   llama.cpp pin change you MUST bump the llamafile version (done:
   0.10.4, submodule commit 8f03833) or the binary silently loads the
   stale dylib and falls back to CPU. Symptom: baseline-CPU speeds
   despite "offloaded 49/49 layers". CAUTION: the same trap applies to
   Metal-source changes WITHOUT a pin change (like fork d4c192d) — the
   binary early-returns "using cached" whenever the dylib file exists, so
   `rm -rf ~/.llamafile/v/0.10.4` after installing a new binary with
   changed Metal sources.
2. **cosmocc make has no header dependency tracking**: after editing a
   header, `rm` the dependent `.o` files or the rebuild is a no-op.
3. **The "no `ggml_metal_init` / empty `MTL :` list" symptom is a red
   herring on 0.10.4**: the dylib's GGML logger isn't wired to llamafile's
   log, so those lines NEVER appear, even when Metal works. The reliable
   Metal-is-live check is `MTL0_Mapped model buffer size` /
   `MTL0 KV buffer size` lines at load. (Debug prints inside the dylib
   need plain `fprintf(stderr, ...)`, not `GGML_LOG_*`.)
4. **Fast Metal printf loop** (no 10-min cosmocc rebuild): edit the
   extracted sources in `~/.llamafile/v/0.10.4/`, compile the dylib
   manually (mirror the cc flags from `llamafile/metal.c` BuildMetal),
   and start the server — an existing dylib suppresses re-extraction.
   You CANNOT just edit the extracted source and `rm` the dylib:
   extraction byte-compares against the zip member and clobbers local
   edits whenever it runs. ~30 s per iteration.
5. **`--recompile` is rejected in `--server` mode** (parser error), even
   though `FLAG_recompile` is honored by BuildMetal. Use the
   delete-the-cache-dir workaround instead.
6. **MTP acceptance can degrade after a slot KV restore** (observed
   2026-06-11 on the packaged artifact: 23% acceptance / 12.7 tok/s on a
   request served right after the smoke test's save→erase→restore cycle,
   vs a reproducible 52% / 17.4-17.7 tok/s on a fresh server). Speed-only —
   verification keeps outputs exact. Suspect: the drafter's per-seq
   cross-batch carryover (speculative.cpp `common_speculative_impl_draft_mtp`)
   not being invalidated on restore. Small, unfixed; investigate if
   restored-slot throughput matters.
7. **`-ub 2048` is broken on Metal** (found 2026-06-11 while packaging):
   any decode fails with `graph_compute ... error -1`, regardless of spec
   type, MTP, ext kernels, or other instances running — pre-existing, NOT
   from the d4c192d fix (verified via GGML_METAL_MV_EXT=1). Ceiling is
   between 1024 (works) and 1536 (fails). The packaged artifact now bakes
   `-ub 1024` (caps packaged embedding inputs at 1024 tokens; serve.sh/CPU
   paths keep GEMMA4_UBATCH=2048). The pre-fix package was only ever
   validated at `-ngl 0`, which is why this never surfaced.

## The working invocation (local paths skip sibling auto-discovery — `-md` is mandatory)

```
bin/llamafile --server \
  -m models/gemma-4-12b-it-qat-q4_0.gguf \
  -md models/mtp-gemma-4-12b-it-qat-q4_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 4 \
  --mmproj models/mmproj-gemma-4-12b-it-qat-q4_0.gguf --no-mmproj-offload \
  -ngl 999 -c 8192 --host 127.0.0.1 --port 8090
```

## Original work plan (all done except where noted)

1. **Load + draft check (Metal, -ngl 999)**: greedy chat (temp 0,
   max_tokens ≥256 — thinking channel), confirm drafting + acceptance >50%
   in slot metrics. Known fallbacks: `--fit off` (fit probe builds an MTP
   ctx without ctx_other and can crash), `-nocb` (one upstream user).
2. **Regression smoke (CPU, prod-like config + draft-mtp flags)**:
   `tests/smoke_test.py` against port 8090 — chat, text+media embeddings,
   multimodal chat, KV save/restore must all still pass.
3. **Bench (Metal)**: `tests/bench_spec.py` — draft-mtp vs ngram-simple vs
   none. Baseline 13.4 tok/s; target >1.5× on prose.
4. **Wire defaults**: `scripts/serve.sh` + `scripts/package.sh` pass
   `-md … --spec-type draft-mtp` when `models/mtp-*.gguf` exists (env
   override to fall back to ngram-simple). `make package`, smoke packaged
   artifact, kill it.
5. **Patch extraction**: DEFERRED by design. The vendor delta lives as 5
   commits inside the `vendor/llamafile` submodule (`1a50723..8f03833`);
   converting them to `lf-*.patch` files would also require reordering
   `make setup` (lf-patches must apply BEFORE the llamafile overlay setup,
   since two commits regenerate overlay patch files). Until then the
   submodule commits must be pushed to a llamafile fork for fresh clones
   to work. Snapshots for future diffing: `/tmp/mtp-snapshots/`.
6. **Restart prod server**: DONE, but relocated — the mm-embedding
   worktree was archived (`2026-06-11-mm-embedding-dev-history.zip`), so
   prod now runs from the main repo
   (`~/Projects/Llamafile-gemma-4-12B-…`) via `GEMMA4_NGL=0 scripts/serve.sh`,
   same flags/port as before.

## Environment constraints (violating these has burned us already)

- **A sibling Claude session shares this machine** (`~/Projects/mm-embedding-gemma4`,
  branch mm-embedding-dev). NEVER `pkill -f` by pattern — kill only PIDs
  you started. Their memory note says our 8090 cycling killed their server
  twice.
- One 12B per GPU; CPU(-ngl 0) + Metal instances coexist (models/ symlinks
  share mmap pages). Metal: `-ngl 999` or `-ngl 0` only, never partial.
  Projector always `--no-mmproj-offload`. Media *embeddings* segfault on
  Metal — exercise them only in `-ngl 0` runs.
- lldb can't debug APE binaries; printf+rebuild
  (`vendor/llamafile/.cosmocc/4.0.2/bin/make -j10 o//llamafile/llamafile`,
  ~10 min) is the only loop.
- BSD `patch` may silently AUTO-REVERSE overlay patches during `make
  setup` — grep setup output for "Reversed" after any pin change.

## Open items beyond this branch

- ~~Submodule commits only local~~ RESOLVED 2026-06-11: the 5 vendor
  commits (`1a50723..8f03833`) are pushed to
  https://github.com/SEBK4C/llamafile branch `mtp-gemma4-drafter`, and
  `.gitmodules` now points at that fork (commit `e35f821`). The nested
  llama.cpp gitlink (04eb4c44) is upstream ggml-org — fetchable as-is.
- Pooled-embeddings bug (patch 0001) still unfixed upstream at 04eb4c446d
  (`GGML_UNUSED(embd_all)` in both kv-cache files).
- 12B drafter untested upstream (author: 31B/26B-A4B only). E4B/E2B fail
  upstream due to centroid LM heads; our 12B verified tied-dense.
- gemma4uv image-perception bug (fine geometry scrambled) is owned by the
  mm-embedding-dev branch — image smoke failures may be THAT bug, not ours;
  compare against prod behavior before blaming MTP changes.
- Drafter reads the target's q4_0 embedding table at graph build — small
  greedy-parity drift vs bf16 transformers reference is expected.
