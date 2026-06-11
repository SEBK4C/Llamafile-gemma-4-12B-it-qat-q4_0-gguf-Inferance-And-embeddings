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

| Config | gen tok/s | vs baseline |
|---|---|---|
| CPU no-spec | 7.95 | — |
| CPU draft-mtp n=4 | 9.08 | **+14%** |
| Metal no-spec | 13.23 | — |
| Metal draft-mtp n=4 | 9.15 | **−31%** |
| Metal draft-mtp n=8 | 11.39 | −14% (acceptance halves to 30%) |
| Metal ngram-simple | 13.18 | ±0 |

**Conclusion: draft-mtp is a CPU-mode win and a Metal loss.** Metal+MTP
throughput equals CPU+MTP to within noise — strong evidence the scheduler
places the MTP/aliased-KV graph segments on CPU, dragging the whole decode
down. Upstream knows Metal MTP underperforms (issues #23752, #23011 —
closed "not a bug"); the CPU-placement mechanism observed here is sharper
than what those issues document and may be worth reporting (by a human —
llama.cpp restricts AI content).

Shipped configuration: ngram-simple stays the default; draft-mtp is opt-in
via `GEMMA4_SPEC=draft-mtp` (serve.sh, auto-adds `-md … --fit off`) or for
the packaged artifact:
`./gemma4-server.llamafile -ngl 0 --fit off --spec-type draft-mtp -md /zip/mtp-gemma-4-12b-it-qat-q4_0.gguf`.
Since prod runs `-ngl 0` anyway (Metal media-embeddings bug), draft-mtp is
the best prod spec config: +14% on neutral prose where ngram gives ~0.

### Two traps found during verification

1. **Version-keyed dylib cache**: llamafile compiles its Metal backend at
   runtime and caches per version (`~/.llamafile/v/X.Y.Z/`). After ANY
   llama.cpp pin change you MUST bump the llamafile version (done:
   0.10.4, submodule commit 8f03833) or the binary silently loads the
   stale dylib and falls back to CPU. Symptom: no `ggml_metal_init` in
   logs, empty `MTL :` feature list, baseline-CPU speeds despite
   "offloaded 49/49 layers".
2. **cosmocc make has no header dependency tracking**: after editing a
   header, `rm` the dependent `.o` files or the rebuild is a no-op.

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
