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
| 6. Verify on M4 | ❌ TODO |
| 7. Bench + package + patch extraction | ❌ TODO |

## The working invocation (local paths skip sibling auto-discovery — `-md` is mandatory)

```
bin/llamafile --server \
  -m models/gemma-4-12b-it-qat-q4_0.gguf \
  -md models/mtp-gemma-4-12b-it-qat-q4_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 4 \
  --mmproj models/mmproj-gemma-4-12b-it-qat-q4_0.gguf --no-mmproj-offload \
  -ngl 999 -c 8192 --host 127.0.0.1 --port 8090
```

## Remaining work (in order)

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
5. **Patch extraction**: snapshot-diff vs `/tmp/mtp-snapshots/` (existence
   unverified after reboot/cleanup — if gone, re-derive: overlay writes are
   working-tree-only in the submodule; local submodule commits are the
   delta).
6. **Restart prod server**: `.scratch/restart-prod-server.sh` holds the
   exact argv of the production instance (mm-embedding worktree, port 8080,
   `-ngl 0`) that was stopped for this window.

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

- 4 commits inside `vendor/llamafile` (`1a50723..05cfc57`: pin bump,
  cuda/vulkan patch regen, ngram-mod overlay-patch drop, mtmd `bool
  placeholder` API fix in chatbot_*.cpp) exist ONLY locally — must be
  pushed to a llamafile fork before the recorded submodule SHA is
  fetchable by anyone else.
- Pooled-embeddings bug (patch 0001) still unfixed upstream at 04eb4c446d
  (`GGML_UNUSED(embd_all)` in both kv-cache files).
- 12B drafter untested upstream (author: 31B/26B-A4B only). E4B/E2B fail
  upstream due to centroid LM heads; our 12B verified tied-dense.
- gemma4uv image-perception bug (fine geometry scrambled) is owned by the
  mm-embedding-dev branch — image smoke failures may be THAT bug, not ours;
  compare against prod behavior before blaming MTP changes.
- Drafter reads the target's q4_0 embedding table at graph build — small
  greedy-parity drift vs bf16 transformers reference is expected.
