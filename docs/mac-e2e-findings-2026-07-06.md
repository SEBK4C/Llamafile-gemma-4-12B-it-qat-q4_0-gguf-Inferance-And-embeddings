# Mac (Apple Silicon) E2E findings — Gemma 4 12B llamafile

**Machine:** MacBook Pro, Apple M1 Pro, 32 GB unified (Metal window 25.5 GB) · macOS Darwin 25.5.0
**Artifacts:** llamafile 0.10.7 standalone + external GGUFs (main / mmproj / MTP drafter), QAT-Q4_0
**Date:** 2026-07-06 · previously the stack was CUDA-tested only (RTX 3080 Ti CT 118); this is the first Mac end-to-end pass plus the start of the Mac autoresearch ratchet (`bench/program-mac.md`).

## Headline results

**All features work on Metal — 13/13 E2E tests PASS** via `./scripts/mac-full-test.sh --start-server`:
chat completions, embeddings (dim 3840, correct semantic ordering), batch-vs-solo embedding
consistency (patch 0001 holds, drift < 1e-6), concurrent mixed load, KV slot save→erase→restore
(219/219 tokens reused), image input through CPU-side mmproj (correct OCR of the test image),
MTP speculative decoding (drafter loads, acceptance 0.91), Metal batch-decode profile.

## Measured baselines (M1 Pro 32 GB)

| metric | value | note |
|---|---|---|
| decode, clean state | **21.5–22.2 tok/s** | `-ngl 999 -fa on`; identical at np1/c4096 and np2/c8192 |
| decode, temp 1.0 battery | **~15 tok/s** | MTP acceptance drops under diverse sampling |
| decode, post-battery | ~16.7 tok/s | residual slot/KV/image state — measure clean-state only |
| MTP acceptance (WAL prose probe) | 0.91 | but see MTP verdict below |
| batch cost (probe_batch_cost) | b=2: 40 ms/tok · b=17: 13 ms/tok · b=65: 8.6 ms/tok | linear-ish small-b slope = the known Metal small-batch matmul issue |
| server cold start to /health | 3–5 s | weights mmap'd |

**MTP is break-even on M1 Pro, not 1.6×.** Despite 0.91 draft acceptance, the batch-2 verify
step costs ~1.75× a single decode (40.2 vs ~46 ms/tok inverse), which cancels the acceptance
gain: 21.5–22.2 tok/s with *or* without the drafter. The M4 numbers in the README don't
transfer to M1 Pro. Keep MTP for telemetry, or strip it (`build-gemma4-fast.sh`) for the same
speed with less memory.

## Bugs found and fixed (all committed)

1. **`serve.sh` OOM'd Metal on launch-default config** — `-ub 2048` exhausts Metal command
   buffers under concurrent slots (`kIOGPUCommandBufferCallbackErrorOutOfMemory`).
   Fix: Darwin-conditional ubatch cap of 1024 (verified with ~800-token pooled embeddings +
   concurrent chat, zero OOM). `scripts/serve.sh`
2. **`smoke_test.py` false-failed on the thinking channel** — `temperature=0` + small
   `max_tokens` returns EMPTY `content` (reasoning consumes the whole budget under greedy).
   Fix: tests use the Gemma 4 official sampler (temp 1.0 / top_k 64 / top_p 0.95) with
   ≥512-token budgets. `tests/smoke_test.py`
3. **Speed-gate ordering in the E2E suite** — measuring after the test battery reads
   ~17 tok/s (residual state) and false-fails the 18 tok/s floor. Fix: gate the clean-state
   measurement taken before the battery; post-battery number is INFO-only. `scripts/mac-full-test.sh`
4. **Bench harness `--max-tokens 768` (CUDA-era) floors every hard probe on Mac** — the
   reasoning channel alone runs ~700+ tokens, content comes back empty, and empty content
   scores hum/soph 1.0 and acc/cal 0. Measured: 768 → 0 content chars; 1536 → full answers;
   enumeration probes need more. Fix: Mac harness forces 3072 and documents it as
   measurement-locked. `bench/mac_serve_bench.py`
5. **Bench harness 180 s request timeout (CUDA-speed assumption) silently drops probes** —
   at ~15 tok/s a probe using the full 3072 budget needs ~205 s+; qa_speed lost 3/5 replicas
   before detection. The dropped probes are exactly the budget-exhausting ones the loop
   studies → baseline would have been biased. Fix: 360 s floor via wrapper. `bench/mac_serve_bench.py`
6. **`thinking_budget_tokens` is a UI-side field the server ignores** — the real per-request
   API in the pinned upstream (ggml-org/llama.cpp `04eb4c4`, `server-task.cpp:497-515`) is
   `reasoning_budget_tokens` + `reasoning_budget_start_tag`/`end_tag`/`message`; Gemma 4's
   thought channel delimiters (from the live `/props` template) are `<|channel>thought\n` …
   `<channel|>`, and `chat_template_kwargs {enable_thinking:false}` is the binary fallback.
   The queued candidate uses the verified fields. `bench/candidates/thinking-budget-512.json`

## Model-behavior findings (Mac ratchet, iterations 1–7)

**The #1 serving-quality failure: empty answers on enumeration asks.** Forensics on the two
`loops` probes with shipped defaults: **10/10 turns** ended `finish=length` with
**content = 0 chars** — the model drafts and re-drafts the list inside `reasoning_content`
(6.0–6.7k chars/turn) until the budget dies, and the user gets nothing. This also produces
detector-artifact "rep" trips: the re-drafted list lines (`'13. Chirpy'` ×5) trip the
whole-line detector on the reasoning channel while `content` never loops.

**DRY holds on Metal; greedy is the loop trigger — CUDA verdict reproduced.** Harness A/B on
the loops battery: shipped DRY-on defaults `rep = 0.5` (all trips = reasoning re-drafts, see
above) vs greedy/DRY-off `rep = 1.0` (true degenerate loops, plus a timeout). Sampler-side
anti-loop insurance works as designed; never serve greedy.

**Next candidate (queued, source-verified, awaiting baseline gates):** cap the thought channel
per-request at 512 tokens (`reasoning_budget_tokens`) so enumeration asks answer instead of
ruminating — hypothesis: fixes empty-content floors without hurting acc on hard probes that
need reasoning.

## Autoresearch ratchet state

- Ledger (`bench/serving-results.tsv`): 3 diagnostic rows (`mac-harness-smoke`,
  `mac-loops-dry`, `mac-loops-greedy`) — none gate.
- **Full 5×5 baseline is running** (detached, `--agent-id mac-baseline`, real GLM-5.2
  Fireworks judge, ~3–4 h) — its `status=baseline` row unlocks gated candidate work.
- Open items: 2 `REPLACE_ME` calibration probes in `bench/probes.json` need the real
  "too-heavy" and "Kitty" examples; judge key is provided per-session (never on disk).
- Orchestration lesson for the loop: background wait-wrappers must not embed watch patterns
  in their own command line (pgrep self-match deadlock) — use file-based waiters polling PIDs.

## Files

| file | role |
|---|---|
| `scripts/mac-full-test.sh` | 13-test Mac E2E suite (`--start-server` for one-shot runs) |
| `bench/mac_serve_bench.py` | Mac wrapper of the frozen harness (localhost, local KV purge, LAT_NORM 25, 3072 budget, 360 s timeout) |
| `bench/program-mac.md` | Mac adaptation of the autoresearch protocol + measured constraints |
| `bench/candidates/thinking-budget-512.json` | first candidate, source-verified reasoning-budget fields |
| `docs/PLATFORM-NOTES.md` | updated compatibility matrix + M1 Pro baseline section |
