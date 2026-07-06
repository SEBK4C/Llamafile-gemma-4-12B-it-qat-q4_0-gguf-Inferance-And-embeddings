# Embedding research-improvement loop (phase-3 sub-program)

Sebastian's directive 2026-07-05: a standing autoresearch loop for the
embedding stack, run on REAL public benchmark data. Same protocol family as
`bench/program.md` (serving defaults) and the voicebench loop: FROZEN
harnesses, ONE mutable knob per iteration, powered A/B, ledger, publish,
ship-gate. Embedding goals are **EM-numbers**; findings stay F-numbers.

## Frozen harnesses (never edit mid-comparison; version-bump instead)

| id | harness | metric | data |
|---|---|---|---|
| H-A | `embed_ab.py --hard` | hit@1 / MRR / margin / ms-doc | 48 synthetic confusable docs, 29 queries (committed) |
| H-B | `real_eval.py people` retrieval | hit@1 / MRR | 14 Flickr30k people photos → enrichment text; query = held-out caption (datasets_real/, local-only) |
| H-C | `embed_real_bench.py` (EM1 builds it) | nDCG@10 / Recall@100 | BEIR NFCorpus (3.6k docs — CPU-feasible per tick) |

H-B is the phase-premise metric: photos findable by natural language
through enrichment text alone. Baseline 2026-07-05 (qwen3-canonical,
docs bare, summary+scene+people composite): **hit@1 0.857 / MRR 0.917**.
H-A baseline: hit@1 0.93 / MRR 0.966. Every candidate runs ALL harnesses.

## Mutable knobs (ONE per iteration)

- EM1 — build H-C (BEIR NFCorpus via datasets-server/resolve URLs; corpus
  embedded once per candidate config, cached by config hash).
- EM2 — doc-side composite: which enrichment fields to embed
  (summary-only vs title+summary vs +scene+people vs +entities). H-B moves.
- EM3 — MRL truncation: 1024 → 512 / 256 dims (index size & speed vs
  quality; Qwen3 is Matryoshka-trained).
- EM4 — task-instruction phrasing per domain (TASK dict wording sweep,
  query side ONLY; docs stay bare — Qwen3 canon).
- EM5 — chunk size for long docs (256/512/1024 tok) once I6 chunker lands.
- EM6 — hybrid fusion weight (RRF k, dense:BM25 ratio) once I9 store lands.
- EM7 — embedder re-match: egemma / nomic / any new small embedder vs
  qwen3-canonical, same 3 harnesses (repeat after any fork ggml sync).

## Protocol per iteration

1. Pick the frontmost EM goal. State the hypothesis in one sentence.
2. Run candidate vs CURRENT SHIPPED config on all available harnesses.
   Replicas: H-A/H-B are deterministic (temp-0 enrichment, fixed corpora) —
   1 run each; H-C is deterministic given corpus — 1 run.
3. Ledger `bench/ingest/embed-research-results.tsv`:
   `date  em_goal  candidate  hA_hit1  hA_mrr  hB_hit1  hB_mrr  hC_ndcg10  ms_doc  dims  verdict  notes`
4. **Ship-gate**: change the deployed sidecar/env only if the candidate is
   ≥ +0.02 on TWO harness primaries with NO regression > 0.01 on the third,
   or equal quality at materially lower cost (≥30% ms/doc or ≥50% index
   size). Ship = update embed.service / ingest config + docs/embeddings.md
   + rollback line. Everything else = documented finding only.
5. Publish data+chart to the HF bench dataset; commit code+ledger+history
   to `cuda-3080ti-optim` (same as phase 2). Raw dataset media NEVER leaves
   the box — metrics and model text output only.

## Standing rules

- Docs embed BARE; instructions on the query side only (Qwen3 canon, F21
  discipline: enum/task lists live in prompts, not only schemas/masks).
- One embedding_model per index generation; changes bump
  `pipeline_version` → full re-embed (envelope records both).
- Embedder + OCR stay CPU-only; GPU belongs to chat + enrichment.
- Known capability boundary (F22): native audio = SPEECH ONLY (WER ~3% on
  LibriSpeech test-clean subset); non-speech description requires a
  tagging sidecar (backlog I15) — do not spend embedding iterations on it.
