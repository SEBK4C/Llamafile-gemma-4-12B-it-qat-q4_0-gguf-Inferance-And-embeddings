# Ingest & Retrieval Guide (phase 3 — consolidated)

Everything-to-enriched-text: any file or API text becomes one `ingest.v1`
envelope — enrichment JSON + fidelity verdict + logical chunks + 1024-dim
vectors — retrievable by natural language from one text vector space.
Every number below has a ledger row in `bench/ingest-results.tsv` and data
on the [HF bench dataset](https://huggingface.co/datasets/SEBK4C/gemma4-serving-bench-data);
the running narrative is `bench/RESEARCH_HISTORY.md` (I/P/Q/EM/T/V goals).

## Why this architecture

Raw multimodal embeddings cluster by modality, not content
(docs/mm-embedding.md: cross-modal retrieval ≈ chance) — but the chat path
extracts content near-perfectly. So: OCR/STT/parse to text → ONE
grammar-constrained enrichment call → deterministic fidelity gate →
hint-guided chunking → a dedicated text embedder. The 12B's own
`/v1/embeddings` stays non-semantic (F9); retrieval vectors come from the
Qwen3-Embedding sidecar (`/embed/v1/embeddings`, fork patch 0019 for
pooling-last).

## Surfaces

| surface | what | where |
|---|---|---|
| `POST /v1/ingest` (v0.6.0 APE) | text → envelope, fully in-file | server, needs baked embedder |
| `bench/ingest/ingest_worker.py` | files (PDF/images/audio/CSV/code) → envelopes, pipelined | Python worker, OCR/EXIF/STT |
| `bench/ingest/hybrid_store.py` | envelopes → FTS5+dense store; dense-first router queries | measurement rig; Qdrant/pg later |
| `/embed/v1/embeddings` `/embed/tokenize` | raw embedding + tokenizer | baked sidecar (or CT embed.service) |

## The measured picture (2026-07-06)

**Throughput** (frozen config: pipeline C=2 · enrichment v1+DRY · PP-OCRv6
medium@8thr · qwen3-last sidecar@8thr):
- 87-file / 8-type batch: **19.4 docs/min, 0 failures**; enrichment holds
  the GPU ~96% busy — it is THE bottleneck (P1/P2/V-1). Single doc ≈ 4.5 s.
- OCR 1.8 s/A4 (CER 0.0000 golden, word-F1 0.925 real FUNSD); embed ~30
  ms/doc; route/chunk ≈ free. Server ceiling ~200 tok/s shared (H7) —
  budget ingest + chat together.

**Enrichment quality**: schema-valid 87/87 at scale (grammar = validity by
construction); prompt-injection probe resisted; CORD-labeled receipts:
key-field recall **100%** end-to-end, gate false-drop **0%**,
hallucinated-number rate **6.7%** (flag-only class).

**Fidelity gate** (deterministic, per-doc): catch-rate **100%** on
composed dates / digit mutations / fabricated amounts / fake names (48
seeded probes); grounding rate stable at **83.4%** across populations;
known residual: unit-swap class (never observed in real output; verifier
second-opinion REJECTED — F26: 50% false-rejects on noisy OCR).

**Retrieval** (42 frozen known-target queries, 87 multimodal docs):
**hit@1 0.929 · hit@3 0.976 · MRR 0.955** — photos by caption, receipts
by total, audio by phrase, PDFs by fact. NFCorpus sanity: nDCG@10 0.3626
(published band for the model class).

## Defaults that are load-bearing (change = re-measure)

- **Query-side instructions, always** — `Instruct: {TASK[domain]}\nQuery:`
  (bare queries: −17% rel nDCG; domain phrasing beats generic +2.7%, EM4).
  Docs embed bare. `task_domain` from enrichment picks the instruction.
- **Doc vector = composite F**: title+summary+scene+entities+people +
  lead-chunk[:400] (EM2: 0.881→0.929 hit@1).
- **Dense-first router**: BM25 fused (3:1, k60) only on digit-bearing
  queries (EM6: always-fusing costs; re-sweep ≥1000 docs).
- **1024 dims** (EM3: 512 = −2.4% rel for half the index — sanctioned for
  large archives; envelope records dims for clean migration).
- **Enrichment sampler**: temp 0 + DRY (F24: greedy+grammar loops inside
  unbounded JSON strings) + `enable_thinking:false` (H8) + byte-identical
  prefix (H10 cache: ~400 tok reused/call).

## Footguns (earned the hard way)

- F16 pooling-last crash → fork patch 0019 (and always `--pooling last`
  for Qwen3). F17 sidecars need `LLAMAFILE_NO_VOICE=1`. F20 VLM verbatim
  errors are plausible-token rewrites → OCR text is authoritative. F21
  enum values must be IN the prompt, not only the grammar. F22 native
  audio is SPEECH-ONLY (WER 3.0%; ESC-50 ≈ 0) and inputs must be 16 kHz
  mono. F23 silent tokenizer fallback under-counts → tokenize units once
  via `/embed/tokenize`. F24 see sampler above. F25 supervisors are
  parent-liveness-tied (SIGKILL-safe). F26 model verify-passes over-reject
  on noisy OCR — the deterministic gate stands alone.

## Reproduce

```sh
V=bench/ingest; B=<gemma-base>; E=$B/embed
python3 $V/ocr.py --bench $V/fixtures                 # OCR CER/speed
python3 $V/enrich.py --base $B --bench                # schema/injection battery
python3 $V/q3_seeded.py                               # gate catch-rates
python3 $V/ingest_worker.py --batch files.txt --out env/ --base $B --embed-base $E
python3 $V/hybrid_store.py build --envelopes env/ --db s.db --embed-base $E
python3 $V/hybrid_store.py eval  --db s.db --embed-base $E   # hit@k/MRR
python3 $V/embed_real_bench.py --embed-base $E        # NFCorpus nDCG
```
