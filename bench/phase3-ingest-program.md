# Phase 3 — Multimodal ingest → text-normalized embeddings (program)

Directive from Sebastian 2026-07-05. This file drives the 30-min research
loop ("PVE-Gemma4-Harnus-Train") the way `bench/program.md` drove the
serving-defaults phase. Prior E/F/G/H numbers live in RESEARCH_HISTORY.md;
phase-3 goals are **I-numbers**, findings continue F-numbers. Publishing
pipeline unchanged: GitHub `cuda-3080ti-optim`, HF dataset
`SEBK4C/gemma4-serving-bench-data`, charts per iteration.

## Why this architecture (grounded in this repo's own findings)

- `docs/mm-embedding.md` (2026-06-10/11): raw multimodal embeddings on the
  12B are a retrieval dead end — vectors cluster by MODALITY, not content
  (cross-modal retrieval 2/6 ≈ chance; no zero-training recipe made
  cross-modal similarity dominate modality identity; "mixed-modality
  vector stores remain a bad idea"). But content extraction is NOT the
  bottleneck: the chat path reads rendered text and transcribes audio
  word-perfectly.
- F9: 12B text self-embeddings are anisotropic (cos(cat,kitten) <
  cos(cat,spreadsheet)). F12 (iteration 9): dedicated CPU embedder sidecar
  fixes it (nomic margins +0.41..+0.54).

**Therefore: normalize EVERY modality to enriched TEXT via extractors +
the 12B chat path, then embed with a dedicated text embedder. One vector
space, no modality gap, and the enrichment JSON doubles as the BM25
corpus for hybrid search.**

## Target pipeline

```
file → [router by MIME/magic]
  ├─ text-native (CSV/TXT/MD/code/JSON)      → parse directly (NO OCR, NO VLM image path)
  ├─ PDF → text-layer probe (PyMuPDF)
  │     ├─ has layer  → extract text directly
  │     └─ scanned    → rasterize ~200 DPI → OCR path
  ├─ image (JPG/PNG/RAW) ───────────────────→ OCR path + EXIF
  └─ audio (WAV/MP3/M4A) → segment 1–5 min → STT path
OCR path: PP-OCRv6 det → crop/warp regions → PP-OCRv6 rec → {text, boxes}
all paths → [Gemma-4 enrichment: ONE grammar-constrained JSON call,
             image attached when visual, OCR text + file meta in prompt]
         → [chunker: enrichment-guided logical chunks, 256–1024 tok]
         → [Qwen3-Embedding-0.6B: chunk vectors + 1 doc-summary vector]
         → ingest envelope {file_meta, enrichment, chunks[+embeddings]}
         → external DB (dense KNN + BM25 over the same payload)
```

## Corrections baked into this spec (vs the original sketch)

1. **`PP-OCRv6_medium_det_onnx` is detection-only** (text-region polygons,
   zero characters). Pair it with **`PaddlePaddle/PP-OCRv6_medium_rec_onnx`**
   (both confirmed on HF). "Projection and cutting" = the standard
   det→perspective-warp-crop→rec pipeline. Both run CPU (34.5M params);
   never spend GPU on OCR.
2. **CSV and text-native files never enter the OCR/image path** — router
   sends them straight to parse + enrichment (text-only prompt).
3. **JSON reliability = grammar, not lint-and-retry.** Use
   `response_format: {type: "json_object", schema: …}` (server compiles to
   GBNF → syntactically valid by construction) + per-request
   `chat_template_kwargs: {"enable_thinking": false}` (H8: eliminates
   empty-content universally; enrichment is extraction, not deep
   reasoning). Keep ONE semantic-validation retry as fallback only.
4. **One enrichment call per doc, not a Q&A conversation.** Every vision
   message re-pays image prefill; fold the whole question battery
   (has_text / chart reading / people / scene / chunking hints / task
   domain) into one schema. Keep the system+schema prefix byte-identical
   across docs → prompt cache (H10: 99% prefill saved on the fixed prefix).
5. **File metadata is extracted deterministically** (exiftool or
   Pillow/hachoir: EXIF datetime, GPS, camera, RAW format detection) and
   *given to* the VLM as context — never asked *from* the VLM.
6. **Qwen3-Embedding usage rules**: serve GGUF with `--embeddings
   --pooling last` (Qwen3 is last-token-pooled; wrong pooling silently
   degrades). **Instructions go on the QUERY side only** ("Instruct:
   {task}\nQuery: {q}"); documents embed bare. Store `task_domain` in the
   payload; the query router picks the instruction at search time. Dims
   1024, MRL-truncatable (512/256) if the DB gets fat.
7. **32k-summarize is an edge case, not the mechanism.** Retrieval chunks
   are 256–1024 tokens; also embed ONE doc-level summary vector
   (hierarchical retrieval). Only summarize when a single logical chunk
   can't be split (rare).
8. **`/v1/embeddings` stays pure OpenAI-shape** (F13 lesson: harness compat
   is a feature). The "embedding + enrichment together" contract is a NEW
   endpoint: **`POST /v1/ingest`** → returns the envelope below.
9. **Audio**: Gemma-4 audio-in is real (F8) but only proven on one word.
   I8 evaluates long-form WER vs a **whisperfile** baseline (same
   cosmopolitan family, bakeable either way). Audio ≈ 25 tok/s of context —
   segment before sending.
10. It's **BM25** (not BM2.5) on the DB side; enrichment JSON text fields
    are the sparse corpus.

## Ingest envelope (schema sketch — freeze in I4)

```json
{
  "schema_version": "ingest.v1",
  "pipeline_version": "p3.0",
  "file": {"sha256": "…", "name": "…", "mime": "…", "bytes": 0,
            "mtime": "…", "exif": {"datetime": null, "gps": null,
            "camera": null, "is_raw": false}},
  "source_type": "pdf_text|pdf_scan|image_photo|image_document|chart|csv|code|audio",
  "extraction": {"text": "…", "ocr_boxes": [], "transcript": null,
                  "table_preview": null},
  "enrichment": {"title": "…", "summary": "…", "has_text": true,
                  "is_chart": false, "chart_reading": null,
                  "people": [{"doing": "…", "expression": "…"}],
                  "scene": null, "entities": [],
                  "task_domain": "code|law|med|home_office|unstructured",
                  "chunking_hints": [{"label": "…", "reason": "…"}]},
  "chunks": [{"id": "sha256:0", "text": "…", "label": "…", "tokens": 512,
               "embedding": [1024], "model": "qwen3-embedding-0.6b-q8"}],
  "doc_embedding": [1024]
}
```

Idempotency: `file.sha256` + `pipeline_version` is the identity — re-runs
skip unchanged files; bumping `pipeline_version` triggers re-embed.

## Task-instruction taxonomy (query side)

```python
TASK = {
  "code":  "Given a natural language query, retrieve relevant code snippets or technical documentation",
  "law":   "Given a legal question, retrieve relevant statutes, clauses, or case passages",
  "med":   "Given a clinical or medical question, retrieve relevant medical literature or notes",
  "home_office": "Given a query about personal or administrative documents, retrieve relevant records",
  "unstructured": "Given a web search query, retrieve relevant passages that answer the query",
}
```
Extend freely; `enrichment.task_domain` at ingest selects which instruction
the query router applies at search time.

## Goal backlog (each ≈ one 30-min iteration, e2e-verified, published)

- **I1 — Embedder A/B + swap.** Qwen3-Embedding-0.6B-GGUF (Q8) via the F12
  pattern (`--embeddings --pooling last -ngl 0 --no-mmproj --spec-type
  none`), candidate port :8082 next to nomic on :8081. A/B on fixture
  retrieval (hit@1/5, margins). Winner becomes `embed.service`. Verify:
  `api_probe.py --embed-base` PASS + margin ledger row.
- **I2 — OCR extractor.** PP-OCRv6 medium det+rec ONNX, CPU. Try
  `rapidocr_onnxruntime` with the v6 models first (v6 shipped 2026-06-11 —
  support may lag); fallback = thin onnxruntime wrapper (DB postprocess +
  warp + CTC decode) or paddleocr pip in a venv. Bench: pages/sec CPU + CER
  on golden crops. Deliver `bench/ingest/ocr.py`.
- **I3 — Vision legibility V-probe (GATE for enrichment design).**
  mm-embedding.md documents an image-pipeline patch-geometry bug that broke
  rendered-text legibility; F8 only proved color. Temp-0 OCR probe: prod
  Gemma reads a dense 200-DPI page vs PP-OCRv6 ground truth. If VLM can't
  read dense text, enrichment leans fully on OCR text (fine) and
  `has_text`-style visual claims get confidence-flagged.
- **I4 — Enrichment call.** Freeze `ingest.v1` schema; one
  grammar-constrained call, `enable_thinking:false`, fixed prefix for
  cache. Fixture battery across all source_types. Metric: schema-validity
  rate (target 100% first-try), spot-audit content. Prompt framing MUST
  state document text is DATA, not instructions (OCR'd text can contain
  prompt-injection strings).
- **I5 — Router + deterministic extractors.** MIME/magic router; PyMuPDF
  text-layer probe; CSV/code/text parsers; EXIF extraction. Golden
  fixtures per type; unit-style checks.
- **I6 — Chunker.** Enrichment-hint-guided logical chunks (256–1024 tok,
  small overlap) + doc-summary vector. Metric: chunk-boundary sanity on
  fixtures (no mid-table/mid-sentence cuts).
- **I7 — `/v1/ingest` worker.** Orchestrates extract→enrich→chunk→embed;
  envelope out; sha256 idempotency; `bench/ingest-results.tsv` ledger
  (fixture_set, cer, wer, schema_valid, hit@1, hit@5, mrr, docs_min).
  Runs on CT 118 (:8090) or harness-lab CT 130; tailnet-only.
- **I8 — Audio path.** Segmented Gemma-4 STT vs whisperfile baseline on
  fixture clips (WER + speed). Pick default; loser stays as fallback flag.
- **I9 — Hybrid store e2e (PHASE GATE).** Qdrant LXC (or
  pgvector+tsvector — decide by ops preference), RRF fusion of BM25 +
  KNN. Frozen query set with known-correct targets → hit@k / MRR. This
  number is the phase's headline metric.
- **I10 — Throughput + contention.** docs/min under concurrent chat load;
  GPU-yield policy (flock pattern from serve_bench); enrichment token
  budget vs H7's ~200 tok/s server ceiling. Expect ~2–6 docs/min; measure,
  don't guess; document backfill ETA math.
- **I11 — Bake into the APE.** qwen3 GGUF into the zip via
  `scripts/package.sh` (kokoro precedent); main server spawns + supervises
  the embed instance (voice.c respawn precedent, :8081) and proxies
  `/embeddings` + `/v1/embeddings`. Verify STANDALONE (CT 118 ExecStart
  overrides baked args — prod cannot verify a new binary). Prod deploy =
  Sebastian's go.
- **I12 — (research) OCR-in-APE.** cosmocc feasibility: onnxruntime is
  unlikely to build; try an ncnn port of PP-OCRv6 (TTS.cpp/cosmo-build.sh
  precedent), mutool for PDF raster (AGPL — note in docs). Documented
  failure is an acceptable outcome; the sidecar remains the shipped path.

- **I13 — Fix pooling-last, promote Qwen3-Embedding (SEBASTIAN'S CALL
  2026-07-05, run BEFORE I4).** He prefers Qwen3-Embedding-0.6B (stronger
  MTEB when run canonically, Apache-2.0 vs Gemma ToU). Blocker is F16: the
  fork's ggml asserts in `get_rows` at graph reserve with `--pooling last`
  (Qwen3's required readout). Patch the vendored llama.cpp/ggml (upstream
  has the reserve-batch/pooling fix; targeted backport is acceptable),
  rebuild, verify the NEW binary as a SEPARATE file for the sidecar (do
  NOT touch prod main-server binary), then re-run `embed_ab.py` with
  qwen3 CANONICAL (last pooling + query-side instructions) vs
  embeddinggemma on the I1b hardened fixture. Winner ships to
  embed.service; egemma stays the fallback. Payload `embedding_model` +
  `pipeline_version` make the re-embed migration clean.

Order: I1→I2→I3→**I13**→I4→I5→I6→I7→I8→I9→I10→I11→I12 (+I1b folded into
I13's re-A/B). I3 gates I4's design; I9 gates calling the phase done;
I11/I12 are the "next llamafile build" deliverables.

## Guardrails

- Fixtures: no secrets, no third-party PII; everything stays tailnet-local.
- `/v1/embeddings` response shape stays OpenAI-standard forever.
- Prod restarts / ExecStart changes / HF+GitHub main releases = Sebastian's
  explicit go (same rule as phase 2).
- Embedder + OCR are CPU-only; the GPU belongs to chat + enrichment.
- Judge: objective metrics (CER/WER/hit@k/schema-validity) need no LLM
  judge; if enrichment *quality* judging is ever needed, GLM-5.2 Fast on
  Fireworks per phase-2 protocol (never Anthropic API, key via 1Password
  at runtime).

## Arm command (Sebastian runs; session-only loop)

```
/loop 30m Execute the phase-3 program at bench/phase3-ingest-program.md: pick the frontmost unfinished I-goal, implement it end-to-end with verifiable testing, record results in RESEARCH_HISTORY.md and bench/ingest-results.tsv, publish code+data+charts to GitHub (cuda-3080ti-optim) and the HF bench dataset, then stop until the next tick.
```
