# v0.6.1 — `/v1/embeddings` is semantically useful by default

One change, requested and to the point: the standard OpenAI embeddings
endpoint now transparently serves the baked Qwen3-Embedding-0.6B sidecar's
**1024-dim retrieval-grade vectors** whenever the embedding payload is
present (it is, in the packaged file). Every OpenAI SDK gets good
embeddings with zero configuration — no more "API-compatible but not
semantically useful" caveat.

- Best use: embed documents bare; prefix queries with
  `Instruct: <task>\nQuery: ` (measured: domain phrasing +2.7% nDCG,
  skipping the instruction −17%).
- Raw native 3840-dim vectors (anisotropic; research only): per-request
  header `X-Raw-Embeddings: 1`, or run with `LLAMAFILE_NO_EMBED=1`.
- `/embed/v1/*` (health, tokenize, embeddings) unchanged; `/v1/ingest`
  unchanged.
- Probe suite on this build: **18 PASS / 0 FAIL / 1 skip** — the
  long-standing documented embeddings FAIL is gone because the endpoint
  is now correct, not because the test got softer.
