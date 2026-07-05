# Embeddings & multi-model serving

**TL;DR: do not use the main model's `/v1/embeddings` for anything semantic.**
Run a tiny dedicated embedding model next to it — the same llamafile binary
serves it. Verified end-to-end 2026-07-05.

## The problem (measured)

The 12B chat model returns valid-looking 3840-dim vectors, but they carry no
usable similarity signal — on 2 of 3 probe pairs the *unrelated* pair scored
**higher** than the related one:

| pair | related | unrelated | margin |
|---|---|---|---|
| cat ~ kitten / spreadsheet | 0.980 | 0.990 | **−0.010** |
| cat-on-mat ~ rug / revenue | 0.869 | 0.849 | +0.020 |
| bake-bread ~ baking / segfault | 0.918 | 0.953 | **−0.035** |

That's expected: decoder-only chat models without contrastive training produce
anisotropic embeddings (everything is cosine ≈ 0.9 to everything). The API
works; the geometry is meaningless. RAG built on it will retrieve noise.

## The fix: a 146 MB sidecar (same binary, second model)

The llamafile doubles as a general llama.cpp server — pass external weights
with `-m` and the baked Gemma model is ignored. nomic-embed-text-v1.5 (Q8_0,
146 MB, 768-dim, CPU-fast):

```bash
curl -L -o /opt/nomic-embed-text-v1.5.Q8_0.gguf \
  "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf"

sh ./gemma4-server.llamafile -m /opt/nomic-embed-text-v1.5.Q8_0.gguf \
   --embeddings --port 8081 --host 127.0.0.1 \
   -ngl 0 -sm layer -ctk f16 -ctv f16 --spec-type none --no-mmproj \
   -c 2048 -ub 512 -np 2 --threads 4
```

The override flags matter: the packaged defaults are tuned for the main GPU
model, so the sidecar must switch off the vision projector (`--no-mmproj`),
speculative decoding (`--spec-type none`), and GPU/KV settings
(`-ngl 0 -sm layer -ctk f16 -ctv f16`).

Same probe pairs through the sidecar:

| pair | related | unrelated | margin |
|---|---|---|---|
| cat ~ kitten / spreadsheet | 0.837 | 0.374 | **+0.463** |
| cat-on-mat ~ rug / revenue | 0.690 | 0.280 | **+0.410** |
| bake-bread ~ baking / segfault | 0.820 | 0.282 | **+0.538** |

Chart: `bench/data/embeddings_compare_20260705.png` · raw data:
[SEBK4C/gemma4-serving-bench-data](https://huggingface.co/datasets/SEBK4C/gemma4-serving-bench-data).

### systemd unit (as deployed on CT 118)

```ini
[Unit]
Description=Embedding sidecar (nomic-embed-text-v1.5 via gemma4 llamafile, CPU)
After=network.target

[Service]
ExecStart=/bin/sh /opt/gemma4-server-gpu.llamafile -m /opt/nomic-embed-text-v1.5.Q8_0.gguf --embeddings --port 8081 --host 127.0.0.1 -ngl 0 -sm layer -ctk f16 -ctv f16 --spec-type none --no-mmproj -c 2048 -ub 512 -np 2 --threads 4
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Chat on `:8080` (GPU) and embeddings on `:8081` (CPU) run side by side; the
sidecar takes ~0.02 s per request and no VRAM.

### Optional: expose it on your tailnet

```bash
tailscale serve --bg --set-path /embed http://127.0.0.1:8081
```

SDK usage: set `base_url` to `https://<your-node>.<tailnet>.ts.net/embed/v1`.

### Notes

- For retrieval workloads, nomic-embed expects task prefixes
  (`search_document: …` / `search_query: …`) for best quality.
- Any GGUF embedding model works the same way (bge, e5, EmbeddingGemma, …).
- Verify your own setup in one command:
  `python3 bench/api_probe.py --base http://127.0.0.1:8080 --embed-base http://127.0.0.1:8081`

## Update 2026-07-05 (phase-3 I1): sidecar is now embeddinggemma-300m

Three-way A/B on a 16-doc / 10-query / 4-triplet fixture
(`bench/ingest/embed_ab.py`, data in `bench/data/embed_ab_20260705.json`),
all models served by this same llamafile on CPU:

| config | dims | hit@1 | margin | ms/doc |
|---|---|---|---|---|
| nomic-raw | 768 | 1.00 | +0.377 | 45 |
| nomic-prefixed (canonical) | 768 | 1.00 | +0.172 | 28 |
| qwen3-0.6B mean-raw | 1024 | 1.00 | +0.131 | 127 |
| qwen3-0.6B mean-instr | 1024 | 1.00 | +0.131 | 122 |
| egemma-raw | 768 | 1.00 | +0.267 | 31 |
| **egemma-prompted (canonical)** | **768** | **1.00** | **+0.350** | **37** |

Retrieval saturated at this fixture size (every config 1.00 — a larger
confusable-heavy corpus is queued), so the call went to canonical-mode
margins + speed + features: **embeddinggemma-300m** wins canonical margins
(+0.350 vs nomic's +0.172), matches nomic's CPU speed, and natively supports
instruction-style prompts (`task: search result | query: …` /
`title: none | text: …`) plus Matryoshka truncation (768→512/256/128) —
both used by the phase-3 ingest design (`bench/phase3-ingest-program.md`).

- **Qwen3-Embedding-0.6B is BLOCKED in this fork (F16)**: its canonical
  `--pooling last` crashes at graph reserve
  (`ggml-cpu/ops.cpp:4914 GGML_ASSERT(i01 >= 0 && i01 < ne01)` in
  `get_rows`); it only runs mean-pooled here, where it is 4× slower with
  the weakest margins. Re-test after syncing the upstream ggml fix.
- **Run embedding sidecars with `LLAMAFILE_NO_VOICE=1`** — without it the
  APE also spawns the baked Kokoro voice server, which fights the main
  server's voice for ports 8078/8079.
- Deployed unit (`/etc/systemd/system/embed.service` in CT 118) now uses
  `Environment=LLAMAFILE_NO_VOICE=1` and
  `-m /opt/embeddinggemma-300m-q8_0.gguf --embeddings --pooling mean
  --no-warmup` + the same CPU-safe overrides. **Revert** = point `-m` back
  to `/opt/nomic-embed-text-v1.5.Q8_0.gguf` (still on disk) and drop
  `--pooling mean --no-warmup`.
