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
