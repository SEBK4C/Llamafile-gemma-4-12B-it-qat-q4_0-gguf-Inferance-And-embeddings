# Session digest — 2026-06-11 (mm-embedding-dev)

One-page compression of a long dev session. Full numbers and methods in
`docs/mm-embedding.md`; datasets in `datasets/`; patches 0009/0010 in
`patches/`. Everything below ran against `gemma-4-12b-it-qat-q4_0` via
this repo's llamafile server on an M4/16GB.

## What shipped

| Artifact | What it does |
|---|---|
| patch 0009 | media embeddings on GPU → clean HTTP 501 instead of a segfault |
| patch 0010 | gemma4uv budget-fill resize (HF-reference parity), bicubic, F32 projector accumulation — port of chippydip's fix (= upstream #24146) |
| `tests/template_sweep.py`, `scale_eval.py`, `span_pooling.py` | prompted-embedding sweep, 32-item retrieval gauntlet, offline span pooling |
| `tests/viz_preprocess.py` | render any image's exact model-input view + per-token contact sheet |
| `datasets/{legacy-224, native-1584, native-tight}` | committed corpora; legacy kept deliberately as the resolution cautionary example |
| `--image-max-tokens 1120` | serve-time flag for OCR work (Google's recommended budget; default 280) |

## Findings, in order of discovery

1. **Prompted embeddings (WS1).** PromptEOL-style wrapping (`this …
   "{content}" means in one word:`, both sides) is the best
   zero-training recipe: audio retrieval r@1 2→7/32, r@5 7→14/32;
   image↔text similarity doubles. Small-N warning made concrete: 6/6 at
   3 topics deflated to ~chance image r@1 at 32 items.
2. **Span pooling (WS2).** Negative at scale; standard mean stays best.
   Bonus mechanics: pooling-none exposes only `[last-media-token +
   trailing text]` rows; media rows are so homogeneous that 7 rows
   reproduce the full mean to 2 decimals.
3. **Metal media-embeddings segfault (WS3).** Async GPU-side fault in
   the embd-input × pooled-embeddings graph; debugger-resistant (lldb
   cannot attach to APE binaries). No upstream match (closest #23072,
   #23643). Shipped the sanctioned interim: refusal instead of crash
   (patch 0009). Re-test after any llama.cpp pin bump past `eef59a764`.
4. **Image pipeline bug (the day's centerpiece).** "Word-perfect OCR"
   from 2026-06-10 did not reproduce; investigation found the runtime
   kept small images at native size while the HF reference always
   budget-fills — small patch grids are out of training distribution
   (model literally reports "cropped/zoomed" views). Fixed by patch
   0010. Remaining weakness is architectural: the 12B *Unified* path is
   encoder-free (raw 48px cells → linear → LM), and the OCR literature
   is unanimous that cross-patch attention before the LM is what makes
   document VLMs work. Best chat-OCR achieved on clean renders: ~62–64%
   word recall — the ceiling, not a config problem.
5. **Resolution research (for PDF ingestion).** Field consensus:
   rasterize at 144–200 DPI (olmOCR: 1288px longest edge; dots.ocr: DPI
   200; Nougat's 96 DPI is the cautionary tale), keep x-height ≥10px,
   size to the model's resize grid. For OCR that must work locally, use
   ViT-path models (Gemma 4 E4B/26B/31B, Qwen2.5-VL-class/olmOCR) — not
   the 12B Unified.
6. **Final gauntlet (legible renders).** Image embedding alignment is
   *insensitive to render legibility* — extraction and embedding are
   decoupled; mean pooling is the bottleneck and prompting/resolution
   are exhausted as levers. **Blank canvas area poisons image
   embeddings** (uniform patches collapse to one vector under
   patch_ln1): 75%-white squares → 0.973 cross-topic similarity vs
   0.884 for content-cropped strips at ¼ the tokens. `native-tight` at
   the default budget is the corpus standard. Audio is bit-reproducible
   throughout and remains the only modality whose content survives
   pooling.

## Open items (ranked)

1. **Learned projection** image→text (ridge/Procrustes on paired data) —
   the only remaining lever for image↔text vector alignment.
2. **Uniform-patch dropping** in clip.cpp (explicit per-patch (x,y) +
   HF's masked padding make it mechanically easy; sparse grids untested).
3. WS3 root cause — re-test repro after the MTP branch's pin bump.
4. Upstreaming patches 0009/0010 (note llama.cpp's no-AI-PR policy —
   needs a human owner; 0010's substance already exists as chippydip's
   branch and upstream issue #24146).

## Operational notes

- Two agent worktrees shared this machine; stray `pkill`s from the
  sibling MTP session killed servers twice → long-lived servers now run
  under a restart-supervisor, and monitor commands must not contain a
  sibling's pkill pattern verbatim.
- One 12B server at a time on 16GB. `--cache-ram 512` for test servers
  (image requests store ~95MB each in the prompt cache at high budgets).
- OCR test discipline: `temperature 0`, `max_tokens 400`, 120s client
  timeout — failed reads ramble to whatever budget they're given.
