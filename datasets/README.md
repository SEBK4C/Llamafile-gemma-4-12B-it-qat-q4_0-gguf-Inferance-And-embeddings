# Cross-modal test corpora (32 sentences × {image, audio, text})

Same 32 sentences (`manifest.json`, source of truth in
`tests/scale_corpus.py`), rendered two ways. Audio is `say` → 16 kHz mono
WAV, regenerable with `tests/scale_corpus.py --audio` (not committed).

## legacy-224/ — kept as a worked example of resolution dependence

224×224 canvas, 26 px Helvetica — the 2026-06-10 recipe. **OCR on these
is ~40% reliable and image embeddings built on them entangle content-
extraction failure with the modality gap.** The gemma4uv runtime
budget-fills every image to the soft-token budget (7.07× bicubic upscale
at the 1120-token OCR budget); interpolation cannot restore detail that
26 px glyphs never carried, and after upscaling each text line straddles
two 48 px patch rows — every glyph horizontally cut, the damaging case
(arXiv 2402.07384). Inspect with:

    uv run --with pillow python3 tests/viz_preprocess.py --max-tokens 1120 datasets/legacy-224/06.png
    # → "scale 7.07x, UPSCALED — detail cannot be recovered"

The 2026-06-11 measurements in docs/mm-embedding.md (baseline/prompteol
@32, span pooling) were taken on this corpus and stand as recorded — but
their image-side numbers measure "modality gap × extraction failure",
which is exactly why this corpus is preserved.

## native-1584/ — sized to the model, patch-aligned

1584×1584 = the exact gemma4uv input at the 1120-token OCR budget
(33×33 patches of 48 px → the budget-fill resize is a no-op). 80 px
Helvetica on a 96 px line pitch starting at y=48, so each text line
occupies exactly two whole patch rows (no horizontal glyph cuts).

**Serve with `--image-max-tokens 1120`** when using these — at the
default 280-token budget the runtime *downscales* them to 768², halving
the pixels per glyph.
