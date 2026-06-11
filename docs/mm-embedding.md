# Multimodal embeddings: the modality gap in Gemma 4 12B

Measured 2026-06-10 on this repo's server (`gemma-4-12b-it-qat-q4_0`,
`--embeddings --pooling mean`, multimodal mmproj loaded, CPU backend —
see the Metal caveat below). Reproduce with `tests/modality_gap.py`.

## Question

If the *same content* enters through different modalities — words as
text, the words rendered on an image, the words spoken aloud — how far
apart do the pooled embeddings sit, and is the modality-specific drift
a consistent offset across topics (i.e. could it be subtracted away)?

## Setup

Three sentences ("topics"): the quick-brown-fox pangram, a pasta-recipe
sentence, a quarterly-revenue sentence. Each embedded three ways through
`/v1/embeddings` on the same server instance:

- **text** — plain string input
- **image** — sentence rendered black-on-white at 26 px onto a 224×224
  PNG (the model's native vision resolution), sent via
  `{"prompt_string": "<media marker from /props>", "multimodal_data": [b64]}`
- **audio** — macOS `say` voice, converted to 16 kHz mono WAV, same format

Content was verified to actually survive each modality before measuring:
an OCR prompt reads the 26 px rendering word-perfectly (an earlier 448 px
rendering downscaled to illegibility — only fragments like "The qui bro"
survived — and was discarded), and the audio transcribes word-perfectly.

## Results

### Embeddings cluster by modality, not content

Cosine similarities (mean-pooled, L2-normalized):

| pair type | range |
|---|---|
| same modality, different topics — text | 0.85 – 0.94 |
| same modality, different topics — image | 0.74 – 0.79 |
| same modality, different topics — audio | 0.55 – 0.66 |
| **same sentence, different modality** | **0.30 – 0.61** |

The image embedding of the fox pangram is closer to the *pasta* text
than to its own text. Raw cross-modal retrieval (nearest text for each
media item): **2/6**, chance-level.

### Drift geometry

Drift vector per topic: `d_mod(t) = e_mod(t) − e_text(t)` on unit vectors.

| modality | ‖d‖ | drift-direction agreement across topics | vs other modality |
|---|---|---|---|
| image | 1.12 – 1.15 | cos 0.69 – 0.81 (semi-parallel) | cos(d_img, d_aud) = 0.48 |
| audio | 0.90 – 1.00 | cos 0.46 – 0.53 (weakly parallel) | — |

So the answer to "is the modality-specific drift the same by topic?" is:
**partially — and more so for vision than audio.** The offsets are huge
(‖d‖ ≈ 1.1 on unit vectors ≈ near-orthogonal displacement), vision's
offset is fairly consistent in direction, audio's is more entangled with
content/prosody, and the two modalities drift in *different* directions.

### Gap correction is not enough

Subtracting the mean per-modality offset (the textbook fix for
CLIP-style contrastive models, whose gap is a clean parallel
translation) improves retrieval only 2/6 → **3/6**. This generative
model's modality gap is not a parallel translation; the residual after
offset removal is still dominated by non-content factors (layout,
voice timbre/prosody) rather than topic.

## Practical guidance

- Don't mix modalities in one vector store with raw cosine — partition
  by modality, or align first (see dev notes).
- Within a single modality, relative ranking carries signal in all three
  modalities (topic structure is visible in each same-modality block).
- ~~Render text at ≥ 26 px when targeting the 224² vision input~~ —
  superseded 2026-06-11: rendered-text legibility is broken at *any* font
  size by an image-pipeline patch-geometry bug (see "Image pipeline
  distortion" below). Always verify with a temperature-0 OCR prompt
  before trusting image embeddings of documents.
- **GPU bug (mitigated)**: media inputs on `/v1/embeddings` crash the GPU
  backend. Since patch 0009 the server refuses them with HTTP 501 instead
  of crashing (chat with media and text embeddings are fine on GPU). Use
  `GEMMA4_NGL=0 make serve` for cross-modal embedding work — details in
  the WS3 section below.

## Prompted embeddings (WS1, measured 2026-06-11)

Hypothesis from dev-note 4: wrapping media in an instruction pulls the
pooled representation toward the text manifold. Tested with
`tests/template_sweep.py` (same wrapper on BOTH sides of the comparison —
text items embedded wrapped too), mean pooling, CPU backend.

3-topic battery (retrieval is nearest-text over 3 candidates, 6 queries):

| config | xmod-img | xmod-aud | retrieval raw | gap-corrected |
|---|---|---|---|---|
| baseline (bare) | 0.34–0.37 | 0.50–0.60 | 2/6 | 3/6 |
| instr-lead (`read this …: {x}`) | 0.41–0.47 | 0.50–0.61 | 2/6 | 5/6 |
| instr-trail (`{x}\nthe … above says:`) | 0.52–0.65 | 0.64–0.80 | 2/6 | 5/6 |
| **prompteol** (`this …: "{x}" means in one word:`) | **0.68–0.76** | 0.60–0.67 | **6/6** | 5/6 |

32-item corpus (`tests/scale_corpus.py` + `tests/scale_eval.py`,
chance r@1 = 1/32 ≈ 3%, chance r@5 ≈ 16%):

| @32 | baseline | prompteol |
|---|---|---|
| image r@1 / r@5 | 1/32 / 4/32 | 2/32 / **11/32** |
| audio r@1 / r@5 | 2/32 / 7/32 | **7/32** / **14/32** |
| image same-topic cross-modal sim | 0.351 | **0.714** |
| audio same-topic cross-modal sim | 0.552 | 0.638 |
| image cross-topic same-modality sim | 0.739 | 0.883 |

Takeaways:

- The PromptEOL-style compressor template **works cross-modally** even
  though it failed for text-text anisotropy (2026-06-10) — different
  mechanism, as suspected. It doubles image→text similarity and is the
  best zero-training recipe found.
- But scale deflates small-N optimism: the perfect 6/6 at 3 topics
  becomes near-chance image r@1 at 32 items (r@5 still ~2× chance);
  **audio keeps a real 3.5× lift** (r@1 7/32). The reason: the wrapper
  also *tightens* the image modality cluster (block sim 0.74→0.88), so
  absolute alignment improves but ranking gains mostly cancel.
- Margins (min same-topic cross-modal − max cross-topic same-modality)
  stay negative everywhere: no recipe makes cross-modal sim dominate
  modality identity. Mixed-modality vector stores remain a bad idea.

## Pooling sensitivity (WS2, measured 2026-06-11)

Researched offline with no server changes: `GEMMA4_POOLING=none` + legacy
`/embedding` returns per-token rows; `tests/span_pooling.py` pools spans
in Python (offline L2 normalization — `embd_normalize` is skipped for
pooling none; one input per request).

**What pooling-none actually exposes** (discovered the hard way): media
chunk tokens are never output-marked (`mtmd-helper.cpp` sets
`batch.logits[i] = false`; the all-outputs override in `llama-batch.cpp`
only fires for pooled types), and leading wrapper text rides inside the
helper's chunks, so the rows you get for a media input are
`[last-media-token] + [trailing wrapper tokens]`. Text row 0 is BOS
(verified content-blind, cos = 1.0 across inputs); media row 0 is
content-dependent (cos ≈ 0.79 across topics) and ≈ the bare-marker
embedding (cos 0.995).

Two useful facts fall out:

1. **Media rows are homogeneous**: mean over the ~7 exposed rows
   reproduces the server's full ~260-row mean to 2 decimals, and the
   last-media-token alone behaves like the full media average. Media-row
   span pooling has nothing more to give; a server-side patch for it is
   not justified on current evidence.
2. **Trailing-instruction-rows-only pooling (`mean_trail`) gives the best
   margins seen anywhere** (3 topics: image −0.159 with prompteol vs
   −0.453 baseline; audio −0.089 with instr-trail vs −0.160) and image
   xmod up to 0.77–0.81 — but raw retrieval does not beat
   prompteol/mean.

32-item check (`span_pooling.py --scale`, prompteol template):

| pooling | image r@1 / r@5 | audio r@1 / r@5 | image margin |
|---|---|---|---|
| mean_all (≈ server mean) | 2/32 / 11/32 | **7/32 / 14/32** | −0.340 |
| mean_trail | 2/32 / **12/32** | 4/32 / 12/32 | **−0.300** |
| last | 1/32 / 7/32 | 2/32 / 10/32 | −0.367 |

(instr-trail template: strictly worse on image; audio `last` reaches
r@5 13/32 with the lowest audio block sim 0.650, but r@1 stays 2/32.)

**Verdict: span pooling is a negative result at scale.** `mean_trail`
buys a slightly better image margin and nothing for retrieval, while
costing audio. The best overall recipe stays **prompteol template ×
standard mean pooling**, which needs no server change at all — so the
per-request-pooling server patch contemplated in MM-prompt.md is NOT
justified on current evidence.

## Image pipeline distortion (discovered 2026-06-11) — read before trusting image numbers

While re-verifying legibility for the 32-item corpus, the OCR spot-check
failed on images that are pixel-identical (same md5) to ones recorded as
word-perfect on 2026-06-10. Systematic probing (temperature 0, both CPU
and Metal, worktree and main-checkout binaries — all identical) shows the
model does not see what is in the file:

- A square 224×224 image with "HELLO" is described as "a very wide,
  horizontal banner" with letters "cropped at top and bottom".
- Multi-line text is read as stacked left-edge slivers: "pasta recipe /
  uses guanciale / and pecorino" comes back as fragments
  `re/pe/us/and/pe/co/ri` — the patch grid is reassembled in the wrong
  shape.
- Coarse *relative* geometry survives (red square top-left / blue circle
  bottom-right is localized correctly), and rotation is perceived
  correctly (rotating the text 90° does not fix reading), so it is not a
  simple transpose — more like a patch-position/stride mismatch.
- Rendering at the model-native patch-aligned size (336×336 = 7×7 patches
  of 48px, which skips the runtime resize entirely) does NOT fix it, so
  the bug is downstream of the resize — in the gemma4uv projector's
  patch-position handling (`pos_x`/`pos_y` learned-position inputs in
  clip.cpp, backported by patch 0005), not in `calc_size_preserved_ratio`.
- Mechanics for reference: mmproj metadata says `image_size=224,
  patch_size=16`, gemma4uv folds the 3× merge into the conv → effective
  48px patches; the runtime enforces a 40-token minimum
  (`set_limit_image_tokens(40, 280)`), so every 224px image is first
  bilinear-upscaled 1.5× to 336². The clip.cpp comment "model performs
  quite poor with small images" is consistent with this bug being
  upstream, not introduced by our patches.

Consequences:

- The 2026-06-10 "chat OCR reads the renderings word-perfectly" claim
  does not reproduce and should be considered wrong (sampling luck at
  temperature 1.0 on 3 short pangram-style sentences).
- **Image-side embedding numbers measure content extraction failure ×
  modality gap, entangled** — they are reproducible but their
  interpretation ("pooling hides what the backbone knows") only holds
  for audio, where transcription does verify. This neatly explains why
  audio beats image on every cross-modal metric at scale.
- Audio numbers are unaffected.
- Fixing the projector patch-position bug is the highest-leverage open
  item for image embeddings — likely worth an upstream issue with the
  sliver-fragment evidence above.

## Image pipeline: resolution research + final verdict (2026-06-11, session 2)

Patch 0010 (port of chippydip's gemma4uv-vision-fix, = upstream issue
#24146) fixed the real runtime divergence: the resize now always fills
the soft-token budget like the HF reference (224² → 768² @ the default
280-token cap), bicubic, with F32 projector accumulation. After that, a
literature + ecosystem sweep and controlled experiments give the final
picture:

- **Budget**: Google's own docs recommend **1120 image tokens for OCR /
  small text** (budgets 70/140/280/560/1120); llama.cpp's 280 default is
  the general-purpose tier (upstream PR #24014 raises it). Our build
  honours `--image-max-tokens 1120` (224² source → 1584², 1089 tokens;
  verified prompt_tokens=1120). Ollama hardcodes 280 (issue #15626).
  Requires all image tokens in one ubatch — serve.sh's `-ub 2048` is OK.
- **Field consensus** (DeepSeek-OCR, GOT-OCR2, Qwen2.5/3-VL, dots.ocr,
  PaddleOCR-VL, Donut/Nougat): every OCR-capable VLM attends across
  patches at 14–16px granularity BEFORE compressing to ~28–64px/token.
  Gemma 4 12B *Unified* is encoder-free (raw 48px cells, linear
  projection, the LM does all stitching) — architecturally the weakest
  configuration in the field for text, matching the encoder-free track
  record (Fuyu, EVE, SOLO, Mono-InternVL all lag on documents).
- **Patch-boundary literature** (arXiv 2402.07384, PIXEL bigram
  rendering): horizontal cuts through text lines hurt (glyph halves end
  up a full patch-row apart in token space); patch-aligned rendering
  measurably helps pixel LMs. Our contact-sheet visualization
  (tests/viz_preprocess.py) showed exactly this on the 224px corpus.
- **Controlled endpoint**: native 1584² render, 80px glyphs, 1120
  tokens, lines patch-row-aligned → clean line structure but content
  words still dropped/mangled ("pasta" lost, "guanciale and pecorino" →
  "and people"); misaligned variant hallucinates. **Verdict: remaining
  weakness is the architecture, not the runtime.** The 12B Unified
  trades OCR fidelity for encoder-free simplicity; that is its ceiling.

Practical recipe for document/PDF work on this stack:

1. Serve with `--image-max-tokens 1120` for any OCR-ish chat use.
2. Rasterize PDFs at 144–200 DPI (olmOCR uses 1288px longest edge;
   MinerU 200 DPI; Nougat's 96 DPI is the cautionary tale), keep
   x-height ≥ 10px at final model resolution.
3. Prefer rendering text at the model input size (multiples of 48px,
   e.g. 1584² at the 1120 budget) so the budget-fill resize is a no-op,
   and align line pitch to the 48px patch grid.
4. For OCR that must actually work, use a ViT-path model (Gemma 4
   E4B/26B/31B = gemma4v, or Qwen2.5-VL-class e.g. olmOCR) — the 12B
   Unified will not get there at any rendering setting.

## Native-resolution gauntlet (2026-06-11, session 2): legibility is NOT the embedding bottleneck

After patch 0010 + the committed native corpora (datasets/), the full
cross-modal battery was re-run on legible renders (OCR word recall
~62–64% in chat, vs ~40% pass-rate on legacy-224). All numbers @32
items, mean pooling, CPU, prompteol template unless noted:

| image corpus / budget | img r@1 | img r@5 | img xmod | img block | audio (unchanged) |
|---|---|---|---|---|---|
| legacy-224 @280 (blurry) | 2/32 | 11/32 | 0.714 | 0.883 | r@1 7, r@5 14, xmod 0.638 |
| native-tight @280 (legible) | 2/32 | 10/32 | 0.656 | 0.884 | identical |
| native-1584 sq @1120 (legible, 75% white) | 1/32 | 10/32 | 0.598 | **0.973** | identical |
| (baseline template, tight @280) | 1/32 | 6/32 | 0.284 | 0.803 | r@1 2, r@5 7 |

Three conclusions, two of them decisive:

1. **Image embedding alignment is insensitive to render legibility.**
   Chat-OCR now extracts most of the content from these renders, yet
   pooled image embeddings carry no more topic signal than they did
   from illegible ones. For images, extraction and embedding quality
   are decoupled problems — the bottleneck is squarely what mean
   pooling exposes, and neither resolution, patch alignment, nor the
   1120-token budget moves it. Closes the "would legible inputs fix
   image embeddings?" question: no.
2. **Blank area poisons image embeddings** (suggested by S., confirmed
   by the square-vs-tight contrast): every uniform patch collapses to
   one vector under patch_ln1, so the 75%-white square renders push
   cross-topic image similarity to **0.973** — near-degenerate — while
   content-cropped strips hold 0.884 at a quarter of the token cost.
   **Tight rendering is mandatory for image embeddings**, and
   native-tight @ the default 280 budget is the recommended corpus
   (square-1584 @1120 is strictly worse and 4x slower).
3. Audio reproduces exactly across binaries/corpora (r@1 7/32, r@5
   14/32) and remains the only modality whose content reliably
   survives pooling. Cross-modal image↔text alignment in vector space
   needs a learned projection (dev note 1) or a different pooling
   mechanism — input quality and prompting are exhausted as levers.

Dev note added by this round: gemma4uv's explicit per-patch (x,y)
positions + the HF reference's masked `(-1,-1)` padding make
**uniform-patch dropping** (skip blank patches in clip.cpp, keep true
coordinates for survivors) mechanically easy — the in-batch analogue of
"KV caching for identical image sections". Untested: sparse grids are
out of training distribution; worth one experiment if image-embedding
work continues.

## Metal media-embeddings crash (WS3, 2026-06-11)

The crash from the 2026-06-10 caveat was debugged as far as the tooling
allows and mitigated:

- **Repro**: Metal (`-ngl 999`) + any media input on `/v1/embeddings` →
  bare SIGSEGV (no GGML_ASSERT), 100%, immediately after the
  "embeddings required but some input tokens were not marked as outputs"
  override. Bisects: all pooling types crash (mean/last), `--spec-type
  none` still crashes, `GGML_METAL_CONCURRENCY_DISABLE` and
  `GGML_METAL_FUSION_DISABLE` don't help. Text embeddings on Metal OK,
  chat+media on Metal OK, media embeddings on CPU OK.
- **It's an async GPU-side fault, not a bad CPU-side copy**: printf
  instrumentation shows every host-side extraction offset in-bounds, and
  the crash point *moves* between runs (sometimes before, sometimes after
  `llama_decode` returns; once it surfaced as a graceful "Compute error"
  HTTP 500 — `ggml_backend_sched_graph_compute_async failed with error
  -1` — proving the error path exists and works when the fault is caught).
  The failing graph is the media-chunk decode with raw F32 embd input ×
  `cparams.embeddings` × all-outputs.
- **Debugger dead ends, for the next attempt**: lldb cannot launch or
  attach to APE binaries usefully (attach hangs; `/bin/sh` launch is
  SIP-denied; loader-direct launch never reaches main), cosmo suppresses
  macOS crash reports, and `MTL_SHADER_VALIDATION=1` reports nothing
  (fault is not a validated kernel OOB).
- **Upstream**: no exact match as of 2026-06-11. Closest fixed:
  PR #23643 (`eef59a764`, 2026-05-29, 3 days after our pin) — embd-batch
  graph node reading null token ids, segfault-on-GPU class. Closest open:
  #23072 (Metal embeddings heap corruption on macOS). A pin bump past
  `eef59a764` (planned by the MTP branch) should be followed by a re-test
  of this repro.
- **Interim ship (patch 0009)**: the server now refuses media-embedding
  requests when GPU offload is active and a GPU device exists — HTTP 501
  with a message pointing at `GEMMA4_NGL=0 make serve` — instead of
  crashing. Verified on Metal: media embedding → clean 501 (server
  stays up), text embeddings and chat+media unaffected; CPU path:
  full smoke suite + this file's baseline battery reproduce exactly.

## Dev notes — open questions for later

1. **Learned alignment instead of mean offset.** The natural next step:
   fit a small projection (ridge / Procrustes / 1-layer MLP) from media
   embeddings to text embeddings on a few hundred paired examples, and
   measure cross-modal retrieval on held-out pairs. Candidate HF
   datasets for pairs:
   - *audio ↔ text*: [`keithito/lj_speech`](https://huggingface.co/datasets/keithito/lj_speech)
     (13k single-speaker clips with transcripts — small, clean, ideal first run);
     [`openslr/librispeech_asr`](https://huggingface.co/datasets/openslr/librispeech_asr)
     (large, multi-speaker — tests whether audio drift inconsistency is
     speaker variance); [`mozilla-foundation/common_voice_17_0`](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0)
     (multi-speaker/multi-lingual, license-friendly).
   - *image ↔ text (captions, not rendered text)*:
     [`nlphuji/flickr30k`](https://huggingface.co/datasets/nlphuji/flickr30k)
     (31k images × 5 captions) — note this measures *semantic* image
     content vs captions, a different (harder) question than our
     rendered-text probe.
   - *rendered-text images*: synthesize from any text corpus with the
     `tests/modality_gap.py --make-assets` recipe (font/size/layout
     jitter would make the probe much stronger).
2. **Is audio drift inconsistency just voice identity?** Re-run with
   multiple `say` voices (and speeds) per sentence: if drift direction
   varies more by voice than by topic, the audio gap decomposes into
   (modality offset) + (speaker offset) — which a per-speaker correction
   could handle.
3. ~~**Pooling sensitivity.**~~ Answered 2026-06-11 — see "Pooling
   sensitivity (WS2)" above: negative at scale; standard mean stays best.
4. ~~**Prompted embeddings.**~~ Answered 2026-06-11 — see "Prompted
   embeddings (WS1)" above: prompteol template is a real zero-training
   win for audio; image gains are capped by the pipeline bug (note the
   premise "chat outputs prove the backbone fully extracts the content"
   turned out to hold only for audio).
5. ~~**Fix the Metal segfault**~~ Mitigated 2026-06-11 (patch 0009
   refuses instead of crashing) — root cause still open, see "Metal
   media-embeddings crash (WS3)" above. Re-test after any llama.cpp pin
   bump past `eef59a764`.
6. **Fix the image-pipeline patch-geometry bug** (see "Image pipeline
   distortion" above) — highest-leverage open item for image embeddings;
   suspect the gemma4uv `pos_x`/`pos_y` learned-position inputs in
   clip.cpp (patch 0005 backport, likely upstream too).
