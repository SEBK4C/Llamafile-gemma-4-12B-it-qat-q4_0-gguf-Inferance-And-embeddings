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
- Render text at ≥ 26 px when targeting the 224² vision input, and
  verify legibility with an OCR prompt before trusting image embeddings
  of documents.
- **Metal bug**: media inputs on `/v1/embeddings` segfault the Metal
  backend (chat with media and text embeddings are fine). Use
  `GEMMA4_NGL=0 make serve` for cross-modal embedding work until fixed.

## ⚠️ Post-publication correction (2026-06-11): image perception is broken

The mm-embedding-dev branch discovered, and we confirmed **on main**, that
the gemma4uv vision path scrambles fine-grained geometry: multi-line text
is perceived as left-edge slivers / hallucinated fragments, on CPU and
Metal alike, at any font size. Our original legibility check was
**confounded**: the fox sentence is the world's most famous pangram, and
the model completed it from priors — non-memorizable sentences (pasta,
money) fail OCR at temperature 0 (e.g. the money image loops on the
sliver "er. (US)"). Coarse layout (left/right color split) survives.

Consequences for this document: the **image-side numbers above entangle
perceptual scrambling with the modality gap** and should be treated as a
lower bound on image-embedding quality; audio is unaffected (transcription
is word-perfect), which likely explains audio beating image on cross-modal
similarity. The text and audio columns stand.

Status: suspect area is the gemma4uv positional-embedding handling
(`tools/mtmd/clip.cpp` pos_x/pos_y + the x/y table split in
`models/gemma4uv.cpp`, from backport patch 0005 — plausibly upstream too).
A clean transpose (x/y swap) is **ruled out**: transposed images are not
readable either. Next step is comparing against the transformers
`Gemma4UnifiedVisionEmbedder` reference — owned by the `mm-embedding-dev`
branch; re-run this experiment after the fix.

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
3. **Pooling sensitivity.** All numbers are mean-pooling. Does `last`
   pooling (or pooling only over the media-token span vs the whole
   sequence) shrink the gap? The media tokens dominate sequence length,
   so mean pooling mostly averages media-token states.
4. **Prompted embeddings.** Does wrapping media in an instruction
   ("transcribe this audio:" / "read this image:") before pooling pull
   the representation toward the text manifold? Cheap to test, could be
   a zero-training win — the OCR/transcription *chat* outputs prove the
   backbone fully extracts the content; the question is purely about
   what pooling exposes.
5. **Fix the Metal segfault** for media + pooled-embeddings batches
   (crash is in graph compute right after the "embeddings required but
   some input tokens were not marked as outputs" override; CPU path is
   correct — diff the graphs or bisect ggml-metal ops to isolate).
