# Kickoff prompt: port Gemma 4's MTP drafter to llama.cpp/llamafile

> Use this file as the opening prompt for the dev sessions on this branch
> (`mtp-gemma4-drafter`, worktree `~/Projects/mtp-gemma4`). It encodes all
> research done on 2026-06-10 so you don't have to rediscover it. The parent
> repo serves Gemma 4 12B (chat+embeddings+image+audio) via a patched
> llamafile; this branch's mission is **true multi-token-prediction
> speculative decoding** using Google's official drafter.

## ⚠️ UPDATE 2026-06-11: upstream merged Gemma4 MTP — plan rewritten

llama.cpp PR [#23398](https://github.com/ggml-org/llama.cpp/pull/23398)
"llama : add Gemma4 MTP" merged 2026-06-07, merge commit
`04eb4c446d22b63449d5dc41c038987d4d8cc3a6`. It implements essentially the
design sketched below (same-repo drafter arch + KV-cache sharing with the
backbone), so **the mission changes from "port the drafter" to "integrate
the upstream implementation into our llamafile build"**. Verified facts:

- New arch `GEMMA4_ASSISTANT` (`src/models/gemma4-assistant.cpp`); the
  converter registers `Gemma4AssistantForCausalLM` **and**
  `Gemma4UnifiedAssistantForCausalLM` — the exact class of our QAT
  checkpoint. It is generic over `backbone_hidden_size` (written as
  `embedding_length_out`), so the 12B/3840 backbone should work.
- **Backbone conversion is unchanged** — our existing
  `gemma-4-12b-it-qat-q4_0.gguf` stays valid; the drafter is a separate
  sibling GGUF.
- Usage: `--spec-type draft-mtp --spec-draft-n-max 4`. Author reports >2×
  on dense models, ~0.59 aggregate acceptance on mtp-bench (DGX Spark).
- Caveat: author tested 31B and 26B-4B; explicitly **not** E4B/E2B; the
  12B is untested upstream — expect it to work (conversion class matches)
  but verify, don't assume.
- Our pin `dbe9c0c` is **208 commits behind** the merge. `a731805ced`
  (source of backport patches 0005/0006) **is an ancestor** of the merge
  commit, so bumping the pin subsumes those two patches.
- The PR touches 31 files of core (`llama-kv-cache*`, `llama-graph`,
  `llama-context`, `llama-memory-hybrid*`, server, conversion) and likely
  depends on refactors among the 208 intermediate commits — **bump the
  pin; do not try to cherry-pick the PR onto dbe9c0c** (fallback only if
  the bump proves unbuildable under cosmocc).

The architecture notes below remain valid as a verification reference
(parity checks, debugging). The original "hard part" / Option A/B analysis
and phase plan are superseded — see the revised plan.

## Mission (revised)

Integrate upstream Gemma4 MTP: bump the vendored llama.cpp to
≥ `04eb4c44`, keep the llamafile overlay + local patch stack building
under cosmocc, convert `google/gemma-4-12B-it-qat-q4_0-unquantized-assistant`
(423M) to a drafter GGUF, and ship `--spec-type draft-mtp` working in the
packaged llamafile. Target hardware: M4/16GB (drafter adds ~0.5–1 GB).

Success criteria, in order:
1. Drafter GGUF loads and produces logits matching the transformers
   reference (token-level parity on a fixed prompt, greedy).
2. Server `--spec-type` path drafts with it; acceptance rate >50% on
   greedy chat; measured wall-clock speedup >1.5× on prose (baseline
   13.4 tok/s on the M4).
3. Embeddings/multimodal/KV-persistence features of the main branch keep
   passing `tests/smoke_test.py`.

## What the drafter actually is (verified against transformers main)

Reference implementation (all confirmed live, quote-level analysis done):
- `src/transformers/models/gemma4_unified_assistant/modeling_gemma4_unified_assistant.py`
- twin: `gemma4_assistant/modeling_gemma4_assistant.py`; inner layers from
  `gemma4/modeling_gemma4.py` (`Gemma4TextAttention`, `Gemma4TextDecoderLayer`)
- drafting loop: `generation/candidate_generator.py` →
  `SinglePositionMultiTokenCandidateGenerator`

Architecture (`Gemma4UnifiedAssistantForCausalLM`, hidden 1024, 4 layers,
backbone_hidden 3840, vocab 262144 — tensor list verified from the QAT
checkpoint `google/gemma-4-12B-it-qat-q4_0-unquantized-assistant`,
single 423M-param safetensors file):

1. **Input**: `inputs_embeds = pre_projection(concat([target_embed(last_token),
   h_backbone], dim=-1))` — 7680→1024, no bias. `target_embed` is the **12B's**
   scaled embedding table (`Gemma4TextScaledWordEmbedding`, ×√3840). The
   backbone hidden state is **post-final-RMSNorm** (`hidden_states[-1]` is
   overwritten with `last_hidden_state` by `@capture_outputs(tie_last_hidden_states=True)`).
2. **Attention**: NO k_proj/v_proj anywhere (`num_kv_shared_layers = 4` forced
   in config `__post_init__`). Every layer cross-attends to
   `shared_kv_states[layer_type]` — the **backbone's** K/V captured from its
   last non-KV-shared layer of each type (`store_full_length_kv`), post
   k_norm+RoPE (K) and post v_norm (V), over the full current context.
   Layers 0-2: sliding_attention (window 1024, 8 KV heads × 256, q_proj
   [4096,1024], q_norm [256], RoPE theta 10000). Layer 3: full_attention
   (1 KV head × 512 — backbone global layers use `attention_k_eq_v` with
   `num_global_key_value_heads=1`, `global_head_dim=512` — q_proj [8192,1024],
   q_norm [512], "proportional" RoPE theta 1e6, partial_rotary_factor 0.25 →
   only 64 of 256 freq pairs real, rest zero-padded = NoPE).
   Attention scale = **1.0** (no 1/√d; q_norm/k_norm handle magnitudes).
   Masks are **bidirectional** against the shared KV (`create_attention_masks`),
   sliding mask `abs(q-kv) <= window` **flipped on the kv axis** (covers most
   recent KV). `position_ids` is **constant** = last seen position, never
   incremented during drafting.
3. **Layer body** (Gemma4TextDecoderLayer): input_layernorm → attn →
   post_attention_layernorm → +residual → pre_feedforward_layernorm →
   gated-GELU MLP (1024↔8192) → post_feedforward_layernorm → +residual →
   **× layer_scalar** (per-layer scalar buffer, last op).
   `Gemma4RMSNorm` multiplies by `weight` directly (NOT Gemma2/3-style
   `1+weight`); v_norm variants are scale-less.
4. **Outputs**: logits = tied lm_head (`model.embed_tokens` [262144,1024])
   on the post-final-norm 1024-d hidden. **Recurrence**: `post_projection`
   (1024→3840) of the same hidden replaces `h_backbone` for the next drafted
   token. Centroid/masked-embedding path is INACTIVE for this checkpoint
   (`use_ordered_embeddings: false`) — implement plain tied head.
   `generation_config` suppresses tokens [258883, 258882] — llama.cpp main
   branch (and our backport patch 0005) already has `suppress_tokens` →
   logits-bias support in `src/models/gemma4.cpp`.
5. **Loop** (assisted generation): backbone runs once with
   `output_hidden_states + return_shared_kv_states`; drafter then greedy-drafts
   up to `num_assistant_tokens` (default ~20, "heuristic" schedule ±) tokens,
   each step re-embedding its own argmax and carrying `post_projection` output;
   backbone verifies all candidates in one forward; next round slices at the
   last accepted token (`n_last_matches`).

## What llama.cpp gives you (SUPERSEDED — written against pin dbe9c0c, before #23398 merged)

The existing `--spec-type mtp` host (`COMMON_SPECULATIVE_TYPE_DRAFT_MTP`,
`common/speculative.cpp` ~line 409) was built for Qwen3.5-style heads and
does NOT match: it feeds **pre-norm** hidden states (staging API
`llama_set_embeddings_pre_norm` in `src/llama-ext.h`), the drafter
self-attends with its own KV cache, and positions advance. Reusable ideas:
- batches carrying BOTH `token` and `embd` (`llama_batch_init` + manual
  token alloc, speculative.cpp:460-463)
- per-seq pending-h carryover / verify-h rollback bookkeeping (lines 419-437)
- server wiring: `ctx_dft` creation, `--spec-type` plumbing, `mtp-*.gguf`
  sibling auto-discovery (`common/download.cpp:619`, `find_best_sibling`
  keyword "mtp-")
- `LLAMA_CONTEXT_TYPE_MTP` + `LLM_GRAPH_TYPE_DECODER_MTP` graph dispatch
  (see `src/models/qwen35.cpp:132` and server-context.cpp ~938-958)
- conversion pattern: `_Qwen35MtpMixin` in `conversion/qwen.py` (separate
  `mtp-*.gguf` export, `--mtp`/`--no-mtp` flags, duplicated embeddings)

## The hard part (SUPERSEDED — upstream solved this in #23398; kept for context)

llama.cpp memory is strictly per-context: a drafter context cannot read the
target context's KV cache. The drafter needs, per drafting round, read access
to the backbone's K/V of **two specific layers** (last non-shared sliding +
last non-shared full-attention) over the whole context — which is exactly
what already sits in the target's `llama_kv_cache_iswa` streams for those
layers, in the right form (K post-RoPE, V post-norm: that is what the cache
stores).

Design sketch (decide early, prototype before polishing):
- **Option A (recommended): same-context drafter.** Merge the drafter into
  the target GGUF as extra layers (à la qwen35 nextn) and run it as a second
  graph type (`LLM_GRAPH_TYPE_DECODER_MTP`) **on the same llama_context**, so
  "cross-context" sharing becomes same-context KV access: the MTP graph
  attends to the KV buffers of backbone layers i_sliding/i_full via new
  graph inputs. Needs: converter merges both checkpoints; hparams record
  which backbone layers to read; `kv_only_nextn`-style cache config so
  drafter layers allocate no KV; careful `llama_set_embeddings`-style toggle
  for the constant-position bidirectional attention. Avoids new memory APIs
  entirely — the graph for layer N may read cache of layer M in the same
  context (precedent: cross-layer KV reuse exists upstream for
  `num_kv_shared_layers` models — CHECK how `gemma4` 12B's own kv-shared
  layers are implemented in `src/models/gemma4.cpp`; the backbone itself
  has `num_kv_shared_layers` in its config, so the mechanism may largely
  exist and only needs exposure to the MTP graph type!).
- **Option B: cross-context KV view.** New core API (e.g.
  `llama_kv_cache_view(src_ctx, layer_id)`) handing a read-only tensor view
  to another context's graph. More general, more invasive, fights the
  scheduler/backend buffer ownership. Only if A dead-ends.

## Phase plan (revised 2026-06-11 for upstream #23398)

1. **Bump the pin.** Advance the nested `llama.cpp` submodule inside
   `vendor/llamafile` from `dbe9c0c` to ≥ `04eb4c446d` (208 commits).
   Snapshot the pre-bump tree first (the snapshot-diff workflow below) —
   Mozilla's overlay modifies many llama.cpp files, so the bump is a
   merge, not a fast-forward, wherever the overlay touched files.
2. **Rebase the patch stack** (`patches/`, applied by
   `scripts/apply-patches.sh`):
   - **Drop 0005/0006** (gemma4uv/ua backport + BUILD.mk) — `a731805ced`
     is in the new pin. Carry forward only 0006's BUILD.mk idea: new
     upstream sources (`gemma4-assistant.cpp`, anything else added in the
     208 commits) need BUILD.mk entries for the cosmocc build.
   - **Re-validate 0001** (pooled-embeddings ubatch fix): #23398 and the
     intervening commits churn `llama-kv-cache-iswa` heavily — check if
     upstream fixed the `embd_all`/`split_equal` bug; rebase otherwise.
   - **Rebase 0004, 0007, 0008** (SWA reuse, slot-save media gating)
     against the new server/kv code.
   - **0002, 0003, lf-0001** are cosmocc/llamafile-side; expect them to
     apply nearly clean.
3. **Build under cosmocc.** `make setup && make build` with the new pin;
   fix BUILD.mk and any cosmocc incompatibilities introduced by the 208
   new commits. This is the main schedule risk.
4. **Convert the drafter.** New `Gemma4AssistantModel` in upstream
   `conversion/gemma.py` accepts `Gemma4UnifiedAssistantForCausalLM` —
   convert `google/gemma-4-12B-it-qat-q4_0-unquantized-assistant`
   (drafter is 423M; q8_0 fine). Name it to match the server's sibling
   auto-discovery (check what `--spec-type draft-mtp` expects in the new
   pin — `find_best_sibling` keyword was "mtp-" at the old pin; re-verify).
5. **Verify.** (a) `tests/smoke_test.py` still passes (embeddings,
   multimodal, KV persistence); (b) drafter loads and `--spec-type
   draft-mtp` drafts on the M4 — remember the Metal traps below (`-ngl
   999` or `0`, never partial); (c) greedy-parity spot-check vs the
   transformers reference if acceptance looks off (the architecture notes
   below are the reference for that debugging); upstream tested 31B and
   26B-4B, NOT the 12B, so don't assume correctness.
6. **Bench + package.** Acceptance/speedup vs the ngram-simple baseline
   (tests/bench_spec.py; target >1.5× over the 13.4 tok/s baseline),
   package the drafter GGUF into the llamafile.

## Assets & repo infra

- Drafter checkpoints: `google/gemma-4-12B-it-assistant` (bf16),
  `google/gemma-4-12B-it-qat-q4_0-unquantized-assistant` (QAT, matches our
  backbone — prefer this). Both single-file safetensors, public.
- Backbone GGUF + mmproj already in `models/` (run `make model`).
- Build: `make setup && make build` (cosmocc; needs vendor's bundled GNU
  make — see Makefile). Local llama.cpp changes are carried as patches in
  `patches/` applied by `scripts/apply-patches.sh` — for this branch's
  large changes, commit directly in the worktree first and extract patches
  when stabilizing (snapshot-diff workflow; beware: mozilla's overlay also
  modifies many llama.cpp files, so per-file `git diff` may contain their
  hunks — diff against pre-edit snapshots).
- Existing local patches 0001-0008 (pooled-embedding split, slot-save dir,
  COSMOCC file IO, SWA reuse, gemma4uv/ua backport, BUILD.mk, slot-save
  media gating) are already applied on this branch's vendored tree state.
- Upstream contribution policy: llama.cpp does not accept predominantly
  AI-generated PRs — keep this work as local patches / fork unless a human
  takes ownership of an upstream submission.

## Known traps (paid for already — don't re-pay)

- The mmproj/assistant configs use `model_type` names that postdate the
  pin; check `conversion/__init__.py` registration tables first.
- `fs_create_directory_with_parents` only creates path components that end
  in a separator (patch 0002 commentary).
- Metal: partial layer offload of the backbone asserts (always `-ngl 999`
  or `-ngl 0`); media+pooled-embeddings segfaults Metal (open bug, README).
- Gemma 4 thinking channel: chat verification with small max_tokens returns
  empty `content` (thought eats the budget) — always budget ≥256 or
  compare `reasoning_content` too.
- Two server instances can't share the 12GB Metal budget on the M4.
- `say`-style background shells: the harness resets cwd between parallel
  tool calls — use absolute paths in scripts.
