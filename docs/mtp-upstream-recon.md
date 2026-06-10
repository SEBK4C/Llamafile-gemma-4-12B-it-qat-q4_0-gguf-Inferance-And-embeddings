# Upstream recon: Gemma4 MTP (`--spec-type draft-mtp`) at merge commit `04eb4c44`

Recon of upstream llama.cpp at `04eb4c446d22b63449d5dc41c038987d4d8cc3a6`
("llama : add Gemma4 MTP", PR #23398, merged 2026-06-07). All file:line
references are against that commit (clone in `.scratch/llama.cpp`).

## 1. How `--spec-type draft-mtp` resolves and loads the drafter

### 1.1 Spec-type parsing and drafter-model resolution

- `"draft-mtp"` maps to `COMMON_SPECULATIVE_TYPE_DRAFT_MTP` in the name
  table at `common/speculative.cpp:28`; the enum lives at
  `common/common.h:163`.
- `common_params_handle_models()` (`common/arg.cpp:437-486`) is where the
  drafter model is resolved:
  - `common/arg.cpp:438-446` — if `draft-mtp` is among
    `params.speculative.types`, it sets `opts.download_mtp = true`.
  - `common/arg.cpp:473-479` — **fallback**: if no draft model was given
    explicitly (`-md` path empty, no `-hfd` repo, no draft URL) and the
    download step discovered an MTP sibling (`res.found_mtp`), the sibling
    becomes `params.speculative.draft.mparams.path`.

### 1.2 Sibling auto-discovery — HF repos ONLY, keyword still `"mtp-"`

- `find_best_sibling()` — `common/download.cpp:579-619`. Picks the best
  sibling GGUF whose **path contains the keyword anywhere**, preferring
  (a) deepest shared directory prefix with the main model file, then
  (b) closest quantization bit-width (`extract_quant_bits` diff).
- `find_best_mtp()` — `common/download.cpp:626-629` — calls it with
  keyword **`"mtp-"`** (unchanged from the old pin).
- `gguf_filename_is_model()` — `common/download.cpp:631-644` — excludes
  filenames containing `mmproj`, `imatrix`, or `mtp-` from being picked
  as the *main* model.
- **Crucial limitation**: discovery only runs on the Hugging Face plan
  path. `get_hf_plan()` (`common/download.cpp:701-755`, `plan.mtp`
  filled at :750-752) enumerates **HF repo files**, and
  `common_download_model()` (`common/download.cpp:784-852`) only builds
  an `hf_plan` when `model.hf_repo` is non-empty (`is_hf`,
  :795-796). A local `-m models/foo.gguf` takes the early-return branch
  at `common/download.cpp:805-808` and **no filesystem sibling scan
  happens**. Reference naming used by the PR author's repo
  (`am17an/Gemma4-31B-it-GGUF`): `mtp-gemma-4-31B-it.gguf` next to
  `Gemma4-31B-Q8_0.gguf`.

  → For our locally-stored backbone the drafter must be passed
  explicitly: `-md models/mtp-gemma-4-12b-it-qat-q4_0.gguf`
  (`--model-draft`/`-md`; or `--spec-draft-hf`/`-hfd` for repo-hosted).
  We keep the `mtp-` prefix anyway so HF-hosted copies of the pair would
  auto-discover, and so the file can never be mistaken for a main model.

### 1.3 `ctx_dft` creation in the server

`tools/server/server-context.cpp`, `init()`:

- :816-895 — `--fit` memory accounting: when `draft-mtp` is enabled it
  pre-reserves VRAM for the draft/MTP context (`cparams_dft.ctx_type =
  LLAMA_CONTEXT_TYPE_MTP`, :845-849). Note: this probe path constructs
  an MTP measurement context **without** `ctx_other`; for
  GEMMA4_ASSISTANT `llama_context` *throws* in that situation
  (`src/llama-context.cpp:91-98`, "this is normal during memory
  fitting") and the server catches it (:891-894, logged as a warning).
  PR followers reported `--fit` probe crashes with MTP on some
  configurations (interaction with later PR #23485) — if model load
  crashes during fitting, retry with `--fit off`.
- :913-960 — **separate-drafter path** (ours): when
  `params_base.speculative.has_dft()` (a draft model path resolved), the
  server loads it as `model_dft`, then creates `ctx_dft` with
  `cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP` (:951-953, only when
  `draft-mtp` is in the spec types) and **`cparams.ctx_other = ctx_tgt`**
  (:955). `params_base.speculative.draft.ctx_tgt/ctx_dft` are wired at
  :958-959.
- :961-985 — *self-hosted* MTP path (no separate drafter; for Qwen3.5/3.6
  style models whose MTP layers live in the main GGUF): creates an MTP
  context **on the target model itself**, also with
  `cparams_mtp.ctx_other = ctx_tgt`. Not our path — Gemma4's drafter is a
  separate GGUF.
- Gate: `common_speculative_init()` only instantiates the MTP impl if
  `params.draft.ctx_dft != nullptr` (`common/speculative.cpp:1373`,
  construction at :1428-1430). If the drafter fails to load you silently
  fall back to whatever other spec types were listed.

### 1.4 KV sharing with the target context

This is the heart of PR #23398. Chain:

1. `llama_context_params.ctx_other` (`include/llama.h:394`);
   `LLAMA_CONTEXT_TYPE_MTP` (`include/llama.h:203`) maps to graph type
   `LLM_GRAPH_TYPE_DECODER_MTP` (`src/llama-context.cpp:28`).
2. `llama_context` ctor: for `LLM_ARCH_GEMMA4_ASSISTANT` it *requires*
   `params.ctx_other` (`src/llama-context.cpp:91-98`) and passes
   `mem_other = llama_get_memory(cparams.ctx_other)` into memory creation
   (`src/llama-context.cpp:313-318`). A context with `ctx_type == MTP`
   but `n_layer_nextn == 0` in the model returns nullptr with a warning
   (`src/llama-context.cpp:3463-3467`).
3. `llama_model::create_memory()` — `src/llama-model.cpp:2151-2179`: for
   `LLM_ARCH_GEMMA4_ASSISTANT` it builds a `llama_kv_cache_iswa` with a
   **`share` callback**:
   - sliding drafter layers → target layer `n_layer(target) - 2`
   - full-attention drafter layer → target layer `n_layer(target) - 1`
4. `llama_kv_cache` ctor — `src/llama-kv-cache.cpp:177-192`: a shared
   layer **aliases the target cache's ggml K/V tensors directly**
   (`layers.push_back(layer_share)`), resolved through
   `other->map_layer_ids[il_share]`. Because the Gemma4 *backbone* cache
   itself maps its trailing KV-shared layers onto the last non-shared
   layer of each type via the `reuse` callback
   (`src/llama-model.cpp:2126-2135`), asking for target layer `n-1`/`n-2`
   lands on the **last non-KV-shared full / sliding layer's buffers** —
   exactly the `store_full_length_kv` semantics of the transformers
   reference. The drafter allocates **zero KV of its own**; "KV sharing"
   is literal tensor aliasing, which also implies drafter and target must
   sit on the same device for those buffers (single-GPU Metal: fine).
   The iswa wrapper plumbs `mem_other` base/swa streams at
   `src/llama-kv-cache-iswa.cpp:64-84`.
5. Host-side draft loop — `common/speculative.cpp:410-` 
   (`common_speculative_impl_draft_mtp`):
   - :450-452 — **runtime gate**: `llama_model_n_embd_out(drafter)` must
     equal `llama_model_n_embd(target)` (`GGML_ASSERT`, "MTP input row
     width must match the target h_nextn width"). For us: 3840.
   - :467-471 — the MTP batch carries **both** `token` and `embd` rows
     (manual `batch.token` malloc).
   - :495-496 — staging API `llama_set_embeddings_nextn(ctx, value,
     masked)` (`src/llama-ext.h:92-104`): target context emits its
     post-final-norm hidden (`h_nextn`), drafter consumes it.
   - :498 — `is_mem_shared = llama_get_ctx_other(ctx_dft) == ctx_tgt`.
   - Per-seq `pending_h` carryover / `verify_h` rollback bookkeeping at
     :423-440 (same design as the old pin's Qwen3.5 host, now generic).
   - Draft sampling is fixed top-k=10 (:474-480), optionally offloaded to
     the backend sampler chain (:483-493).

## 2. Drafter GGUF: required metadata and loader validation

### 2.1 Converter (`conversion/gemma.py:788-795`)

`Gemma4AssistantModel` (registered for `Gemma4AssistantForCausalLM` and
`Gemma4UnifiedAssistantForCausalLM`, see also
`conversion/__init__.py:78,82`) subclasses `Gemma4Model` and adds exactly
two keys:

- `gemma4-assistant.embedding_length_out` = `backbone_hidden_size`
  (**3840** for the 12B) — `gguf-py/gguf/constants.py:107`,
  `src/llama-arch.cpp:173`.
- `gemma4-assistant.nextn_predict_layers` = `block_count` (**4**) —
  `gguf-py/gguf/constants.py:129`, `src/llama-arch.cpp:198`.

Everything else (sliding-window pattern, per-layer-type rope theta
10000/1e6, window 1024, head dims 256/512, RMS eps, vocab) comes from the
shared `Gemma4Model` path reading the assistant's `config.json`. Arch
name string: `"gemma4-assistant"` (`src/llama-arch.cpp:60`).

### 2.2 Loader hparams (`src/models/gemma4-assistant.cpp:3-22`)

- `n_embd_inp_impl = n_embd_out()` — the *input* row width is the
  backbone hidden (pre_projection consumes `concat(embed, h_backbone)`).
- Reads sliding-window pattern per layer, `nextn_predict_layers`
  (asserted `== n_layer_all`, :15), `rope_freq_base_swa`, sliding window,
  RMS eps, swa K/V head sizes. `f_attention_scale` forced to **1.0**
  (:12) — matches the reference (no 1/sqrt(d)).

### 2.3 The three `runtime_error` validation gates
(`src/models/gemma4-assistant.cpp:27-35`, in `load_arch_tensors`)

1. :28 — `n_embd_head_k == n_embd_head_v` (full-attn head 512/512).
2. :31 — `n_embd_head_k_swa == n_embd_head_v_swa` (sliding head 256/256).
3. :34 — `n_embd_out() != n_embd` — i.e. `embedding_length_out` **must**
   be present and different from the drafter's own hidden size (1024);
   it must carry the target hidden (3840). A drafter converted without
   `backbone_hidden_size` fails here.

Plus the host-side gate from §1.4: `n_embd_out(drafter) == n_embd(target)`
(3840 vs 3840 — would catch pairing the 12B drafter with a different
backbone).

### 2.4 Expected tensors (loader, `gemma4-assistant.cpp:37-81`)

- `token_embd.weight` `[1024, 262144]`, duplicated as tied `output`.
- `output_norm.weight` `[1024]`.
- `nextn_proj_post.weight` `[1024, 3840]` (post_projection) and
  `nextn_proj_pre.weight` (blk.0) `[7680, 1024]` (pre_projection,
  2×3840 → 1024).
- Per layer (4): `attn_norm`, `wq`, `wo`, `attn_q_norm`,
  `attn_post_norm`, `out_scale` (`layer_scalar`, shape `{1}`),
  `ffn_norm/gate/up/down`, `ffn_post_norm`; `rope_freqs` only on
  non-SWA layers (partial-rotary freq factors for the NoPE-ish global
  layer). **No `wk`/`wv` anywhere** — the graph builds Q-only attention
  (`build_attn(..., Qcur, nullptr, nullptr, ...)`,
  `gemma4-assistant.cpp:155-156`) against the aliased target KV.
- Graph output: `t_logits` (tied head) **and** `t_h_nextn`
  (`nextn_proj_post` of the post-norm hidden) — the recurrence input for
  the next drafted token (`gemma4-assistant.cpp:195-203`).

## 3. Flags and single-GPU Metal notes

- `--spec-type draft-mtp` — `tools/server/README.md:257`,
  `docs/speculative.md:250`.
- `--spec-draft-n-max N` — `common/arg.cpp:3574-3578`; tokens drafted per
  round. PR author benched with 4 (~0.59 aggregate acceptance, >2x on
  dense 31B). Old `--draft-max` was removed (`common/arg.cpp:3786-3788`).
- `--spec-draft-device` (`-devd`) — `common/arg.cpp:3612`; only needed
  multi-GPU (PR note: pair with `-sm layer`). Irrelevant on a single
  Metal device — and given KV sharing is tensor aliasing, drafter layers
  *must* live with the target's KV anyway.
- `--spec-draft-type-k/-v` (`common/arg.cpp:3520,3533`) set the MTP
  context's cache types — mostly moot since shared layers alias target
  buffers.
- Known-good workarounds from PR thread: `--fit off` if the fit probe
  crashes (see §1.3); one user needed `-nocb` (inherited master issue).

## 4. Why E4B/E2B are unsupported — and 12B risk assessment

From the PR thread: the **E4B/E2B assistants use a centroid/sparse
("masked embedding") LM head** — conversion dies on
`masked_embedding.centroids.weight` (no tensor mapping), and the ordered
/centroid embedding path is simply not implemented. The 31B and 26B-A4B
assistants use a **tied dense head**.

Our 12B QAT assistant checkpoint has `use_ordered_embeddings: false`
(tied dense head, verified in MTP-prompt.md notes), so it is in the
*supported* family even though upstream never tested the 12B. The
checkpoint was downloaded and inspected (2026-06-11): 48 tensors,
**no** `masked_embedding.*` / centroid tensors, no `k_proj`/`v_proj`,
`use_ordered_embeddings: false`, `tie_word_embeddings: true` — clean.

12B-specific assumptions, all **verified against our GGUFs**:

1. **Layer-pattern assumption in the `share` callback**
   (`src/llama-model.cpp:2155-2163`): target layer `n-1` must be
   full-attention and `n-2` sliding, routed via the backbone's
   `map_layer_ids` to full-context KV buffers. Verified on
   `models/gemma-4-12b-it-qat-q4_0.gguf`: 48 layers, sliding-window
   pattern tail `[..., swa, swa, full]` → layer 47 full (1 KV head ×
   512), layer 46 swa (8 KV heads × 256). The 12B backbone GGUF has
   `gemma4.attention.shared_kv_layers = 0` (unlike the 31B), so
   `n_layer_kv_from_start = 48`, the `reuse` map is identity, and
   `share` lands directly on layers 46/47's own full-context buffers —
   semantically identical to the reference's "last non-KV-shared layer
   of each type". Note backbone layer 47 has `attn_k` but **no
   `attn_v`** tensor: that's the `attention_k_eq_v` global layer (V is
   K), already handled by the gemma4 backbone graph.
2. **Head-geometry**: drafter full-attn layer (blk.3, q `[1024,8192]`,
   q_norm `[512]`) matches backbone layer 47 (1 × 512); drafter sliding
   layers match layer 46 (8 × 256). Verified from both GGUFs' metadata
   (`head_count_kv`, `key_length{,_swa}`). The loader gates in §2.3 only
   check K==V sizes, not the cross-model pairing — a mismatch would have
   surfaced as a ggml shape assert at graph build, not a clean error.
3. `embedding_length_out` = 3840 from the checkpoint's
   `backbone_hidden_size` — verified post-conversion (§6).
4. The drafter reads the **target model's** scaled token-embedding table
   at graph build (`model_other->tok_embd`,
   `gemma4-assistant.cpp:112-115`) — so the drafter GGUF's own (tied)
   embedding is used only for the LM head; input embedding quality
   follows the backbone's q4_0 table. Nothing to do, just good to know
   for parity debugging.

## 5. Practical invocation for this repo

```
llama-server -m models/gemma-4-12b-it-qat-q4_0.gguf \
             -md models/mtp-gemma-4-12b-it-qat-q4_0.gguf \
             --spec-type draft-mtp --spec-draft-n-max 4 \
             -ngl 999   # Metal: all-or-nothing offload (repo-known trap)
```

(`-md` is required because local paths get no sibling auto-discovery,
§1.2. If the pair is ever published to HF in one repo, the `mtp-` prefix
makes `-hf <repo>` auto-discover the drafter when `--spec-type draft-mtp`
is set.)

## 6. Drafter conversion record (task #5, done 2026-06-11)

- Source: `google/gemma-4-12B-it-qat-q4_0-unquantized-assistant`
  (single 845 MB bf16 safetensors, `Gemma4UnifiedAssistantForCausalLM`).
- Converter: upstream `convert_hf_to_gguf.py` at `04eb4c44`,
  `--outtype q8_0`.
- One local fixup was needed: the checkpoint's `tokenizer_config.json`
  ships `"extra_special_tokens": []` (a list); transformers 4.57.6
  (pulled in by `requirements/requirements-convert_hf_to_gguf.txt`)
  requires a dict and crashes in `_set_model_specific_special_tokens`.
  Changed to `{}` in the downloaded copy (both are "empty") — re-apply
  if the checkpoint is ever re-downloaded.
- Output: **`models/mtp-gemma-4-12b-it-qat-q4_0.gguf`** (449 MB, q8_0,
  49 tensors). Name = backbone filename with the `mtp-` discovery
  prefix (§1.2), mirroring the PR author's published convention.
- Sanity check (gguf-py, all PASS):
  - `general.architecture = gemma4-assistant`, file_type q8_0
  - `embedding_length_out = 3840`, `embedding_length = 1024`,
    `block_count = 4`, `nextn_predict_layers = 4`, vocab tokens 262144
  - `nextn.pre_projection.weight` `[7680, 1024]`,
    `nextn.post_projection.weight` `[1024, 3840]`,
    `blk.{0..3}.layer_output_scale.weight` `[1]` present
  - NO `attn_k`/`attn_v` tensors anywhere
  - tied `token_embd.weight` `[262144, 1024]` (no separate
    `output.weight`)
  - sliding pattern `[T,T,T,F]`, `head_count_kv [8,8,8,1]`,
    key/value length 512 (full) / 256 (swa), rope 1e6 / 1e4,
    `rope_freqs.weight [256]` (partial rotary 0.25 freq factors),
    `shared_kv_layers = 4`

