# Release checklist — Gemma 4 12B llamafile

**Rule zero: nothing is tagged, published to HF, or merged to `main` unless every
supported platform's E2E suite has passed against the exact artifact being
shipped.** Branch pushes for review are fine; the gate is on release.

Supported platforms: **macOS/Metal (Apple Silicon)** · **Linux/CUDA (NVIDIA)** ·
**CPU-only fallback (any arch)**.

Every unchecked box blocks the release. Each item lists the command or evidence
that satisfies it — "it worked on my box" is not evidence; logs and numbers are.

## 1 · Build integrity

- [ ] **Vendor pins are reachable on their remotes.** `git submodule update --init`
      from a FRESH clone succeeds. (v0.5.0 shipped with `vendor/llamafile`
      pinned to `337ca45`, which was never pushed — the repo could not be
      rebuilt from source.)
- [ ] **Patches resolve.** `./scripts/apply-patches.sh` ends green: every patch
      `applied` or `already applied`, zero `does not apply`. If the fork
      overlay pre-applies them, spot-check distinctive hunks are present
      instead, and record which model you used in the release notes.
- [ ] **Both args profiles staged.** `unzip -l dist/gemma4-server.llamafile`
      lists `.args`, `.args.xnu`, `ui-config.json`, both GGUFs, and the MTP
      drafter.
- [ ] **Args content reviewed against release-notes claims.** `unzip -p
      dist/gemma4-server.llamafile .args` (and `.args.xnu`) — every feature the
      notes claim ships enabled is actually in the args. (v0.5.0's notes said
      "MTP enabled by default"; the shipped `.args` had no `--spec-type` block.)

## 2 · Per-platform E2E — the artifact, not the source tree

Run against the **packaged file bare** (no flags), so the baked defaults are
what's under test.

### macOS / Metal
- [ ] `./scripts/mac-full-test.sh` → **13/13 PASS** (chat, embeddings,
      batch-vs-solo consistency, concurrent mixed load, KV save/erase/restore,
      image via CPU mmproj, MTP load, batch-cost profile).
- [ ] Startup log confirms the Metal profile resolved: `-c 8192`, `-fa on`,
      `--no-mmproj-offload`, drafter loaded (`adding speculative implementation
      'draft-mtp'`).
- [ ] **No Metal command-buffer OOM** (`kIOGPUCommandBufferCallbackErrorOutOfMemory`)
      anywhere in the log after the concurrent stress test.

### Linux / CUDA
- [ ] `python3 tests/smoke_test.py` full PASS on the reference GPU (CT 118 class).
- [ ] Startup log confirms the CUDA profile resolved (its own `.args`, DSO
      extracted and `cuInit` OK, patches 0016/0017 present in any rebuilt DSO —
      without them small embedding batches return silent garbage on sm_86).
- [ ] Audio path spot-check with a **speech** sample (synthetic tones are not a
      valid regression signal).

### CPU-only fallback
- [ ] Portable invocation from PLATFORM-NOTES (`-ngl 0 --gpu disable -sm layer
      -ctk f16 -ctv f16 --flash-attn off --spec-type none`) boots, serves one
      chat and one embedding.

## 3 · Web UI defaults (both platforms)

- [ ] `GET /` returns the web UI, **not** JSON 404 (empty `dist/` in the build
      silently ships without the UI).
- [ ] `/props` → `ui_settings.systemMessage` is the Constitution prompt
      (~1.1 k chars) and `default_generation_settings` carries the tuned
      sampler (temp 1.0, top_k 64, top_p 0.95, min_p 0.01, DRY 0.8/1.75/2/-1,
      repeat_penalty 1.0).
- [ ] Raw API call **without** a system message is not prompt-shaped — the
      system prompt must stay WebUI-only, never injected into `/v1` requests.
- [ ] Voice/controls injection intact if shipped: `python3 -c
      "print(open('bin/llamafile','rb').read().count(b'g4v-btn'))"` ≥ 1
      (assets embed as hex — `strings | grep` is unreliable).

## 4 · Performance within tested norms

Measure **clean-state** — first requests after a fresh server start. A battery
of prior tests depresses the same probe ~20% (residual slot/KV state); that is
an artifact, not a regression.

| platform | gate | tested norm |
|---|---|---|
| M1 Pro 32 GB Metal | **≥ 18 tok/s** | 21–22 tok/s (MTP or not — break-even there) |
| M4-class Metal | ≥ 18 tok/s | 21.5 prose / 25 edit with MTP (1.5–1.6×) |
| RTX 3080 Ti-class CUDA | **≥ 85 tok/s** | 90–110 tok/s honest chat decode |
| CPU-only (any) | boots + responds | ~4–5× slower than Metal same-machine |

- [ ] Platform gates met (record numbers in the release notes).
- [ ] MTP drafter *loads* on both GPU platforms (acceptance line in log). Do
      **not** gate Mac speed on MTP gains — it is break-even on M1 Pro.
- [ ] Embedding batch-vs-solo cosine drift < 1e-4 (patch 0001 regression).
- [ ] KV persistence roundtrip: save → erase → restore → `cache_n` reuse ≥ 100.

## 5 · Serving-quality gates (autoresearch ledger)

- [ ] A `status=baseline` row exists in `bench/serving-results.tsv` for the
      release config (per platform where the config differs).
- [ ] Any defaults change (system prompt / sampler / args) shipped only if its
      candidate row **passed gates**: acc ≥ baseline − 0.05, cal ≥ baseline,
      rep ≤ baseline, composite strictly improves — on BOTH configs in the
      config matrix.
- [ ] No `REPLACE_ME` placeholders left in `bench/probes.json` for any metric
      the release notes cite.
- [ ] KV dirs purged before every measurement run (`.gemma4-kv*`, `.kvcache*`)
      — autosave restores across config changes and masks fixes.

## 6 · Release hygiene

- [ ] No secrets in the diff (`git diff origin/main | grep -iE "fw_|sk-|api_key
      *=" ` returns nothing; judge keys live in env/1Password only).
- [ ] HF publish: repack `.args` via `zipalign -j0` only (append-only — never
      rewrite the zip); keep `.args.bak`; verify the download runs bare on a
      Mac and a CUDA box before announcing.
- [ ] Rollback documented: previous artifact retained + one-line revert for
      any service ExecStart change; purge KV dirs after rollback too.
- [ ] Release notes: platform-specific numbers stated separately (M1 Pro ≠ M4
      ≠ CUDA — one blended number misleads users on every platform).

---
*Checklist born 2026-07-06 from the Mac E2E pass: the packaged artifact ran
~6 tok/s on Apple Silicon with a bare UI because CUDA-tuned baked args were
the only profile, the release notes claimed MTP that the args had lost, and
the vendor pin could not be fetched. Every box above traces to a real failure.*
