#!/usr/bin/env python3
"""Mac Metal variant of the frozen serving-defaults benchmark.

Adapts serve_bench.py for Apple Silicon / Metal without touching the frozen
harness.  Three Mac-specific overrides:

    LAT_NORM = 25.0    Metal + MTP ceiling (≈21-25 tok/s on M4)
    DEF_SERVER = "http://127.0.0.1:8080"  (local, not CT 118)
    purge_kv()  = local rm -rf of $ROOT/.kvcache (no pct/LXC)

Everything else — probe battery, judge, ledger, gates, coordination — is
identical to the CUDA run, so candidates are scored on the same axes and
gate epsilons, just normalised against Metal-class latency.

Usage (same as serve_bench.py):
    # First: start the server in another terminal
    make serve          # or: GEMMA4_SPEC=none make serve   (no MTP)

    # Seed a Mac baseline row:
    python3 bench/mac_serve_bench.py --baseline

    # Run a candidate:
    python3 bench/mac_serve_bench.py --candidate bench/candidates/my-cand.json

    # Full protocol (5 replicas × 5 turns):
    python3 bench/mac_serve_bench.py --replicas 5 --depth 5

    # No Fireworks key? Skip judging by passing a dummy key (gates on acc/cal/rep only):
    FIREWORKS_API_KEY=skip python3 bench/mac_serve_bench.py --baseline
"""
import shutil
import sys
from pathlib import Path

# ── Mac overrides (patched before serve_bench.main() runs) ──────────────────
ROOT = Path(__file__).resolve().parent.parent
KV_DIR = ROOT / ".kvcache"

# Make serve_bench importable from its sibling directory.
sys.path.insert(0, str(Path(__file__).parent))
import serve_bench  # noqa: E402

serve_bench.DEF_SERVER = "http://127.0.0.1:8080"

# Metal + MTP ceiling.  Measured M4: ~21.5 tok/s prose, ~25 tok/s edit tasks.
# Using 25 so a well-tuned MTP run can reach lat_norm ≈ 0.86 (vs 1.0 on CUDA).
serve_bench.LAT_NORM = 25.0


def _purge_kv_mac():
    """Local KV purge — replaces the LXC `pct exec` in serve_bench.purge_kv."""
    try:
        if KV_DIR.exists():
            shutil.rmtree(KV_DIR)
        KV_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[kv] purged {KV_DIR}", file=sys.stderr)
    except Exception as e:              # noqa: BLE001
        print(f"[kv] purge failed (non-fatal): {e}", file=sys.stderr)


serve_bench.purge_kv = _purge_kv_mac

# gemma_chat's default timeout=180s assumes CUDA speed (90-110 tok/s). At Mac
# battery speed (~15 tok/s) a probe that exhausts the 3072-token budget needs
# ~205s+ and times out, silently dropping exactly the budget-exhausting probes
# the loop cares about (observed: qa_speed 3/5 replicas lost, lp_count greedy).
_orig_gemma_chat = serve_bench.gemma_chat


def _gemma_chat_mac(server, messages, sampler, max_tokens, timeout=180):
    return _orig_gemma_chat(server, messages, sampler, max_tokens,
                            timeout=max(timeout, 360))


serve_bench.gemma_chat = _gemma_chat_mac

# Gemma-4's reasoning channel exhausts the CUDA-era 768-token default before
# emitting content on hard probes (measured on this Mac 2026-07-06: 768 ->
# 0 content chars / ~2.5k reasoning chars; 1536 -> full answers). Empty
# content scores hum/soph 1.0 and acc/cal 0, silently flooring the run.
# Enumeration probes (lp_synonyms) are worse: the model re-drafts the list
# in reasoning and hits finish=length at 2048 with content STILL empty.
# 3072 gives those probes room to answer. LOCKED: the budget is part of the
# measurement — never change it between a baseline and its candidates.
MAC_MAX_TOKENS = "3072"

# ── re-use the exact same main() ────────────────────────────────────────────
if __name__ == "__main__":
    if "--max-tokens" not in sys.argv:
        sys.argv += ["--max-tokens", MAC_MAX_TOKENS]
    serve_bench.main()
