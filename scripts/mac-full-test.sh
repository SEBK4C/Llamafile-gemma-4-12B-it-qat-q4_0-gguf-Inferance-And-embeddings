#!/usr/bin/env bash
# mac-full-test.sh — Mac Metal end-to-end test suite for Gemma 4 12B llamafile.
#
# Tests all features documented for Mac in docs/PLATFORM-NOTES.md.
# The server must be running before this script is called, OR pass --start-server
# to have the script launch it (background, killed on exit).
#
# Usage:
#   ./scripts/mac-full-test.sh                 # against already-running server
#   ./scripts/mac-full-test.sh --start-server  # launch server, test, shut down
#   ./scripts/mac-full-test.sh --port 8081     # non-default port
#   ./scripts/mac-full-test.sh --no-mtp        # skip MTP speed check
#
# Output: bench/mac-test-results.log  (PASS/FAIL per test + timing)
# Exit:   0 = all pass, 1 = one or more failures
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${PORT:-8080}"
HOST="127.0.0.1"
BASE_URL="http://${HOST}:${PORT}"
LOG="${ROOT}/bench/mac-test-results.log"
START_SERVER=0
SKIP_MTP=0
SERVER_PID=""

for arg in "$@"; do
  case "$arg" in
    --start-server) START_SERVER=1 ;;
    --no-mtp)       SKIP_MTP=1 ;;
    --port=*)       PORT="${arg#--port=}"; BASE_URL="http://${HOST}:${PORT}" ;;
  esac
done

PASS=0; FAIL=0

ok()   { echo "  PASS: $*"; PASS=$((PASS+1)); echo "PASS  $*" >> "$LOG"; }
fail() { echo "  FAIL: $*"; FAIL=$((FAIL+1)); echo "FAIL  $*" >> "$LOG"; }
sep()  { echo ""; echo "── $* ──"; echo "" >> "$LOG"; echo "── $* ──" >> "$LOG"; }

cleanup() {
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ── 0. pre-flight ─────────────────────────────────────────────────────────────
echo "Mac Metal E2E test suite — $(date)" | tee "$LOG"
echo "Server: $BASE_URL" | tee -a "$LOG"
echo ""

BIN="${ROOT}/bin/llamafile"
MODEL="${ROOT}/models/gemma-4-12b-it-qat-q4_0.gguf"
MMPROJ="${ROOT}/models/mmproj-gemma-4-12b-it-qat-q4_0.gguf"
MTP="${ROOT}/models/mtp-gemma-4-12b-it-qat-q4_0.gguf"

sep "Pre-flight checks"
[ -x "$BIN" ]    && ok "binary present: $BIN" || fail "missing binary: $BIN"
[ -f "$MODEL" ]  && ok "model present (~$(du -sh "$(realpath "$MODEL")" 2>/dev/null | cut -f1 || echo "?"))" || fail "missing model: $MODEL"
[ -f "$MMPROJ" ] && ok "mmproj present" || fail "missing mmproj: $MMPROJ"
[ -f "$MTP" ]    && ok "MTP drafter present" || fail "missing MTP: $MTP"

# ── 1. server start (optional) ───────────────────────────────────────────────
if [ "$START_SERVER" = "1" ]; then
  sep "Starting server"
  KV_DIR="${ROOT}/.kvcache-test"
  rm -rf "$KV_DIR" && mkdir -p "$KV_DIR"
  # Launch through the production serve.sh so the test covers the real
  # config path (serve.sh caps ubatch at 1024 on Darwin — Metal command
  # buffers OOM at 2048 on M1 Pro 32 GB).
  GEMMA4_KV_DIR="$KV_DIR" GEMMA4_PORT="$PORT" \
    "${ROOT}/scripts/serve.sh" -fa on \
    > /tmp/gemma4-mac-server.log 2>&1 &
  SERVER_PID=$!
  echo "  Server PID: $SERVER_PID (log: /tmp/gemma4-mac-server.log)"

  echo -n "  Waiting for /health "
  for i in $(seq 1 180); do
    if [ "$(curl -s -o /dev/null -w '%{http_code}' "${BASE_URL}/health" 2>/dev/null)" = "200" ]; then
      echo " ready (${i}s)"
      ok "server started in ${i}s"
      break
    fi
    echo -n "."
    sleep 1
    if [ "$i" = "180" ]; then
      echo " TIMEOUT"
      fail "server did not become ready in 180s"
      echo "Server log tail:"; tail -20 /tmp/gemma4-mac-server.log
      exit 1
    fi
  done
else
  sep "Checking running server"
  STATUS=$(curl -s -o /dev/null -w '%{http_code}' "${BASE_URL}/health" 2>/dev/null || echo "000")
  if [ "$STATUS" = "200" ]; then
    ok "server reachable at $BASE_URL"
  else
    fail "server not reachable (HTTP $STATUS) — start it with 'make serve' or pass --start-server"
    exit 1
  fi
fi

# ── 2. Metal GPU detection ────────────────────────────────────────────────────
sep "Metal / hardware detection"
INFO=$(curl -s "${BASE_URL}/props" 2>/dev/null || echo "{}")
if echo "$INFO" | python3 -c "import json,sys; d=json.load(sys.stdin)" 2>/dev/null; then
  ok "/props endpoint responds"
  if echo "$INFO" | grep -qi "metal\|gpu\|ngl"; then
    ok "GPU/Metal references in /props"
  else
    echo "  INFO: /props does not explicitly mention Metal (may still be GPU-accelerated)"
  fi
else
  fail "/props returned invalid JSON or timed out"
fi

# ── 3. clean-state generation speed (MUST run before the battery) ─────────────
# Running this after multimodal/smoke tests reads ~17 t/s instead of ~21.5:
# residual slot KV state and image tokens depress the measurement. Only this
# clean-state number is gated; the post-battery re-check below is INFO-only.
sep "Generation speed (clean state)"
if [ "$SKIP_MTP" = "0" ]; then
  measure_speed() {
    python3 - <<PYEOF
import json, urllib.request
prompt = ("Explain in depth how a write-ahead log guarantees durability "
          "and how group commit amortises fsync cost across concurrent "
          "transactions. Be thorough and technical.")
body = json.dumps({
    "prompt": prompt, "n_predict": 200, "temperature": 0.8,
    "top_p": 0.95, "cache_prompt": False, "ignore_eos": True, "stream": False
}).encode()
req = urllib.request.Request("$BASE_URL/completion", body,
                             {"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=300) as r:
    resp = json.load(r)
print(f'{resp.get("timings", {}).get("predicted_per_second", 0):.2f}')
PYEOF
  }
  measure_speed >/dev/null 2>&1 || true     # warmup
  SPEED=$(measure_speed) || SPEED=0
  echo "  generation speed: ${SPEED} tok/s"
  # M1 Pro measured healthy decode: 21-22 tok/s (with OR without MTP — MTP is
  # break-even on M1 Metal, batch-2 verify costs ~1.75x single decode).
  # Below 18 = something regressed (GPU offload, config); below 13 = CPU-only.
  if python3 -c "import sys; sys.exit(0 if float('${SPEED}') >= 18 else 1)"; then
    ok "generation speed ${SPEED} tok/s (healthy Metal decode; M1 Pro baseline 21-22)"
  elif python3 -c "import sys; sys.exit(0 if float('${SPEED}') >= 13 else 1)"; then
    fail "generation speed ${SPEED} tok/s — below the 18 tok/s healthy floor (regression?)"
  else
    fail "generation speed ${SPEED} tok/s — likely CPU-only; check -ngl/Metal offload"
  fi
  ACC=$(grep -oE "draft acceptance = [0-9.]+" /tmp/gemma4-mac-server.log 2>/dev/null | tail -1 | grep -oE "[0-9.]+$" || true)
  if [ -n "$ACC" ]; then
    ok "MTP draft acceptance: $ACC (drafter active; 0.91 on this prompt when healthy)"
  else
    echo "  INFO: no draft-acceptance lines in server log (MTP inactive or external server)"
  fi
else
  echo "  SKIP: --no-mtp passed"
fi

# ── 4. smoke tests (canonical suite from tests/smoke_test.py) ─────────────────
sep "Smoke tests (chat + embeddings + concurrent + KV)"
if python3 "${ROOT}/tests/smoke_test.py" --url "$BASE_URL" 2>&1 | tee /tmp/smoke.log; then
  ok "smoke_test.py PASSED"
else
  fail "smoke_test.py FAILED (see /tmp/smoke.log)"
fi

# ── 4. embedding correctness: batch vs solo consistency ───────────────────────
sep "Embedding batch-vs-solo consistency (SWA iSWA regression)"
if python3 - <<PYEOF
import sys, json, math, urllib.request
url = "$BASE_URL"
texts = [
    "content-defined chunking and Merkle manifests",
    "write-ahead log durability and group commit",
    "B-tree page splits and balance guarantees",
    "Metal GPU scheduling and command buffer submission",
]
def embed(t):
    body = json.dumps({"input": t if isinstance(t, list) else [t]}).encode()
    r = urllib.request.Request(url + "/v1/embeddings", body,
                               {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=120) as resp:
        data = sorted(json.load(resp)["data"], key=lambda d: d["index"])
    return [d["embedding"] for d in data]

def cos(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    return dot / (sum(x*x for x in a)**0.5 * sum(y*y for y in b)**0.5)

batch  = embed(texts)
solo   = [embed(t)[0] for t in texts]
drifts = [cos(b, s) for b,s in zip(batch, solo)]
worst  = min(drifts)
print(f"  batch-vs-solo cosines: {[f'{d:.4f}' for d in drifts]}")
print(f"  worst drift: {worst:.6f}")
if worst < 0.9999:
    print("FAIL: batch embedding drifted from solo (SWA iSWA regression?)")
    sys.exit(1)
print("PASS: batch-vs-solo consistent")
PYEOF
then
  ok "batch-vs-solo embedding consistency"
else
  fail "batch-vs-solo drift detected (patch 0001 regression?)"
fi

# ── 5. multimodal: image input ────────────────────────────────────────────────
sep "Multimodal: image input via mmproj (CPU offload)"
IMG="${ROOT}/datasets/legacy-224/00.png"
if [ ! -f "$IMG" ]; then
  IMG="${ROOT}/images.jpeg"  # fallback to Downloads copy if dataset not present
fi
if [ -f "$IMG" ]; then
  IMG_B64=$(python3 -c "import base64; print(base64.b64encode(open('$IMG','rb').read()).decode())")
  MM_RESULT=$(python3 - <<PYEOF
import json, sys, urllib.request
url = "$BASE_URL"
b64 = "$IMG_B64"
body = json.dumps({
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": "Describe this image in one short sentence."}
        ]
    }],
    "temperature": 1.0, "top_k": 64, "top_p": 0.95,
    "max_tokens": 1024, "stream": False
    # temp=0 + small budget returns empty content on this thinking model:
    # the reasoning channel consumes the whole budget before the answer.
}).encode()
req = urllib.request.Request(url + "/v1/chat/completions", body,
                             {"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=120) as r:
    resp = json.load(r)
print(resp["choices"][0]["message"].get("content",""))
PYEOF
  ) || MM_RESULT=""
  if [ -n "$MM_RESULT" ]; then
    ok "multimodal image input: got '$MM_RESULT'"
  else
    fail "multimodal image input: empty response"
  fi
else
  echo "  SKIP: no test image found (datasets/ not populated)"
fi

# ── 6. post-battery speed re-check (INFO only, not gated) ────────────────────
sep "Generation speed after battery (residual-state INFO)"
if [ "$SKIP_MTP" = "0" ]; then
  SPEED2=$(measure_speed) || SPEED2=0
  echo "  post-battery speed: ${SPEED2} tok/s (clean-state was ${SPEED:-?};"
  echo "  a drop here reflects residual slot/KV/image state, not a regression)"
else
  echo "  SKIP: --no-mtp passed"
fi

# ── 7. Metal batch decode cost profile ───────────────────────────────────────
sep "Metal batch decode cost profile"
if python3 "${ROOT}/tests/probe_batch_cost.py" --port "$PORT" --host "$HOST" \
     --batches "1,2,4,8,16,32,64" 2>&1 | tee /tmp/batch-cost.log; then
  ok "batch cost profile completed (results in /tmp/batch-cost.log)"
else
  fail "batch cost profile failed"
fi

# ── 8. concurrent load: embedding + chat ─────────────────────────────────────
sep "Concurrent mixed load (stress)"
if python3 - <<PYEOF
import json, threading, urllib.request, sys
url = "$BASE_URL"
errors = []

def chat():
    body = json.dumps({"messages":[{"role":"user","content":"Count from 1 to 5, digits only."}],
                        "temperature":0,"max_tokens":64}).encode()
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url+"/v1/chat/completions", body,
                                   {"Content-Type":"application/json"}), timeout=120) as r:
            json.load(r)
    except Exception as e:
        errors.append(("chat", str(e)))

def embed(i):
    body = json.dumps({"input": [f"concurrent stress probe {i}"]}).encode()
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url+"/v1/embeddings", body,
                                   {"Content-Type":"application/json"}), timeout=60) as r:
            json.load(r)
    except Exception as e:
        errors.append((f"embed{i}", str(e)))

threads = [threading.Thread(target=chat)] + [
    threading.Thread(target=embed, args=(i,)) for i in range(4)
]
for t in threads: t.start()
for t in threads: t.join()
if errors:
    print(f"FAIL: concurrent errors: {errors}")
    sys.exit(1)
print("PASS: 5/5 concurrent requests succeeded")
PYEOF
then
  ok "concurrent mixed load (5 threads)"
else
  fail "concurrent mixed load failed"
fi

# ── summary ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo "  Results: ${PASS} PASS  |  ${FAIL} FAIL"
echo "  Log: $LOG"
echo "════════════════════════════════════════"
echo "" >> "$LOG"
echo "TOTAL: ${PASS} PASS  ${FAIL} FAIL  ($(date))" >> "$LOG"

[ "$FAIL" -eq 0 ]
