#!/bin/sh
# Generate package/.prewarm-xnu-slot-0.bin — the shipped system-prompt KV
# state for the Mac Metal profile.
#
# Boots the server with the EXACT .args.xnu geometry (-c 8192 -np 2 ->
# 4096/slot, f16 KV), sends one 1-token request carrying the WebUI default
# system prompt (package/ui-config.json), then stops gracefully so the
# autosave (patch 0011) writes the slot state; that file ships in the zip
# and is extracted by autorestore on Xnu at first run.
#
# Rerun after ANY change to the system prompt, chat template, model
# weights, or the .args.xnu context geometry — the state encodes all four.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${PREWARM_PORT:-8097}"
KV="$(mktemp -d)"

"${ROOT}/bin/llamafile" --server \
    -m "${ROOT}/models/gemma-4-12b-it-qat-q4_0.gguf" \
    --mmproj "${ROOT}/models/mmproj-gemma-4-12b-it-qat-q4_0.gguf" --no-mmproj-offload \
    --embeddings --pooling mean \
    -ngl 999 -fa on -c 8192 -np 2 -b 1024 -ub 1024 \
    --slot-save-path "$KV" \
    --host 127.0.0.1 --port "$PORT" >/dev/null 2>&1 &
PID=$!
trap 'kill "$PID" 2>/dev/null || true; rm -rf "$KV"' EXIT

i=0
until [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" 2>/dev/null)" = "200" ]; do
    i=$((i+1)); [ "$i" -gt 180 ] && { echo "error: server never became ready" >&2; exit 1; }
    sleep 1
done

python3 - "${ROOT}/package/ui-config.json" "http://127.0.0.1:${PORT}" <<'PYEOF'
import json, sys, urllib.request
cfg = json.load(open(sys.argv[1]))
body = json.dumps({
    "messages": [{"role": "system", "content": cfg["systemMessage"]},
                  {"role": "user", "content": "Hi"}],
    "max_tokens": 1, "temperature": 0, "cache_prompt": True, "stream": False,
}).encode()
r = urllib.request.urlopen(urllib.request.Request(
    sys.argv[2] + "/v1/chat/completions", body,
    {"Content-Type": "application/json"}), timeout=600)
t = json.loads(r.read()).get("timings", {})
print(f"warm request done: prompt_n={t.get('prompt_n')}")
PYEOF

# graceful stop -> slots_autosave writes autosave-slot-*.bin
kill -INT "$PID"
wait "$PID" 2>/dev/null || true
trap 'rm -rf "$KV"' EXIT

# the request lands on either slot (LRU); state files are seq-portable, so
# whichever slot autosaved becomes the shipped slot-0 prewarm
SRC=""
for f in "${KV}"/autosave-slot-*.bin; do
    [ -f "$f" ] && SRC="$f" && break
done
[ -n "$SRC" ] || { echo "error: autosave produced no slot state in $KV" >&2; exit 1; }
cp "$SRC" "${ROOT}/package/.prewarm-xnu-slot-0.bin"
ls -la "${ROOT}/package/.prewarm-xnu-slot-0.bin"
echo "done — re-run scripts/package.sh to bake it"
