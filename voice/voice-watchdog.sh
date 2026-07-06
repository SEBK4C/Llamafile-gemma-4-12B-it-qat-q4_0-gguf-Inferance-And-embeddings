#!/bin/sh
# Supervise one baked Kokoro TTS instance: respawn on death AND on WEDGE.
# A wedged instance (hung synthesis worker → all http threads blocked) still
# LISTENS but never answers — process-liveness supervision can't catch it,
# so probe /health and force-respawn after 3 consecutive timeouts.
#   $1 = tts-server APE   $2 = kokoro GGUF   $3 = port   $4 = parent pid
# The watchdog is tied to the MAIN SERVER'S lifetime ($4): atexit-based
# cleanup never runs when the main process is SIGKILLed, which used to leak
# watchdog+instance forever and made the next launch fight for the port
# (F25). Empty $4 = legacy behaviour (run unsupervised).
APE="$1"; GGUF="$2"; PORT="$3"; PARENT="${4:-}"
parent_alive() { [ -z "$PARENT" ] || kill -0 "$PARENT" 2>/dev/null; }
NP="nice -n -5"; command -v nice >/dev/null 2>&1 || NP=""
HAVE_CURL=0; command -v curl >/dev/null 2>&1 && HAVE_CURL=1
while parent_alive; do
    $NP /bin/sh "$APE" -mp "$GGUF" --port "$PORT" &
    PID=$!
    sleep 8
    FAILS=0
    while kill -0 "$PID" 2>/dev/null; do
        sleep 10
        if ! parent_alive; then
            kill -9 "$PID" 2>/dev/null
            exit 0
        fi
        if [ "$HAVE_CURL" = 1 ]; then
            if curl -s -m 6 -o /dev/null "http://127.0.0.1:$PORT/health"; then
                FAILS=0
            else
                FAILS=$((FAILS + 1))
            fi
            if [ "$FAILS" -ge 3 ]; then
                logger -t g4voice "watchdog: instance on :$PORT wedged (3 health timeouts) - force respawn" 2>/dev/null
                kill -9 "$PID" 2>/dev/null
                break
            fi
        fi
    done
    wait "$PID" 2>/dev/null
    sleep 2
done
