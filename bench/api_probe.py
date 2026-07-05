#!/usr/bin/env python3
"""api_probe.py — one-command end-to-end test of every API endpoint + modality
of a gemma4-server.llamafile (or any llama.cpp-based server).

    python3 bench/api_probe.py --base http://127.0.0.1:8080

Tests text / vision / audio-in / TTS / embeddings across the OpenAI-compatible
(chat completions, completions, embeddings, Responses API), Anthropic-compatible
(/v1/messages + count_tokens), and native (llama.cpp /completion, /tokenize)
surfaces, with wall-clock speed numbers for each. Pure stdlib — no pip installs.

Exit code = number of FAILED tests (skips don't count). Writes a JSON + TSV
report to --out (default: bench/data/).

NOTE ON EXPECTATIONS: Gemma 4 12B is a small local model. These probes verify
the API contract and measure speed on YOUR hardware — they are not a capability
benchmark. Agentic coding harnesses will run, but do not expect frontier-model
coding quality from a 12B QAT-Q4 model.
"""
import argparse, base64, io, json, math, ssl, struct, sys, time, urllib.request, urllib.error, wave, zlib

# ---------------------------------------------------------------- helpers

def http(base, path, method="GET", body=None, timeout=180, headers=None, stream=False):
    url = base.rstrip("/") + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    ctx = ssl.create_default_context()
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout, context=ctx)
        if stream:
            return resp, t0  # caller iterates + closes
        raw = resp.read()
        return {"code": resp.status, "wall": time.time() - t0, "raw": raw}
    except urllib.error.HTTPError as e:
        return {"code": e.code, "wall": time.time() - t0, "raw": e.read()}
    except Exception as e:
        return {"code": 0, "wall": time.time() - t0, "raw": str(e).encode(), "exc": True}

def jbody(r):
    try:
        return json.loads(r["raw"].decode())
    except Exception:
        return {}

def sse_events(resp, t0, first_token_key=None):
    """Iterate an SSE stream; return (event_counts, ttft, wall, n_data)."""
    counts, ttft, n_data = {}, None, 0
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if line.startswith("event:"):
            ev = line.split(":", 1)[1].strip()
            counts[ev] = counts.get(ev, 0) + 1
        elif line.startswith("data:"):
            n_data += 1
            if ttft is None:
                payload = line[5:].strip()
                if payload and payload != "[DONE]":
                    ttft = time.time() - t0
    wall = time.time() - t0
    resp.close()
    return counts, ttft, wall, n_data

def make_png(rgb=(0xE0, 0x20, 0x20), w=96, h=96):
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

# ---------------------------------------------------------------- probe suite

class Suite:
    def __init__(self, base, max_tokens, embed_base=None):
        self.base, self.max_tokens = base, max_tokens
        self.embed_base = embed_base
        self.results, self.props = [], {}
        self.tts_wav = None

    def rec(self, test, endpoint, ok, wall, detail, **extra):
        row = {"test": test, "endpoint": endpoint,
               "status": "PASS" if ok is True else ("SKIP" if ok is None else "FAIL"),
               "wall_s": None if wall is None else round(wall, 3), "detail": str(detail)[:220]}
        row.update({k: v for k, v in extra.items() if v is not None})
        self.results.append(row)
        mark = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️ "}[row["status"]]
        w = "" if wall is None else f" [{wall:.2f}s]"
        print(f"{mark} {test:<22} {endpoint:<28}{w} {row['detail'][:110]}")

    # --- basics
    def t_health(self):
        r = http(self.base, "/health", timeout=15)
        self.rec("health", "GET /health", jbody(r).get("status") == "ok", r["wall"], jbody(r))

    def t_props(self):
        r = http(self.base, "/props", timeout=15)
        p = jbody(r)
        self.props = p
        mod = p.get("modalities", {})
        ok = r["code"] == 200 and "model_alias" in p
        self.rec("props", "GET /props", ok, r["wall"],
                 f"model={p.get('model_alias')} ctx={p.get('default_generation_settings',{}).get('n_ctx')} "
                 f"vision={mod.get('vision')} audio={mod.get('audio')}")

    def t_models(self):
        r = http(self.base, "/v1/models", timeout=15)
        b = jbody(r)
        names = [m.get("name") or m.get("id") for m in b.get("models", b.get("data", []))]
        self.rec("models", "GET /v1/models", bool(names), r["wall"], names)

    def t_tokenize(self):
        r = http(self.base, "/tokenize", "POST", {"content": "hello world"}, timeout=15)
        toks = jbody(r).get("tokens", [])
        r2 = http(self.base, "/detokenize", "POST", {"tokens": toks}, timeout=15)
        text = jbody(r2).get("content", "")
        self.rec("tokenize_roundtrip", "POST /tokenize+/detokenize",
                 bool(toks) and "hello" in text.lower(), r["wall"] + r2["wall"],
                 f"{len(toks)} tokens -> {text!r}")

    # --- text generation surfaces
    def t_chat(self):
        body = {"max_tokens": self.max_tokens, "cache_prompt": False,
                "messages": [{"role": "user", "content": "In one short sentence, why is the sky blue?"}]}
        r = http(self.base, "/v1/chat/completions", "POST", body)
        b = jbody(r)
        msg = (b.get("choices") or [{}])[0].get("message", {})
        u = b.get("usage", {})
        tps = round(u.get("completion_tokens", 0) / r["wall"], 1) if r["wall"] else None
        self.rec("chat_completions", "POST /v1/chat/completions",
                 r["code"] == 200 and bool(msg.get("content")), r["wall"],
                 f"content={msg.get('content','')!r:.90} tok/s(wall)={tps}",
                 gen_tokens=u.get("completion_tokens"), tok_s_wall=tps)

    def t_chat_stream(self):
        body = {"max_tokens": 128, "stream": True, "cache_prompt": False,
                "messages": [{"role": "user", "content": "Count from 1 to 5."}]}
        try:
            resp, t0 = http(self.base, "/v1/chat/completions", "POST", body, stream=True)
        except Exception as e:
            return self.rec("chat_stream", "POST /v1/chat/completions", False, None, e)
        if isinstance(resp, dict):
            return self.rec("chat_stream", "POST /v1/chat/completions", False, resp["wall"], resp["raw"][:120])
        counts, ttft, wall, n = sse_events(resp, t0)
        self.rec("chat_stream", "POST /v1/chat/completions (SSE)", n > 2, wall,
                 f"chunks={n} ttft={ttft:.2f}s" if ttft else f"chunks={n}", ttft_s=round(ttft, 3) if ttft else None)

    def t_completions_v1(self):
        r = http(self.base, "/v1/completions", "POST",
                 {"prompt": "The capital of France is", "max_tokens": 16, "cache_prompt": False})
        b = jbody(r)
        text = (b.get("choices") or [{}])[0].get("text", "")
        self.rec("completions_v1", "POST /v1/completions", r["code"] == 200 and bool(text.strip()),
                 r["wall"], repr(text)[:90])

    def t_completion_native(self):
        r = http(self.base, "/completion", "POST",
                 {"prompt": "The capital of France is", "n_predict": 16, "cache_prompt": False})
        b = jbody(r)
        tps = (b.get("timings") or {}).get("predicted_per_second")
        self.rec("completion_native", "POST /completion", r["code"] == 200 and bool(b.get("content", "").strip()),
                 r["wall"], f"content={b.get('content','')!r:.60} server-tok/s={round(tps,1) if tps else '?'}",
                 tok_s_server=round(tps, 1) if tps else None)

    # --- modern/agent surfaces
    def t_messages(self):
        body = {"model": "local", "max_tokens": self.max_tokens,
                "system": "Answer concisely.",
                "messages": [{"role": "user", "content": "Say exactly: PONG"}]}
        r = http(self.base, "/v1/messages", "POST", body,
                 headers={"x-api-key": "none", "anthropic-version": "2023-06-01"})
        b = jbody(r)
        kinds = [blk.get("type") for blk in b.get("content", [])]
        text = " ".join(blk.get("text", "") for blk in b.get("content", []) if blk.get("type") == "text")
        ok = r["code"] == 200 and b.get("type") == "message" and "text" in kinds
        self.rec("messages_anthropic", "POST /v1/messages", ok, r["wall"],
                 f"blocks={kinds} text={text!r:.60} stop={b.get('stop_reason')}")

    def t_messages_stream(self):
        body = {"model": "local", "max_tokens": 128, "stream": True,
                "messages": [{"role": "user", "content": "Count from 1 to 5."}]}
        try:
            resp, t0 = http(self.base, "/v1/messages", "POST", body, stream=True)
        except Exception as e:
            return self.rec("messages_stream", "POST /v1/messages", False, None, e)
        if isinstance(resp, dict):
            return self.rec("messages_stream", "POST /v1/messages", False, resp["wall"], resp["raw"][:120])
        counts, ttft, wall, n = sse_events(resp, t0)
        need = {"message_start", "content_block_delta", "message_stop"}
        self.rec("messages_stream", "POST /v1/messages (SSE)", need.issubset(counts), wall,
                 f"events={counts} ttft={ttft:.2f}s" if ttft else f"events={counts}",
                 ttft_s=round(ttft, 3) if ttft else None)

    def t_count_tokens(self):
        r = http(self.base, "/v1/messages/count_tokens", "POST",
                 {"model": "local", "messages": [{"role": "user", "content": "hello world"}]}, timeout=20)
        n = jbody(r).get("input_tokens")
        self.rec("count_tokens", "POST /v1/messages/count_tokens",
                 isinstance(n, int) and n > 0, r["wall"], f"input_tokens={n}")

    def t_responses(self):
        r = http(self.base, "/v1/responses", "POST",
                 {"model": "local", "input": "Say exactly: PONG", "max_output_tokens": self.max_tokens})
        b = jbody(r)
        types = [o.get("type") for o in b.get("output", [])]
        txt = ""
        for o in b.get("output", []):
            if o.get("type") == "message":
                for c in o.get("content", []):
                    txt += c.get("text", "")
        ok = r["code"] == 200 and b.get("object") == "response" and "message" in types
        self.rec("responses_api", "POST /v1/responses", ok, r["wall"],
                 f"output_types={types} text={txt!r:.60}")

    # --- embeddings
    def t_embeddings(self):
        r = http(self.base, "/v1/embeddings", "POST",
                 {"input": ["cat", "kitten", "spreadsheet"], "model": "local"}, timeout=60)
        b = jbody(r)
        data = b.get("data", [])
        if r["code"] != 200 or len(data) != 3:
            return self.rec("embeddings", "POST /v1/embeddings", False, r["wall"], jbody(r))
        vecs = [d["embedding"] for d in data]
        if isinstance(vecs[0][0], list):  # pooling=none → per-token; take mean
            vecs = [[sum(col) / len(col) for col in zip(*v)] for v in vecs]
        sim_kk = cosine(vecs[0], vecs[1])   # cat~kitten
        sim_ks = cosine(vecs[0], vecs[2])   # cat~spreadsheet
        ok = sim_kk > sim_ks
        self.rec("embeddings", "POST /v1/embeddings", ok, r["wall"],
                 f"dims={len(vecs[0])} cos(cat,kitten)={sim_kk:.3f} > cos(cat,spreadsheet)={sim_ks:.3f} -> semantic sanity",
                 dims=len(vecs[0]), cos_close=round(sim_kk, 4), cos_far=round(sim_ks, 4))

    def t_embeddings_sidecar(self):
        """Optional dedicated embedding server (see docs/embeddings.md)."""
        if not self.embed_base:
            return self.rec("embeddings_sidecar", "POST <embed-base>/v1/embeddings", None, None,
                            "no --embed-base given — skipped (run a dedicated embedder; the main model's embeddings are not semantic)")
        r = http(self.embed_base, "/v1/embeddings", "POST",
                 {"input": ["cat", "kitten", "spreadsheet"], "model": "embed"}, timeout=60)
        b = jbody(r)
        data = b.get("data", [])
        if r["code"] != 200 or len(data) != 3:
            return self.rec("embeddings_sidecar", "POST <embed-base>/v1/embeddings", False, r["wall"], jbody(r))
        vecs = [d["embedding"] for d in data]
        if isinstance(vecs[0][0], list):
            vecs = [[sum(col) / len(col) for col in zip(*v)] for v in vecs]
        sim_kk, sim_ks = cosine(vecs[0], vecs[1]), cosine(vecs[0], vecs[2])
        self.rec("embeddings_sidecar", "POST <embed-base>/v1/embeddings", sim_kk > sim_ks, r["wall"],
                 f"dims={len(vecs[0])} cos(cat,kitten)={sim_kk:.3f} > cos(cat,spreadsheet)={sim_ks:.3f}",
                 dims=len(vecs[0]), cos_close=round(sim_kk, 4), cos_far=round(sim_ks, 4))

    # --- multimodal
    def t_vision(self):
        if not self.props.get("modalities", {}).get("vision"):
            return self.rec("vision_input", "POST /v1/chat/completions", None, None, "server reports vision=false — skipped")
        img = base64.b64encode(make_png()).decode()
        body = {"max_tokens": self.max_tokens, "cache_prompt": False, "messages": [{"role": "user", "content": [
            {"type": "text", "text": "What color is this square? Answer with one word."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}}]}]}
        r = http(self.base, "/v1/chat/completions", "POST", body)
        content = ((jbody(r).get("choices") or [{}])[0].get("message", {}).get("content") or "")
        self.rec("vision_input", "POST /v1/chat/completions", "red" in content.lower(), r["wall"],
                 f"expected 'red', got {content!r:.60}")

    def t_tts(self):
        r = http(self.base, "/tts/v1/audio/speech", "POST",
                 {"input": "The secret word is banana.", "voice": "af_heart", "response_format": "wav"}, timeout=120)
        if r["code"] == 404:
            return self.rec("tts_speech", "POST /tts/v1/audio/speech", None, r["wall"],
                            "no /tts on this build (voice not baked) — skipped")
        ok = r["code"] == 200 and r["raw"][:4] == b"RIFF"
        dur = rtf = None
        if ok:
            self.tts_wav = r["raw"]
            with wave.open(io.BytesIO(r["raw"])) as w:
                dur = w.getnframes() / w.getframerate()
            rtf = round(dur / r["wall"], 2) if r["wall"] else None
        self.rec("tts_speech", "POST /tts/v1/audio/speech", ok, r["wall"],
                 f"{len(r['raw'])}B wav, {dur and round(dur,2)}s audio, {rtf}x realtime", rtf=rtf)

    def t_audio_in(self):
        if not self.props.get("modalities", {}).get("audio"):
            return self.rec("audio_input", "POST /v1/chat/completions", None, None, "server reports audio=false — skipped")
        if not self.tts_wav:
            return self.rec("audio_input", "POST /v1/chat/completions", None, None,
                            "no speech sample (TTS unavailable) — skipped; supply a WAV to test manually")
        aud = base64.b64encode(self.tts_wav).decode()
        body = {"max_tokens": self.max_tokens, "cache_prompt": False, "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Listen to this audio. What is the secret word? Answer with one word."},
            {"type": "input_audio", "input_audio": {"data": aud, "format": "wav"}}]}]}
        r = http(self.base, "/v1/chat/completions", "POST", body)
        content = ((jbody(r).get("choices") or [{}])[0].get("message", {}).get("content") or "")
        self.rec("audio_input", "POST /v1/chat/completions", "banana" in content.lower(), r["wall"],
                 f"TTS->ears round-trip; expected 'banana', got {content!r:.50}")

    # --- documented-unsupported (expect clean 501s, not crashes)
    def t_unsupported(self):
        for name, path, method, body in [
            ("metrics", "/metrics", "GET", None),
            ("rerank", "/v1/rerank", "POST", {"query": "q", "documents": ["d"]}),
            ("infill", "/infill", "POST", {"input_prefix": "a", "input_suffix": "b"})]:
            r = http(self.base, path, method, body, timeout=20)
            self.rec(f"unsupported_{name}", f"{method} {path}", r["code"] == 501, r["wall"],
                     f"expect 501 (off by default / model lacks FIM), got {r['code']}")

    def run(self, quick=False):
        for t in [self.t_health, self.t_props, self.t_models, self.t_tokenize,
                  self.t_chat, self.t_chat_stream, self.t_completions_v1, self.t_completion_native,
                  self.t_messages, self.t_messages_stream, self.t_count_tokens, self.t_responses,
                  self.t_embeddings, self.t_embeddings_sidecar]:
            t()
        if not quick:
            self.t_vision(); self.t_tts(); self.t_audio_in()
        self.t_unsupported()
        return self.results


def main():
    ap = argparse.ArgumentParser(description="E2E probe of all gemma4-server API endpoints + modalities")
    ap.add_argument("--base", default="http://127.0.0.1:8080", help="server base URL")
    ap.add_argument("--out", default=None, help="directory for JSON/TSV report (default: no files)")
    ap.add_argument("--quick", action="store_true", help="skip vision/TTS/audio tests")
    ap.add_argument("--max-tokens", type=int, default=600,
                    help="per-request budget (Gemma 4 spends tokens on hidden reasoning first — keep >=400)")
    ap.add_argument("--embed-base", default=None,
                    help="base URL of a dedicated embedding server (e.g. http://127.0.0.1:8081); see docs/embeddings.md")
    args = ap.parse_args()

    print(f"# api_probe — target {args.base}\n")
    s = Suite(args.base, args.max_tokens, embed_base=args.embed_base)
    t0 = time.time()
    results = s.run(quick=args.quick)
    total = time.time() - t0

    npass = sum(1 for r in results if r["status"] == "PASS")
    nfail = sum(1 for r in results if r["status"] == "FAIL")
    nskip = sum(1 for r in results if r["status"] == "SKIP")
    print(f"\n== {npass} PASS / {nfail} FAIL / {nskip} SKIP in {total:.1f}s "
          f"— model {s.props.get('model_alias')} ==")

    if args.out:
        import os
        os.makedirs(args.out, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        meta = {"stamp": stamp, "model": s.props.get("model_alias"),
                "modalities": s.props.get("modalities"), "total_wall_s": round(total, 1),
                "pass": npass, "fail": nfail, "skip": nskip, "results": results}
        jpath = os.path.join(args.out, f"api_probe_{stamp}.json")
        with open(jpath, "w") as f:
            json.dump(meta, f, indent=1)
        tpath = os.path.join(args.out, f"api_probe_{stamp}.tsv")
        keys = ["test", "endpoint", "status", "wall_s", "ttft_s", "tok_s_wall", "tok_s_server", "detail"]
        with open(tpath, "w") as f:
            f.write("\t".join(keys) + "\n")
            for r in results:
                f.write("\t".join(str(r.get(k, "")) for k in keys) + "\n")
        print(f"report: {jpath}\n        {tpath}")
    sys.exit(min(nfail, 125))


if __name__ == "__main__":
    main()
