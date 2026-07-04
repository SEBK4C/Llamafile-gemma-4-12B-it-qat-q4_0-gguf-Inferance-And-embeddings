#!/usr/bin/env python3
"""FROZEN voice-conversation benchmark (autoresearch protocol: this file is
the ground-truth eval and must not be modified by experiments).

Runs a fixed 3-turn conversation against the production voice UI with
auto-speak on, a controllable fake microphone (real VAD path), and prints a
grep-able metric block:

    first_audio_ms_t1: 4210
    first_audio_ms_t2: 3900
    p50_first_audio_ms: 4055
    barge_stop_ms: 240
    transcript_similarity: 0.91
    voice_score: 71.3

voice_score = 100 - p50_first_audio_ms/100 - barge_stop_ms/50
              - (1-transcript_similarity)*30   (higher is better)

Usage: pwenv/bin/python voicebench.py [--url URL] [--out DIR]
Screenshots + WAVs land in DIR (crash triage only — not part of scoring).
"""
import argparse, asyncio, base64, difflib, json, statistics, sys, time, urllib.request
from pathlib import Path
from playwright.async_api import async_playwright

DEF_URL = "https://sebk4c-gemma-4-12b-it-q4.bunny-sunfish.ts.net/"
TURNS = [
    "In one short sentence, what are lighthouses for?",
    "Name three famous lighthouses, one short sentence each.",
    "Briefly, why do sailors still value lighthouses despite GPS?",
]

FAKE_MIC = r"""
(() => {
  // controllable fake mic: real MediaStream from an oscillator so the UI's
  // actual VAD/energy code runs; window.__vbMic.speak(ms) produces "speech"
  const real = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
  let ctx = null, osc = null, gain = null, dest = null;
  navigator.mediaDevices.getUserMedia = async (c) => {
    if (!c || !c.audio) return real(c);
    ctx = new AudioContext();
    osc = ctx.createOscillator(); gain = ctx.createGain(); dest = ctx.createMediaStreamDestination();
    osc.frequency.value = 220; gain.gain.value = 0.0;
    osc.connect(gain); gain.connect(dest); osc.start();
    return dest.stream;
  };
  window.__vbMic = {
    // play real speech (decoded WAV) through the fake mic — lets tests hold
    // actual conversations with the model instead of feeding it sine tones
    async speakBuffer(b64) {
      if (!gain) return false;
      if (ctx.state !== 'running') await ctx.resume();
      const bin = atob(b64), arr = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
      const buf = await ctx.decodeAudioData(arr.buffer);
      const src = ctx.createBufferSource();
      src.buffer = buf; src.connect(dest);
      src.start();
      return buf.duration;
    },
    async speak(ms) {
      if (!gain) return false;
      if (ctx.state !== 'running') await ctx.resume();   // headless contexts start suspended
      gain.gain.value = 0.4;
      setTimeout(() => { gain.gain.value = 0.0; }, ms);
      return true;
    },
    state() { return ctx ? ctx.state : 'none'; },
    ready() { return !!gain; }
  };
  // track every AudioContext so the harness can resume ones created
  // outside a user gesture (headless autostart leaves them suspended)
  const OrigAC = window.AudioContext;
  window.__vbCtxs = [];
  window.AudioContext = class extends OrigAC {
    constructor(...a) { super(...a); window.__vbCtxs.push(this); }
  };
  // audio playback probes
  window.__vbEvents = [];
  const origPlay = HTMLMediaElement.prototype.play;
  HTMLMediaElement.prototype.play = function(...a) {
    this.addEventListener('playing', () => window.__vbEvents.push({ t: performance.now(), ev: 'playing' }), { once: true });
    this.addEventListener('pause', () => window.__vbEvents.push({ t: performance.now(), ev: 'pause' }), { once: true });
    return origPlay.apply(this, a);
  };
})();
"""

def judge_similarity(wav_bytes, expected_text, api):
    """LLM-as-judge: transcribe the first spoken chunk, fuzzy-compare."""
    try:
        body = json.dumps({
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Transcribe this audio exactly, word for word. Output only the transcription."},
                {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav_bytes).decode(), "format": "wav"}}]}],
            "temperature": 0, "max_tokens": 150,
            "chat_template_kwargs": {"enable_thinking": False}})
        r = urllib.request.urlopen(urllib.request.Request(
            api + "/v1/chat/completions", body.encode(),
            {"Content-Type": "application/json"}), timeout=300)
        heard = json.load(r)["choices"][0]["message"]["content"].strip().lower()
        norm = lambda s: " ".join("".join(c for c in s.lower() if c.isalnum() or c.isspace()).split())
        h, e = norm(heard), norm(expected_text)
        e = e[:max(len(h), 40)]
        return difflib.SequenceMatcher(None, h, e).ratio(), heard
    except Exception as ex:
        return 0.0, f"judge-error: {ex}"

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEF_URL)
    ap.add_argument("--out", default="/tmp/voicebench")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    api = args.url.rstrip("/")

    async with async_playwright() as pw:
        b = await pw.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])
        page = await b.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)[:200]))
        await page.add_init_script(FAKE_MIC)
        # auto-speak on, auto-listen off (we drive turns by keyboard; mic used for barge only)
        await page.add_init_script(
            "localStorage.setItem('gemma-autospeak','1');"
            "localStorage.setItem('gemma-automode','1');"
            "localStorage.setItem('gemma-temp0','1');")
        await page.goto(args.url, wait_until="networkidle", timeout=60000)
        await page.wait_for_timeout(2500)
        await page.evaluate("window.__vbCtxs.forEach(c => { try { c.resume() } catch {} })")

        first_audio = []
        captured = []
        for i, prompt in enumerate(TURNS):
            await page.evaluate("window.__vbEvents.length = 0")
            ta = page.locator("textarea").first
            await ta.click(); await ta.fill(prompt)
            t_send = time.monotonic()
            await page.evaluate("window.__vbSend = performance.now()")
            await page.keyboard.press("Enter")
            # wait for first 'playing' event (auto-speak live reader)
            ms = None
            for _ in range(600):
                evs = await page.evaluate("window.__vbEvents")
                pl = [e for e in evs if e["ev"] == "playing"]
                if pl:
                    ms = round(pl[0]["t"] - await page.evaluate("window.__vbSend"))
                    break
                await page.wait_for_timeout(100)
            first_audio.append(ms if ms is not None else 60000)
            # let it speak a bit, capture reply text
            await page.wait_for_timeout(1500)
            cap = await page.evaluate(
                "(document.querySelector('[aria-label=\"Assistant message with actions\"]:last-of-type .agentic-text')||{}).textContent || ''")
            captured.append(cap)
            if i < len(TURNS) - 1:
                # wait for generation + speech to settle before next turn
                for _ in range(240):
                    if (await page.locator('[aria-label="Stop generation"]').count()) == 0:
                        break
                    await page.wait_for_timeout(500)
                await page.wait_for_timeout(4000)
                await page.evaluate("document.querySelectorAll('audio').forEach(a=>{try{a.pause()}catch{}})")
            await page.screenshot(path=str(out / f"turn{i+1}.png"))

        # ── barge-in stop latency (turn 3 just started speaking: its first
        # chunk is ≥10 words ≈ 4s of audio, so 300ms in is always mid-speech) ──
        barge_ms = None
        try:
            await page.wait_for_timeout(300)
            await page.evaluate("window.__vbCtxs.forEach(c => { try { c.resume() } catch {} })")
            ok = await page.evaluate("window.__vbMic && window.__vbMic.ready()")
            if ok:
                await page.evaluate(
                    "window.__vbEvents.length = 0; window.__vbBarge = performance.now(); window.__vbMic.speak(1500)")
                for _ in range(200):
                    evs = await page.evaluate("window.__vbEvents")
                    pa = [e for e in evs if e["ev"] == "pause"]
                    if pa:
                        barge_ms = round(pa[0]["t"] - await page.evaluate("window.__vbBarge"))
                        break
                    await page.wait_for_timeout(25)
            else:
                errors.append("barge: fake mic never initialized (VAD not armed?)")
        except Exception as ex:
            errors.append(f"barge: {ex}")
        # ── voice-turn first audio: a clean SPOKEN question through the fake
        # mic (real VAD → auto-send → auto-speak); measures the actual
        # conversational-loop latency, distinct from typed-turn latency
        voice_ms = None
        try:
            def on_req(req):
                if "/chat/completions" in req.url and req.post_data and '"input_audio"' in req.post_data:
                    asyncio.ensure_future(page.evaluate("if (!window.__vbReqT) window.__vbReqT = performance.now()"))
            page.on("request", on_req)
            # let the barge fallout settle: generation idle, playback stopped
            for _ in range(360):
                busy = await page.locator('[aria-label="Stop generation"]').count()
                if not busy:
                    break
                await page.wait_for_timeout(500)
            await page.wait_for_timeout(6000)
            await page.evaluate("window.__vbEvents.length = 0; window.__vbReqT = 0")
            body = json.dumps({"input": "What color is the sky on a clear day? Answer in one short sentence.", "voice": "af_heart"}).encode()
            wav64 = base64.b64encode(urllib.request.urlopen(urllib.request.Request(
                api + "/tts/v1/audio/speech", body,
                {"Content-Type": "application/json"}), timeout=300).read()).decode()
            await page.evaluate(f"window.__vbMic.speakBuffer('{wav64}')")
            t0 = 0
            for _ in range(600):   # utterance + countdown + send
                t0 = await page.evaluate("window.__vbReqT")
                if t0:
                    break
                await page.wait_for_timeout(100)
            if t0:
                for _ in range(600):
                    evs = await page.evaluate("window.__vbEvents")
                    pl = [e for e in evs if e["ev"] == "playing" and e["t"] > t0]
                    if pl:
                        voice_ms = round(pl[0]["t"] - t0)
                        break
                    await page.wait_for_timeout(100)
        except Exception as ex:
            errors.append(f"voice-turn: {ex}")
        await page.screenshot(path=str(out / "final.png"))
        await b.close()

    # ── judge: synthesize the reply's first words on /tts, then transcribe ──
    sim = 0.0
    if captured and captured[0]:
        words = " ".join(captured[0].split()[:12])
        try:
            body = json.dumps({"input": words, "voice": "af_heart"}).encode()
            r = urllib.request.urlopen(urllib.request.Request(
                api + "/tts/v1/audio/speech", body,
                {"Content-Type": "application/json"}), timeout=300)
            wav = r.read()
            (out / "judge_input.wav").write_bytes(wav)
            sim, heard = judge_similarity(wav, words, api)
            (out / "turn1_heard.txt").write_text(f"expected: {words}\nheard:    {heard}")
        except Exception as ex:
            (out / "turn1_heard.txt").write_text(f"synth-error: {ex}")

    ok = [m for m in first_audio if m < 60000]
    p50 = round(statistics.median(ok)) if ok else 60000
    bm = barge_ms if barge_ms is not None else 5000
    score = round(100 - p50 / 100 - bm / 50 - (1 - sim) * 30, 1)
    for i, m in enumerate(first_audio):
        print(f"first_audio_ms_t{i+1}: {m}")
    print(f"p50_first_audio_ms: {p50}")
    print(f"barge_stop_ms: {bm}")
    print(f"voice_first_audio_ms: {voice_ms if voice_ms is not None else 60000}")
    print(f"transcript_similarity: {round(sim, 2)}")
    print(f"voice_score: {score}")
    if errors:
        print(f"page_errors: {len(errors)} (see triage)", file=sys.stderr)
        for e in errors[:5]:
            print("  " + e, file=sys.stderr)

asyncio.run(main())
