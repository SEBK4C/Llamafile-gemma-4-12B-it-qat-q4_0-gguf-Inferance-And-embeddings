"""Zero-dependency client for the dual-mode Gemma 4 llamafile server.

One server instance, two capabilities:

    >>> c = Gemma4Client()
    >>> c.chat([{"role": "user", "content": "Why is the sky blue?"}])
    'Rayleigh scattering ...'
    >>> v = c.embed(["the sky is blue", "der Himmel ist blau"])
    >>> cosine(v[0], v[1])
    0.93...
"""

import json
import math
import urllib.request


class Gemma4Client:
    def __init__(self, base_url="http://127.0.0.1:8080", timeout=300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path, payload):
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def chat(self, messages, **kwargs):
        """OpenAI-style chat completion; returns the assistant text."""
        payload = {"messages": messages, **kwargs}
        out = self._post("/v1/chat/completions", payload)
        return out["choices"][0]["message"]["content"]

    def embed(self, texts):
        """OpenAI-style embeddings; returns one vector per input text."""
        if isinstance(texts, str):
            texts = [texts]
        out = self._post("/v1/embeddings", {"input": texts})
        data = sorted(out["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data]

    def health(self):
        with urllib.request.urlopen(self.base_url + "/health", timeout=10) as resp:
            return json.loads(resp.read())

    # On-disk KV cache (server must run with --slot-save-path, which our
    # serve script and packaged llamafile do by default). Saved state lives
    # in the hidden cache dir and survives server restarts.

    def save_slot(self, filename, slot=0):
        """Persist a slot's KV cache (processed prompt state) to disk."""
        return self._post(f"/slots/{slot}?action=save", {"filename": filename})

    def restore_slot(self, filename, slot=0):
        """Load a previously saved KV cache into a slot."""
        return self._post(f"/slots/{slot}?action=restore", {"filename": filename})

    def erase_slot(self, slot=0):
        """Clear a slot's KV cache."""
        return self._post(f"/slots/{slot}?action=erase", {})


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb)
