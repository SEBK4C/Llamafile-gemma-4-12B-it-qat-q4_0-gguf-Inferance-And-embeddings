"""Benchmark generation speed of the running server on two workloads:
freeform prose, and an edit task where drafts can copy from the prompt
(the best case for self-speculation)."""

import json
import sys
import urllib.request

SNIPPET = '''\
{
  "name": "gemma4-server",
  "version": "1.0.0",
  "description": "Dual-mode llamafile server",
  "endpoints": ["/v1/chat/completions", "/v1/embeddings", "/health"],
  "defaults": {"ctx": 8192, "slots": 2, "ubatch": 2048, "pooling": "mean"},
  "license": "Apache-2.0"
}'''

CASES = {
    "prose": ("Explain in two short paragraphs why the sky is blue.", 360),
    "edit":  (f"Here is a JSON file:\n```json\n{SNIPPET}\n```\n"
              "Reply with exactly the same JSON, changing only the version to 2.0.0.", 400),
}


def run(label, prompt, max_tokens, url):
    req = urllib.request.Request(url + "/v1/chat/completions",
        data=json.dumps({"messages": [{"role": "user", "content": prompt}],
                         "temperature": 0, "max_tokens": max_tokens}).encode(),
        headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=600).read())
    t = r["timings"]
    print(f"{label:6} gen={t['predicted_n']:4d} tok  "
          f"speed={t['predicted_per_second']:6.2f} tok/s  "
          f"draft_n={t.get('draft_n', 0)}  draft_acc={t.get('draft_n_accepted', 0)}")
    return t["predicted_per_second"]


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
    tag = sys.argv[2] if len(sys.argv) > 2 else "run"
    print(f"== {tag} ==")
    for label, (prompt, n) in CASES.items():
        run(label, prompt, n, url)
