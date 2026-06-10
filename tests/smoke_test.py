"""Smoke test: chat completions and embeddings from the SAME server instance.

Run `make serve` in another terminal (or pass --url), then: make test
"""

import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "client"))

from gemma4_client import Gemma4Client, cosine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    args = ap.parse_args()
    c = Gemma4Client(args.url)

    print(f"[1/4] health @ {args.url} ...", flush=True)
    print("      ", c.health())

    print("[2/4] chat completion ...", flush=True)
    t = time.time()
    answer = c.chat(
        [{"role": "user", "content": "Reply with exactly one word: the color of a cloudless daytime sky."}],
        temperature=0, max_tokens=256,  # leave room for the thinking channel
    )
    print(f"       {answer!r}  ({time.time()-t:.1f}s)")
    assert "blue" in answer.lower(), f"unexpected chat answer: {answer!r}"

    print("[3/4] embeddings ...", flush=True)
    t = time.time()
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A quick brown fox leaped over a lazy dog.",
        "Quarterly revenue grew nine percent in the third fiscal quarter.",
    ]
    vecs = c.embed(texts)
    dims = {len(v) for v in vecs}
    sim_related = cosine(vecs[0], vecs[1])
    sim_unrelated = cosine(vecs[0], vecs[2])
    print(f"       dim={dims.pop()}, sim(paraphrase)={sim_related:.3f}, "
          f"sim(unrelated)={sim_unrelated:.3f}  ({time.time()-t:.1f}s)")
    assert not dims, "embedding dimensions differ between inputs"
    assert sim_related > sim_unrelated, (
        f"semantic ordering violated: {sim_related:.3f} <= {sim_unrelated:.3f}")

    # regression check for patches/0001: batching unequal-length inputs must
    # not change the vectors (the iSWA cache used to pool only the final
    # ubatch of a split sequence)
    solo = [c.embed(t)[0] for t in texts]
    for i, (s, b) in enumerate(zip(solo, vecs)):
        drift = cosine(s, b)
        assert drift > 0.9999, f"batched embedding {i} drifted from solo: {drift:.4f}"
    print("       batch-vs-solo consistency OK")

    print("[4/4] concurrent mixed load (chat + embeddings in flight together) ...",
          flush=True)
    results, errors = {}, []

    def do_chat():
        try:
            results["chat"] = c.chat(
                [{"role": "user", "content": "Count from 1 to 10, digits only."}],
                temperature=0, max_tokens=512,
            )
        except Exception as e:  # noqa: BLE001
            errors.append(("chat", e))

    def do_embed(i):
        try:
            results[f"embed{i}"] = c.embed(f"concurrent embedding probe #{i}")[0]
        except Exception as e:  # noqa: BLE001
            errors.append((f"embed{i}", e))

    threads = [threading.Thread(target=do_chat)] + [
        threading.Thread(target=do_embed, args=(i,)) for i in range(4)
    ]
    t = time.time()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors, f"concurrent failures: {errors}"
    assert len(results) == 5
    print(f"       5/5 requests OK ({time.time()-t:.1f}s)")

    print("[5/5] KV cache persistence (save -> erase -> restore -> reuse) ...",
          flush=True)
    kv_msgs = [{"role": "system", "content": "You are a pirate. " * 40},
               {"role": "user", "content": "Say ahoy."}]
    c.erase_slot(0), c.erase_slot(1)  # so only the pirate state exists below
    first = c._post("/v1/chat/completions",
                    {"messages": kv_msgs, "temperature": 0, "max_tokens": 64})
    # the request may land on either slot; save whichever holds tokens
    for slot in (0, 1):
        if c.save_slot("smoke-kv.bin", slot=slot)["n_saved"] > 0:
            break
    else:
        raise AssertionError("no slot held KV state after the request")
    c.erase_slot(0), c.erase_slot(1)
    c.restore_slot("smoke-kv.bin", slot=slot)
    again = c._post("/v1/chat/completions",
                    {"messages": kv_msgs, "temperature": 0, "max_tokens": 64})
    t = again["timings"]
    print(f"       restored to slot {slot}: cache_n={t['cache_n']}, prompt_n={t['prompt_n']}")
    assert t["cache_n"] >= 100, f"restored KV state barely reused: cache_n={t['cache_n']}"
    assert again["choices"][0]["message"] == first["choices"][0]["message"], \
        "answer differs after KV restore"

    print("\nPASS — one instance served inference and embeddings.")


if __name__ == "__main__":
    main()
