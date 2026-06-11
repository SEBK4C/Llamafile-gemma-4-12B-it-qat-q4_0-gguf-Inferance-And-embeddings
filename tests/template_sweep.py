"""WS1 sweep: prompted-embedding templates over the modality-gap battery.

Runs a fixed set of named template configs (same wrapper applied on BOTH
sides of the comparison — media wrapped AND text wrapped) against a running
server and tabulates the metrics that matter: cross-modal same-topic
similarity, same-modality cross-topic similarity, retrieval, margin.

Server: GEMMA4_NGL=0 make serve   (media embeddings are CPU-only until WS3)

    python3 tests/template_sweep.py [--url ...] [--only name1,name2]
"""

import argparse
import json

import modality_gap as mg

CONFIGS = {
    # bare inputs — replicates the 2026-06-10 baseline numbers
    "baseline": {
        "t_text":  "{text}",
        "t_image": "{marker}",
        "t_audio": "{marker}",
    },
    # leading instruction, marker last (instruction tokens come first, so
    # under causal attention they cannot see the media rows — included as
    # the control for position sensitivity)
    "instr-lead": {
        "t_text":  "read this text: {text}",
        "t_image": "read the text in this image: {marker}",
        "t_audio": "transcribe this audio: {marker}",
    },
    # marker FIRST + trailing instruction: trailing text tokens attend to
    # all media rows, so mean pooling picks up content-conditioned text
    # states and last pooling sees a token that summarizes the media
    "instr-trail": {
        "t_text":  "{text}\nthe text above says:",
        "t_image": "{marker}\nthe text in the image above says:",
        "t_audio": "{marker}\nthe audio above says:",
    },
    # PromptEOL-style one-word compressor (note: PromptEOL alone did NOT
    # fix text-text anisotropy in this model, 2026-06-10 — testing the
    # cross-modal manifold-pulling mechanism, which is different)
    "prompteol": {
        "t_text":  'this sentence: "{text}" means in one word:',
        "t_image": 'the text in this image: "{marker}" means in one word:',
        "t_audio": 'this audio: "{marker}" means in one word:',
    },
}


def run(url, names):
    rows = {}
    for name in names:
        cfg = CONFIGS[name]
        print(f"== {name} ==", flush=True)
        E = mg.embed_battery(url, **cfg)
        rows[name] = mg.metrics(E)
    return rows


def tabulate(rows):
    def rng(xs):
        return f"{min(xs):.2f}-{max(xs):.2f}"
    print(f"\n{'config':<12} {'xmod-img':>10} {'xmod-aud':>10} "
          f"{'blk-img':>10} {'blk-aud':>10} {'ret-raw':>8} {'ret-cor':>8} "
          f"{'mrg-img':>8} {'mrg-aud':>8}")
    for name, m in rows.items():
        print(f"{name:<12} {rng(m['xmodal']['image']):>10} {rng(m['xmodal']['audio']):>10} "
              f"{rng(m['block']['image']):>10} {rng(m['block']['audio']):>10} "
              f"{m['retrieval_raw']:>6}/{m['retrieval_total']} "
              f"{m['retrieval_corrected']:>6}/{m['retrieval_total']} "
              f"{m['margin']['image']:>+8.3f} {m['margin']['audio']:>+8.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--only", default=None, help="comma-separated config names")
    ap.add_argument("--json", default=None, help="write raw metrics JSON here")
    args = ap.parse_args()
    names = args.only.split(",") if args.only else list(CONFIGS)
    rows = run(args.url, names)
    tabulate(rows)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)
        print("wrote", args.json)
