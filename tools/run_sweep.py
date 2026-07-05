"""Stream one long generation per model and record the inter-token latency at each position, so the
per-token decode cost can be plotted against the length of the KV cache it attends to. Uses a fixed
short prompt and ignore_eos with a large n_predict, so the whole run is decode over a growing
context. Several reps per model. The first few gaps (adjacent to prefill) are dropped. Writes
results/<model>.jsonl with one row per token. Exact monotonic-clock inter-token timing.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request

PROMPT = "Write an extremely long and detailed continuous story, never stopping:"
WARMUP_GAPS = 3


def stream_gaps(port: int, npred: int, seed: int) -> list[float]:
    body = json.dumps({"prompt": PROMPT, "n_predict": npred, "temperature": 0.7, "seed": seed,
                       "stream": True, "cache_prompt": False, "ignore_eos": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    gaps: list[float] = []
    prev: int | None = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            now = time.perf_counter_ns()
            if prev is not None:
                gaps.append((now - prev) / 1e6)
            prev = now
    return gaps[WARMUP_GAPS:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port15", type=int, default=8001)
    ap.add_argument("--port05", type=int, default=8002)
    ap.add_argument("--npred", type=int, default=6000)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    for port, model in ((args.port15, "qwen15b"), (args.port05, "qwen05b")):
        rows: list[dict[str, object]] = []
        for rep in range(args.reps):
            gaps = stream_gaps(port, args.npred, seed=1 + rep)
            for pos, g in enumerate(gaps):
                rows.append({"model": model, "rep": rep, "pos": pos + WARMUP_GAPS, "gap_ms": g})
            print(f"  {model} rep{rep}: {len(gaps)} gaps", flush=True)
        out = f"results/{model}.jsonl"
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"# SWEEP_DONE {model} {len(rows)} rows -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
