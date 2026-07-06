#!/usr/bin/env python3
"""Independent verification of the decodedrift headline. Shares NO code with src/drift.py or
analyze.py: it re-reads the raw per-token rows, computes binned medians and a straight-line fit with
numpy's polyfit (a different solver from the hand-rolled OLS), and asserts (1) the slope is positive
with a bootstrap CI excluding zero for both models (decode latency drifts up with context); (2) the
relative drift over the run is materially above zero; (3) the bigger model has the larger absolute
slope. Exit non-zero on mismatch. Run in the gate.
"""
from __future__ import annotations

import json
import sys

import numpy as np

WIDTH = 250


def binned(pos: list[int], gap: list[float]) -> tuple[list[float], list[float]]:
    buckets: dict[int, list[float]] = {}
    for p, g in zip(pos, gap):
        buckets.setdefault(p // WIDTH, []).append(g)
    xs: list[float] = []
    ys: list[float] = []
    for b in sorted(buckets):
        xs.append(b * WIDTH + WIDTH / 2.0)
        ys.append(float(np.median(buckets[b])))
    return xs, ys


def main() -> int:
    slopes: dict[str, float] = {}
    ok = True
    for m in ("qwen15b", "qwen05b"):
        rows = [json.loads(x) for x in open(f"results/{m}.jsonl") if x.strip()]
        reps = sorted({int(r["rep"]) for r in rows})
        pos = [int(r["pos"]) for r in rows]
        gap = [float(r["gap_ms"]) for r in rows]
        xs, ys = binned(pos, gap)
        c, a = np.polyfit(np.array(xs), np.array(ys), 1)  # independent solver
        pmax = max(pos)
        drift = (a + c * pmax - (a + 0.0)) / (a + 0.0)
        # bootstrap slope over reps
        rng = np.random.default_rng(7)
        bs = []
        for _ in range(500):
            pick = rng.choice(reps, size=len(reps), replace=True)
            keep = set(int(p) for p in pick)
            sp = [int(r["pos"]) for r in rows if r["rep"] in keep]
            sg = [float(r["gap_ms"]) for r in rows if r["rep"] in keep]
            bx, by = binned(sp, sg)
            bs.append(float(np.polyfit(np.array(bx), np.array(by), 1)[0]))
        c_lo = float(np.quantile(bs, 0.025))
        slopes[m] = float(c)
        pos_slope = c > 0 and c_lo > 0
        real_drift = drift > 0.03
        print(f"  {m}: slope {c:.3e} (CI-lo {c_lo:.3e}) pos={pos_slope} | drift {drift:.1%} "
              f"real={real_drift}")
        ok = ok and pos_slope and real_drift
    bigger_steeper = slopes["qwen15b"] > slopes["qwen05b"]
    print(f"  bigger model has larger absolute slope ({slopes['qwen15b']:.3e} > "
          f"{slopes['qwen05b']:.3e}): {bigger_steeper}")
    ok = ok and bigger_steeper
    if ok:
        print("VERIFY OK: per-token decode latency drifts up linearly with KV-cache growth; the "
              "bigger model drifts faster in absolute terms")
        return 0
    print("VERIFY FAILED", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
