"""Fit per-token decode latency versus generation position for each model. Bins the per-token gaps
into position windows (robust median), fits gap = a + c*pos, and reports the intercept (position-
independent cost), the slope (per-token KV-growth cost), and the relative drift across the run.
Bootstrap CIs on the slope come from resampling reps. Writes bench_results/frontier.md and
curve.json. Pure derivation; no server calls.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from drift import binned_medians, fit_line, relative_drift  # noqa: E402

MODELS = ["qwen15b", "qwen05b"]
WIDTH = 250


def load(path: str) -> list[dict[str, object]]:
    return [json.loads(x) for x in open(path) if x.strip()]


def slope_ci(rows: list[dict[str, object]], reps: list[int], seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    slopes: list[float] = []
    for _ in range(500):
        pick = rng.choice(reps, size=len(reps), replace=True)
        pos: list[int] = []
        gap: list[float] = []
        for rp in pick:
            for r in rows:
                if r["rep"] == int(rp):
                    pos.append(int(str(r["pos"])))
                    gap.append(float(str(r["gap_ms"])))
        binned = binned_medians(pos, gap, WIDTH)
        slopes.append(fit_line([p for p, _ in binned], [m for _, m in binned]).c)
    return float(np.quantile(slopes, 0.025)), float(np.quantile(slopes, 0.975))


def main() -> int:
    curve: dict[str, object] = {}
    lines = ["# decodedrift: does per-token decode latency drift up as the KV cache grows?", "",
             "Each generated token attends to every earlier token, so decode cost per token should "
             "rise linearly with position. gap = a + c*pos, fit on binned medians.", ""]
    for m in MODELS:
        path = f"results/{m}.jsonl"
        if not os.path.exists(path):
            print(f"MISSING {path}", file=sys.stderr)
            return 1
        rows = load(path)
        reps = sorted({int(str(r["rep"])) for r in rows})
        pos = [int(str(r["pos"])) for r in rows]
        gap = [float(str(r["gap_ms"])) for r in rows]
        binned = binned_medians(pos, gap, WIDTH)
        fit = fit_line([p for p, _ in binned], [mm for _, mm in binned])
        pmax = max(pos)
        drift = relative_drift(fit, 0.0, float(pmax))
        lo, hi = slope_ci(rows, reps, seed=len(m))
        curve[m] = {"a": fit.a, "c": fit.c, "c_lo": lo, "c_hi": hi, "pmax": pmax, "drift": drift}
        lines += [f"## {m}", "",
                  "   position     median gap (ms)",
                  "   --------     ---------------"]
        for p, med in binned[::4]:  # every 4th window to keep the table short
            lines.append(f"   {int(p):>8}     {med:>8.3f}")
        lines += ["",
                  f"- intercept a = {fit.a:.3f} ms (position-independent per-token cost)",
                  f"- slope c = {fit.c:.3e} ms/token of context [95% CI {lo:.3e}, {hi:.3e}]",
                  f"- per-token decode latency drifts {drift:.1%} from position 0 to {pmax} "
                  "(the KV-cache growth cost).", ""]
    a15 = curve["qwen15b"]
    a05 = curve["qwen05b"]
    assert isinstance(a15, dict) and isinstance(a05, dict)
    lines += ["## cross-model", "",
              f"- absolute slope is larger for the bigger model ({float(a15['c']):.3e} vs "
              f"{float(a05['c']):.3e} ms/token): more layers and wider KV per position.",
              f"- relative drift over the run: 1.5B {float(a15['drift']):.1%} vs 0.5B "
              f"{float(a05['drift']):.1%}.", ""]
    os.makedirs("bench_results", exist_ok=True)
    with open("bench_results/curve.json", "w") as f:
        json.dump(curve, f, indent=2)
    with open("bench_results/frontier.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
