"""drift core: fit per-token decode latency as a function of generation position to a straight line
gap(pos) = a + c*pos. The intercept a is the position-independent cost (weights, feed-forward); the
slope c is the per-position cost of attending to a KV cache that grows by one entry each token. From
the fit it reports the slope, the relative drift over a span, and a robust binned-median summary so
outlier tokens do not dominate. Closed-form ordinary least squares, no external solver. Pure
functions; no I/O, no server.
"""
from __future__ import annotations

from typing import NamedTuple


class Line(NamedTuple):
    a: float  # intercept, ms (position-independent per-token cost)
    c: float  # slope, ms per token of context (KV-growth cost)


def fit_line(xs: list[float], ys: list[float]) -> Line:
    """Ordinary least squares y = a + c*x in closed form. Requires >= 2 distinct x."""
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("need >= 2 paired points")
    n = float(len(xs))
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0.0:
        raise ValueError("degenerate x (all equal)")
    c = (n * sxy - sx * sy) / denom
    a = (sy - c * sx) / n
    return Line(a, c)


def relative_drift(line: Line, x_lo: float, x_hi: float) -> float:
    """Fractional increase in per-token latency from position x_lo to x_hi under the fit."""
    lo = line.a + line.c * x_lo
    hi = line.a + line.c * x_hi
    return (hi - lo) / lo if lo > 0.0 else 0.0


def binned_medians(positions: list[int], gaps: list[float], width: int) -> list[tuple[float, float]]:
    """Median gap in consecutive position windows of the given width: robust to per-token spikes.
    Returns (window_center, median_gap) pairs for windows that contain data."""
    if width < 1:
        raise ValueError("width must be >= 1")
    buckets: dict[int, list[float]] = {}
    for p, g in zip(positions, gaps):
        buckets.setdefault(p // width, []).append(g)
    out: list[tuple[float, float]] = []
    for b in sorted(buckets):
        vals = sorted(buckets[b])
        k = len(vals)
        med = vals[k // 2] if k % 2 else (vals[k // 2 - 1] + vals[k // 2]) / 2.0
        out.append((b * width + width / 2.0, med))
    return out
