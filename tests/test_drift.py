from __future__ import annotations

from drift import Line, binned_medians, fit_line, relative_drift


def test_fit_recovers_line() -> None:
    a, c = 6.0, 1.5e-4
    xs = [float(i) for i in range(0, 5000, 250)]
    ys = [a + c * x for x in xs]
    fit = fit_line(xs, ys)
    assert abs(fit.a - a) < 1e-9
    assert abs(fit.c - c) < 1e-12


def test_fit_flat_has_zero_slope() -> None:
    xs = [0.0, 100.0, 200.0, 300.0]
    ys = [6.0, 6.0, 6.0, 6.0]
    fit = fit_line(xs, ys)
    assert abs(fit.c) < 1e-12
    assert abs(fit.a - 6.0) < 1e-9


def test_relative_drift() -> None:
    line = Line(6.0, 1.2e-4)
    # (6 + 1.2e-4*5000)/(6) - 1 = 0.6/6 = 0.10
    assert abs(relative_drift(line, 0.0, 5000.0) - 0.10) < 1e-9


def test_binned_medians() -> None:
    pos = [0, 1, 2, 100, 101, 102]
    gaps = [1.0, 3.0, 2.0, 10.0, 30.0, 20.0]
    out = binned_medians(pos, gaps, width=100)
    assert out[0] == (50.0, 2.0)     # median of [1,3,2]
    assert out[1] == (150.0, 20.0)   # median of [10,30,20]


def test_fit_requires_two_points() -> None:
    threw = False
    try:
        fit_line([1.0], [2.0])
    except ValueError:
        threw = True
    assert threw
