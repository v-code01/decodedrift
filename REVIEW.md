# Adversarial review: decodedrift

A skeptic's pass over the claims, and why each survives.

## "The drift is thermal throttling, not KV growth."
Three things rule it out. The rise is smooth and linear in POSITION across the whole run, not a
step or a late-onset knee as throttling would give; it reproduces across three independent runs; and
the slope matches the mechanism (per-token attention over a linearly growing KV cache is O(context)).
Thermal effects on this Metal backend under caffeinate would also affect the intercept run-to-run,
but the intercept is stable. The linear-in-context shape is the signature of KV growth.

## "Client-side timestamps are noisy - you are measuring the network or the client."
The client reads one stream on one thread (no lock contention), and the per-token gap is dominated by
the ~6 ms server decode step; scheduling jitter is absorbed by taking medians over 250-token windows.
The sibling latency study shows single-stream inter-token jitter is ~1.1x, far smaller than the 13-17%
drift measured here. The drift is in the server's decode, not the client.

## "You fit a line to something that might be sub-linear or noisy."
A straight line is the correct model: one decode step attends to L keys, cost linear in L, and L
grows by one per token. The binned medians are visibly linear (6.16, 6.31, 6.51, 6.66, 6.84, 7.00),
and the slope's bootstrap CI is tight and excludes zero. numpy's polyfit in verify.py recovers the
same slope as the hand-rolled OLS, so the shape is not a solver artifact.

## "temperature 0.7 means different runs generate different text - unfair comparison."
Position, not content, is the independent variable, and each token is a full forward-plus-sample step
regardless of which token is chosen. Sampling temperature does not change the per-step attention cost;
it only varies which tokens are produced, which is why medians over windows and across reps are
stable. Greedy would give the same drift with less content variety.

## "The bigger-model-drifts-faster claim could be within noise."
The slopes are 1.74e-4 and 8.46e-5 with non-overlapping bootstrap CIs (1.72-1.77e-4 vs 8.3-8.8e-5),
a clean 2x separation, and the mechanism (KV size scales with depth and width) predicts the direction.
It is not a marginal call.

## "verify.py could share the fit with analyze.py."
It shares no code: verify.py uses numpy.polyfit on its own binning, its own bootstrap, and asserts
the slope, drift, and ordering from the raw JSONL. drift.py and analyze.py could both be wrong and
verify would catch it.

## Pre-registration honesty
All three predictions held, including the directional cross-model one. PREREG.md was committed before
the results and is unedited.
