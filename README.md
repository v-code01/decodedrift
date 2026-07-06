# decodedrift: does per-token decode latency drift up as the KV cache grows?

Generating a token means attending to every token before it. As a generation gets longer, the KV
cache each new token reads grows by one entry per step, so decode should get slower the further in
you are. This streams long generations, timestamps every token, and fits the per-token latency to
`gap = a + c*pos` (a = position-independent cost, c = per-position KV-growth cost). Two model sizes
(1.5B and 0.5B, Q4_K_M), monotonic-clock inter-token timing. It is the decode-time complement to the
prefill attention wall.

## Pre-registration

Three predictions were committed to git (`PREREG.md`) before the sweep: (1) per-token decode latency
drifts up linearly with position (slope positive, CI excludes zero); (2) the drift is material over
a long generation; (3) the bigger model drifts faster in absolute terms. **All three held.**

## Result

Median inter-token latency by position (robust binned medians):

```
 position   1.5B ms/token   0.5B ms/token
    125         6.155           3.694
   1125         6.306           3.755
   2125         6.505           3.874
   3125         6.659           3.958
   4125         6.838           4.019
   5125         6.998           4.104
```

- **1.5B:** intercept a = 6.11 ms, slope c = 1.74e-4 ms per token of context [95% CI 1.72e-4,
  1.77e-4], drift **+17.1%** from position 0 to ~6000.
- **0.5B:** intercept a = 3.67 ms, slope c = 8.46e-5 ms per token of context [95% CI 8.27e-5,
  8.76e-5], drift **+13.8%**.

1. **Per-token decode latency drifts up linearly with position.** For both models the slope is
   positive with a tight CI that excludes zero. Decoding the 6000th token takes 13-17% longer than
   the first, purely because attention reads a longer KV cache each step. The intercept is the
   position-independent cost (weight streaming and feed-forward, which the sibling roofline study
   shows is the memory-bound floor); the slope is the KV-growth tax on top.

2. **The bigger model drifts faster, in both absolute and relative terms.** Its slope is 2x the
   smaller model's (1.74e-4 vs 8.46e-5 ms/token) and its relative drift is larger (17.1% vs 13.8%),
   because the KV cache scales with depth and width - more layers and wider per-position state to
   read. So a larger model pays a steeper long-generation latency penalty, not just a higher
   baseline.

**The one-line finding:** long generations get slower per token as they go, linearly, at a rate set
by KV-cache size - so a fixed per-token latency budget silently erodes over a long response, and more
so for bigger models. This is the decode-time twin of the quadratic prefill wall.

## Reproduce

The servers must run with a context large enough for the generation and a single slot:

```
./reproduce.sh 8001 8002        # PORT_15B PORT_05B (the ports your servers run on)
./scripts/gate.sh               # ruff, mypy --strict, pytest, ASCII, independent verify
```

`tools/verify.py` refits with numpy's polyfit (independent of the hand-rolled OLS in `src/drift.py`)
and re-asserts the positive slope, the drift, and the cross-model ordering from the raw per-token
rows.

## Limitations and falsifiers

- One backend (llama.cpp on Apple GPU, flash attention on), one build, Q4_K_M, two sizes, ~6000
  generated tokens, temperature 0.7. Flash attention reduces the constant, not the linear-in-context
  scaling of the per-token read.
- Inter-token gaps are client-side monotonic-clock timestamps on a single stream; the first few gaps
  (adjacent to prefill) are dropped, and medians over 250-token windows absorb per-token spikes.
- **Falsifier:** if the slope's CI included zero, decode latency would be flat and finding 1 wrong.
  It excludes zero for both models.
- **Falsifier:** if the smaller model had the larger absolute slope, finding 2 would be wrong; it is
  8.5e-5 vs 1.74e-4.

MIT licensed. The oracle is monotonic-clock timing plus a closed-form fit; no model-quality judgement.
