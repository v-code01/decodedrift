# Pre-registered predictions (committed before the authoritative sweep)

Study: whether per-token decode latency drifts upward over a long generation, because each new token
attends to a KV cache that has grown by one entry per token. Per-token inter-token latency is
measured by streaming a long generation and timestamping each token; the gap is fit to gap = a + c*
pos (a = position-independent cost, c = per-position KV-growth cost). Two model sizes (1.5B and 0.5B,
Q4_K_M). A pilot to ~6k tokens informed these predictions.

Predictions:

1. **Per-token decode latency drifts up linearly with position.** The slope c is positive and its
   bootstrap CI excludes zero for both models: decoding token N takes measurably longer than token
   10 because attention reads a longer KV cache.

2. **The drift is material over a long generation** - at least a few percent from the start of the
   generation to several thousand tokens in.

3. **The bigger model drifts faster in absolute terms** (larger slope c), because it has more layers
   and a wider KV cache to read per position.

Falsifiers:
- If the slope's CI included zero, prediction 1 is wrong (decode latency would be flat).
- If the relative drift were negligible (< 3%), prediction 2 is wrong.
- If the smaller model had the larger absolute slope, prediction 3 is wrong.

If the data contradicts any prediction, the finding as measured is what ships; this file is not
edited after the results are in.
