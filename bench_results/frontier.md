# decodedrift: does per-token decode latency drift up as the KV cache grows?

Each generated token attends to every earlier token, so decode cost per token should rise linearly with position. gap = a + c*pos, fit on binned medians.

## qwen15b

   position     median gap (ms)
   --------     ---------------
        125        6.155
       1125        6.306
       2125        6.505
       3125        6.659
       4125        6.838
       5125        6.998

- intercept a = 6.112 ms (position-independent per-token cost)
- slope c = 1.743e-04 ms/token of context [95% CI 1.719e-04, 1.773e-04]
- per-token decode latency drifts 17.1% from position 0 to 5999 (the KV-cache growth cost).

## qwen05b

   position     median gap (ms)
   --------     ---------------
        125        3.694
       1125        3.755
       2125        3.874
       3125        3.958
       4125        4.019
       5125        4.104

- intercept a = 3.670 ms (position-independent per-token cost)
- slope c = 8.457e-05 ms/token of context [95% CI 8.274e-05, 8.758e-05]
- per-token decode latency drifts 13.8% from position 0 to 5999 (the KV-cache growth cost).

## cross-model

- absolute slope is larger for the bigger model (1.743e-04 vs 8.457e-05 ms/token): more layers and wider KV per position.
- relative drift over the run: 1.5B 17.1% vs 0.5B 13.8%.

