# Method v5: development fit

8,628 development field-model pairs.

## (a) Intervention on the D3 development population

1,219 fields (each on its routed action). Net repairs, paired split-group intervals.

| reflection + verifier | net | minus current | combination minus this | repairs | breaks | OMOP net | other net |
|---|--:|--:|--:|--:|--:|--:|--:|
| cur_chk | +302 [261.4, 347.6] | +0 [0.0, 0.0] | +76 [56.8, 94.8] | 304 | 2 | +8 | +294 |
| cur_either | +317 [275.3, 362.5] | +15 [8.2, 22.9] | +61 [44.0, 76.7] | 320 | 3 | +8 | +309 |
| new_chk | +355 [308.4, 401.3] | +53 [38.6, 66.7] | +23 [12.6, 34.2] | 356 | 1 | +8 | +347 |
| new_either | +378 [329.2, 425.1] | +76 [56.8, 94.8] | +0 [0.0, 0.0] | 381 | 3 | +9 | +369 |

Selected: new_either.

## (c) Benefit router

C = 0.1; tau = 0.0071; cross-fitted dev accuracy 95.39% at a routed share of 23.0%.

| model | fields | keep | v5 (cross-fitted) | every field | routed |
|---|--:|--:|--:|--:|--:|
| gptoss | 2,157 | 93.14 | 96.29 | 96.20 | 19.3% |
| llama4 | 2,157 | 92.21 | 96.01 | 95.87 | 24.5% |
| granite | 2,157 | 89.52 | 92.30 | 92.35 | 25.1% |
| qwen35 | 2,157 | 90.08 | 96.94 | 96.62 | 23.1% |

Margin router (comparison), t_m refitted, v4.1 t_s: gptoss t_m=0.54; llama4 t_m=0.2; granite t_m=0.6; qwen35 t_m=0.55
