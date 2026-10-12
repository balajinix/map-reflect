# v5 cost accounting: collection against standalone (offline)

Collection costs (what the collector spent; a widen outcome copied from NT costs nothing extra) are what the frozen fit and the scored test use. Standalone costs charge a copied widen outcome the NT action's calls, as a deployed strategy would pay (v5.standalone_costs). No model call; nothing frozen or scored is rewritten.

Frozen tau 0.007133. Refit under collection costs: 0.007133; under standalone costs: 0.007133 (C 0.1, frozen C 0.1). Unchanged: True.

## Development

8,628 field-model pairs; 7,318 copied widen outcomes, 5,812 of them where widen is the routed action.

| strategy | calls per field, collection | calls per field, standalone |
|---|--:|--:|
| v5 (benefit router) | 0.1720 | 0.2959 |
| margin router (t_m refitted) | 0.2156 | 0.5746 |
| margin router (v4.1 thresholds) | 0.1926 | 0.4067 |
| struct (widen action) | 0.1748 | 0.1748 |
| every field (routed action) | 0.3755 | 1.0821 |

## Core test

13,384 field-model pairs; 1,623 copied widen outcomes, 0 of them where widen is the routed action.

| strategy | calls per field, collection | calls per field, standalone |
|---|--:|--:|
| v5 (benefit router) | 0.4382 | 0.4382 |
| margin router (t_m refitted) | 0.6863 | 0.6863 |
| margin router (v4.1 thresholds) | 0.5433 | 0.5433 |
| struct (widen action) | 0.3693 | 0.3693 |
| every field (routed action) | 1.1285 | 1.1285 |

## Held-out test

3,652 field-model pairs; 15 copied widen outcomes, 0 of them where widen is the routed action.

| strategy | calls per field, collection | calls per field, standalone |
|---|--:|--:|
| v5 (benefit router) | 1.0334 | 1.0334 |
| margin router (t_m refitted) | 1.0331 | 1.0331 |
| margin router (v4.1 thresholds) | 0.9463 | 0.9463 |
| struct (widen action) | 0.9209 | 0.9209 |
| every field (routed action) | 1.1517 | 1.1517 |

## Scored test facts

| fact | committed | recomputed (collection) |
|---|--:|--:|
| F-VFiveMainCalls | 0.44 | 0.4382 |
| F-VFiveMarginCalls | 0.69 | 0.6863 |
| F-VFiveMarginOldCalls | 0.54 | 0.5433 |
| F-VFiveStructCalls | 0.37 | 0.3693 |
| F-VFiveEveryCalls | 1.13 | 1.1285 |
