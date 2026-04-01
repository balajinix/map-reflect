# Result Provenance Audit — Complete Findings

**Date:** 2026-03-31
**Auditor:** Claude Code session
**Triggered by:** Discovery that WatsonX temperature=0 is non-deterministic

---

## Executive Summary

We audited the source files for all numbers in the paper. The main findings:

1. **Table 7 (ITOpsGraph ablation): VERIFIED.** All 9 numbers come from
   `results/signal_ablation/*/metrics.accuracy`. The stored accuracy accounts
   for reflected predictions. All configs share the same MAP cache (verified:
   only 2-3 of 1419 predictions differ between configs). Internally consistent.

2. **Table 8 (OMOP ablation): VERIFIED.** Numbers come from
   `results/reflection_ablation_20260228_130716/*.log` (Feb 28 run). Only 5
   configs were run (base, self, error, neighborhood, uq_sampling). Same batch.

3. **Tables 3-4 (main results): MIXED PROVENANCE.** The paper numbers come from
   multiple generations of result files. Some match `mapreflect_*` files (Feb-Mar),
   others match older `ue_reflect_*` files (Jan-Feb), and some don't match any
   file we can find. See detailed breakdown below.

4. **WatsonX non-determinism:** 40.9% of top-1 predictions change between cached
   and uncached runs, causing ~5pp accuracy swings. Any delta < 5pp is within noise.
   However, this does NOT affect the ablation (Table 7) because all configs share
   the same cached MAP responses.

---

## Table 7: Signal Ablation (ITOpsGraph, Llama)

### Source files
All at `results/signal_ablation/<config>/mapreflect_itopsgraph_meta-llama_llama-3-3-70b-instruct.json`

### How paper numbers were computed
The paper reads `metrics.accuracy` from each JSON file. This differs from
recomputing accuracy from `predictions.initial_ranking[0]` because the stored
metric accounts for **reflected predictions** (the runner updates the prediction
after reflection, and `metrics.accuracy` counts the final prediction).

### Verification

| Paper Config | File Config | Paper Field% | Stored `metrics.accuracy` | Match |
|---|---|---|---|---|
| Base (no reflection) | base | 88.5 | 88.5% | YES |
| Self-reflection | self_refl | 88.6 | 88.6% | YES |
| Error-triggered | error_refl | 85.1 | 85.1% | YES |
| UQ-triggered | uq_refl | 88.4 | 88.4% | YES |
| Neighbourhood-triggered | nbr_refl | 83.2 | 83.2% | YES |
| Error + UQ | error_uq | 87.0 | 87.0% | YES |
| Error + Nbr | error_nbr | 85.7 | 85.7% | YES |
| UQ + Nbr | uq_nbr | 83.8 | 83.8% | YES |
| Induced (all signals) | induced_refl | 86.7 | 86.7% | YES |

### Batch coherence
- All 9 configs created Feb 19-20
- 5 configs (base, error_uq, error_nbr, induced_refl + some) were copied from
  the old `results/ablation/` run (Feb 19)
- 4 configs (self_refl, uq_refl, nbr_refl, uq_nbr) were run fresh (Feb 20)
- **All share the same MAP cache** — verified by comparing base predictions with
  uq_refl, self_refl, error_refl: only 2-3 of ~965 non-empty predictions differ

### Important note on recomputation
If you recompute accuracy by reading `predictions[field].initial_ranking[0].id`
and comparing to GT, you get DIFFERENT numbers for reflected configs (e.g.,
error_refl recomputes to 88.6% instead of stored 85.1%). This is because
`initial_ranking` stores the pre-reflection ranking, while `metrics.accuracy`
uses the post-reflection prediction. The paper correctly uses the post-reflection
number.

### UQ gate firing
- The paper claims "UQ gate fires on 1.0% of fields (15/1,429)"
- The uq_refl file shows **20 reflected fields** (20/1419 = 1.4%)
- The discrepancy is because the file uses 1419 fields (not 1429) and the
  count is 20 (not 15). The paper number is approximate.

---

## Table 8: OMOP Ablation (CMS, MIMIC, Synthea)

### Source files
Log files at `results/reflection_ablation_20260228_130716/OMOP_*.log`

### Numbers extracted from logs

| Config | CMS (145 fields) | MIMIC (119 fields) | Synthea (101 fields) |
|---|---|---|---|
| base | 111 (76.6%) | 71 (59.7%) | 101 (100.0%) |
| self_reflect | 114 (78.6%) | 70 (58.8%) | 91 (90.1%) |
| error_reflect | 109 (75.2%) | 82 (68.9%) | 92 (91.1%) |
| neighborhood_reflect | 115 (79.3%) | 71 (59.7%) | 98 (97.0%) |
| uq_sampling_reflect | 111 (76.6%) | 71 (59.7%) | 101 (100.0%) |

### Notes
- Only 5 configs were run (not the full 9-config factorial)
- All run Feb 28 13:28–15:11 in a single batch
- These used the older `ue_reflect.py` code, not the current `map_reflect_runner.py`
- A second OMOP ablation was run Mar 28-29 with 9 configs using the new runner,
  but all showed 0.0% accuracy due to a `predict_fields` evaluation bug

---

## Tables 3-4: Main Results

### Source file inventory

There are THREE generations of result files:

**Generation 1: `ue_reflect_*` files (Jan 29-31)**
- Naming: `ue_reflect_{Dataset}_{model}.json` (e.g., `ue_reflect_NL2GQL_llama.json`)
- Location: `map_reflect/results/` (both local and remote)
- Code: older `ue_reflect.py` pipeline
- Coverage: ITOps (20-289 samples), ITOpsGraph (370-521), NL2GQL, ToolBench,
  OMOP×3 — all with Granite, Llama, Mistral

**Generation 2: `ue_reflect_*` files with full model IDs (Feb 12-18)**
- Naming: `ue_reflect_{dataset}_{full_model_id}.json`
- Location: `map_reflect/results/`
- Code: updated `ue_reflect.py` with new model naming
- Coverage: ITOpsGraph (Llama Feb 12, Granite Feb 18), NL2GQL, OMOP (Llama Feb 13)

**Generation 3: `mapreflect_*` files (Feb 28 – Mar 25)**
- Naming: `mapreflect_{dataset}_{full_model_id}.json`
- Location: `map_reflect/results/`
- Code: current `map_reflect_runner.py` with batch inference
- Coverage: All datasets, all models

### Which files produced which paper numbers?

| Dataset | Model | Paper Type% | Paper Field% | Likely Source | Evidence |
|---|---|---|---|---|---|
| ITOpsGraph | Llama | 100.0 | 80.0 | **UNKNOWN** | Gen3 gives 90.8/83.6, Gen2 gives 97.7/89.9, Gen1 gives 96.5/61.7 — none match |
| ITOpsGraph | Granite | 73.3 | 66.6 | **Gen1 ~match** | Gen1 ITOps_granite: type=100, field=65.6 (field close) |
| ITOpsGraph | Mistral | 96.7 | 87.7 | **Gen3 ~match** | Gen3: type=94.0, field=85.4 (close but not exact) |
| NL2GQL | Llama | 100.0 | 90.8 | **UNKNOWN** | Gen1: 0/59.5, Gen2: 0/77.0, Gen3: 0/0 — none match |
| NL2GQL | Granite | 57.9 | 91.3 | **UNKNOWN** | Gen1: 0/0 — no match |
| OMOP-CMS | Llama | 100.0 | 76.6 | **Gen3** | Gen3 Feb 28: type=100, field=76.6 — exact match |
| OMOP-MIMIC | Llama | 100.0 | 59.7 | **Gen3** | Gen3 Feb 28: type=100, field=59.7 — exact match |
| OMOP-MIMIC | Mistral | 52.2 | 25.0 | **Gen1** | Gen1: type=52.2, field=25.2 — close match |
| ToolBench | Mistral | 98.6 | 98.4 | **Gen1 ~match** | Gen1: type=98.6/field=96.9 (type matches, field close) |
| Spider | Llama | 95.0 | 79.2 | **UNKNOWN** | Gen3: type=0 (broken), field=75.2 — doesn't match |

### Key observations

1. **OMOP Llama numbers verified:** CMS and MIMIC Llama come from Gen3 `mapreflect_*`
   files dated Feb 28. These are the same files used for the OMOP ablation.

2. **Many paper numbers cannot be traced to any existing file.** The ITOpsGraph
   Llama numbers (100.0% type, 80.0% field), NL2GQL numbers (90.8% field), and
   Spider numbers (95.0% type) don't match any generation. These may come from:
   - A different evaluation script that computes F1 differently
   - An interactive notebook or manual computation
   - Result files that have been overwritten or lost

3. **The paper mixes Gen1 and Gen3 results.** OMOP-MIMIC Mistral (25.0%) matches
   Gen1 (Jan 30), while OMOP-CMS Llama (76.6%) matches Gen3 (Feb 28). These used
   different code versions.

---

## WatsonX Non-Determinism

### Measurement
Compared cached base (from simba_uq/itopsgraph_base, Mar 30) with uncached base
(from simba_uq/itopsgraph_uncached_base, Mar 31):

| Metric | Value |
|---|---|
| Cached base accuracy | 86.8% (1240/1429) |
| Uncached base accuracy | 81.9% (1171/1429) |
| Predictions that changed | 584/1429 (40.9%) |
| Accuracy delta | 4.9pp |

### Impact on paper claims
- **Table 7 ablation: NOT affected.** All configs share the same MAP cache, so
  the relative ordering is trustworthy even if absolute values shift.
- **Tables 3-4 main results: AFFECTED.** Any cross-run comparison (e.g., Map-Reflect
  vs MatchMaker) could include up to 5pp of API variance. However, most winning
  margins in the paper are >10pp, so the directional claims are likely robust.
- **SIMBA UQ experiment: Uncached base = 81.9% vs SIMBA = 81.0%.** The -0.9pp
  is within noise. SIMBA is neutral on accuracy.

---

## Recommendations

### For the ablation (Tables 7-8)
**No action needed.** The numbers are verified, internally consistent, and come
from a single cached batch. The relative ordering is trustworthy.

### For main results (Tables 3-4)
**Action needed.** Many numbers cannot be traced to source files. Options:
1. **Re-run all main results** with the current `map_reflect_runner.py` in a single
   batch per dataset to get consistent numbers. This is the cleanest approach.
2. **Document the mixed provenance** and note that absolute values may vary ±5pp
   across runs but directional claims (MR > MatchMaker > GNN) are robust.

### For UQ claims
- The paper says "15 fields" but the file shows 20. Update to 20.
- The paper says "1,429 fields" but the ablation uses 1,419. Clarify which
  dataset version is used where (1429 = itopsgraph_dataset.jsonl with 521 samples,
  1419 = an earlier version or different sample count).
