# Experiments Needed for Paper Tables

**Date:** 2026-03-31
**Context:** Result provenance audit revealed mixed-version files and inconsistencies.
All experiments below should be run with the current `map_reflect_runner.py` code
to ensure consistency.

---

## Priority 1 — Consistency (must fix before submission)

### 1. Map-Reflect ITOpsGraph × 3 models (main table baseline)
- **Why:** Ablation base=88.5% (Feb 19, 1419 fields) vs main table=79.8% (Mar 18, 1429 fields).
  Different runs, different field counts, 8.7pp gap from WatsonX non-determinism.
- **Run:** MR on ITOpsGraph (521 samples, 1429 fields), Granite + Llama + Mistral, no reflection
- **Cost:** 3 × 1429 MAP calls ≈ 30 min

### 2. ITOpsGraph 9-config signal ablation (Table 7 refresh)
- **Why:** Current ablation uses 1419 fields (Feb 19-20). Main table uses 1429.
  Need same dataset version. Also need single-batch run for internal consistency.
- **Run:** 9 configs (base, self_refl, error_refl, uq_refl, nbr_refl,
  error_uq, error_nbr, uq_nbr, induced_refl) on ITOpsGraph 521 samples
- **Cost:** 9 × 1429 MAP calls + reflection calls ≈ 2-3 hours
- **Critical:** Run ALL configs using the same MAP cache so deltas are meaningful

---

## Priority 2 — Missing/broken cells

### 3. Map-Reflect NL2GQL Granite
- **Why:** ue_reflect Granite produces 0% — likely Granite model can't handle NL2GQL format
- **Run:** MR Granite on NL2GQL (437 samples) with current runner
- **Cost:** 437 MAP calls ≈ 5 min

### 4. Map-Reflect ToolBench Granite + Llama (full dataset)
- **Why:** Granite has only 5 samples (100% but meaningless). Llama has parsing error.
- **Run:** MR Granite + Llama on ToolBench (3190 samples)
- **Cost:** 2 × 3190 MAP calls ≈ 30 min

### 5. Investigate MatchMaker NL2GQL 0% and ITOpsGraph low type accuracy
- **Why:** MM produces predictions on NL2GQL (257-273 per model) but none match GT.
  MM ITOpsGraph type accuracy is 20-23% despite making 985-1182 predictions.
- **Action:** Check if this is an evaluation/format bug or genuine poor performance.
  May need to fix MM NL2GQL evaluation to handle format differences.
- **Cost:** No new runs, just evaluation code investigation

---

## Priority 3 — Old ue_reflect numbers

### 6-8. Map-Reflect OMOP Granite + Mistral (all 3 OMOP datasets)
- **Why:** Current Granite/Mistral numbers (18-50%) come from old ue_reflect (Jan 29-30)
  with adversarial reflection prompt. Current code should perform much better.
- **Run:** MR Granite + Mistral on OMOP-CMS (38), OMOP-MIMIC (46), OMOP-Synthea (12)
- **Cost:** 6 runs × ~100-150 fields ≈ 30 min total

### 9. OMOP 9-config ablation × 3 datasets (Table 8 refresh)
- **Why:** Only 5 configs exist from Feb 28. Mar 28-29 re-run shows 0% (evaluation bug).
- **Run:** 9 configs × 3 datasets (CMS, MIMIC, Synthea)
- **Cost:** 27 runs × 100-150 fields ≈ 1-2 hours

### 10. MatchMaker OMOP-MIMIC Mistral + OMOP-Synthea Mistral
- **Why:** No _full files exist for these cells
- **Run:** MM Mistral on OMOP-MIMIC and OMOP-Synthea
- **Cost:** 2 runs ≈ 10 min

---

## What does NOT need re-running

- **AnyTool (all datasets):** Generated in this session, verified, consistent
- **GNN (all datasets):** Hardcoded from GNN evaluation scripts, verified
- **Map-Reflect OMOP Llama:** mapreflect Feb 28 files verified (76.6% CMS, 59.7% MIMIC, 100% Synthea)
- **Map-Reflect Spider (all 3 models):** mapreflect Mar 25 files, consistent
- **MatchMaker Spider (all 3 models):** Mar 23 files, consistent
- **MatchMaker OMOP Granite + Llama:** Feb 27 _full files, consistent
- **MatchMaker ToolBench (all 3 models):** Feb-Mar files, verified
- **SchemaNet tables:** Separate from Tables 2-3, already verified

---

## Execution plan

### Batch 1 (Priority 1, ~3 hours):
```bash
# Run on remote: bganesa1@ai4graphs.vpc.cloud9.ibm.com
# 1. MR ITOpsGraph × 3 models (no reflection)
# 2. ITOpsGraph 9-config ablation
# Total: ~4300 MAP calls + reflections
```

### Batch 2 (Priority 2, ~1 hour):
```bash
# 3. MR NL2GQL Granite
# 4. MR ToolBench Granite + Llama
# 5. Debug MM NL2GQL and MM ITOpsGraph type acc
```

### Batch 3 (Priority 3, ~2 hours):
```bash
# 6-8. MR OMOP Granite + Mistral (6 runs)
# 9. OMOP 9-config ablation (27 runs)
# 10. MM OMOP Mistral (2 runs)
```

**Total estimated time: ~6-8 hours on WatsonX**
**Total estimated API calls: ~30,000-40,000**
