# OMOP Signal Ablation Results

## Date: 2026-03-28
## Model: Llama-3.3-70B (watsonx)
## Datasets: OMOP-CMS (38 samples, 145 fields), OMOP-MIMIC (46 samples, 119 fields)

---

## Results

### OMOP-CMS (38 samples, 145 fields)

| Config | Fields Correct | Total | Accuracy |
|--------|---------------|-------|----------|
| Base (no reflection) | 24 | 145 | 16.6% |
| UQ-triggered | 24 | 145 | 16.6% |
| Self-reflection | 103 | 145 | 71.0% |
| Error-triggered | 103 | 145 | 71.0% |
| Neighbourhood | 102 | 145 | 70.3% |
| Error + UQ | 102 | 145 | 70.3% |
| Error + Neighbourhood | 103 | 145 | 71.0% |
| UQ + Neighbourhood | 103 | 145 | 71.0% |
| Induced (all three) | 103 | 145 | 71.0% |

### OMOP-MIMIC (46 samples, 119 fields)

| Config | Fields Correct | Total | Accuracy |
|--------|---------------|-------|----------|
| Base (no reflection) | 28 | 119 | 23.5% |
| UQ-triggered | 28 | 119 | 23.5% |
| Self-reflection | 51 | 118 | 43.2% |
| Error-triggered | 53 | 118 | 44.9% |
| Neighbourhood | 51 | 118 | 43.2% |
| Error + UQ | 51 | 118 | 43.2% |
| Error + Neighbourhood | 51 | 118 | 43.2% |
| UQ + Neighbourhood | 51 | 118 | 43.2% |
| Induced (all three) | 53 | 118 | 44.9% |

---

## Key Observations

### 1. Reflection helps dramatically on OMOP (opposite of ITOpsGraph)

On ITOpsGraph, any reflection signal **hurts** accuracy (base 88.0% → induced 73.3%).
On OMOP, reflection **massively improves** accuracy:
- OMOP-CMS: 16.6% → 71.0% (+54.4 pp)
- OMOP-MIMIC: 23.5% → 44.9% (+21.4 pp)

### 2. UQ-triggered reflection does NOT fire on OMOP

UQ-triggered reflection produces identical results to the base (no reflection) on both datasets:
- OMOP-CMS: 16.6% = 16.6%
- OMOP-MIMIC: 23.5% = 23.5%

This means the UQ gate never fires — the model's heuristic uncertainty scores are below τ_H = 0.25 even though the predictions are wrong. The model is **confidently wrong** on OMOP.

### 3. All reflection signals are equally effective on OMOP

On OMOP-CMS, self-reflection, error-triggered, and induced all produce ~71.0%.
On OMOP-MIMIC, error-triggered is slightly better (44.9%) than self/neighbourhood (43.2%).
There is no signal that hurts — all reflection helps.

### 4. Why the difference from ITOpsGraph?

**ITOpsGraph**: 12 products with domain-specific terminology. The base model is already quite confident and mostly correct (88%). Reflection fires on already-correct predictions and over-corrects them.

**OMOP**: Single target schema (OMOP CDM) with healthcare-specific column names (person_id, provider_npi, visit_occurrence_id). The base model maps poorly because:
- Healthcare terminology is specialized
- Many source fields map to the same OMOP column (many-to-one is legitimate)
- The model needs the reflection pass to reconsider its initial (wrong) greedy mapping

### 5. Implications for the paper

The current paper claims "UQ-triggered reflection is the only safe default." This is true for ITOpsGraph but **not for OMOP** — on OMOP, UQ-triggered reflection is safe (doesn't hurt) but also doesn't help (never fires). The paper already notes this:

> "Table 8 shows that UQ benefit is highest where the model has moderate initial confidence (MIMIC), and lowest where it is uniformly confident (Synthea)."

But the full 9-config ablation reveals that **unconditional reflection is actually the best strategy on OMOP** — a finding that contrasts with ITOpsGraph. This suggests the optimal reflection strategy is dataset-dependent:
- **ITOpsGraph** (confident model, mostly correct): UQ-only → avoid over-correction
- **OMOP** (uncertain model, mostly wrong): full reflection → improve from low base

### 6. The UQ calibration problem

The UQ gate doesn't fire on OMOP because the heuristic uncertainty (score spread among candidates) is low — the model assigns high confidence to its (wrong) top candidate. This is a **calibration** issue:
- The model is confidently wrong on healthcare terminology
- The entropy-based UQ metric doesn't detect this because the confidence scores are concentrated, not dispersed
- A sampling-based UQ method (generating multiple predictions and measuring consistency) would likely detect this better

This supports the paper's note that "the UQ entropy trigger uses prompt-elicited confidence scores rather than the sampling-based aggregation in SIMBA."

---

## Comparison with ITOpsGraph Ablation (Table 7 in paper)

| Config | ITOpsGraph | OMOP-CMS | OMOP-MIMIC |
|--------|-----------|----------|------------|
| Base | 88.0% | 16.6% | 23.5% |
| UQ-triggered | **88.0%** | 16.6% | 23.5% |
| Self-reflection | 82.6% | **71.0%** | 43.2% |
| Error-triggered | 73.1% | **71.0%** | **44.9%** |
| Neighbourhood | 80.9% | 70.3% | 43.2% |
| Error + UQ | 74.5% | 70.3% | 43.2% |
| Induced (all) | 73.3% | **71.0%** | **44.9%** |

**Key takeaway**: The optimal reflection strategy depends on the base model's initial accuracy. When the base is strong (88%), reflection hurts. When the base is weak (16-23%), reflection helps massively. UQ-triggered reflection is the safe middle ground — it never hurts, but it may not fire when needed.
