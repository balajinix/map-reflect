# SIMBA-Style Sampling UQ Experiment

## Background

The current Map-Reflect UQ trigger uses single-pass entropy over prompt-elicited
confidence scores at temperature=0. This has two problems:
1. Temperature=0 means no stochasticity — scores are deterministic
2. Scores are uncalibrated — the model reports its own confidence, which may be
   systematically overconfident on unfamiliar domains (OMOP healthcare vocabulary)

The SIMBA paper (Bhattacharjya et al.) and the Consistency Hypothesis paper
(Xiao et al.) both motivate measuring uncertainty via **generation consistency**:
run the MAP phase k times at temperature > 0, measure how often the top-1
prediction changes. High inconsistency = genuine uncertainty.

This experiment implements and evaluates that approach.

---

## What to implement

### Step 1: Sampling-based MAP phase

For each source field f, instead of one MAP call at temperature=0:
- Run the MAP phase **k=3 times** at **temperature=0.7**, seed varied (42, 43, 44)
- Each run produces a ranked candidate list with a top-1 prediction
- Record: top-1 prediction from each of the 3 runs

### Step 2: Consistency score

For each field f, compute:

```
consistency(f) = fraction of run pairs where top-1 prediction agrees
               = |{(i,j): i<j, pred_i == pred_j}| / C(k,2)
```

For k=3, C(3,2)=3, so consistency ∈ {0, 1/3, 2/3, 1}
- consistency=1.0: all 3 runs agree → low uncertainty
- consistency=0.0: all 3 runs disagree → high uncertainty

Disagreement rate = 1 - consistency. Use this as the UQ signal.

### Step 3: Gate threshold

Gate fires when **disagreement_rate > τ_D**, where τ_D is determined the same
way as τ_H in the current implementation:

```
τ_D = max(0.25, μ_D + 1.5 * σ_D)
```

where μ_D and σ_D are computed over all fields in the dataset on the held-out
validation split.

### Step 4: Reflection

When the gate fires, run the standard REFLECT phase (same as current). The only
change is what triggers it — disagreement_rate instead of entropy.

For the reflected prediction, use the **majority vote** from the 3 samples as
the initial candidate (instead of top-1 from the single-pass run), then apply
the score update equation as normal.

---

## Datasets to run

**Priority 1 — ITOpsGraph** (521 eval samples, 1,429 fields)
- This is the main benchmark. Current entropy gate fires on 1% of fields (+0.2pp).
- Expected: disagreement gate fires on more fields, potentially larger gain.
- Cost: ~3x current MAP phase inference = ~90 minutes on WatsonX

**Priority 2 — OMOP-CMS** (38 samples, 145 fields)
- Current entropy gate fires on 28% of fields but gives +0.7pp (near-zero gain).
- Hypothesis: model is confidently wrong on healthcare vocabulary. Sampling
  will reveal inconsistency that entropy cannot detect.
- Cost: tiny — 145 fields × 3 samples = 435 MAP calls total

**Priority 3 — OMOP-MIMIC** (46 samples, 119 fields)
- Same hypothesis as CMS.
- Cost: negligible

---

## Evaluation

For each dataset, report a comparison table:

| Method | Gate fires (%) | Field recall (%) | Delta vs base |
|--------|---------------|-----------------|---------------|
| Base (no reflection) | 0% | X | — |
| Entropy gate (current) | Y% | X+a | +a pp |
| Disagreement gate (SIMBA) | Z% | X+b | +b pp |

The key number is whether b > a on OMOP-CMS/MIMIC, and whether b ≥ a on ITOpsGraph.

Also report:
- Correlation between disagreement_rate and MAP accuracy (analogous to the
  quartile calibration we have for entropy: Q1-Q4 accuracy vs disagreement bucket)
- How many fields have disagreement_rate=0 (all 3 samples agree) — this is the
  "confident" bucket

---

## What success looks like

**Strong result** (best case):
- OMOP-CMS: disagreement gate fires on ~30-40% of fields AND gives +3-5pp
  (showing the model IS inconsistent on healthcare vocabulary when sampled)
- ITOpsGraph: disagreement gate gives ≥ +0.2pp (at least as good as entropy)

**Acceptable result**:
- OMOP-CMS: disagreement gate fires and gives ≥ +1pp improvement over base
  (better than entropy's +0.7pp)
- ITOpsGraph: disagreement gate gives ≥ 0pp (doesn't hurt)

**Null result**:
- Both gates behave identically — the model is equally consistent/inconsistent
  as it is confident/unconfident. This would mean the model is well-calibrated
  and prompt-elicited entropy IS measuring real uncertainty, not just overconfidence.
  This is also a publishable finding.

---

## Implementation notes

- Use the existing MAP phase code, just add a loop over k seeds at temperature=0.7
- The 3 MAP calls per field are independent — can be parallelised
- Store intermediate results so you can re-run the gate threshold analysis
  without re-running inference
- The REFLECT phase code is unchanged — only the trigger changes

---

## Output needed for paper update

1. For each dataset: base / entropy-gate / disagreement-gate field recall table
2. Disagreement rate distribution plot (analogous to chart_entropy_distribution.png)
3. Calibration: accuracy by disagreement bucket (Q1-Q4, analogous to existing
   quartile analysis)
4. Gate firing rates: what % of fields have disagreement_rate > τ_D vs > τ_H

If SIMBA-style UQ shows improvement over entropy-based UQ on OMOP, the paper
will update §3.2 to present disagreement-rate as the primary UQ method, with
entropy as the cheap single-pass approximation.
