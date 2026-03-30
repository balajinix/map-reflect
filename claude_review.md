# Map-Reflect: Reviewer Observations

Status legend: `[ ]` open · `[~]` in progress · `[x]` resolved

---

## HIGH severity

### OBS-1: Central claim is near-null
**Issue:** UQ improvement is +0.2pp on ITOpsGraph, +1.7pp on OMOP-MIMIC, 0pp on NL2GQL, −7.9pp on OMOP-Synthea. The paper frames "not degrading accuracy" as a positive result, but this is indistinguishable from noise on most datasets. The abstract says "outperforms baselines" without qualification — which is false on Spider field F1 (MR 81.9% vs MatchMaker 85.8%), NL2GQL field F1 (MR 90.8% vs MatchMaker Granite 91.3%), and SchemaNet where AnyTool ties or beats Map-Reflect.
**Location:** Abstract, §4.1, §4.3 ablation framing
**Resolution needed:** Either find datasets or configurations where UQ yields meaningful gains, or reframe the central claim honestly as "a safe default that never hurts" rather than a performance improvement. Fix the abstract.
- [ ] Fix abstract benchmark count (says five, paper has seven)
- [ ] Fix abstract "outperforms baselines" — qualify or remove
- [ ] Reframe UQ claim in §1 and conclusion

---

### OBS-2: ITOpsGraph* ground truth circularity
**Issue:** Map-Reflect (three LLM backbones = 3 out of 8 votes) is one of the eight systems used to construct the ITOpsGraph* ground truth via ensemble agreement. Retention threshold is "predicted by at least two method/model combinations" — 2/8 is a low bar, and Map-Reflect has 3/8 votes by default. Map-Reflect is then evaluated on this benchmark and achieves 100% type accuracy. The benchmark may be structurally biased toward Map-Reflect's mapping style.
**Location:** §4.4 ITOpsGraph* benchmark construction paragraph
**Resolution needed:** Report per-pair agreement statistics (how many of 786 pairs were agreed by all 8 vs. exactly 2). Ideally show Map-Reflect performance on the subset agreed by non-Map-Reflect methods only.
- [ ] Add agreement statistics to ITOpsGraph* construction paragraph
- [ ] Discuss circularity risk explicitly (or run held-out construction evaluation)

---

### OBS-3: Confidence scores are not probabilities; entropy is uncalibrated
**Issue:** Confidence scores are prompt-elicited (LLM asked to score 0–1), then normalised to sum to 1, then Shannon entropy is computed. Normalising uncalibrated outputs does not produce probabilities. SIMBA uses pairwise generation similarity across independent samples — this paper does neither. The entropy distribution showing 98% of fields below τ_H is consistent with both "model is calibrated" and "model systematically overconfident." Cannot distinguish without calibration analysis.
**Location:** §3.2 UQ trigger, §4.3 hyperparameter sensitivity, Limitations
**Resolution needed:** Either (a) run a lightweight calibration check — compare entropy-ranked buckets against actual accuracy to verify high-entropy fields are indeed more often wrong, or (b) strengthen the disclaimer in Limitations that the SIMBA grounding is theoretical motivation only.
- [ ] Check: do high-entropy fields have lower MAP accuracy than low-entropy fields? (this is the key empirical test)
- [ ] Add calibration discussion to §3.2 or Limitations accordingly

---

### OBS-4: OMOP full ablation missing; signal heatmap uses estimated values
**Issue:** Table 8 shows only base vs. UQ on OMOP (3 rows). The full 9-configuration ablation was run on ITOpsGraph and NL2GQL but not OMOP-CMS/MIMIC. The key question — does error-triggered reflection hurt OMOP as badly as ITOpsGraph? — is unanswered. Worse, Figure 9 (signal heatmap) caption says "OMOP values estimated from ITOpsGraph deltas" — a figure with estimated rather than measured values in a data-focused venue is not acceptable.
**Location:** §4.3 Table 8, Figure 9 caption
**Resolution needed:** Run full 9-configuration ablation on OMOP-CMS and OMOP-MIMIC (EXP-4 is in progress on remote). Update Table 8 to match Table 7 structure. Remove "estimated" from Figure 9 caption or remove OMOP from the heatmap until real numbers are available.
- [ ] Awaiting EXP-4 results from remote
- [ ] Update Table 8 with full 9-config OMOP results
- [ ] Fix Figure 9 caption — remove "estimated" claim

---

## MEDIUM severity

### OBS-5: Spider GNN has in-domain training; comparison is unfair
**Issue:** The Spider GNN is trained on Spider FK graph with `maps_to` edges. Map-Reflect is zero-shot on Spider. Comparing a method with domain-specific structural supervision to a zero-shot LLM method and noting the GNN's competitive field F1 (77.6%) is not an apples-to-apples comparison. The observation is interesting but the framing implies it reflects on Map-Reflect's weakness rather than the GNN's privilege.
**Location:** §4.1 Spider discussion paragraph, §4 GNN baseline description
**Resolution needed:** The GNN baseline description already notes this. Make the framing in the Spider discussion paragraph explicit: "the GNN benefits from FK-derived training signal unavailable to zero-shot methods; this is an upper bound on structural supervision, not a fair comparison."
- [ ] Revise Spider discussion paragraph framing

---

### OBS-6: Signal 2 described in detail then recommended not to use
**Issue:** Section 3.2 gives Signal 2 (Error Analysis) detailed technical treatment — six error patterns with specific penalty values, schema deployment, query execution. The ablation shows error-triggered reflection is the worst signal (−14.9pp). The recommended configuration excludes it entirely. This creates a mismatch: substantial paper real estate is spent on something the paper itself says should not be used.
**Location:** §3.2 Signal 2, §4.3 ablation discussion
**Resolution needed:** Either (a) cut Signal 2 description to 2–3 sentences explaining what it is and why it was explored, moving detail to an appendix, or (b) show a scenario (e.g., cold-start with no validation split) where error signals help and UQ is unavailable. Option (a) is faster.
- [ ] Decide: cut Signal 2 description or find redeeming scenario
- [ ] If cut: condense to ~3 sentences

---

### OBS-7: Pipeline description disconnected from evaluation protocol
**Issue:** §3 describes a system that processes "user utterances" through a context extractor that "splits $u$ into source and destination phrases." The evaluation uses structured oracle utterances like "map turbonomic Action.actionID to concert PackageRecommendation.recommendation\_id." The context extractor stage is never validated and may be irrelevant to all reported results. The "five stages" framing suggests a more general NL system than what is actually evaluated.
**Location:** §3 five-stages pipeline paragraph, §4.2 oracle mode description
**Resolution needed:** Clarify in §3 that in oracle mode the context extractor is bypassed (source/target are already identified in the utterance) and that the five-stage description applies to the full retrieval-mode pipeline. State this explicitly.
- [ ] Add one sentence to §3 pipeline paragraph clarifying oracle vs retrieval execution paths

---

### OBS-8: No inter-annotator agreement for wiki-based ground truth
**Issue:** ITOpsGraph ground truth is auto-populated from Map-Reflect output, then SME-validated. No inter-annotator agreement (IAA) coefficient is reported. The Many-to-many boundary disagreements failure mode (§4.5) describes situations where SME reviewers disagree, but gives no sense of how frequent this is or what Kappa/F1 agreement looks like. Without IAA the benchmark's reliability is unknown — and the risk of Map-Reflect-style bias in the ground truth parallels OBS-2.
**Location:** §3.5 Wiki Architecture / Annotation Workflow, §4.5 Error Analysis
**Resolution needed:** Report IAA on a sample (e.g., 50–100 type pairs reviewed independently by two SMEs). Cohen's Kappa or pairwise F1 would suffice. If this cannot be done before submission, add an explicit note on the limitation.
- [ ] Compute IAA on ITOpsGraph sample (or add explicit limitation note)

---

### OBS-9: Hyperparameter choices unjustified
Three values are stated without justification or sensitivity analysis:
- **k=5** hops for neighbourhood traversal: chosen "empirically" but no sensitivity shown. Given that neighbourhood signal hurts accuracy, k choice may matter.
- **τ_p=0.5** pruning threshold: directly controls many-to-many coverage; never ablated.
- **Floor of 0.25** in adaptive τ_H formula: formula yields 0.175 but is clamped to 0.25. The choice of 0.25 as the floor is not explained.

**Location:** §3.3 Neighbourhood Graph Construction (k=5), §3.2 score update paragraph (τ_p), §3.2 UQ trigger (floor)
**Resolution needed:** Add 1–2 sentences justifying each. For τ_p=0.5: show that the many-to-many F1 in Table 2 is not sensitive to thresholds in [0.4, 0.6]. For floor(0.25): explain it was chosen to ensure at least a few fields are inspectable for UQ calibration purposes.
- [ ] Add justification for k=5
- [ ] Add justification for τ_p=0.5 (sensitivity ±0.1)
- [ ] Add justification for floor(0.25)

---

## LOW severity

### OBS-10: "Clustering" mentioned once, never seen again
**Issue:** §3 intro says "Map-Reflect performs domain and structural analysis to create clusters." Clustering does not appear anywhere else — not in the algorithm, not in experiments, not in ablations. This is either a vestige of an earlier system design or an undescribed component.
**Location:** §3 first paragraph
**Resolution needed:** Either remove the clustering mention, or add a sentence explaining what clustering is and where it fits in the pipeline.
- [ ] Remove or explain the clustering reference in §3

---

### OBS-11: Efficiency reported as call count, not latency
**Issue:** §4.3 Efficiency says "1–2 LLM calls per source field" vs. AnyTool's 43.3. Call count is not latency. A single Llama-3.3-70B call on a 2,000-token prompt may take longer than a GPT-4 tool call in AnyTool. For a systems venue, this needs wall-clock time or tokens per sample.
**Location:** §4.3 Efficiency paragraph
**Resolution needed:** Report median latency per sample (seconds) and/or median token count per sample for Map-Reflect vs. AnyTool under matched backbone. Even a rough estimate (e.g., "~8s per sample on A100 vs. AnyTool's ~45s") is better than call counts alone.
- [ ] Add latency/token estimate to efficiency paragraph

---

### OBS-12: Ablation scope mismatch — one-per-field vs. many-to-many
**Issue:** Ablation uses one-prediction-per-field variant (precision fixed at 86.2%) while main results use many-to-many pruning (τ_p=0.5). These are different evaluation modes. The ablation does not show whether UQ-triggered reflection improves the full many-to-many F1 in Tables 2–3. No "Map phase only (no reflection)" row appears in Tables 2–3, so the delta from reflection is invisible in the main results.
**Location:** §4.3 Ablation scope paragraph, Tables 2–3
**Resolution needed:** Add a footnote or prose sentence in §4.1 giving the Map-phase-only field F1 on ITOpsGraph (the base row from the ablation table is 88.0% recall on the one-per-field variant, but the equivalent many-to-many F1 is not stated anywhere).
- [ ] State Map-phase-only (no reflection) field F1 in oracle mode results prose for ITOpsGraph
- [ ] Note the ablation/main-results scope difference more explicitly in §4.3

---

### OBS-13: NL2GQL ablation truncated without explanation
**Issue:** NL2GQL section of Table 7 shows only 3 of 9 configurations (Base, UQ-triggered, Self-reflection). No explanation given. Reviewer will question whether the other 6 configurations were run.
**Location:** Table 7 (tab:ablation_components) NL2GQL section
**Resolution needed:** Either run the full 9 configurations on NL2GQL and add rows, or add a sentence explaining why only 3 are shown (e.g., "NL2GQL results are shown for the three configurations of interest; the remaining six configurations show the same pattern as ITOpsGraph and are omitted for space").
- [ ] Run remaining 6 NL2GQL ablation configs, or add explanation sentence

---

## Tracking

| ID | Severity | Status | Owner |
|---|---|---|---|
| OBS-1 | HIGH | [ ] | — |
| OBS-2 | HIGH | [ ] | — |
| OBS-3 | HIGH | [ ] | — |
| OBS-4 | HIGH | [~] | EXP-4 running |
| OBS-5 | MEDIUM | [ ] | — |
| OBS-6 | MEDIUM | [ ] | — |
| OBS-7 | MEDIUM | [ ] | — |
| OBS-8 | MEDIUM | [ ] | — |
| OBS-9 | MEDIUM | [ ] | — |
| OBS-10 | LOW | [ ] | — |
| OBS-11 | LOW | [ ] | — |
| OBS-12 | LOW | [ ] | — |
| OBS-13 | LOW | [ ] | — |
