# Map-Reflect: Missing Experiments
# Priority order: HIGH = needed for submission, MEDIUM = strengthens claims, LOW = future work
# Last updated: 2026-03-28

## STATUS SUMMARY
# EXP-1: DONE — paper updated, MatchMaker excluded from retrieval table
# EXP-2: PARTIALLY DONE — MatchMaker on 246/786 subset, documented with dagger
# EXP-3: DATA EXISTS — entropy values in results, need extraction + plot script
# EXP-4: PARTIALLY DONE — OMOP ablation dir has UQ results, need other 8 configs
# EXP-5: CODE EXISTS — LLMatch at ./llmatch/LLMatch/, not yet run
# EXP-6: CHARACTERIZED — MatchMaker fails on agent init with expanded catalog, documented in Limitations
# EXP-7: SKIP — future work
# EXP-8: SKIP — future work
#
# PAPER STATUS (after Claude Desktop cleanup):
# - tau_H = 0.25 filled in with adaptive formula
# - Mistral corrected to "Mistral Medium 25.05"
# - ITOpsGraph* in dataset table (786 pairs, 3.9 avg fields)
# - Spider GNN: type 39.6%, field 77.6% — in both tables
# - ITOpsGraph* tables: MR (57.0/44.6/56.6), MM† (29.4/33.2/28.6), GNN (20.9)
# - All [VALUE] and [AUTHORS] placeholders resolved
# - FIX comments added for traceability

## HIGH — Fixes incoherence in current paper

### EXP-1: Remove MatchMaker from retrieval table (already done in paper)
# No experiment needed — MatchMaker has no retrieval mode and was removed from Table 4.
# Confirm: retrieval table now shows Map-Reflect only (Llama, Mistral, Granite).

### EXP-2: Run MatchMaker on full ITOpsGraph* (786 type pairs)
# Currently only 246-sample subset because MatchMaker crashes on expanded catalog.
# Goal: debug agent initialisation failure or formally declare it unable to run.
# If it cannot run: update Tables 5 and 6 to show MatchMaker as N/A (not partial subset).
# If it can run: populate full MatchMaker rows in Tables 5 and 6.

## HIGH — Directly addresses reviewer's central UQ objection

### EXP-3: UQ entropy distribution plot (bimodality evidence)
# Run Map-Reflect MAP phase on ITOpsGraph validation split (20% held-out).
# For each source field, collect Shannon entropy H over top-n candidate confidences.
# Plot histogram of H values. Expected: bimodal — spike near 0 (confident) and
# spike near log(n) (uncertain). This justifies why tau_H threshold is robust.
# Output: one figure (chart_entropy_distribution.png) for inclusion in §4.3.
# Also report: fraction of fields above tau_H=0.25 (the gate firing rate).

## MEDIUM — Strengthens ablation completeness

### EXP-4: Full signal ablation on OMOP (all 9 configurations)
# Table 8 (tab:ablation_omop) currently shows only base vs UQ-triggered on OMOP.
# Run all 9 configurations from Table 7 (tab:ablation_components) on OMOP-CMS and
# OMOP-MIMIC (skip Synthea, only 12 samples).
# Configurations: Base, UQ-triggered, Self-reflection, UQ+Nbhd, Neighbourhood-only,
#   Error+UQ, Error+Nbhd, Induced (all), Error-only.
# Metric: field recall (one-prediction-per-field variant, same as ITOpsGraph ablation).
# Output: new rows in tab:ablation_omop or a separate table.

## MEDIUM — Resolves LLMatch comparison gap

### EXP-5: LLMatch on ITOpsGraph and NL2GQL under matched backbone
# Currently LLMatch appears only in SchemaNet (Table 10) using GPT-4o-mini.
# We compare with LLMatch on SchemaNet but cannot claim many-to-many superiority
# without running it on ITOpsGraph (the many-to-many benchmark).
# Run LLMatch with Llama-3.3-70B backbone on:
#   - ITOpsGraph (521 samples, oracle mode)
#   - NL2GQL (437 samples, oracle mode)
# Report type-level accuracy and field F1 alongside Map-Reflect.
# If LLMatch code does not support Llama backend, report GPT-4o-mini numbers
# and add a footnote noting the backbone difference.

## MEDIUM — Addresses MatchMaker partial results on ITOpsGraph*

### EXP-6: Characterise MatchMaker ITOpsGraph* failure mode
# MatchMaker currently runs on 246/786 samples of ITOpsGraph*.
# Determine: does it fail on specific product pairs, schema sizes, or randomly?
# Goal: either fix the agent initialisation issue so it runs on all 786 pairs,
# or produce a reproducible characterisation of which 540 pairs it cannot handle
# so the 246-sample subset can be described precisely (e.g., "pairs involving
# only observability products" or "pairs with <20 fields per type").

## LOW — Future work, not needed for submission

### EXP-7: Retrieval mode with large confusable catalog (100-200 APIs)
# Current retrieval pools: ITOpsGraph 12 products, NL2GQL 54 schemas.
# Construct a hard-distractor evaluation: add semantically similar decoy schemas
# to the catalog and measure API accuracy degradation.
# Relevant to reviewer Q7 on retrieval scalability.

### EXP-8: Set-level many-to-many precision/recall per field
# For fields in ITOpsGraph with multiple SME-validated targets (e.g., Alert->Incident
# AND Alert->Event), compute set-level P/R: what fraction of valid targets does
# Map-Reflect recover, and what fraction of its predictions are valid?
# Requires: export from wiki of all valid target sets per source field.
# Relevant to reviewer Q8.
