# Retrieval Mode Experiments

## Overview

In **retrieval mode**, the target schema is NOT specified in the utterance. The method
must discover the correct target from a catalogue of candidate schemas using semantic
search. This contrasts with **oracle mode** where both source and target are named.

## Utterance Format

| Dataset | Oracle utterance | Retrieval utterance |
|---------|-----------------|-------------------|
| ITOpsGraph | "Map cloudability StringFilterInput to turbonomic FilterInput" | "find types that can be mapped to cloudability StringFilterInput" |
| NL2GQL | "map concert data to stadium data" | "find tables that can be joined with concert" |
| ToolBench | "Map Foreca Weather Nowcast to Cloud Cast Get humidty" | "Find APIs that can be mapped with Foreca Weather Nowcast" |
| OMOP-CMS | "Map OMOP PERSON to CMS BENEFICIARYSUMMARY" | "find OMOP tables that map to omop person" |
| OMOP-MIMIC | "Map OMOP PERSON to MIMIC PATIENTS" | "find OMOP tables that map to omop person" |
| OMOP-Synthea | "Map OMOP PERSON to SYNTHEA PATIENTS" | "find OMOP tables that map to omop person" |
| Spider | "map cite data to publication data" | "find tables that can be joined with cite" |

## Dataset Files

Retrieval mode datasets are in `data/retrieval_mode/`:
- `itopsgraph_dataset.jsonl` (521 samples) — original dataset utterances are already retrieval-style
- `nl2gql_dataset.jsonl` (47 samples) — subset with retrieval utterances
- `toolbench_eval_dataset.jsonl` (3190 samples) — full ToolBench with retrieval utterances
- `omop_cms_dataset.jsonl` (38 samples) — original utterances are retrieval-style
- `omop_mimic_dataset.jsonl` (46 samples) — original utterances are retrieval-style
- `omop_synthea_dataset.jsonl` (12 samples) — original utterances are retrieval-style
- `spider_dataset.jsonl` (745 samples) — created 2026-03-30 from oracle dataset

## Methods Compared

### Map-Reflect (retrieval)
Uses hybrid search (ChromaDB sentence-transformers + BM25 + Reciprocal Rank Fusion)
to discover target schemas from the property lookup catalogue. Triggered automatically
when utterance matches retrieval patterns ("find types that can be mapped to...").

Run via: `--data-dir data/retrieval_mode`

### AnyTool (retrieval)
Uses 3-tier agent hierarchy (meta-agent -> category-agents -> tool-agents) to explore
the hierarchy and discover target APIs. Source product excluded from search.

Run via: `--retrieval` flag in `anytool_faithful.py`

## AnyTool Retrieval Results (Llama-3.3-70B) — Completed 2026-03-30

### Type-Level F1 (%)

| Dataset | Precision | Recall | F1 |
|---------|-----------|--------|-----|
| ITOpsGraph | 56.9 | 37.0 | 44.9 |
| NL2GQL | 19.8 | 14.0 | 16.4 |
| ToolBench | 45.0 | 45.0 | 45.0 |
| OMOP-CMS | 93.9 | 81.6 | 87.3 |
| OMOP-MIMIC | 50.0 | 37.0 | 42.5 |
| OMOP-Synthea | 100.0 | 100.0 | 100.0 |
| Spider | 56.7 | 26.0 | 35.7 |

### Field-Level F1 (%)

| Dataset | Precision | Recall | F1 |
|---------|-----------|--------|-----|
| ITOpsGraph | 71.2 | 47.7 | 57.1 |
| NL2GQL | 19.8 | 14.0 | 16.4 |
| ToolBench | 32.8 | 32.8 | 32.8 |
| OMOP-CMS | 52.8 | 38.6 | 44.6 |
| OMOP-MIMIC | 32.5 | 22.7 | 26.7 |
| OMOP-Synthea | 60.0 | 50.5 | 54.8 |
| Spider | 57.1 | 25.5 | 35.2 |

## Map-Reflect Retrieval Results (Llama-3.3-70B) — Completed 2026-03-30

### Type-Level F1 (%)

| Dataset | F1 |
|---------|-----|
| ITOpsGraph | 89.8 |
| NL2GQL | 91.4 |
| ToolBench | 98.1 |
| OMOP-CMS | 92.1 |
| OMOP-MIMIC | 58.7 |
| OMOP-Synthea | 100.0 |
| Spider | 80.3 |

### Field-Level F1 (%)

| Dataset | F1 |
|---------|-----|
| ITOpsGraph | 87.6 |
| NL2GQL | 91.4 |
| ToolBench | 97.6 |
| OMOP-CMS | 75.9 |
| OMOP-MIMIC | 46.2 |
| OMOP-Synthea | 53.5 |
| Spider | 77.3 |

## Side-by-Side Comparison (Retrieval Mode, Llama-3.3-70B)

### Type-Level F1 (%)

| Dataset | AnyTool | Map-Reflect |
|---------|---------|-------------|
| ITOpsGraph | 44.9 | **89.8** |
| NL2GQL | 16.4 | **91.4** |
| ToolBench | 45.0 | **98.1** |
| OMOP-CMS | 87.3 | **92.1** |
| OMOP-MIMIC | 42.5 | **58.7** |
| OMOP-Synthea | **100.0** | **100.0** |
| Spider | 35.7 | **80.3** |

### Field-Level F1 (%)

| Dataset | AnyTool | Map-Reflect |
|---------|---------|-------------|
| ITOpsGraph | 57.1 | **87.6** |
| NL2GQL | 16.4 | **91.4** |
| ToolBench | 32.8 | **97.6** |
| OMOP-CMS | 44.6 | **75.9** |
| OMOP-MIMIC | 26.7 | **46.2** |
| OMOP-Synthea | **54.8** | 53.5 |
| Spider | 35.2 | **77.3** |

## Notes

- Map-Reflect dominates in retrieval mode across all datasets, winning 6/7 on type F1
  and 6/7 on field F1. The only near-tie is OMOP-Synthea field F1 (AnyTool 54.8% vs MR 53.5%).
- AnyTool retrieval on ITOpsGraph field F1 (57.1%) is higher than its oracle (48.4%),
  possibly because the meta-agent explores more diverse targets without the oracle hint
  constraining the search.
- NL2GQL and Spider degrade significantly for AnyTool in retrieval mode — discovering
  the right target from a large pool without hints is challenging for agent-based methods.
  Map-Reflect's hybrid search (ChromaDB + BM25 + RRF) handles this much better.
- OMOP retrieval works well for both methods because there are only 2 categories
  (OMOP + source schema), making target discovery straightforward.
