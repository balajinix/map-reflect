# Mistral Experiment Tracker

Model: `mistralai/mistral-medium-2505`

## Priority 1 — Core Paper Tables (Tables 3-4)

| # | Method | Dataset | Samples | Status | Notes |
|---|--------|---------|---------|--------|-------|
| 1 | MatchMaker | OMOP-CMS | 38 | pending | |
| 2 | MatchMaker | OMOP-MIMIC | 46 | pending | |
| 3 | MatchMaker | OMOP-Synthea | 12 | pending | |
| 4 | MatchMaker | ITOpsGraph | 521 | pending | |
| 5 | MatchMaker | ToolBench | 3190 | pending | |
| 6 | AnyTool | ITOpsGraph | 521 | pending | |

## Priority 2 — SchemaNet Tables (Tables 5-6)

| # | Method | Dataset | Samples | Status | Notes |
|---|--------|---------|---------|--------|-------|
| 7 | Map-Reflect | SchemaNet (7 pairs) | ~408 | pending | |
| 8 | AnyTool | SchemaNet (7 pairs) | ~408 | pending | |
| 9 | MatchMaker | SchemaNet (7 pairs) | ~408 | pending | |

## Already Done (Mistral results exist)

| Method | Dataset | Result File |
|--------|---------|-------------|
| Map-Reflect | ITOpsGraph | ITOpsGraph_mistralai_mistral-medium-2505.json |
| Map-Reflect | NL2GQL | NL2GQL_mistral.json |
| Map-Reflect | ToolBench | ue_reflect_ToolBench_mistral.json |
| Map-Reflect | OMOP-CMS | OMOP_CMS_mistral.json |
| Map-Reflect | OMOP-MIMIC | OMOP_MIMIC_mistral.json |
| Map-Reflect | OMOP-Synthea | OMOP_Synthea_mistral.json |
| AnyTool | NL2GQL | NL2GQL_mistral.json |
| AnyTool | OMOP-CMS | OMOP_CMS_mistral.json |
| AnyTool | OMOP-MIMIC | OMOP_MIMIC_mistral.json |
| AnyTool | OMOP-Synthea | OMOP_Synthea_mistral.json |
| MatchMaker | NL2GQL | NL2GQL_mistral.json |
