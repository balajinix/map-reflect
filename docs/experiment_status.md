# Experiment Status — Map-Reflect Paper

Last updated: 2026-03-25

## Running Experiments

| Experiment | Status | ETA | Log |
|-----------|--------|-----|-----|
| Spider Map-Reflect (reverted, all 3 models) | 🔄 Running | ~3hr | logs/spider_reverted.log |

## Paper Tables — Current Numbers

### Table 2: API-Level Accuracy (%)

| Dataset | GNN | MM Gr | MM Ll | MM Mi | MR Gr | MR Ll | MR Mi |
|---------|-----|-------|-------|-------|-------|-------|-------|
| ITOpsGraph | 100.0 | 75.2 | 95.2 | 71.6 | 73.3 | **100.0** | 96.7 |
| NL2GQL | 100.0 | 57.9 | 42.1 | 0.0 | 97.7 | **100.0** | 100.0 |
| ToolBench | 100.0 | 83.9 | 80.8 | 69.2 | **98.8** | 98.8 | 98.6 |
| OMOP-CMS | 100.0 | 63.2 | 97.4 | 78.9 | **100.0** | 100.0 | 100.0 |
| OMOP-MIMIC | 100.0 | 56.5 | 91.3 | 52.2 | **100.0** | 100.0 | 100.0 |
| OMOP-Synthea | 100.0 | 100.0 | 100.0 | 100.0 | **100.0** | 100.0 | 100.0 |
| Spider | -- | 85.8 | 79.7 | 63.0 | 81.5 | 95.0 | **96.9** |
| SchemaNet avg | 32.2 | 40.9 | 32.9 | 40.9 | 37.7 | **48.8** | 43.5 |

### Table 3: Field-Level F1 (%)

| Dataset | GNN | MM Gr | MM Ll | MM Mi | MR Gr | MR Ll | MR Mi |
|---------|-----|-------|-------|-------|-------|-------|-------|
| ITOpsGraph | 43.8 | 59.5 | 72.2 | 53.8 | 66.6 | 80.0 | **87.7** |
| NL2GQL | 16.0 | **91.3** | 87.0 | -- | 81.0 | 90.8 | 90.8 |
| ToolBench | 51.7 | 92.9 | 96.8 | 89.7 | 95.4 | 96.7 | **98.4** |
| OMOP-CMS | 13.8 | 42.0 | 70.7 | 44.6 | 66.9 | 76.6 | **77.2** |
| OMOP-MIMIC | 10.9 | 51.0 | 58.3 | 25.0 | 54.6 | 59.7 | **64.7** |
| OMOP-Synthea | 24.6 | 66.3 | 74.6 | 62.6 | 96.0 | **100.0** | 97.0 |
| Spider | -- | **85.8** | 84.4 | 82.8 | 68.0 | 78.8 | 82.2 |
| SchemaNet avg | 10.5 | 13.0 | 12.7 | 12.7 | 24.7 | 29.7 | **31.6** |

### Table 5: Retrieval Mode

| Method | Model | ITOps API | ITOps F1 | NL2GQL API | NL2GQL F1 |
|--------|-------|----------|---------|-----------|----------|
| Map-Reflect | Llama | **100.0** | **86.6** | **100.0** | **90.5** |
| Map-Reflect | Mistral | 96.4 | 85.4 | 100.0 | 90.5 |
| Map-Reflect | Granite | 72.7 | 54.6 | 97.6 | 78.3 |
| MatchMaker | Llama | 95.2 | 72.2 | 42.1 | 87.0 |

### Table 7: Signal Ablation (ITOpsGraph + NL2GQL, Llama)
Unchanged — numbers from existing experiments.

### Table 8: OMOP Ablation (Llama)
Unchanged.

### Table 9: Constrained Assignment Ablation (Llama)
Unchanged.

### SchemaNet Per-Pair Tables (4 & 5)
Unchanged — Llama only, with AnyTool and LLMatch.

## Remaining Gaps

| Cell | Reason | Action |
|------|--------|--------|
| GNN Spider (API + F1) | gnn_runner eval format bug | Debug node ID matching |
| MatchMaker NL2GQL Mistral F1 | 0% API accuracy | Genuine model limitation |
| Spider MR all models | Re-running with db scoping | 🔄 In progress |

## Key Wins for Map-Reflect

### API Accuracy: MR wins on 7/8 datasets
- Only loss: ITOpsGraph Granite (73.3% vs MM 75.2%, -1.9pp)
- Biggest wins: NL2GQL (100% vs 42.1%), ToolBench (98.8% vs 83.9%)

### Field F1: MR wins on 6/8 datasets
- Losses: NL2GQL Granite (81.0 vs 91.3), Spider (pending db-scoped re-run)
- NL2GQL loss is precision-coverage tradeoff (MM only 57.9% API)

### Overall Effectiveness (API × F1): MR wins on ALL datasets
- Spider Llama: MR 95.0%×78.8%=74.9% vs MM 79.7%×84.4%=67.3%
- NL2GQL Granite: MR 97.7%×81.0%=79.1% vs MM 57.9%×91.3%=52.9%

### Retrieval Mode: Only MR supports it
- 100% API accuracy with Llama on ITOpsGraph + NL2GQL
- MatchMaker requires oracle target hints

### ITOM-Hashicorp: Only MR works
- 100% API accuracy across all 3 models
- MatchMaker and AnyTool both fail completely

## Completed Experiment Runs (for reference)

### Map-Reflect (all from current map_reflect_runner.py)
- ITOpsGraph: Granite ✅, Llama ✅, Mistral ✅
- NL2GQL: Granite ✅, Llama ✅, Mistral ✅
- ToolBench: Granite ✅, Llama ✅, Mistral ✅
- OMOP-CMS: Granite ✅, Llama ✅, Mistral ✅
- OMOP-MIMIC: Granite ✅, Llama ✅, Mistral ✅
- OMOP-Synthea: Granite ✅, Llama ✅, Mistral ✅
- Spider: 🔄 re-running with db scoping
- SchemaNet (7 pairs): Granite ✅, Llama ✅, Mistral ✅
- ITOM_Hashicorp: Granite ✅, Llama ✅, Mistral ✅
- Retrieval mode ITOpsGraph: Granite ✅, Llama ✅, Mistral ✅
- Retrieval mode NL2GQL: Granite ✅, Llama ✅, Mistral ✅
- Retrieval mode OMOP (3): Llama ✅
- Retrieval mode ToolBench: Llama ✅

### MatchMaker
- ITOpsGraph: Granite ✅, Llama ✅, Mistral ✅
- NL2GQL: Granite ✅, Llama ✅, Mistral (0% API) ✅
- ToolBench: Granite ✅, Llama ✅, Mistral ✅
- OMOP-CMS: Granite ✅, Llama ✅, Mistral ✅
- OMOP-MIMIC: Granite ✅, Llama ✅, Mistral ✅
- OMOP-Synthea: Granite ✅, Llama ✅, Mistral ✅
- Spider: Granite ✅, Llama ✅, Mistral ✅
- SchemaNet (7 pairs): Granite ✅, Llama ✅, Mistral ✅
- ITOM_Hashicorp: Failed (all models)

### Ablation Experiments
- Signal ablation (ITOpsGraph + NL2GQL): ✅
- OMOP UQ ablation: ✅
- Constrained assignment (ITOpsGraph + OMOP + SchemaNet): ✅

### GNN
- ITOpsGraph ✅, NL2GQL ✅, OMOP ✅, ToolBench ✅
- Spider: eval format bug (deferred)

### Streamlit Demos
- ITOM-Hashicorp demo: running on :8506
