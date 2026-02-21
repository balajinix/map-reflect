# Handoff Note: Map-Reflect Ablation Experiments

## Project Overview
**Repository**: `graphql-schema-mapping` (IBM GitHub)
**Purpose**: Schema mapping between GraphQL APIs using LLM-based Map-Reflect framework
**Paper**: VLDB submission in `paper/vldb.tex`

---

## Experiments Completed

### 1. Signal-Based Ablation Study
**Goal**: Test each reflection trigger mechanism independently
**Dataset**: ITOpsGraph (1419 field mappings across 200 API pairs)
**Model**: `watsonx/meta-llama/llama-3-3-70b-instruct`

| Configuration | Accuracy | Description |
|---------------|----------|-------------|
| base | 88.5% | No reflection |
| error_refl | 85.1% | Reflect on validation errors only |
| error_uq | 87.0% | Error + UQ signals (no reflection) |
| error_nbr | 85.7% | Error + neighborhood signals |
| induced_refl | 86.7% | Induce errors then reflect |
| self_refl | 86.8% | Always reflect (no signals) |
| uq_refl | 88.4% | UQ-triggered reflection |
| nbr_refl | 86.9% | Neighborhood-triggered reflection |
| uq_nbr | 88.0% | Both UQ + neighborhood triggers |

**Results location**: `results/signal_ablation/`
**Script**: `run_signal_ablation_new.sh`

### 2. UQ Threshold Ablation Study
**Goal**: Find optimal uncertainty threshold for triggering reflection
**Key insight**: Lower thresholds trigger more reflections but don't improve accuracy

| Threshold | Reflections | % Triggered | Accuracy |
|-----------|-------------|-------------|----------|
| 0.08 | 668 | 47% | 88.5% |
| 0.10 | 408 | 29% | 88.6% |
| 0.12 | 255 | 18% | 88.6% |
| **0.15** | **58** | **4%** | **88.7%** |
| 0.25 (default) | 20 | 1% | 88.4% |

**Results location**: `results/uq_threshold_ablation/`
**Script**: `run_uq_threshold_ablation.sh`

---

## Key Code Changes

### `map_reflect/map_reflect_runner.py`
Added UQ threshold configuration:
```python
# In MapReflectConfig dataclass
uq_threshold: Optional[float] = None  # Fixed threshold
uq_threshold_percentile: Optional[float] = None  # Percentile-based

# CLI arguments
--uq-threshold FLOAT      # Fixed threshold (e.g., 0.15)
--uq-threshold-percentile FLOAT  # Percentile (e.g., 75)
```

### `paper/vldb.tex`
- Updated Table 5 (`tab:ablation_components`) with signal ablation results
- Added Table 6 (`tab:uq_threshold`) with UQ threshold ablation
- Added analysis explaining counterintuitive finding: fewer targeted reflections outperform aggressive reflection

---

## File Locations

| Type | Location | Git Status |
|------|----------|------------|
| Code | `map_reflect/` | Tracked |
| Paper | `paper/vldb.tex` | Tracked |
| Experiment scripts | `run_*.sh` | Tracked |
| Results | `results/` | .gitignore (rsync to remote) |
| Data | `data/` | .gitignore (rsync to remote) |

**Remote sync target**: `ai4graphs.vpc.cloud9.ibm.com:~/graphql-schema-mapping/`

---

## Data Files

- `data/itopsgraph_api_lookup.json` - API property lookup for ITOpsGraph
- `data/itopsgraph_coverage_eval_dataset.jsonl` - Evaluation dataset
- `data/oracle_mode/` - Oracle mode experiment data
- `data/retrieval_mode/` - Retrieval mode experiment data

---

## Git State

**Latest commit**: `08e89ef` - "Add signal-based and UQ threshold ablation experiments"
**Branch**: `main` (pushed)
**Author**: Balaji Ganesan (no co-author)

---

## Potential Next Steps

1. **Run retrieval mode experiments** - Test with API retrieval instead of oracle pairs
2. **Cross-dataset validation** - Run on ToolBench dataset
3. **Model ablation** - Test with different LLMs (GPT-4, Claude)
4. **Paper writing** - Complete results section, add figures from `paper/charts/`

---

## Commands to Resume

```bash
# Check experiment results
python3 -c "import json; d=json.load(open('results/signal_ablation/uq_refl/mapreflect_itopsgraph_meta-llama_llama-3-3-70b-instruct.json')); print(f\"Accuracy: {d['metrics']['accuracy']*100:.1f}%\")"

# Run new experiment
python -m map_reflect.map_reflect_runner \
  --datasets ITOpsGraph \
  --models watsonx/meta-llama/llama-3-3-70b-instruct \
  --property-lookup-file data/itopsgraph_api_lookup.json \
  --data-dir data \
  --use-uncertainty \
  --use-uq-reflection \
  --uq-threshold 0.15

# Sync results to remote
rsync -avz --progress results/ ai4graphs.vpc.cloud9.ibm.com:~/graphql-schema-mapping/results/
```

---

## Understanding the Codebase

### Reflection operates at field-level, not API-level
- Oracle mode: API pairs are pre-determined
- Reflection helps when field mapping within an API pair is uncertain
- UQ measures confidence in field-to-field mappings

### Key finding from ablations
Aggressive reflection (low UQ threshold) doesn't help because:
1. Most mappings are already correct
2. Reflection can flip correct predictions to incorrect
3. Targeted reflection (high threshold) only touches truly uncertain cases
