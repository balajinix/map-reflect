# Evaluation Scripts for Schema Mapping

This directory contains all evaluation and accuracy calculation scripts for the schema mapping experiments.

## Overview

The evaluation framework supports multiple approaches:

| Script | Purpose | Metrics |
|--------|---------|---------|
| `calc_accuracy.py` | Primary accuracy calculator | API-level, Field-level accuracy |
| `eval_schema_mapping.py` | Result aggregator with LaTeX output | Aggregated statistics, LaTeX tables |
| `similarity_evaluator.py` | Embedding-based evaluation | Cosine similarity scores |
| `llm_evaluator.py` | LLM-as-judge (single file) | Equivalence, Joinability scores |
| `llm_eval_batch.py` | LLM-as-judge (batch) | Batch equivalence/joinability |

---

## Primary Evaluation Scripts

### `calc_accuracy.py`

**Main accuracy calculator for all methods.** Computes API-level and field-level accuracy by reading results from each method's `results/` directory.

**Metrics:**
- **API-level accuracy**: Correct when predicted mapping contains fields from expected source AND target products
- **Field-level accuracy**: Correct field mappings, calculated ONLY on samples where API selection was correct

**Usage:**
```bash
# Calculate accuracy for all methods and datasets
python eval/calc_accuracy.py

# Specific methods and datasets
python eval/calc_accuracy.py --methods map_reflect,gnn --datasets ITOps,NL2GQL

# Output formats
python eval/calc_accuracy.py --output markdown  # Default - markdown tables
python eval/calc_accuracy.py --output csv       # CSV format
python eval/calc_accuracy.py --output json      # JSON format
```

**Output:** Prints two tables:
1. API-level (Product) Accuracy - by method/model/dataset
2. Field-level Accuracy (Given Correct API) - by method/model/dataset

---

### `eval_schema_mapping.py`

**Result aggregator that generates LaTeX tables.** Collects results from method directories, recalculates accuracy, and generates publication-ready LaTeX tables.

**Usage:**
```bash
# Aggregate results for all combinations
python eval/eval_schema_mapping.py \
    --datasets ITOps NL2GQL OMOP_CMS \
    --methods SMT ReAct GNN AnyTool MapReflect \
    --models granite llama mistral-large \
    --output-dir ./results

# Called by aggregate_results.sh wrapper
./aggregate_results.sh --datasets itops,nl2gql --methods smt,react
```

**Output:**
- `results/results_tables.tex` - LaTeX tables for paper
- `results/evaluation_statistics.json` - JSON statistics
- Console output with detailed breakdown by dataset/model

---

## Alternative Evaluation Methods

### `similarity_evaluator.py`

**Embedding-based evaluation using cosine similarity.** Uses sentence-transformers to compute similarity between source and target field descriptions.

**Methodology:**
1. Encode field descriptions using `paraphrase-MiniLM-L6-v2`
2. Compute cosine similarity between embeddings
3. Classify as correct if similarity > 0.3 threshold

**Usage:**
```bash
python eval/similarity_evaluator.py \
    --mapping_file path/to/mapping.json \
    --api_lookup_file data/itops_api_lookup.json \
    --output similarity_results.json

# Or use the shell wrapper
./eval/similarity_evaluator.sh
```

**Output:** JSON with similarity scores for each field mapping

---

### `llm_evaluator.py`

**LLM-as-judge for single mapping files.** Uses WatsonX to evaluate whether field mappings are semantically equivalent or joinable.

**Metrics:**
- **equivalence_score** (0-1): Do both fields serve the same purpose?
- **joinability_score** (0-1): Can one field's output enrich the other?
- **explanation**: LLM reasoning for scores

**Usage:**
```bash
python eval/llm_evaluator.py \
    --mapping_file path/to/mapping.json \
    --api_lookup_file data/itops_api_lookup.json \
    --output llm_eval_results.json
```

**Requires:** `WATSONX_APIKEY` and `WATSONX_PROJECT_ID` in environment

---

### `llm_eval_batch.py`

**Batch LLM evaluation for entire directories.** Processes all mapping files and generates aggregate metrics.

**Usage:**
```bash
python eval/llm_eval_batch.py \
    --mapping_dir data/mappings_output/ \
    --api_lookup_file data/itops_api_lookup.json \
    --output all_results.json

# Or use the shell wrapper
./eval/llm_eval_batch.sh
```

**Output:**
- JSON with all evaluation results
- Console report with positive rates for equivalence and joinability (threshold > 0.5)

---

## Data Files

| File | Description |
|------|-------------|
| `api_lookup.json` | Enriched API metadata used for evaluations |
| `all_results.json` | Output from LLM batch evaluation |
| `similarity_eval_output.json` | Output from similarity evaluation |
| `.env` | Environment variables (WatsonX credentials) |

---

## Accuracy Calculation Methodology

### API-Level Accuracy

Measures whether the method identified the correct source and target products (schemas).

```
API Accuracy = (# samples with correct source AND target product) / (total samples)
```

A prediction is correct if the predicted mapping contains at least one field from both:
- The expected source product/schema
- The expected target product/schema

### Field-Level Accuracy

Measures field mapping correctness, **conditional on correct API selection**.

```
Field Accuracy = (# correctly mapped fields) / (total ground truth fields)
                 [only for samples where API was correct]
```

Field matching supports:
- Exact match
- Case-insensitive match
- Partial match (field name without schema prefix)

---

## Integration with Main Pipeline

The evaluation scripts are called by:

1. **`aggregate_results.sh`** - Wrapper for `eval_schema_mapping.py`
   ```bash
   ./aggregate_results.sh --datasets itops,nl2gql
   ```

2. **`run_evaluations.sh`** - References `calc_accuracy.py` for final metrics
   ```bash
   ./run_evaluations.sh --method map_reflect --dataset ITOps
   ```

---

## Method Result Directories

Evaluation scripts read from these directories:

| Method | Results Location |
|--------|------------------|
| SMT | `smt/results/` |
| ReAct | `react/results/` |
| GNN | `gnn/results/` |
| AnyTool | `anytool/results/` |
| Map-Reflect | `map_reflect/results/` |

Expected file naming: `{dataset}_{model}.jsonl` or `{dataset}_{model}.json`
