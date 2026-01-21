# GraphQL Schema Mapping

Evaluation framework for schema mapping methods, supporting the VLDB paper "Map-Reflect: GraphQL Schema Mapping using Self Reflection".

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run experiments (dry-run to preview)
./run.sh --dry-run

# Run specific method/dataset/model
./run.sh --method map_reflect --dataset ITOps --model granite

# Calculate accuracy
./run.sh --accuracy

# Full pipeline: run + aggregate + accuracy
./run.sh --all
```

## Methods

| Method | Description | Directory |
|--------|-------------|-----------|
| **Map-Reflect** | Our proposed self-reflection method | `map_reflect/` |
| **ReAct** | Agent-based approach with tool use | `react/` |
| **SMT** | Schema Matching with Transformers | `smt/` |
| **AnyTool** | Hierarchical tool selection | `anytool/` |
| **GNN** | Graph Neural Network approach | `gnn/` |

## Datasets

| Dataset | Description |
|---------|-------------|
| **ITOps** | IT operations monitoring (Concert, Instana, Turbonomic) |
| **NL2GQL** | Natural language to GraphQL scenarios |
| **OMOP_CMS** | OMOP healthcare with CMS data |
| **OMOP_MIMIC** | OMOP healthcare with MIMIC data |
| **OMOP_Synthea** | OMOP healthcare with Synthea data |

## Models

| Model | ID |
|-------|-----|
| **Granite** | `watsonx/ibm/granite-4-h-small` |
| **Llama** | `watsonx/meta-llama/llama-3-3-70b-instruct` |
| **Mistral** | `mistralai/mistral-medium-2505` |

## Project Structure

```
graphql-schema-mapping/
├── run.sh                 # Unified experiment runner
├── requirements.txt       # Python dependencies
│
├── map_reflect/           # Map-Reflect method
│   ├── map_reflect_runner.py
│   └── results/
│
├── react/                 # ReAct method
│   ├── react_runner.py
│   └── results/
│
├── smt/                   # SMT method
│   ├── smt_runner.py
│   └── results/
│
├── anytool/               # AnyTool method
│   ├── anytool_runner.py
│   └── results/
│
├── gnn/                   # GNN method
│   ├── gnn_runner.py
│   └── results/
│
├── eval/                  # Evaluation scripts
│   ├── calc_accuracy.py       # Accuracy calculation
│   ├── eval_schema_mapping.py # Result aggregation
│   ├── similarity_evaluator.py
│   ├── llm_evaluator.py
│   └── README.md
│
├── data/                  # Data files
│   ├── *_api_lookup.json      # API metadata per dataset
│   ├── *_hierarchy.json       # Hierarchy files (AnyTool)
│   └── *_graphql/             # GraphQL schemas
│
├── results/               # Paper artifacts
│   ├── vldb.tex
│   └── RESULTS_SUMMARY.md
│
└── scripts/               # Utility scripts
    └── merge_mappings.py
```

## Running Experiments

### Using the Unified Runner

```bash
# Run all experiments (default behavior)
./run.sh

# Filter by method, dataset, or model
./run.sh --method map_reflect
./run.sh --dataset ITOps
./run.sh --model granite
./run.sh --method react --dataset OMOP_CMS --model llama

# Preview without executing
./run.sh --dry-run

# Quick mode (reduced delays)
./run.sh --quick --method map_reflect --dataset ITOps

# Run on remote machine
./run.sh --remote
```

### Aggregating Results

```bash
# Aggregate results and generate LaTeX tables
./run.sh --aggregate

# Calculate accuracy metrics
./run.sh --accuracy

# Full pipeline
./run.sh --all
```

### Direct Script Execution

Each method can also be run directly:

```bash
# Map-Reflect
cd map_reflect && python map_reflect_runner.py \
    --datasets ITOps \
    --models watsonx/ibm/granite-4-h-small \
    --data-dir ../data

# ReAct
cd react && python react_runner.py \
    --datasets ITOps \
    --models granite

# Accuracy calculation
python eval/calc_accuracy.py --datasets ITOps --output markdown
```

## Evaluation

### Accuracy Metrics

- **API-level accuracy**: Correct source AND target product identification
- **Field-level accuracy**: Correct field mappings (conditional on correct API)

See `eval/README.md` for detailed documentation of evaluation scripts.

### Output Format

All methods produce results in JSONL format:

```json
{
  "sample_id": "itops_001",
  "predicted_mapping": {"source_field": {"target_property": "target_field", "justification": "..."}},
  "ground_truth_mapping": {"source_field": "target_field"},
  "expected_source": {"id": "source_schema"},
  "expected_target": {"id": "target_schema"},
  "accuracy": 0.85,
  "execution_time": 2.3,
  "success": true
}
```

## Environment Setup

Create a `.env` file with API credentials:

```bash
WATSONX_APIKEY=your_api_key
WATSONX_PROJECT_ID=your_project_id
MISTRAL_API_KEY=your_mistral_key
```

## Results

Results are saved in each method's `results/` directory:
- `{dataset}_{model}.jsonl` - Raw results
- `{dataset}_{model}.json` - Aggregated results

LaTeX tables for the paper are generated in `results/results_tables.tex`.
