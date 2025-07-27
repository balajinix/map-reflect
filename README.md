# GraphQL Schema Mapping

Code for evaluating different schema mapping methods for the AAAI paper "Map-Reflect: GraphQL Schema Mapping using Self Reflection". The evaluation compares 5 methods across 3 datasets using 3 language models.

## Methods

1. **SMT** (Schema Mapping Toolkit) - Baseline method
2. **AnyTool** - Baseline method  
3. **ReAct** - ICSE version baseline
4. **GNN** - Graph Neural Network approach
5. **Map-Reflect** - Our proposed self-reflection method

## Datasets

1. **ITOps** - IT operations monitoring schemas
2. **NL2GQL** - Natural language to GraphQL scenarios  
3. **GraphQLRestBench** - GraphQL-REST API mapping challenges

## Models

1. **Granite** - IBM's foundation model
2. **Llama** - Meta's open-source model
3. **Qwen** - Alibaba's multilingual model

---

## General Workflow Pattern

Each method follows this standardized pattern:

### Phase 1: Data Preparation
1. **Natural Language Input**: Start with utterances describing schema mapping tasks
2. **Schema Retrieval**: Use Milvus/Chroma vector database to retrieve relevant API schemas
3. **File Generation**: Populate `source.json` and `target.json` files
4. **Ground Truth**: Ensure ground truth mappings are available

### Phase 2: Method Execution  
1. **Method-Specific Processing**: Run the specific algorithm/toolkit
2. **Result Generation**: Produce predicted schema mappings
3. **Evaluation**: Calculate accuracy against ground truth
4. **Output Formatting**: Save results in standardized JSON format

### Phase 3: Aggregation
1. **Collection**: Gather all method results
2. **Analysis**: Run comprehensive evaluation script
3. **Reporting**: Generate LaTeX tables for paper

---

## Method-Specific Workflows

### 1. SMT (Schema Mapping Toolkit)

**Directory Structure:**
```
smt/
├── benchmark_driver/
├── cookbooks/
├── toolkit/
├── results/           # Output directory
├── source.json        # Generated from retrieval
├── target.json        # Generated from retrieval
├── Makefile
└── smt_runner.py      # Our execution script
```

**Execution Steps:**
1. **Setup**: `make install` in SMT directory
2. **Data Preparation**: 
   - Generate `source.json` and `target.json` from vector retrieval
   - Ensure proper schema format expected by SMT toolkit
3. **Execution**: Run SMT toolkit using `toolkit.solve.solve()` function
4. **Output**: Generate `results/{dataset}_{model}.json`

**Key Integration Points:**
- Uses existing `toolkit.solve.solve()` API
- Requires `source_schema_dict` and `target_schema_dict` parameters
- Uses `pipeline_name="genai_mapping"` (configurable)
- Needs adaptation of result parsing based on SMT output format

### 2. AnyTool

**Expected Directory Structure:**
```
anytool/
├── [AnyTool codebase]
├── results/           # Output directory
└── anytool_runner.py  # Our execution script
```

**Execution Steps:**
1. **TBD**: Pending AnyTool codebase details
2. **Data Preparation**: Format input for AnyTool requirements
3. **Execution**: Run AnyTool algorithm
4. **Output**: Generate `results/{dataset}_{model}.json`

### 3. ReAct (ICSE Version)

**Expected Directory Structure:**
```
react/
├── [ReAct codebase]
├── results/           # Output directory  
└── react_runner.py    # Our execution script
```

**Execution Steps:**
1. **TBD**: Pending ReAct codebase details
2. **Data Preparation**: Format input for ReAct requirements
3. **Execution**: Run ReAct reasoning algorithm
4. **Output**: Generate `results/{dataset}_{model}.json`

### 4. GNN

**Expected Directory Structure:**
```
gnn/
├── [GNN codebase]
├── results/           # Output directory
└── gnn_runner.py      # Our execution script  
```

**Execution Steps:**
1. **TBD**: Pending GNN codebase details
2. **Data Preparation**: Create graph representations of schemas
3. **Execution**: Run GNN-based mapping prediction
4. **Output**: Generate `results/{dataset}_{model}.json`

### 5. Map-Reflect

**Expected Directory Structure:**
```
map_reflect/
├── [Map-Reflect codebase]
├── results/           # Output directory
└── map_reflect_runner.py  # Our execution script
```

**Execution Steps:**
1. **TBD**: Pending Map-Reflect implementation details
2. **Data Preparation**: Format input for self-reflection framework
3. **Execution**: Run iterative self-reflection mapping
4. **Output**: Generate `results/{dataset}_{model}.json`

---

## Data Preparation Pipeline

### Vector Database Integration

**Components Needed:**
1. **Vector Store**: Milvus or Chroma database containing API schemas
2. **Embeddings**: Schema embeddings for similarity search
3. **Retrieval Pipeline**: Query → relevant schemas → JSON files

**Process:**
1. **Input**: Natural language utterance describing mapping task
2. **Embedding**: Convert utterance to vector representation
3. **Search**: Find relevant source and target schemas in vector DB
4. **Extraction**: Extract schema definitions 
5. **Formatting**: Convert to method-specific JSON format
6. **Ground Truth**: Associate with known correct mappings

### Schema Format Standardization

**Source/Target JSON Structure:**
```json
{
  "schema_type": "graphql|rest|openapi",
  "schema_definition": {
    // Method-specific schema representation
  },
  "metadata": {
    "dataset": "ITOps|NL2GQL|GraphQLRestBench",
    "source_api": "...",
    "target_api": "...",
    "description": "..."
  }
}
```

---

## Standardized Output Format

All methods must produce results in this format:

```json
{
  "metadata": {
    "dataset": "ITOps",
    "method": "SMT", 
    "model": "granite",
    "execution_time": 45.2,
    "timestamp": "2025-01-15T10:30:00Z",
    "total_samples": 20,
    "successful_samples": 18,
    "overall_accuracy": 0.75
  },
  "results": [
    {
      "sample_id": "itops_001",
      "predicted_mapping": {
        "source_field": "target_field"
      },
      "accuracy": 0.85,
      "ground_truth_mapping": {
        "source_field": "target_field"
      },
      "source_schema": "...",
      "target_schema": "...", 
      "success": true,
      "error_message": null
    }
  ]
}
```

---

## Execution Sequence

### Full Evaluation Run

```bash
# 1. Data preparation (run once)
python prepare_evaluation_data.py --datasets ITOps NL2GQL GraphQLRestBench

# 2. Run each method
cd smt && python smt_runner.py && cd ..
cd anytool && python anytool_runner.py && cd ..  
cd react && python react_runner.py && cd ..
cd gnn && python gnn_runner.py && cd ..
cd map_reflect && python map_reflect_runner.py && cd ..

# 3. Generate evaluation report
python eval_schema_mapping.py --generate-latex
```

### Quick Test Run

```bash
# Quick test with minimal data
python prepare_evaluation_data.py --quick
cd smt && python smt_runner.py --quick && cd ..
python eval_schema_mapping.py --quick --generate-latex
```
