# 🧪 Evaluation Scripts for Schema Mapping

This directory contains scripts to **evaluate the quality of API field mappings** using two methods:

1. **Cosine Similarity Evaluation** – Lightweight, embedding-based evaluation
2. **LLM-based Judging** – Uses IBM WatsonX or another LLM to judge mappings on joinability and equivalence

---

## 📁 Files Overview
```
evaluation-scripts/
├── .env                        # Environment variables 
├── api_lookup.json             # Enriched API metadata (used in all evaluations)
├── all_results.json            # Output of LLM-based batch evaluation
├── similarity_eval_output.json # Output of similarity-based evaluation
├── similarity_evaluator.py     # Evaluates mappings using cosine similarity
├── similarity_evaluator.sh     # Bash wrapper for similarity_evaluator.py
├── llm_evaluator.py            # Single mapping evaluation using LLM
├── llm_eval_batch.py           # Batch mapping evaluation using LLM
├── llm_eval_batch.sh           # Bash wrapper for llm_eval_batch.py
└── README.md                 
```

---

## 🧠 Evaluation Methods

### 🔹 1. `similarity_evaluator.py`

**Description**:  
Evaluates a schema mapping file using sentence-transformer embeddings.  
Each field mapping is classified as **correct** if the cosine similarity between source and target fields > **0.3**.

**Workflow**:
- Input: `--mapping_file`, `--api_lookup_file`, `--output_file`
- For each mapping:
  - Retrieve `source` and `target` fields from `api_lookup.json`
  - Compute cosine similarity using SentenceTransformers
  - Classify as correct (if sim > 0.3) or incorrect
- Output accuracy = (correct / total)

**Run**:
`bash evaluation-scripts/similarity_evaluator.sh`

### 2. `llm_eval_batch.py`

**Description:**
Uses a foundation model (LLM) to evaluate whether two API fields are functionally equivalent or joinable.

**Functionality:**
- Loads a directory of schema mapping JSONs (--mapping_dir)
- Uses api_lookup.json to fetch metadata for each field
- Sends formatted prompt to WatsonX model

Returns:

- equivalence_score (0–1)
- joinability_score (0–1)
- explanation for each decision

Classification:

- Mappings with score ≥ 0.5 are treated as correct
- Prints accuracy report for both metrics

**Run**:
`bash evaluation-scripts/llm_eval_batch.sh`