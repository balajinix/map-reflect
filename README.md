# 🧠 Schema Mapping

This repository contains modular components for a multi-agent system designed for GraphQL schema mapping and knowledge extraction. It includes an schema mapping baselines, information retrieval for enhancing descriptions and evaluation scripts.

---

## 📁 Folder Overview

### `schema_mapping_agent/`
- **Purpose**: Core LangGraph-based agent that performs GraphQL schema mapping.
- **Key Features**:
  - A2A-compatible communication using the agent protocol.
  - Dynamically invokes external tools (like `populate_neo4j`, `populate_wiki`) from a separate MCP server.
  - Detects user intent and routes tasks accordingly in a multi-agent workflow.
  - Has been contributed to the solis-dal-ai repository. 

---

### `docs_info/`
- **Purpose**: Enrich incomplete API schema descriptions.
- **Key Features**:
  - Uses a Retrieval-Augmented Generation (RAG) model to extract relevant informtation from raw API documentation.
  - Generates an enriched `api_lookup.json` file used downstream for better field-level mapping and for evaluation.

---

### `mappings_output/`
- **Purpose**: Stores output of schema mappings performed by the schema-mapping-toolkit.
- **Contents**:
  - Mappings are also used to populate a **Neo4j graph** and a **schema wiki** via agent tools.
  - These mappings are used to populate the graph for training the GNN.

---

### `evaluation-scripts/`
- **Purpose**: Evaluate the quality of generated mappings.
- **Key Features**:
  - **Cosine Similarity**: Measures semantic similarity between enriched descriptions.
  - **LLM-as-a-Judge**: Uses a foundation model to evaluate whether source–target field mappings are valid using joinability and equivalence scoring.
  - Accuracy scores are computed for batch mapping files.

---

### `gnn_baseline/`
- **Purpose**: Predict links using Graph Neural Networks.
- **Key Features**:
  - Constructs a knowledge graph from all mappings.
  - Implements GCN, GraphSAGE, and GAT models using PyTorch Geometric.
  - Evaluates link prediction performance using AUC and average precision.

---

## 🔁 End-to-End Pipeline

1. Enrich API schema descriptions → `docs_info/`
2. Generate mappings using agent → `schema_mapping_agent/`
3. Store all mapping outputs → `mappings_output/`
4. Evaluate quality of results → `evaluation-scripts/`
5. Link prediction GNN → `gnn_baseline/`

---

## 🧠 Built With

- **LangGraph** – Agent orchestration
- **FastAPI + A2A** – Protocol for multi-agent communication
- **Neo4j** – Graph database to store mappings
- **MediaWiki** – For schema visualization
- **PyTorch Geometric** – GNN modeling
- **WatsonX / LLM APIs** – For judgment and enrichment

