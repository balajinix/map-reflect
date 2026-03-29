# AnyTool Faithful Implementation

## Date: 2026-03-29

## Background

The original AnyTool implementation in `anytool_agent.py` was a flat sequential
LangGraph chain — NOT the architecture described in the AnyTool paper (Du et al.,
arXiv:2402.04253). This document describes the faithful reimplementation in
`anytool_faithful.py`.

## Original AnyTool Paper Architecture

The AnyTool paper describes a 3-tier hierarchical agent system:

```
Meta-Agent
  → creates Category Agents (one per relevant category)
    → create Tool Agents (one per subset of tools)
      → search APIs and add to global API Candidate Pool
Solver
  → uses API Candidate Pool to resolve query
Self-Reflection
  → re-activates agents bottom-up if Solver fails
```

Key components:
1. **3-tier API database**: `{category → tool → api_list_names}`
2. **GPT-4 function calling**: each agent has specific functions it can call
3. **Multi-threaded execution**: agents run in parallel
4. **DFSDT Solver**: Depth-First Search Decision Tree for resolving queries
5. **Self-reflection**: re-activates agents in ascending order on failure

Source: `/Users/balaji/code/AnyTool/` (original paper code)

## Our Old Implementation (anytool_agent.py) — BROKEN

A flat sequential LangGraph pipeline:
```
get_products → select_product → get_source_types → select_source_type →
get_source_properties → get_target_types → select_target_type →
get_target_properties → analyze_properties
```

Problems:
- **No hierarchical agents** — just one linear chain
- **No function calling** — text completion only
- **No self-reflection loop** — no re-activation on failure
- **No API candidate pool** — direct type selection
- **Mapped source to itself** — no target product exploration
- Result: **0% accuracy** on all non-SchemaNet datasets

## Faithful Reimplementation (anytool_faithful.py)

### Architecture

```
Phase 1: Meta-Agent (function calling)
  - Input: query + list of 11 categories
  - Functions: get_tools_in_category(), get_tool_descriptions(), select_category(), finish_search()
  - Output: list of relevant categories

Phase 2: Tool Agents (function calling, one per category)
  - Input: query + tools in category
  - Functions: get_apis_in_tool(), get_api_details(), add_api_to_pool(), finish_search()
  - Output: API candidate pool (list of {category, tool, api_name})
  - KEY FIX: excludes source product from target search

Phase 3: Solver (direct LLM prompt)
  - Input: source fields + full target product field list
  - Output: {source_field: {target_property, justification}}

Self-Reflection:
  - If solver returns empty, broaden category search and retry
  - max_reflection=1 (configurable)
```

### 3-Tier Hierarchy Construction

Built by `build_hierarchy.py` from `itopsgraph_api_lookup.json`:

| Level | Maps to | Count |
|-------|---------|-------|
| Category | Domain (Observability, Governance, etc.) | 11 |
| Tool | Product (instana, concert, aws, etc.) | 17 |
| API | Type.field (Entity.uuid, Application.id, etc.) | 11,110 |

Category assignments:
- Observability: instana
- Optimization: turbonomic
- Governance: concert
- FinOps: cloudability
- Networking: ns1
- AI_Orchestration: watsonx-orchestrate
- Cloud_Infrastructure: aws, azure, gcp
- Infrastructure_Graph: infragraph, base, model, edge, resource
- Image_Management: packer
- Cloud_Platform: hcp
- Infrastructure_as_Code: hcp_terraform

### LLM Integration

- Uses **LiteLLM** with WatsonX backend (not OpenAI)
- Function calling via `tools` parameter (works with Llama-3.3-70B)
- Retry logic: 3 retries with exponential backoff (2s, 4s, 6s)
- Max 8 steps per agent (reduced from 15 for efficiency)

### Key Differences from Paper

| Aspect | Paper (GPT-4) | Our Implementation (Llama) |
|--------|--------------|---------------------------|
| LLM | GPT-4 with function calling | Llama-3.3-70B via WatsonX |
| Hierarchy source | RapidAPI categories | Product domain categories |
| Task | API tool calling | Schema field mapping |
| Solver | DFSDT with API execution | Direct LLM field matching |
| Threading | Multi-threaded (16 semaphore) | Sequential |
| API pool size | Up to 64 APIs | All fields in target product |
| Self-reflection depth | Up to 20 rounds | 1 round |

### Results

| Dataset | Metric | Result |
|---------|--------|--------|
| ITOpsGraph (50 samples) | Field accuracy (flexible) | **32.8%** (19/58) |
| ITOpsGraph (50 samples) | Type accuracy | 25.9% |
| ITOpsGraph (50 samples) | Raw accuracy | 13.8% (8/58) |

For comparison: Map-Reflect (Llama) achieves 80.0% field F1 on full ITOpsGraph.

### Files

- `anytool_faithful.py` — main implementation (agents, solver, runner)
- `build_hierarchy.py` — converts API lookup to 3-tier format
- `tool_data.json` — category → tool → api_list_names
- `api_details.json` — detailed API descriptions
- `category_tool_details.json` — category and tool descriptions
- `results/` — evaluation output files
