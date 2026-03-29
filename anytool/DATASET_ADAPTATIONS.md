# AnyTool Dataset Adaptations

## Date: 2026-03-29

This document tracks all dataset adaptations made for running AnyTool
across different evaluation datasets.

---

## ITOpsGraph (11,110 fields, 17 products, 521 samples)

### Hierarchy Construction
- **Source**: `data/itopsgraph_api_lookup.json`
- **Script**: `anytool/build_hierarchy.py`
- **Output**: `anytool/tool_data.json`, `anytool/api_details.json`, `anytool/category_tool_details.json`
- **Structure**: 11 categories → 17 products → 11,110 API fields
- **Category mapping**: Manual assignment of products to domain categories
  (Observability, Governance, Optimization, FinOps, etc.)

### Evaluation
- **Dataset file**: `data/itopsgraph_dataset.jsonl` (521 samples)
- **Subset tested**: 50 samples (first 50 in dataset order)
- **Evaluation script**: `scripts/fix_anytool_eval.py` (flexible field name matching)
- **Matching strategy**: Tries exact match, reversed direction, field-name-only match,
  and reversed field-name match to handle format differences between AnyTool's
  predictions and ground truth keys.

### Results (50-sample subset)
- Raw accuracy: 13.8% (8/58)
- Flexible matching: 32.8% (19/58)
- Type accuracy: 25.9%

### Known Issues
- AnyTool sometimes maps to the source product instead of the target
- Meta-agent doesn't always select both source and target categories
- Solver receives top-100 target fields (truncated for context length)

---

## OMOP-CMS (38 samples, 145 fields)

### Hierarchy Construction
- **Source**: Same `itopsgraph_api_lookup.json` (includes OMOP types)
- **Note**: OMOP has a single target schema (OMOP CDM), making the
  hierarchical retrieval simpler — only one target product to explore.

### Adaptation Required
- OMOP utterances: "find OMOP tables that map to omop person"
- Source is the OMOP CDM type, target is a specific product (CMS, MIMIC, Synthea)
- The hierarchy needs to include CMS/MIMIC/Synthea product schemas
- These are in the API lookup under their respective product names

### Status
- [ ] Run AnyTool on OMOP-CMS
- [ ] Run AnyTool on OMOP-MIMIC
- [ ] Run AnyTool on OMOP-Synthea

---

## OMOP-MIMIC (46 samples, 119 fields)

### Status
- [ ] Pending (same hierarchy as OMOP-CMS)

---

## OMOP-Synthea (12 samples, 58 fields)

### Status
- [ ] Pending (same hierarchy as OMOP-CMS)

---

## NL2GQL (437 samples, 54 schemas)

### Adaptation Required
- NL2GQL uses GraphQL schemas (different product names than ITOpsGraph)
- Need to build separate hierarchy from NL2GQL API lookup
- Products: 54 individual GraphQL schemas (e.g., countries, spacex, anilist)

### Status
- [ ] Build NL2GQL hierarchy
- [ ] Run AnyTool on NL2GQL

---

## ToolBench (3,190 samples, 50 categories)

### Adaptation Required
- ToolBench uses REST APIs from RapidAPI — closest to AnyTool's original setting
- Need hierarchy from ToolBench API specifications
- AnyTool was originally designed for this kind of data

### Status
- [ ] Build ToolBench hierarchy
- [ ] Run AnyTool on ToolBench

---

## Spider (745 samples, 164 databases)

### Adaptation Required
- Spider uses SQL tables with foreign key relationships
- Need hierarchy: database → tables → columns
- Products = databases, types = tables, APIs = columns

### Status
- [ ] Build Spider hierarchy
- [ ] Run AnyTool on Spider

---

## SchemaNet (377 samples, 7 schema pairs)

### Status
- **Already completed** using `run_anytool_schemanet.py` (separate evaluation)
- Uses different evaluation logic (query-based, not dataset-based)
- Results in paper Tables 9-10

---

## LLMatch Adaptation

### MongoDB Integration
- **Script**: `llmatch/load_itopsgraph_to_mongodb.py`
- **Data loaded**: 11,110 OntologySchemaRewrite records + 55 OntologyAlignmentGroundTruth records
- **MongoDB**: `mongodb://localhost:27017/llmatch`
- **Runner**: `llmatch/run_llmatch_itopsgraph.py`

### Results (3 ITOpsGraph experiments)
- instana → concert: F1=0.000 (0/92)
- instana → turbonomic: F1=0.179 (10/92)
- instana → cloudability: F1=0.148 (12/85)
- **Average F1: 0.109** (22/269)

### Known Issues
- LLMatch's column-matching approach assumes flat relational schemas
- No structural understanding of nested GraphQL/protobuf types
- Maps one field at a time without cross-field context
