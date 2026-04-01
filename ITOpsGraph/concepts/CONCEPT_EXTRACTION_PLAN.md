# Concept Extraction Pipeline Plan

## Overview

Build a reusable pipeline to extract concepts from documentation and populate a concept graph in Neo4j. The pipeline separates API-specific documentation from general product documentation, handles structured schemas via direct parsing (not markdown conversion), and provides quality metrics at each stage. Neo4j serves as the unified store for both graph data and vector embeddings.

---

## 1. Directory Structure

```
/Users/balaji/code/solis-dal-dataset/nl2cypher/
│
├── data/                           # Existing - schemas, mappings
│
├── docs/                           # Top-level docs directory
│   │
│   ├── api_docs/                   # API-specific documentation
│   │   ├── raw/                    # Original files
│   │   │   ├── graphql/           # .graphql schema files
│   │   │   ├── openapi/           # OpenAPI/Swagger specs
│   │   │   └── proto/             # .proto files
│   │   └── processed/              # Parsed structured output (JSON)
│   │
│   ├── product_docs/               # General product documentation
│   │   ├── raw/                    # Original files
│   │   │   ├── pdf/               # PDF manuals, guides
│   │   │   └── html/              # Scraped HTML pages
│   │   └── processed/              # Docling output (markdown)
│   │
│   └── external/                   # Third-party/reference docs
│       ├── raw/
│       └── processed/
│
├── concepts/                       # Concept extraction pipeline
│   │
│   ├── config/                     # Pipeline configuration
│   │   ├── entity_types.yaml      # GLiNER2 entity definitions
│   │   ├── relation_types.yaml    # Relation type definitions
│   │   ├── synonyms.yaml          # Domain synonym dictionaries
│   │   └── pipeline.yaml          # Processing parameters
│   │
│   ├── pipeline/                   # Reusable pipeline code
│   │   ├── __init__.py
│   │   ├── convert.py             # Docling conversion
│   │   ├── chunker.py             # Section-aware chunking
│   │   ├── ner.py                 # Entity extraction (NER)
│   │   ├── resolve.py             # Entity resolution/dedup
│   │   ├── relations.py           # Relation discovery
│   │   ├── loader.py              # Neo4j loading (graph + vectors)
│   │   └── dedup.py               # Document-hash idempotency
│   │
│   ├── extractors/                 # NER tool wrappers
│   │   ├── __init__.py
│   │   ├── gliner2_extractor.py   # GLiNER2 (primary)
│   │   ├── spacy_extractor.py     # spaCy (fallback)
│   │   └── llm_extractor.py       # LLM augmentation
│   │
│   ├── parsers/                    # Schema-specific parsers
│   │   ├── __init__.py
│   │   ├── graphql_parser.py      # GraphQL schema parser
│   │   ├── openapi_parser.py      # OpenAPI/Swagger parser
│   │   └── proto_parser.py        # Protobuf parser
│   │
│   ├── relations/                  # Relation discovery
│   │   ├── __init__.py
│   │   ├── dependency.py          # Dependency parse patterns
│   │   ├── cooccurrence.py        # Co-occurrence analysis
│   │   └── llm_classifier.py      # LLM relation classification
│   │
│   ├── output/                     # Pipeline outputs (per run)
│   │   ├── entities/              # Extracted entities (JSON)
│   │   ├── relations/             # Discovered relations (JSON)
│   │   └── metrics/               # Quality metrics
│   │
│   ├── evaluation/                 # Quality evaluation
│   │   ├── __init__.py
│   │   ├── metrics.py             # Metric calculations
│   │   └── reports/               # Generated reports
│   │
│   └── scripts/                    # CLI entry points
│       ├── run_pipeline.py        # Full pipeline
│       ├── convert_docs.py        # Conversion only
│       ├── extract_entities.py    # NER only
│       └── evaluate.py            # Run evaluation
│
└── wiki/docs/                      # Existing (to be migrated)
    └── documentation/              # Current location
```

---

## 2. Pipeline Stages

### Document Conversion & Schema Parsing

The pipeline handles two fundamentally different input types with separate paths:

#### Unstructured Document Conversion (Docling)

**Tool:** [Docling](https://github.com/docling-project/docling)

**Input:** PDF, HTML, DOCX files
**Output:** Structured Markdown with preserved hierarchy

**Why Docling:**
- Supports both PDF and HTML (single tool)
- Uses Granite-Docling (258M params) for accurate extraction
- Preserves document structure (headers, tables, code blocks)
- Runs fully local (no API calls)
- Outputs to Markdown, JSON, or DocTags

#### Structured Schema Parsing (Native Parsers)

**Input:** GraphQL schemas, OpenAPI/Swagger specs, Proto files
**Output:** Structured JSON with types, fields, descriptions, relationships

These files are already structured and should NOT be converted to markdown (which loses structure). Instead, use native parsers:

- **GraphQL:** Parse `.graphql` files to extract types, fields, arguments, descriptions, and inter-type relationships. Reuse existing ITOpsGraph GraphQL parsing.
- **OpenAPI:** Parse specs to extract endpoints, request/response schemas, parameters, and model definitions. Use `prance` or `openapi-spec-validator`.
- **Proto:** Parse `.proto` files to extract messages, fields, services, and enums. Use `grpcio-tools` or `proto-schema-parser`.

The parsed output feeds directly into NER with richer structure than markdown would provide.

#### Section-Aware Chunking

After conversion, all text content is chunked before NER extraction:

- **Chunk by document structure:** Use Docling's header hierarchy to split at section boundaries, not arbitrary token windows.
- **Overlap strategy:** Include the parent section header with each chunk for context.
- **Chunk size target:** 512-1024 tokens per chunk (GLiNER2's effective window).
- **Metadata preservation:** Each chunk retains its source document, section path, and position.

**Quality Metrics (Conversion):**
| Metric | Description |
|--------|-------------|
| `docs_converted` | Documents processed by Docling |
| `schemas_parsed` | Schema files parsed natively |
| `chunks_created` | Total chunks after splitting |
| `avg_chunk_tokens` | Average tokens per chunk |
| `chars_extracted` | Total characters extracted |
| `tables_detected` | Number of tables found |
| `headers_found` | Number of section headers |
| `conversion_time` | Processing time in seconds |

---

### Entity Extraction (NER + LLM)

**Primary Tool:** [GLiNER2](https://github.com/fastino-ai/GLiNER2)

**Why GLiNER2 over GLiNER/Stanza/spaCy:**
- **Unified multi-task model** - NER + Text Classification + Structured Data + Relation Extraction in one 205M parameter model
- **Schema-driven interface** - Define extraction schemas declaratively
- **Smaller & faster** - 205M params vs ~500M for original GLiNER
- **Competitive accuracy** - Matches GPT-4o on CrossNER benchmark (0.590 vs 0.599 F1)
- **Zero-shot custom entity types** - No retraining needed
- **CPU-efficient** - Runs locally without GPU
- **Built-in relation extraction** - Can extract relations alongside entities

#### Two-Pass Extraction Strategy

Entity types mix abstraction levels (e.g., "Technology" is broad, "Container" is specific). Use a two-pass approach:

**Pass 1 - Fine-Grained Extraction:**
Extract specific, concrete entities that GLiNER2 can reliably identify.

```yaml
# config/entity_types.yaml

# Pass 1: Fine-grained entities
fine_grained:
  itops_resources:
    - Metric
    - Alert
    - Policy
    - Action
    - Workload
    - Cluster
    - Container
    - Instance
    - Endpoint
    - Dashboard
    - Namespace

  technologies:
    - Service
    - Tool
    - Platform
    - Protocol
    - API
    - Framework

  operational:
    - Cost
    - SLA
    - Threshold
    - Configuration
    - Deployment
    - Pipeline
```

**Pass 2 - Hierarchy Classification (LLM):**
Classify extracted entities into broader categories.

```yaml
# Pass 2: Hierarchy classification (LLM-assigned)
hierarchy:
  - Technology        # Broad category
  - Feature           # Product capability
  - Capability        # Abstract function
  - Resource          # Infrastructure component
  - Component         # System building block
  - Monitoring        # Observability concept
  - Optimization      # Performance/cost concept
  - Integration       # Cross-system concept
```

**LLM Augmentation (applied per chunk):**
1. Validate extracted entities against chunk context
2. Merge obvious duplicates within chunk
3. Assign hierarchy classification
4. Add missing entities the NER missed

**Quality Metrics (NER):**
| Metric | Description |
|--------|-------------|
| `entities_total` | Total entities extracted |
| `entities_per_chunk` | Average entities per chunk |
| `entity_type_dist` | Distribution by fine-grained type |
| `hierarchy_dist` | Distribution by hierarchy category |
| `confidence_avg` | Average extraction confidence |
| `confidence_dist` | Confidence score distribution |
| `llm_additions` | Entities added by LLM |
| `llm_removals` | Entities rejected by LLM |
| `pass1_count` | Entities from GLiNER2 pass |
| `pass2_reclassified` | Entities reclassified by hierarchy |

---

### Entity Resolution (Cross-Product Aware)

**Goal:** Deduplicate and canonicalize entities, with special handling for cross-product synonyms in ITOps.

#### Resolution Approaches (in order):

1. **Exact matching** - Normalize case, whitespace, punctuation
2. **Domain synonym matching** - Use curated ITOps synonym dictionary
3. **Fuzzy matching** - Edit distance, Jaro-Winkler (threshold > 0.85)
4. **Embedding similarity** - Sentence-BERT cosine similarity (threshold > 0.80), stored in Neo4j vector index
5. **LLM verification** - Confirm proposed merges for borderline cases

#### Cross-Product Synonym Dictionary

In ITOps, the same concept appears across products with different names. Maintain a curated synonym dictionary:

```yaml
# config/synonyms.yaml
cross_product_synonyms:
  virtual_machine:
    canonical: "Virtual Machine"
    aliases:
      - "VM"
      - "EC2 Instance"          # AWS
      - "Compute Instance"      # GCP
      - "Azure VM"              # Azure
      - "Virtual Server"        # Generic
      - "vServer"               # IBM

  container:
    canonical: "Container"
    aliases:
      - "Docker Container"
      - "Pod"                   # Kubernetes (close but distinct - flag for review)
      - "Container Instance"    # AWS ECS

  load_balancer:
    canonical: "Load Balancer"
    aliases:
      - "ELB"                   # AWS
      - "ALB"                   # AWS
      - "Azure Load Balancer"   # Azure
      - "Cloud Load Balancing"  # GCP

  # ... extend as pipeline discovers new synonyms
```

**Output:**
```json
{
  "canonical": "Kubernetes Cluster",
  "aliases": ["k8s cluster", "K8S Cluster", "kubernetes-cluster"],
  "mentions": 47,
  "source_docs": ["doc1.md", "doc2.md"],
  "source_products": ["instana", "turbonomic"],
  "confidence": 0.95,
  "resolution_method": "synonym_dict+embedding"
}
```

**Quality Metrics (Resolution):**
| Metric | Description |
|--------|-------------|
| `duplicates_found` | Number of duplicate groups |
| `entities_merged` | Total entities merged |
| `merge_ratio` | Merged / Original entities |
| `synonym_matches` | Merges from synonym dict |
| `fuzzy_matches` | Merges from fuzzy matching |
| `embedding_matches` | Merges from embedding similarity |
| `llm_verified` | Merges confirmed by LLM |
| `llm_rejected` | Proposed merges rejected by LLM |
| `cross_product_merges` | Merges across different products |
| `canonical_coverage` | % with clean canonical form |

---

### Relation Discovery

#### Dependency Parse Patterns

Extract high-precision typed relations directly from sentence structure using spaCy dependency parsing:

**Verb Pattern Extraction:**
```
"Turbonomic monitors Kubernetes clusters" -> (Turbonomic)-[MONITORS]->(Kubernetes Cluster)
"Instana traces application requests"     -> (Instana)-[TRACES]->(Application Request)
"Prometheus collects metrics from pods"    -> (Prometheus)-[COLLECTS_FROM]->(Pod)
```

**Pattern Templates:**
```yaml
# Relation patterns from dependency parse
verb_patterns:
  monitors: MONITORS
  manages: MANAGES
  deploys: DEPLOYS_TO
  integrates: INTEGRATES_WITH
  requires: REQUIRES
  enables: ENABLES
  triggers: TRIGGERS
  scales: SCALES
  collects: COLLECTS
  sends: SENDS_TO
  receives: RECEIVES_FROM
  stores: STORES_IN
```

**Prepositional Patterns:**
```
"metrics from Prometheus" -> (Metric)-[SOURCED_FROM]->(Prometheus)
"alerts in PagerDuty"    -> (Alert)-[MANAGED_IN]->(PagerDuty)
"deployed on Kubernetes"  -> (Application)-[DEPLOYED_ON]->(Kubernetes)
```

#### Co-occurrence Analysis

Build initial weighted graph from:
- **Sentence co-occurrence** - Concepts in same sentence (strongest signal)
- **Section co-occurrence** - Concepts in same section (medium)
- **Document co-occurrence** - Concepts in same document (weakest)

**Edge Weight Calculation:**
```
weight = 0.6 * sentence_cooc + 0.3 * section_cooc + 0.1 * doc_cooc
```

#### LLM Relation Classification

For discovered edges (from both dep-parse and co-occurrence), classify relation type:

```yaml
# config/relation_types.yaml
hierarchical:
  - IS_A           # Taxonomy (Prometheus IS_A Monitoring Tool)
  - PART_OF        # Composition (Pod PART_OF Cluster)
  - INSTANCE_OF    # Instantiation (my-cluster INSTANCE_OF Cluster)

functional:
  - ENABLES        # A enables B
  - REQUIRES       # A requires B
  - USES           # A uses B
  - PRODUCES       # A produces B
  - MONITORS       # A monitors B
  - MANAGES        # A manages B

associative:
  - RELATED_TO     # General association
  - SIMILAR_TO     # Semantic similarity
  - ALTERNATIVE_TO # Can substitute
```

#### GLiNER2 Built-in Relation Extraction

GLiNER2 supports relation extraction natively. For high-confidence entity pairs, use GLiNER2's relation extraction mode before falling back to dependency parsing:

```python
# GLiNER2 schema-based relation extraction
schema = {
    "entities": ["Service", "Platform", "Metric"],
    "relations": [
        {"head": "Service", "relation": "MONITORS", "tail": "Platform"},
        {"head": "Service", "relation": "COLLECTS", "tail": "Metric"}
    ]
}
```

#### GNN-based Discovery (Phase 2 - Deferred)

Use Graph Neural Networks to discover latent relationships after the initial graph is populated. This requires sufficient graph data to train meaningfully.

- **GCN** (Graph Convolutional Network)
- **GAT** (Graph Attention Network) - better for weighted neighbors
- Deferred until the concept graph has >500 nodes and >2000 edges

**Quality Metrics (Relations):**
| Metric | Description |
|--------|-------------|
| `edges_gliner2` | Edges from GLiNER2 relation extraction |
| `edges_dep_parse` | Edges from dependency parsing |
| `edges_cooccurrence` | Edges from co-occurrence |
| `edges_total` | Total unique edges |
| `dep_parse_patterns` | Distinct verb patterns matched |
| `avg_edge_weight` | Average edge confidence |
| `relation_type_dist` | Distribution by relation type |
| `graph_density` | Edge count / possible edges |
| `connected_components` | Number of disconnected subgraphs |

---

### Neo4j Loading (Unified Graph + Vector Store)

Neo4j 5.x serves as the unified store for both the concept graph and vector embeddings, eliminating the need for a separate vector database.

**Schema:**

```cypher
// Concept node with embedding
CREATE (c:Concept {
    name: "Kubernetes Cluster",
    canonical_name: "kubernetes_cluster",
    description: "A set of nodes that run containerized applications",
    category: "Resource",
    hierarchy: "Infrastructure",
    source_docs: ["turbonomic.pdf", "instana.pdf"],
    source_products: ["turbonomic", "instana"],
    extraction_method: "GLiNER2+LLM",
    confidence: 0.92,
    doc_hash: "sha256:abc123",
    created_at: datetime(),
    updated_at: datetime()
})

// Vector index for embeddings
CREATE VECTOR INDEX concept_embeddings IF NOT EXISTS
FOR (c:Concept) ON (c.embedding)
OPTIONS {indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine'
}}

// Store embedding on concept node
SET c.embedding = $embedding_vector

// Similarity search
CALL db.index.vector.queryNodes('concept_embeddings', 10, $query_vector)
YIELD node, score
RETURN node.name, score

// Concept-to-Concept relationships
(c1:Concept)-[:RELATED_TO {weight: 0.85, method: "cooccurrence"}]->(c2:Concept)
(c1:Concept)-[:IS_A]->(c2:Concept)
(c1:Concept)-[:PART_OF]->(c2:Concept)
(c1:Concept)-[:ENABLES]->(c2:Concept)
(c1:Concept)-[:MONITORS {method: "gliner2", source_sentence: "..."}]->(c2:Concept)

// Source traceability
(c:Concept)-[:MENTIONED_IN {section: "Overview", chunk_id: "ch_042"}]->(d:Document)

// API Graph integration (HAS_CONCEPT edges)
(p:Product)-[:HAS_CONCEPT {weight: 0.9}]->(c:Concept)
(t:Type)-[:HAS_CONCEPT {weight: 0.8}]->(c:Concept)
(f:Field)-[:HAS_CONCEPT {weight: 0.7}]->(c:Concept)
```

**Idempotency Strategy:**

Each document is hashed (SHA-256 of content). On re-run:
1. Check if `doc_hash` exists in the graph
2. If unchanged: skip processing
3. If changed: remove old entities from that document, re-extract
4. Dedup key for entities: `(canonical_name, source_doc, span_start, span_end)`

```python
# pipeline/dedup.py
import hashlib

def doc_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()

def should_process(neo4j_session, doc_path: str, content_hash: str) -> bool:
    result = neo4j_session.run(
        "MATCH (d:Document {path: $path}) RETURN d.content_hash AS hash",
        path=doc_path
    )
    record = result.single()
    if record is None:
        return True  # New document
    return record["hash"] != content_hash  # Changed document
```

**Linking Strategy (with confidence thresholds):**
1. **Product-level links** (threshold > 0.7) - Name matching, document provenance
2. **Type-level links** (threshold > 0.8) - Type/Concept name similarity + description embedding
3. **Field-level links** (threshold > 0.85) - Strict name + context matching to avoid noise

**Quality Metrics (Loading):**
| Metric | Description |
|--------|-------------|
| `concepts_created` | New Concept nodes |
| `concepts_updated` | Existing Concept nodes updated |
| `concepts_skipped` | Unchanged documents skipped |
| `concept_edges` | Concept-to-Concept edges |
| `embeddings_stored` | Vectors stored in Neo4j index |
| `api_links_product` | Product-HAS_CONCEPT edges |
| `api_links_type` | Type-HAS_CONCEPT edges |
| `api_links_field` | Field-HAS_CONCEPT edges |
| `mentioned_in_edges` | MENTIONED_IN traceability edges |
| `orphan_concepts` | Concepts with no API links |
| `load_time` | Total loading time |

---

## 3. Quality Evaluation Framework

### Per-Run Report

Each pipeline run generates:

```
concepts/output/metrics/run_20260210_143022/
├── conversion_metrics.json
├── ner_metrics.json
├── resolution_metrics.json
├── relations_metrics.json
├── loading_metrics.json
└── summary.json
```

### Summary Report Format

```json
{
  "run_id": "20260210_143022",
  "input": {
    "docs_processed": 15,
    "docs_skipped": 3,
    "schemas_parsed": 8,
    "total_chars": 2450000,
    "total_chunks": 487,
    "doc_types": {"pdf": 10, "html": 5, "graphql": 3, "openapi": 4, "proto": 1}
  },
  "output": {
    "concepts_extracted": 342,
    "concepts_after_resolution": 287,
    "cross_product_merges": 23,
    "relations_discovered": 1205,
    "gliner2_relations": 180,
    "dep_parse_relations": 340,
    "cooccurrence_relations": 685,
    "api_links_created": 534,
    "embeddings_stored": 287
  },
  "quality": {
    "extraction_confidence_avg": 0.84,
    "resolution_merge_ratio": 0.16,
    "relation_confidence_avg": 0.72,
    "api_link_coverage": 0.78,
    "dep_parse_precision_est": 0.88
  },
  "timing": {
    "conversion_seconds": 120,
    "ner_seconds": 85,
    "resolution_seconds": 30,
    "relations_seconds": 150,
    "loading_seconds": 25,
    "total_seconds": 410
  }
}
```

### Evaluation Dimensions

| Dimension | Metrics | Target |
|-----------|---------|--------|
| **Coverage** | Entities per doc, API link % | High |
| **Precision** | LLM validation rate, Manual sample | > 0.85 |
| **Consistency** | Duplicate ratio, Canonical coverage | Low dups |
| **Connectivity** | Graph density, Components | Connected |
| **Cross-Product** | Cross-product merge count, Synonym coverage | Growing |
| **Performance** | Time per doc, Memory usage | Scalable |
| **Idempotency** | Docs skipped on re-run, No duplicate nodes | 100% |

---

## 4. Dependencies

```
# Document processing
docling>=2.0.0
pymupdf>=1.23.0          # Fallback PDF

# Schema parsing
graphql-core>=3.2.0      # GraphQL schema parsing
prance>=23.0.0           # OpenAPI parsing
openapi-spec-validator>=0.7.0
grpcio-tools>=1.60.0     # Proto parsing (or proto-schema-parser)

# NER extraction
gliner2                  # GLiNER2 unified multi-task model
spacy>=3.7.0             # Fallback NER + dependency parsing
transformers>=4.35.0

# Embeddings
sentence-transformers>=2.2.0

# GNN (Phase 2 - optional)
# torch>=2.0.0
# torch-geometric>=2.4.0

# LLM
openai>=1.0.0            # Or: ollama, vllm for local

# Graph + Vector
neo4j>=5.11.0            # Requires 5.11+ for vector index

# Utilities
pyyaml>=6.0
tqdm>=4.65.0
pandas>=2.0.0
```

---

## 5. CLI Usage

```bash
# Full pipeline
python -m concepts.scripts.run_pipeline \
    --input docs/product_docs/raw/pdf/ \
    --schemas docs/api_docs/raw/ \
    --output concepts/output/ \
    --config concepts/config/pipeline.yaml

# Conversion only: Document conversion + schema parsing
python -m concepts.scripts.convert_docs \
    --input docs/product_docs/raw/pdf/ \
    --schemas docs/api_docs/raw/ \
    --output docs/product_docs/processed/ \
    --format markdown \
    --chunk-size 768

# NER only: Entity extraction
python -m concepts.scripts.extract_entities \
    --input docs/product_docs/processed/ \
    --output concepts/output/entities/ \
    --extractor gliner2 \
    --entity-types concepts/config/entity_types.yaml \
    --two-pass

# Evaluate run
python -m concepts.scripts.evaluate \
    --run-dir concepts/output/metrics/run_20260210_143022/ \
    --output-report concepts/evaluation/reports/
```

---

## 6. Migration Plan

### Step 1: Create new directory structure
```bash
mkdir -p docs/{api_docs,product_docs,external}/{raw,processed}
mkdir -p docs/api_docs/raw/{graphql,openapi,proto}
mkdir -p docs/product_docs/raw/{pdf,html}
mkdir -p concepts/{config,pipeline,extractors,parsers,relations,output,evaluation}
mkdir -p concepts/output/{entities,relations,metrics}
```

### Step 2: Move existing files
```bash
# API docs
mv schemas/graphql/* docs/api_docs/raw/graphql/
mv schemas/proto/* docs/api_docs/raw/proto/
mv wiki/docs/documentation/openapi/* docs/api_docs/raw/openapi/

# Product docs
mv wiki/docs/documentation/pdf/* docs/product_docs/raw/pdf/
mv wiki/docs/documentation/html/* docs/product_docs/raw/html/
mv wiki/docs/documentation/markdown/* docs/product_docs/processed/
```

### Step 3: Update existing code references
- `wiki/docs/content_loader.py` -> point to new paths
- `wiki/docs/pipeline.py` -> deprecate in favor of new pipeline

---

## 7. Implementation Order

1. [x] Create directory structure
2. [x] Migrate existing files
3. [x] Implement `convert.py` (Docling conversion) + `chunker.py`
4. [x] Test on sample PDF/HTML documents (with fallback extractors)
5. [ ] Implement schema parsers (reuse existing GraphQL parser)
6. [x] Implement `ner.py` (GLiNER2 extraction, two-pass)
7. [x] Implement NER quality metrics (NERMetrics class)
8. [x] Implement `resolve.py` (entity resolution with synonym dict)
9. [x] Build initial `synonyms.yaml` from known ITOps products (50+ entries)
10. [x] Implement `relations.py` (GLiNER2 relations + dependency parse + co-occurrence)
11. [x] Implement LLM relation classification (integrated in relations.py)
12. [x] Implement `loader.py` (Neo4j loading with vector index + idempotency)
13. [ ] Implement per-run quality reporting
14. [ ] End-to-end pipeline test on full corpus
15. [ ] Iterate based on metrics

### Phase 2 (after initial pipeline is stable):
- [ ] GNN-based relation discovery
- [ ] Human-in-the-loop review UI (Streamlit) for merge/relation approval
- [ ] Automated synonym dictionary expansion from resolved entities

---

## 8. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **GLiNER2 over GLiNER** | Unified multi-task model (NER + relations + classification) in 205M params. Smaller, faster, matches GPT-4o accuracy. Built-in relation extraction reduces pipeline complexity. |
| **Neo4j as unified store** | Graph + vector in one DB. No ChromaDB dependency. Neo4j 5.11+ has native vector index with cosine similarity. Embeddings stored as node properties. |
| **Schema files parsed natively** | GraphQL/OpenAPI/Proto are structured data. Converting to markdown loses structure. Native parsers extract types, fields, and relationships directly. |
| **Section-aware chunking** | GLiNER2 has token limits. Splitting by document structure (headers) preserves context better than naive token windows. |
| **Two-pass entity extraction** | Avoids asking GLiNER2 to handle mixed abstraction levels. Fine-grained NER first, then LLM hierarchy classification. |
| **Cross-product synonym dicts** | ITOps has product-specific terminology for identical concepts. Curated dictionaries catch what embedding similarity misses. |
| **GLiNER2 relations + dependency parse** | GLiNER2 provides schema-driven relation extraction. Dependency parse adds verb/prep patterns for additional coverage. GNN deferred to Phase 2. |
| **Idempotent pipeline** | Document-hash dedup keys. Re-running on same docs skips unchanged, updates changed. No duplicate nodes. |
| **MENTIONED_IN traceability** | Every concept links back to source document and section. Enables provenance queries and debugging. |

---

## 9. References

- [Docling GitHub](https://github.com/docling-project/docling)
- [GLiNER2 GitHub](https://github.com/fastino-ai/GLiNER2)
- [GLiNER2 Paper](https://arxiv.org/html/2507.18546v1)
- [KGGen Paper](https://arxiv.org/html/2502.09956v1)
- [Neo4j Vector Index Docs](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/)
- [Neo4j Knowledge Graph Builder](https://neo4j.com/labs/genai-ecosystem/llm-graph-builder/)
