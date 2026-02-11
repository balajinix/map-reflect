# Schema Files Usage Guide

This document describes how `hierarchy.json` and `property_lookup.json` are used by the api_mapping codebase for schema mapping operations.

## File Overview

| File | Purpose | Access Pattern |
|------|---------|----------------|
| `hierarchy.json` | Nested tree structure | Browse by product → type → fields |
| `property_lookup.json` | Flat key-value store | Direct lookup by qualified field name |

---

## 1. hierarchy.json

### Structure
```json
{
  "product_name": {
    "TypeName": [
      {
        "name": "product_TypeName.field_name",
        "description": "Field description",
        "type": "String",
        "schema_component": "TypeName"
      }
    ]
  }
}
```

### Primary Use Cases

#### A. Agent Tool Functions
Used by LLM agents to browse and understand schema structure:

```python
# Get all types for a product
def get_types_for_product(product: str) -> list:
    return [
        {
            'type_name': type_name,
            'property_count': len(properties),
        }
        for type_name, properties in hierarchy.get(product, {}).items()
    ]

# Get all fields for a specific type
def get_properties_for_type(product: str, type_name: str) -> list:
    properties = hierarchy.get(product, {}).get(type_name, [])
    return [{
        'name': prop['name'],
        'description': prop['description'],
        'data_type': prop['type']
    } for prop in properties]
```

#### B. Schema Traversal
Navigate the schema tree for mapping candidates:

```python
# Find all types across products
for product, types in hierarchy.items():
    for type_name, fields in types.items():
        for field in fields:
            # Process each field
            pass
```

#### C. Context Building for LLM Prompts
Build rich context strings for mapping prompts:

```python
def build_type_context(product: str, type_name: str) -> str:
    fields = hierarchy[product][type_name]
    lines = [f"Type: {type_name} (Product: {product})"]
    for f in fields:
        lines.append(f"  - {f['name']}: {f['type']} - {f['description']}")
    return "\n".join(lines)
```

---

## 2. property_lookup.json

### Structure
```json
{
  "product_TypeName.field_name": {
    "product_name": "product",
    "type_name": "TypeName",
    "field_name": "field_name",
    "description": "Field description"
  }
}
```

### Primary Use Cases

#### A. Direct Field Lookup
Fast O(1) access to field metadata:

```python
def get_field_info(qualified_name: str) -> dict:
    return property_lookup.get(qualified_name, {})

# Example
info = get_field_info("concert_BuildArtifact.id")
# Returns: {"product_name": "concert", "type_name": "BuildArtifact", ...}
```

#### B. Embedding Generation for Similarity Search
Generate embeddings for all fields to enable semantic matching:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Build embedding corpus
field_texts = []
field_keys = []
for key, info in property_lookup.items():
    # Combine name and description for richer embeddings
    text = f"{info['type_name']}.{info['field_name']}: {info['description']}"
    field_texts.append(text)
    field_keys.append(key)

embeddings = model.encode(field_texts)
```

#### C. ChromaDB Vector Store Indexing
Index fields for semantic retrieval:

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.create_collection("schema_fields")

for key, info in property_lookup.items():
    collection.add(
        ids=[key],
        documents=[f"{info['field_name']}: {info['description']}"],
        metadatas=[{
            "product": info['product_name'],
            "type": info['type_name'],
            "field": info['field_name']
        }]
    )

# Query similar fields
results = collection.query(
    query_texts=["user identifier"],
    n_results=10
)
```

#### D. Mapping Candidate Retrieval
Find potential mapping targets:

```python
def find_candidates(source_field: str, target_product: str, top_k: int = 5) -> list:
    """Find candidate fields in target product that might map to source field."""
    source_info = property_lookup[source_field]
    
    # Filter to target product
    candidates = [
        (key, info) for key, info in property_lookup.items()
        if info['product_name'] == target_product
    ]
    
    # Score by semantic similarity (simplified)
    # In practice, use embeddings
    return candidates[:top_k]
```

---

## 3. Common Patterns

### Loading the Files
```python
import json

with open('data/hierarchy.json') as f:
    hierarchy = json.load(f)

with open('data/property_lookup.json') as f:
    property_lookup = json.load(f)
```

### Qualified Name Format
```
{product}_{TypeName}.{field_name}

Examples:
- concert_BuildArtifact.id
- instana_Entity.name  
- aws_AwsEc2Instance.tenancy
- turbonomic_Action.actionType
```

### Cross-Product Mapping Workflow
```python
def map_fields(source_product: str, target_product: str):
    """Map fields from source product to target product."""
    
    # 1. Get all source fields from hierarchy
    source_fields = []
    for type_name, fields in hierarchy[source_product].items():
        source_fields.extend(fields)
    
    # 2. For each source field, find candidates in target
    mappings = []
    for source in source_fields:
        # Use property_lookup for fast access to target fields
        candidates = [
            key for key, info in property_lookup.items()
            if info['product_name'] == target_product
        ]
        
        # 3. Score candidates (using embeddings, LLM, or rules)
        best_match = score_and_rank(source, candidates)
        
        if best_match:
            mappings.append({
                'source': source['name'],
                'target': best_match,
                'confidence': best_match['score']
            })
    
    return mappings
```

---

## 4. File Statistics

| Metric | Value |
|--------|-------|
| Products | 12 |
| Types | 1,679 |
| Fields | 11,031 |
| hierarchy.json size | 2.7 MB |
| property_lookup.json size | 2.5 MB |

### Per-Product Breakdown
| Product | Types | Fields |
|---------|-------|--------|
| instana | 1,128 | 6,842 |
| concert | 274 | 2,015 |
| cloudability | 40 | 1,169 |
| turbonomic | 59 | 393 |
| infragraph | 69 | 311 |
| watsonx-orchestrate | 57 | 218 |
| aws | 21 | 65 |
| ns1 | 13 | 59 |
| hcp_packer | 9 | 56 |
| hcp_terraform | 6 | 27 |
| azure | 2 | 6 |
| gcp | 1 | 1 |

---

## 5. Integration Points

### With LLM Agents
- **AnyTool Agent**: Uses hierarchy for tool functions (get_types, get_properties)
- **MapReflect Agent**: Uses property_lookup for semantic retrieval via ChromaDB

### With Embedding Models
- Generate embeddings from `{field_name}: {description}` text
- Store in vector database for similarity search

### With Evaluation
- Compare predicted mappings against ground truth
- Use qualified names as unique identifiers

