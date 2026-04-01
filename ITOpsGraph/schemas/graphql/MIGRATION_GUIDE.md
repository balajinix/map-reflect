# Migration Guide: Using New solis-graphql-schemas

This document details the changes required in the **api_mapping** and **graphql-schema-mapping** repositories to use the new schemas from `solis-graphql-schemas`.

---

## Architecture Overview

```
┌─────────────────────────────────┐
│  solis-graphql-schemas          │  Source of truth (93 files, 72,253 lines)
│  (StepZen GraphQL format)       │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  api_mapping                    │  Converts GraphQL → JSON
│  - subschema_selector.py        │
│  - toolkit_utils.py             │
│  - convert_schema_toolkit_*.py  │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  JSON Output Files              │
│  - hierarchy.json               │  Grouped by product/type
│  - api_lookup.json              │  Flat property index
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  graphql-schema-mapping         │  Schema mapping experiments
│  - smt/                         │
│  - gnn/                         │
│  - match_maker/                 │
│  - react/                       │
└─────────────────────────────────┘
```

---

## Key Changes in New Schemas

### 1. Product Names Changed

| Old Name | New Name | Notes |
|----------|----------|-------|
| `concert_*` | `concert/openapi/...` | Nested directory structure |
| `turbo_*` | `turbo/...` | Simplified structure |
| `instana_*` | `instana/openapi/...` | Nested with events subdir |
| `apptio_*` | `apptio/cloudability/...` | Extensive reporting module |
| `wxo_*` | `watsonx-orchestrate/...` | Renamed directory |
| (none) | `ns1/...` | **NEW PRODUCT** |

### 2. StepZen Directives

The new schemas use StepZen-specific directives that the parser must handle:

```graphql
# @sdl - Schema composition directive
schema @sdl(files: ["file1.graphql", "file2.graphql"]) {
  query: Query
}

# @rest - REST endpoint mapping
field: Type @rest(
  endpoint: "https://api.example.com/$id"
  configuration: "configName"
  headers: [{ name: "authorization", value: "bearer $token" }]
)

# @sequence - Multi-step query execution
query @sequence(
  steps: [
    { query: "_accessToken" }
    { query: "_actualQuery", arguments: [...] }
  ]
)

# @tool - LLM tool metadata
schema @tool(
  name: "tool-name"
  description: "Tool description"
  graphql: [{ expose: true, types: "Query", fields: "..." }]
)
```

### 3. Directory Structure

```
solis-graphql-schemas/
└── dev/products/
    ├── apptio/cloudability/
    │   ├── index.graphql              # @sdl composition file
    │   ├── budgets/
    │   ├── clusters/
    │   ├── recommendations/
    │   └── reporting/
    │       ├── llm_*.graphql          # LLM-friendly schemas
    │       └── operations/            # Sample queries
    ├── concert/
    │   ├── openapi/schema/            # Main schemas
    │   └── postgres/                  # DB schemas
    ├── instana/
    │   ├── openapi/schema/
    │   │   ├── applications/
    │   │   └── infrastructure/
    │   └── events/schema/
    ├── ns1/pulsar/                    # NEW PRODUCT
    ├── turbo/
    │   ├── entities/
    │   ├── markets/
    │   └── search/
    └── watsonx-orchestrate/
        ├── chats/
        ├── skillsets/
        └── threads/
```

---

## Changes Required in api_mapping

### 1. Update `src/utils/subschema_selector.py`

#### Current Limitations
The current regex patterns don't handle:
- `@rest` directive with complex arguments
- `@sequence` directive
- `@sdl` composition directives
- Triple-quoted descriptions (`"""..."""`)
- Arguments with `$variable` interpolation

#### Required Updates

```python
# Add new patterns for StepZen directives
STEPZEN_DIRECTIVE_PATTERNS = {
    'rest': re.compile(r'@rest\s*\(([^)]+)\)', re.DOTALL),
    'sequence': re.compile(r'@sequence\s*\(([^)]+)\)', re.DOTALL),
    'sdl': re.compile(r'@sdl\s*\(([^)]+)\)', re.DOTALL),
    'tool': re.compile(r'@tool\s*\(([^)]+)\)', re.DOTALL),
}

# Update field extraction to strip directives
def extract_field_with_directive(line):
    """Extract field name and type, ignoring directives."""
    # Remove directives before parsing
    clean_line = re.sub(r'@\w+\s*\([^)]*\)', '', line)
    # ... existing field parsing logic
```

#### Handle Underscore-Prefixed Fields

StepZen uses `_fieldName` for internal implementation fields (authentication, etc.):

```python
def is_public_field(field_name):
    """Check if field should be exposed in API."""
    return not field_name.startswith('_')

def filter_public_fields(fields):
    """Filter to only public API fields."""
    return [f for f in fields if is_public_field(f['name'])]
```

### 2. Update `src/utils/toolkit_utils.py`

#### Update Product Detection

```python
PRODUCT_MAPPING = {
    'apptio': 'Apptio',
    'cloudability': 'Apptio',
    'concert': 'Concert',
    'instana': 'Instana',
    'turbo': 'Turbonomic',
    'ns1': 'NS1',           # NEW
    'pulsar': 'NS1',        # NEW
    'watsonx-orchestrate': 'Watson Orchestrate',
    'wxo': 'Watson Orchestrate',
}

def detect_product(file_path):
    """Detect product from file path."""
    path_lower = str(file_path).lower()
    for key, product in PRODUCT_MAPPING.items():
        if key in path_lower:
            return product
    return 'Unknown'
```

#### Handle Nested Directory Structure

```python
def get_schema_component_name(file_path, type_name):
    """Generate schema component name from path and type.

    Example:
        file_path: apptio/cloudability/reporting/llm_outputs.graphql
        type_name: CostReport
        → apptio_cloudability_CostReport
    """
    parts = Path(file_path).parts
    # Find product directory index
    products = ['apptio', 'concert', 'instana', 'ns1', 'turbo', 'watsonx-orchestrate']
    for i, part in enumerate(parts):
        if part in products:
            # Include subdirectory for context
            prefix_parts = parts[i:i+2]  # e.g., ['apptio', 'cloudability']
            prefix = '_'.join(prefix_parts)
            return f"{prefix}_{type_name}"
    return type_name
```

### 3. Create New Converter Scripts

#### `src/convert_schema_toolkit_apptio.py`

```python
#!/usr/bin/env python3
"""Convert Apptio/Cloudability schemas to JSON format."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from utils.subschema_selector import parse_graphql_schema
from utils.toolkit_utils import build_fields, traverse, generate_json

SCHEMA_DIR = Path("../solis-graphql-schemas/dev/products/apptio")
OUTPUT_DIR = Path("output/toolkit_data")

def convert_apptio_schemas():
    """Convert all Apptio schemas."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all .graphql files (excluding index.graphql composition files)
    schema_files = [
        f for f in SCHEMA_DIR.rglob("*.graphql")
        if f.name != "index.graphql"
    ]

    for schema_file in schema_files:
        # Skip build directories
        if '/build/' in str(schema_file):
            continue

        print(f"Processing: {schema_file}")

        parsed = parse_graphql_schema(schema_file)
        # ... generate JSON

if __name__ == "__main__":
    convert_apptio_schemas()
```

#### Create converters for each product:
- `convert_schema_toolkit_concert.py`
- `convert_schema_toolkit_instana.py`
- `convert_schema_toolkit_turbo.py`
- `convert_schema_toolkit_ns1.py` (NEW)
- `convert_schema_toolkit_wxo.py`

### 4. Update `src/utils/itops.py`

#### Update `create_hierarchical_mapping()`

```python
def create_hierarchical_mapping(json_files_dir):
    """Create hierarchy.json with new product structure."""

    hierarchy = {
        'apptio': {},      # Apptio/Cloudability
        'concert': {},
        'instana': {},
        'ns1': {},         # NEW PRODUCT
        'turbo': {},       # Note: 'turbo' not 'turbonomic'
        'wxo': {},         # Watson Orchestrate
    }

    # ... process files
```

---

## Changes Required in graphql-schema-mapping

### 1. Update `smt/smt_schema_indexer.py`

#### Update Domain Mappings

```python
# Line 75-79: Update domain_mappings
self.domain_mappings = {
    "turbo": ["performance", "optimization", "virtual", "entity", "market"],
    "instana": ["monitoring", "application", "performance", "infrastructure", "topology"],
    "concert": ["security", "vulnerability", "scanning", "compliance", "application"],
    "apptio": ["cost", "budget", "cloud", "financial", "recommendation", "reporting"],
    "ns1": ["dns", "traffic", "pulsar", "performance", "routing"],  # NEW
    "wxo": ["chat", "skill", "thread", "agent", "orchestrate"],
}
```

### 2. Update `gnn/gnn_schema_indexer.py`

#### Update Domain Keywords (Line 215-219)

```python
domain_mappings = {
    "turbo": ["performance", "optimization", "virtual", "disk", "storage", "entity", "market", "statistics", "metrics"],
    "instana": ["monitoring", "infrastructure", "application", "performance", "apm", "observability", "topology", "entities"],
    "concert": ["security", "vulnerability", "scanning", "assessment", "compliance", "risk", "applications"],
    "apptio": ["cost", "financial", "cloud", "billing", "optimization", "analysis", "budget", "reporting"],
    "ns1": ["dns", "traffic", "routing", "performance", "pulsar"],  # NEW
    "wxo": ["chat", "conversation", "skill", "automation", "agent"],
}
```

### 3. Update `match_maker/match_maker_agent.py`

#### Update Product Name Parsing (Line 193-208)

```python
def set_source_target_from_utterance(self, utterance: str):
    """Parse utterance to identify source/target schemas."""

    # Updated product list
    products = [
        'Apptio', 'Cloudability',
        'Concert',
        'Instana',
        'NS1', 'Pulsar',  # NEW
        'Turbonomic', 'Turbo',
        'Watson Orchestrate', 'WXO',
    ]

    prompt = f"""Parse this schema mapping request...
Available products in the system: {', '.join(products)}
...
"""
```

### 4. Update `data/itops_hierarchy.json`

Regenerate using updated api_mapping with new product structure:

```json
{
    "apptio": {
        "CostReport": [...],
        "Budget": [...],
        "Cluster": [...],
        "Recommendation": [...]
    },
    "concert": {
        "Application": [...],
        "Vulnerability": [...],
        "Exposure": [...]
    },
    "instana": {
        "Application": [...],
        "Infrastructure": [...],
        "Event": [...]
    },
    "ns1": {
        "PulsarJob": [...],
        "Performance": [...]
    },
    "turbo": {
        "Entity": [...],
        "Market": [...],
        "Action": [...]
    },
    "wxo": {
        "Chat": [...],
        "Skill": [...],
        "Thread": [...]
    }
}
```

### 5. Update `data/itops_api_lookup.json`

Regenerate with new naming convention:

```json
{
    "apptio_cloudability_CostReport.total_cost": {
        "product_name": "Apptio",
        "type_name": "CostReport",
        "field_name": "total_cost",
        "description": "..."
    },
    "ns1_pulsar_Job.status": {
        "product_name": "NS1",
        "type_name": "Job",
        "field_name": "status",
        "description": "..."
    }
}
```

---

## Migration Steps

### Step 1: Update api_mapping Repository

1. Copy new schemas to api_mapping's graphql directory:
```bash
cp -r solis-graphql-schemas/dev/products/* api_mapping/graphql/schema/
```

2. Update `subschema_selector.py` to handle StepZen directives

3. Update `toolkit_utils.py` for nested directory structure

4. Create/update converter scripts for each product

5. Run conversion:
```bash
cd api_mapping
python src/convert_schema_toolkit_apptio.py
python src/convert_schema_toolkit_concert.py
python src/convert_schema_toolkit_instana.py
python src/convert_schema_toolkit_turbo.py
python src/convert_schema_toolkit_ns1.py
python src/convert_schema_toolkit_wxo.py
```

6. Generate hierarchy and lookup files:
```bash
python src/utils/itops.py
```

### Step 2: Update graphql-schema-mapping Repository

1. Copy generated JSON files:
```bash
cp api_mapping/output/itops_hierarchy.json graphql-schema-mapping/data/
cp api_mapping/output/itops_api_lookup.json graphql-schema-mapping/data/
```

2. Update domain mappings in:
   - `smt/smt_schema_indexer.py`
   - `gnn/gnn_schema_indexer.py`
   - `match_maker/match_maker_agent.py`

3. Rebuild ChromaDB indexes:
```bash
cd graphql-schema-mapping
python smt/smt_schema_indexer.py --schema-dir data/itops_graphql --force-reindex
```

4. Re-run experiments:
```bash
python eval_schema_mapping.py --datasets ITOps
```

---

## New Data Volume Comparison

| Product | Old (graphql-schema-mapping) | New (solis-graphql-schemas) | Change |
|---------|------------------------------|------------------------------|--------|
| Apptio | 224 lines, 3 types | **6,470 lines, ~50 types** | 29x |
| Concert | 1,390 lines | **18,017 lines** | 13x |
| Instana | 4,955 lines | **44,643 lines** | 9x |
| Turbo | 7,775 lines | 1,526 lines | 0.2x |
| WXO | 430 lines | **1,026 lines** | 2.4x |
| NS1 | (none) | **571 lines** | NEW |
| **Total** | ~14,774 lines | **72,253 lines** | **4.9x** |

Note: Turbonomic has fewer lines because the previous source included extensive recovered schemas that may have been outdated.

---

## Testing After Migration

### 1. Verify JSON Generation

```python
# Test that hierarchy.json has all products
import json
with open('data/itops_hierarchy.json') as f:
    hierarchy = json.load(f)

expected_products = ['apptio', 'concert', 'instana', 'ns1', 'turbo', 'wxo']
for product in expected_products:
    assert product in hierarchy, f"Missing product: {product}"
    print(f"{product}: {len(hierarchy[product])} types")
```

### 2. Verify Schema Indexing

```bash
python gnn/gnn_schema_indexer.py \
    --schema-dir data/itops_graphql \
    --test-query "find cost optimization recommendations" \
    --verbose
```

### 3. Run Sample Mapping

```bash
python match_maker/match_maker_runner.py \
    --utterance "map apptio cost reports to instana metrics" \
    --verbose
```

---

## Potential Issues

### 1. StepZen Directive Parsing
The StepZen directives use complex nested syntax. The regex parser may need multiple passes or a proper GraphQL parser library.

### 2. Internal Fields
Fields prefixed with `_` are internal to StepZen (authentication, etc.) and should be filtered from public API representations.

### 3. Product Name Consistency
Ensure consistent naming between:
- Directory names (`turbo` vs `turbonomic`)
- API lookup keys (`turbo_Entity.id`)
- Domain mappings in experiments

### 4. Description Generation
The new schemas may have different documentation patterns. Update RAG pipelines accordingly.
