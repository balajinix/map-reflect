# ToolBench REST API Schema Mapping Dataset

## Overview

This dataset contains **3,190 schema mapping samples** derived from ToolBench REST APIs. The task is to map parameters between REST APIs from different providers in the same domain.

## Task Description

Given a **source API** with its parameter schema, find the corresponding parameters in a **target API** from a different provider offering similar functionality.

**Example:**
- Source: `Foreca Weather / Nowcast` with params `[location, tempunit, windunit, ...]`
- Target: `Cloud Cast / Get current weather` with params `[city]`
- Mapping: `location → city`

## Dataset Statistics

| Category | Samples | Description |
|----------|---------|-------------|
| Sports | 725 | Sports data, scores, player stats APIs |
| Finance | 700 | Stock data, crypto, financial APIs |
| Weather | 435 | Weather forecast, air quality APIs |
| Social | 338 | Social media platform APIs |
| Media | 221 | Video, audio, streaming APIs |
| eCommerce | 193 | Product, shopping, marketplace APIs |
| Entertainment | 193 | Games, movies, trivia APIs |
| Music | 120 | Music streaming, lyrics APIs |
| Location | 89 | Geocoding, maps, places APIs |
| Travel | 81 | Flights, hotels, transport APIs |
| News_Media | 36 | News aggregation APIs |
| Food | 24 | Restaurant, nutrition APIs |
| Financial | 20 | Banking, payments APIs |
| Health_and_Fitness | 15 | Health tracking APIs |
| **Total** | **3,190** | |

## Data Format

### JSON Format (`toolbench_schema_mapping.json`)

```json
{
  "description": "REST API schema mapping dataset from ToolBench",
  "task": "Given a source API schema, find corresponding parameters in target API schema",
  "total_samples": 3190,
  "categories": ["Weather", "Finance", ...],
  "samples": [
    {
      "category": "Weather",
      "source_api": {
        "tool_name": "Foreca Weather",
        "api_name": "Nowcast",
        "api_description": "3-hour forecast in 15-minute time steps...",
        "category": "Weather",
        "method": "GET",
        "parameters": [
          {
            "name": "location",
            "type": "string",
            "description": "",
            "required": true,
            "default": ""
          },
          ...
        ]
      },
      "target_api": {
        "tool_name": "Cloud Cast",
        "api_name": "Get humidty",
        ...
      },
      "ground_truth_mappings": {
        "location": "city"
      },
      "mapping_count": 1
    },
    ...
  ]
}
```

### JSONL Format (`toolbench_dataset.jsonl`)

Each line is a JSON object with the same structure as above (without the top-level wrapper).

## Parameter Types

Parameters have the following types:
- `STRING` - Text values
- `NUMBER` - Numeric values
- `BOOLEAN` - True/false values
- `OBJECT` - Nested objects
- `DATE` - Date values (various formats)

## Ground Truth Mappings

Mappings are created based on:
1. **Exact name matches** (case-insensitive)
2. **Semantic equivalences** including:
   - Location: `lat`↔`latitude`, `lon`↔`lng`↔`longitude`, `city`↔`location`↔`q`
   - Time: `start`↔`from`↔`startDate`, `end`↔`to`↔`endDate`
   - Identifiers: `id`↔`uuid`↔`identifier`
   - Pagination: `page`↔`pageNum`, `limit`↔`count`↔`size`↔`perPage`
   - Units: `units`↔`tempunit`, `currency`↔`curr`
   - Language: `lang`↔`language`↔`locale`

## Source Data

This dataset is derived from [ToolBench](https://github.com/OpenBMB/ToolBench):
- `G1_instruction.json` - 200 samples
- `G2_instruction.json` - 200 samples
- `G3_instruction.json` - 100 samples

Total: **2,024 unique APIs** across **43 categories** from **~600 providers**.

## Usage

### Loading the Dataset

```python
import json

# Load JSON format
with open('data/toolbench_schema_mapping.json') as f:
    dataset = json.load(f)

samples = dataset['samples']
print(f"Total samples: {len(samples)}")

# Load JSONL format
samples = []
with open('data/toolbench_dataset.jsonl') as f:
    for line in f:
        samples.append(json.loads(line))
```

### Running Evaluation

```bash
# Run Map-Reflect on ToolBench dataset
python map_reflect/map_reflect_runner.py \
    --datasets ToolBench \
    --models watsonx/ibm/granite-3-3-8b-instruct
```

## Key Differences from Database Schema Mapping

| Aspect | Database Schema | REST API Schema |
|--------|-----------------|-----------------|
| Schema Source | Database tables/columns | API parameters |
| Context | Foreign key relationships | API descriptions |
| Cardinality | 1:1 column mapping | 1:1 or 1:N param mapping |
| Semantics | Table/column names | Parameter names + descriptions |
| Structure | Flat or nested tables | Flat or nested JSON |

## Citation

If you use this dataset, please cite:

```bibtex
@misc{toolbench_schema_mapping,
  title={ToolBench REST API Schema Mapping Dataset},
  author={Based on ToolBench by Qin et al.},
  year={2024},
  note={Derived from ToolBench for schema mapping research}
}

@article{qin2023toolllm,
  title={ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs},
  author={Qin, Yujia and others},
  journal={arXiv preprint arXiv:2307.16789},
  year={2023}
}
```
