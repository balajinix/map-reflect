#!/usr/bin/env python3
import json

input_file = "itops_dataset_raw.jsonl"
output_file = "itops_dataset.jsonl"

def extract_schemas(result_name, properties):
    # Source schema is the prefix before "__" if present
    source_schema = result_name.split("__")[0] if "__" in result_name else result_name

    # Target schema: pick the first mapTable from properties
    target_schema = ""
    for mappings in properties.values():
        if mappings and isinstance(mappings, list):
            target_schema = mappings[0].get("mapTable", "")
            break

    return source_schema, target_schema

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for idx, line in enumerate(infile):
        data = json.loads(line.strip())
        utterance = data.get("utterance", "")
        result = data.get("result", {})
        result_name = result.get("name", "")
        properties = result.get("properties", {})

        # Extract source and target schemas
        source_schema, target_schema = extract_schemas(result_name, properties)

        # Build flat mapping dict: src_field → first mapProperty
        mapping = {}
        for src_field, mappings in properties.items():
            if mappings and isinstance(mappings, list):
                mapping[src_field] = mappings[0].get("mapProperty", "")

        # Construct output line
        converted = {
            "id": data.get("id", f"itops_{idx+1:03d}"),
            "utterance": utterance,
            "source_schema": source_schema,
            "target_schema": target_schema,
            "mapping": mapping,
            "metadata": data.get("metadata", {})
        }

        outfile.write(json.dumps(converted) + "\n")

print(f"✅ Converted unified format to raw JSONL: {output_file}")
