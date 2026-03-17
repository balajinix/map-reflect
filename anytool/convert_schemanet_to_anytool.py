#!/usr/bin/env python3
"""
Convert SchemaNet data to AnyTool format.

AnyTool expects:
- tool_data.json: {category: {tool: {api_list_names: [...]}}}
- api_details.json: {category: {tool: {api_list: [{name, description, ...}]}}}
- category_tool_details.json: {category: {description, tools: {tool: {description}}}}
- Query file: [{query_id, query, gt_api_list: [{category_name, tool_name, api_name}]}]

For SchemaNet:
- Categories = target schema name (e.g., "omop", "bank2")
- Tools = target types (e.g., "person", "branch")
- APIs = target fields (e.g., "person_id", "branch_name")
- Query = source field description with task to find matching target field
"""

import json
import os
from pathlib import Path


def convert_schemanet_to_anytool(dataset_name, schemanet_dir, output_dir):
    """Convert a SchemaNet dataset to AnyTool format."""

    # Load SchemaNet data
    api_lookup_file = Path(schemanet_dir) / f"{dataset_name}_api_lookup.json"
    dataset_file = Path(schemanet_dir) / f"{dataset_name}_dataset.jsonl"

    with open(api_lookup_file) as f:
        api_lookup = json.load(f)

    samples = []
    with open(dataset_file) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    # Extract target schema name
    target_schema = samples[0]['target_schema']

    # Build AnyTool database structure
    # Category = target_schema (e.g., "omop")
    # Tool = type name (e.g., "person")
    # API = field name (e.g., "person_id")

    tool_data = {target_schema: {}}
    api_details = {target_schema: {}}
    category_tool_details = {
        target_schema: {
            "description": f"Schema {target_schema} containing database tables and fields",
            "tools": {}
        }
    }

    for type_id, type_data in api_lookup.items():
        type_name = type_data.get('name', type_id.replace(f'{target_schema}_', ''))
        type_desc = type_data.get('description', f'Table {type_name}')

        # Tool data
        tool_data[target_schema][type_name] = {
            "api_list_names": []
        }

        # API details
        api_details[target_schema][type_name] = {
            "tool_name": type_name,
            "tool_description": type_desc,
            "api_list": []
        }

        # Category tool details
        category_tool_details[target_schema]["tools"][type_name] = {
            "description": type_desc
        }

        # Add fields as APIs
        for field_name, field_data in type_data.get('fields', {}).items():
            field_desc = field_data.get('description', f'Field {field_name}')
            field_type = field_data.get('type', 'unknown')

            tool_data[target_schema][type_name]["api_list_names"].append(field_name)

            api_details[target_schema][type_name]["api_list"].append({
                "name": field_name,
                "description": f"{field_desc} (type: {field_type})",
                "method": "GET",
                "required_parameters": [],
                "optional_parameters": []
            })

    # Create query file from samples
    queries = []
    query_id = 0

    for sample in samples:
        source_schema = sample['source_schema']

        for source_field, target_field in sample.get('mapping', {}).items():
            # Parse source field
            src_parts = source_field.split('.')
            src_type = src_parts[0].replace(f'{source_schema}_', '')
            src_field_name = src_parts[1] if len(src_parts) > 1 else src_parts[0]

            # Parse target field
            tgt_parts = target_field.split('.')
            tgt_type = tgt_parts[0].replace(f'{target_schema}_', '')
            tgt_field_name = tgt_parts[1] if len(tgt_parts) > 1 else tgt_parts[0]

            # Create query
            query_text = f"Find the field in {target_schema} schema that corresponds to {src_field_name} from {src_type} table in {source_schema} schema. The source field is {source_field}."

            queries.append({
                "query_id": str(query_id),
                "query": query_text,
                "source_field": source_field,
                "expected_target_field": target_field,
                "gt_api_list": [
                    {
                        "category_name": target_schema,
                        "tool_name": tgt_type,
                        "api_name": tgt_field_name
                    }
                ]
            })
            query_id += 1

    # Save files
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'tool_data.json'), 'w') as f:
        json.dump(tool_data, f, indent=2)

    with open(os.path.join(output_dir, 'api_details.json'), 'w') as f:
        json.dump(api_details, f, indent=2)

    with open(os.path.join(output_dir, 'category_tool_details.json'), 'w') as f:
        json.dump(category_tool_details, f, indent=2)

    with open(os.path.join(output_dir, f'{dataset_name}_queries.json'), 'w') as f:
        json.dump(queries, f, indent=2)

    print(f"Converted {dataset_name}:")
    print(f"  Categories: {len(tool_data)}")
    print(f"  Tools (types): {sum(len(tools) for tools in tool_data.values())}")
    print(f"  APIs (fields): {sum(len(t['api_list_names']) for cat in tool_data.values() for t in cat.values())}")
    print(f"  Queries: {len(queries)}")
    print(f"  Output: {output_dir}")

    return queries


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    script_dir = Path(__file__).resolve().parent.parent
    parser.add_argument('--schemanet-dir', type=str, default=str(script_dir / 'data' / 'schemanet'))
    parser.add_argument('--output-dir', type=str, default=str(script_dir / 'data' / 'schemanet' / 'anytool_format'))
    args = parser.parse_args()

    convert_schemanet_to_anytool(
        args.dataset,
        args.schemanet_dir,
        args.output_dir
    )


if __name__ == "__main__":
    main()
