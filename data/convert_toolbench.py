#!/usr/bin/env python3
"""
Convert ToolBench dataset to standard schema mapping format.
Creates both the dataset file and lookup file for experiments.
"""

import json
from pathlib import Path
from collections import defaultdict


def convert_toolbench_dataset():
    """Convert ToolBench to standard format."""
    input_file = Path(__file__).parent / "toolbench_dataset.jsonl"
    output_file = Path(__file__).parent / "toolbench_eval_dataset.jsonl"
    lookup_file = Path(__file__).parent / "toolbench_api_lookup.json"

    samples = []
    lookup = {}

    with open(input_file) as f:
        for i, line in enumerate(f):
            data = json.loads(line)

            source_api = data.get("source_api", {})
            target_api = data.get("target_api", {})
            mappings = data.get("ground_truth_mappings", {})
            category = data.get("category", "")

            # Create source schema ID
            source_tool = source_api.get("tool_name", "").replace(" ", "_")
            source_name = source_api.get("api_name", "").replace(" ", "_")
            source_id = f"{source_tool}_{source_name}"

            # Create target schema ID
            target_tool = target_api.get("tool_name", "").replace(" ", "_")
            target_name = target_api.get("api_name", "").replace(" ", "_")
            target_id = f"{target_tool}_{target_name}"

            # Add source parameters to lookup
            for param in source_api.get("parameters", []):
                param_name = param.get("name", "")
                full_name = f"{source_id}.{param_name}"
                lookup[full_name] = {
                    "name": full_name,
                    "description": param.get("description", "") or f"Parameter {param_name} of {source_api.get('api_name', '')}",
                    "type": param.get("type", "STRING"),
                    "required": param.get("required", False),
                    "api": source_api.get("api_name", ""),
                    "tool": source_api.get("tool_name", ""),
                    "category": category
                }

            # Add target parameters to lookup
            for param in target_api.get("parameters", []):
                param_name = param.get("name", "")
                full_name = f"{target_id}.{param_name}"
                lookup[full_name] = {
                    "name": full_name,
                    "description": param.get("description", "") or f"Parameter {param_name} of {target_api.get('api_name', '')}",
                    "type": param.get("type", "STRING"),
                    "required": param.get("required", False),
                    "api": target_api.get("api_name", ""),
                    "tool": target_api.get("tool_name", ""),
                    "category": category
                }

            # Convert ground truth mappings to use full parameter names
            converted_mappings = {}
            for src_param, tgt_param in mappings.items():
                src_full = f"{source_id}.{src_param}"
                tgt_full = f"{target_id}.{tgt_param}"
                converted_mappings[src_full] = tgt_full

            # Create utterance
            utterance = f"Map parameters from {source_api.get('tool_name', '')} {source_api.get('api_name', '')} to {target_api.get('tool_name', '')} {target_api.get('api_name', '')}"

            # Create target schema with properties
            target_properties = []
            for param in target_api.get("parameters", []):
                param_name = param.get("name", "")
                full_name = f"{target_id}.{param_name}"
                target_properties.append({
                    "name": full_name,
                    "description": param.get("description", "") or f"Parameter {param_name}"
                })

            sample = {
                "sample_id": f"toolbench_{i:04d}",
                "utterance": utterance,
                "category": category,
                "expected_source": source_id,
                "expected_target": {
                    "id": target_id,
                    "name": target_id,
                    "description": target_api.get("api_description", ""),
                    "properties": target_properties
                },
                "ground_truth_mapping": converted_mappings,
                "source_api": source_api,
                "target_api": target_api
            }
            samples.append(sample)

    # Write converted dataset
    with open(output_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    print(f"Converted {len(samples)} samples to {output_file}")

    # Write lookup file
    with open(lookup_file, 'w') as f:
        json.dump(lookup, f, indent=2)

    print(f"Created lookup with {len(lookup)} parameters in {lookup_file}")

    # Print category distribution
    categories = defaultdict(int)
    for s in samples:
        categories[s["category"]] += 1

    print("\nCategory distribution:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    convert_toolbench_dataset()
