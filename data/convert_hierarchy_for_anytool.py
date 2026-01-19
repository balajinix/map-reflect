#!/usr/bin/env python3
"""Convert OMOP hierarchy file format for AnyTool compatibility.

Our format:
{
  "OMOP": {
    "PERSON": {
      "fields": [{"name": "field1", ...}],
      "description": "..."
    }
  }
}

AnyTool expects:
{
  "OMOP": {
    "PERSON": [{"name": "field1", ...}]  # Just the fields list
  }
}
"""

import json
import sys
from pathlib import Path

def convert_hierarchy(input_file: str, output_file: str = None):
    """Convert hierarchy file format for AnyTool."""
    with open(input_file) as f:
        data = json.load(f)

    converted = {}
    for product_name, product_data in data.items():
        converted[product_name] = {}
        if isinstance(product_data, dict):
            for type_name, type_data in product_data.items():
                if isinstance(type_data, dict) and 'fields' in type_data:
                    # Convert from {fields: [...], description: "..."} to just [...]
                    converted[product_name][type_name] = type_data['fields']
                elif isinstance(type_data, list):
                    # Already in the right format
                    converted[product_name][type_name] = type_data
                else:
                    print(f"Warning: Unexpected type_data format for {product_name}.{type_name}: {type(type_data)}")
                    converted[product_name][type_name] = []

    if output_file is None:
        # Default output: replace .json with _anytool.json
        output_file = input_file.replace('.json', '_anytool.json')

    with open(output_file, 'w') as f:
        json.dump(converted, f, indent=2)

    print(f"Converted {input_file} -> {output_file}")
    return output_file

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_hierarchy_for_anytool.py <input_file> [output_file]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    convert_hierarchy(input_file, output_file)
