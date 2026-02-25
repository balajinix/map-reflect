#!/usr/bin/env python3
"""Convert OMOP hierarchy file format for AnyTool compatibility.

AnyTool agent (anytool_agent.py) expects properties as OBJECTS with name/description/type:
{
  "OMOP": {
    "PERSON": [
      {"name": "person_id", "description": "Primary key", "type": "INTEGER"},
      {"name": "gender_concept_id", "description": "Gender concept", "type": "INTEGER"}
    ]
  }
}

But our hierarchy files have properties as STRINGS:
{
  "OMOP": {
    "PERSON": ["person_id", "gender_concept_id", ...]
  }
}

This script converts string lists to object lists, enriching with descriptions from API lookup.
"""

import json
import sys
from pathlib import Path

def convert_hierarchy(input_file: str, output_file: str = None, api_lookup_file: str = None):
    """Convert hierarchy file format for AnyTool.

    Args:
        input_file: Input hierarchy JSON file
        output_file: Output file (defaults to input_anytool_v2.json)
        api_lookup_file: Optional API lookup file to get field descriptions
    """
    with open(input_file) as f:
        data = json.load(f)

    # Load API lookup for descriptions if available
    api_lookup = {}
    if api_lookup_file:
        try:
            with open(api_lookup_file) as f:
                api_data = json.load(f)
                for prop in api_data.get('properties', []):
                    name = prop.get('name', '').lower()
                    if name:
                        api_lookup[name] = prop
        except Exception as e:
            print(f"Warning: Could not load API lookup: {e}")

    converted = {}
    for product_name, product_data in data.items():
        converted[product_name] = {}
        if isinstance(product_data, dict):
            for type_name, type_data in product_data.items():
                # Get field list based on input format
                if isinstance(type_data, dict) and 'fields' in type_data:
                    fields = type_data['fields']
                    type_desc = type_data.get('description', '')
                elif isinstance(type_data, list):
                    fields = type_data
                    type_desc = ''
                else:
                    print(f"Warning: Unexpected type_data format for {product_name}.{type_name}: {type(type_data)}")
                    fields = []
                    type_desc = ''

                # Convert each field to object format
                field_objects = []
                for field in fields:
                    if isinstance(field, dict):
                        # Already an object
                        field_objects.append(field)
                    elif isinstance(field, str):
                        # String - convert to object with description
                        # Look up description in API lookup
                        lookup_key = f"{product_name.lower()}_{type_name.lower()}.{field}".lower()
                        field_info = api_lookup.get(lookup_key, {})
                        if not field_info:
                            field_info = api_lookup.get(field.lower(), {})

                        field_objects.append({
                            "name": field,
                            "description": field_info.get('description', f"{field} field in {type_name}"),
                            "type": field_info.get('type', 'STRING')
                        })

                converted[product_name][type_name] = field_objects

    if output_file is None:
        # Default output: replace .json with _anytool_v2.json
        output_file = input_file.replace('.json', '_anytool_v2.json')

    with open(output_file, 'w') as f:
        json.dump(converted, f, indent=2)

    print(f"Converted {input_file} -> {output_file}")

    # Print summary
    for prod in converted:
        total_fields = sum(len(converted[prod][t]) for t in converted[prod])
        print(f"  {prod}: {len(converted[prod])} types, {total_fields} fields")

    return output_file


def convert_all_omop():
    """Convert all OMOP hierarchy files."""
    data_dir = Path(__file__).parent

    conversions = [
        ('omop_cms_hierarchy.json', 'omop_cms_hierarchy_anytool_v2.json', 'omop_cms_api_lookup.json'),
        ('omop_mimic_hierarchy.json', 'omop_mimic_hierarchy_anytool_v2.json', 'omop_mimic_api_lookup.json'),
        ('omop_synthea_hierarchy.json', 'omop_synthea_hierarchy_anytool_v2.json', 'omop_synthea_api_lookup.json'),
    ]

    for input_name, output_name, lookup_name in conversions:
        input_file = data_dir / input_name
        output_file = data_dir / output_name
        lookup_file = data_dir / lookup_name

        if input_file.exists():
            convert_hierarchy(
                str(input_file),
                str(output_file),
                str(lookup_file) if lookup_file.exists() else None
            )
        else:
            print(f"Skipping {input_name} - file not found")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python convert_hierarchy_for_anytool.py <input_file> [output_file] [api_lookup_file]")
        print("  python convert_hierarchy_for_anytool.py --all  # Convert all OMOP files")
        sys.exit(1)

    if sys.argv[1] == '--all':
        convert_all_omop()
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        api_lookup = sys.argv[3] if len(sys.argv) > 3 else None
        convert_hierarchy(input_file, output_file, api_lookup)
