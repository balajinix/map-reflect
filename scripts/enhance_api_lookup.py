#!/usr/bin/env python3
"""
Enhance the itopsgraph_api_lookup.json with descriptions from GraphQL schemas.

Uses the graphql_extractor from nl2cypher to get richer type and field descriptions.
"""

import json
import sys
from pathlib import Path

# Add nl2cypher to path
NL2CYPHER_PATH = Path("/Users/balaji/code/solis-dal-dataset/nl2cypher")
sys.path.insert(0, str(NL2CYPHER_PATH))

from wiki.docs.graphql_extractor import build_product_descriptions

DATA_DIR = Path(__file__).parent.parent / "data"
API_LOOKUP_PATH = DATA_DIR / "itopsgraph_api_lookup.json"
# Note: build_product_descriptions expects the parent of 'graphql/' directory
GRAPHQL_SCHEMA_DIR = NL2CYPHER_PATH / "schemas"  # Not schemas/graphql!
ALL_PRODUCTS_SCHEMA = NL2CYPHER_PATH / "data" / "all_products_schema.json"


def load_all_products_schema() -> dict:
    """Load the all_products_schema.json file."""
    if ALL_PRODUCTS_SCHEMA.exists():
        with open(ALL_PRODUCTS_SCHEMA) as f:
            return json.load(f)
    return {}


def main():
    # Load current api_lookup
    print(f"Loading API lookup from {API_LOOKUP_PATH}...")
    with open(API_LOOKUP_PATH) as f:
        api_lookup = json.load(f)
    print(f"  Loaded {len(api_lookup)} entries")

    # Count current descriptions
    initial_with_desc = sum(1 for v in api_lookup.values() if v.get("description"))
    initial_with_type_desc = sum(1 for v in api_lookup.values() if v.get("type_description"))
    print(f"  Fields with descriptions: {initial_with_desc}")
    print(f"  Fields with type descriptions: {initial_with_type_desc}")

    # Load GraphQL descriptions
    print(f"\nLoading GraphQL descriptions from {GRAPHQL_SCHEMA_DIR}...")
    graphql_docs = build_product_descriptions(GRAPHQL_SCHEMA_DIR)
    total_types = sum(len(p) for p in graphql_docs.values())
    print(f"  Loaded {total_types} types from {len(graphql_docs)} products")

    # Load schema for additional type descriptions
    print(f"\nLoading all_products_schema from {ALL_PRODUCTS_SCHEMA}...")
    schema_data = load_all_products_schema()
    if "products" in schema_data:
        print(f"  Loaded {len(schema_data['products'])} products")

    # Enhance api_lookup
    print("\nEnhancing API lookup...")
    enhanced_count = 0
    type_desc_added = 0

    for key, value in api_lookup.items():
        product = value.get("product_name", "")
        type_name = value.get("type_name", "")
        field_name = value.get("field_name", "")

        # Try to add field description
        if not value.get("description") or value.get("description") == field_name:
            # Try GraphQL docs
            if product in graphql_docs and type_name in graphql_docs[product]:
                type_info = graphql_docs[product][type_name]
                if isinstance(type_info, dict) and "fields" in type_info:
                    if field_name in type_info["fields"]:
                        desc = type_info["fields"][field_name]
                        if desc and desc != field_name:
                            value["description"] = desc
                            enhanced_count += 1

            # Try schema data
            if not value.get("description") or value.get("description") == field_name:
                if "products" in schema_data:
                    product_data = schema_data["products"].get(product, {})
                    for t in product_data.get("types", []):
                        if t.get("name") == type_name:
                            for f in t.get("fields", []):
                                if f.get("name") == field_name:
                                    desc = f.get("description", "")
                                    if desc and desc != field_name:
                                        value["description"] = desc
                                        enhanced_count += 1
                                    break
                            break

        # Try to add type description
        if not value.get("type_description"):
            # Try GraphQL docs
            if product in graphql_docs and type_name in graphql_docs[product]:
                type_info = graphql_docs[product][type_name]
                if isinstance(type_info, dict) and type_info.get("description"):
                    value["type_description"] = type_info["description"]
                    type_desc_added += 1
                elif isinstance(type_info, str) and type_info:
                    value["type_description"] = type_info
                    type_desc_added += 1

            # Try schema data
            if not value.get("type_description"):
                if "products" in schema_data:
                    product_data = schema_data["products"].get(product, {})
                    for t in product_data.get("types", []):
                        if t.get("name") == type_name:
                            desc = t.get("description", "")
                            if desc:
                                value["type_description"] = desc
                                type_desc_added += 1
                            break

    print(f"  Enhanced {enhanced_count} field descriptions")
    print(f"  Added {type_desc_added} type descriptions")

    # Count final descriptions
    final_with_desc = sum(1 for v in api_lookup.values() if v.get("description"))
    final_with_type_desc = sum(1 for v in api_lookup.values() if v.get("type_description"))
    print(f"\nFinal counts:")
    print(f"  Fields with descriptions: {final_with_desc} (was {initial_with_desc})")
    print(f"  Fields with type descriptions: {final_with_type_desc} (was {initial_with_type_desc})")

    # Save enhanced lookup
    output_path = API_LOOKUP_PATH  # Overwrite original
    print(f"\nSaving enhanced lookup to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(api_lookup, f, indent=2)
    print("Done!")

    # Show sample enhanced entry
    print("\n" + "="*60)
    print("Sample enhanced entries:")
    print("="*60)

    count = 0
    for key, value in api_lookup.items():
        if value.get("description") and value.get("type_description"):
            print(f"\nKey: {key}")
            print(f"  Type: {value.get('type_name')}")
            print(f"  Field: {value.get('field_name')}")
            print(f"  Type desc: {value.get('type_description', '')[:100]}...")
            print(f"  Field desc: {value.get('description', '')[:100]}...")
            count += 1
            if count >= 3:
                break


if __name__ == "__main__":
    main()
