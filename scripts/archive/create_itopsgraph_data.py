#!/usr/bin/env python3
"""
Create ITOpsGraph dataset files from nl2cypher repository.

This script:
1. Converts GraphQL and Proto schemas to .graphql files
2. Generates itopsgraph_api_lookup.json
3. Generates itopsgraph_hierarchy.json

Usage:
    python scripts/create_itopsgraph_data.py --nl2cypher-dir /path/to/nl2cypher
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


def load_graphql_schema(filepath: Path) -> Dict[str, Any]:
    """Load a GraphQL schema JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_proto_schema(filepath: Path) -> Dict[str, Any]:
    """Load a proto-derived schema JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def type_to_graphql(type_data: Dict[str, Any], product: str) -> str:
    """Convert a type definition to GraphQL SDL format."""
    kind = type_data.get('kind', 'OBJECT')
    name = type_data.get('name', '')
    description = type_data.get('description', '')
    fields = type_data.get('fields', [])

    lines = []

    # Add description if present
    if description:
        lines.append(f'"""{description}"""')

    if kind == 'ENUM':
        lines.append(f'enum {name} {{')
        for field in fields:
            field_name = field.get('name', '')
            field_desc = field.get('description', '')
            if field_desc:
                lines.append(f'  """{field_desc}"""')
            lines.append(f'  {field_name}')
        lines.append('}')

    elif kind in ('OBJECT', 'MESSAGE', 'INPUT_OBJECT'):
        type_keyword = 'input' if kind == 'INPUT_OBJECT' else 'type'
        implements = type_data.get('implements')
        implements_str = f' implements {" & ".join(implements)}' if implements else ''

        lines.append(f'{type_keyword} {name}{implements_str} {{')
        for field in fields:
            field_name = field.get('name', '')
            field_type = field.get('type', 'String')
            field_desc = field.get('description', '')
            is_list = field.get('is_list', False)
            is_required = field.get('is_required', False)

            # Format the type
            if is_list:
                formatted_type = f'[{field_type}]'
            else:
                formatted_type = field_type

            if is_required:
                formatted_type = f'{formatted_type}!'

            if field_desc:
                lines.append(f'  """{field_desc}"""')
            lines.append(f'  {field_name}: {formatted_type}')
        lines.append('}')

    elif kind == 'INTERFACE':
        lines.append(f'interface {name} {{')
        for field in fields:
            field_name = field.get('name', '')
            field_type = field.get('type', 'String')
            is_list = field.get('is_list', False)
            is_required = field.get('is_required', False)

            if is_list:
                formatted_type = f'[{field_type}]'
            else:
                formatted_type = field_type

            if is_required:
                formatted_type = f'{formatted_type}!'

            lines.append(f'  {field_name}: {formatted_type}')
        lines.append('}')

    elif kind == 'SCALAR':
        lines.append(f'scalar {name}')

    elif kind == 'UNION':
        possible_types = type_data.get('possibleTypes', [])
        if possible_types:
            lines.append(f'union {name} = {" | ".join(possible_types)}')
        else:
            lines.append(f'union {name}')

    return '\n'.join(lines)


def convert_schemas_to_graphql(
    nl2cypher_dir: Path,
    output_dir: Path
) -> Dict[str, Dict[str, Any]]:
    """
    Convert all schemas to .graphql files.

    Returns: Dictionary of {product: {type_name: type_data}}
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_schemas = {}

    # 1. Process GraphQL schema JSON files
    graphql_schemas_dir = nl2cypher_dir / 'data'
    for schema_file in graphql_schemas_dir.glob('graphql_*_schema.json'):
        print(f"Processing GraphQL schema: {schema_file.name}")
        schema_data = load_graphql_schema(schema_file)

        product = schema_data.get('product', schema_file.stem.replace('graphql_', '').replace('_schema', ''))
        types = schema_data.get('types', [])

        all_schemas[product] = {}

        for type_data in types:
            type_name = type_data.get('name', '')
            if not type_name or type_name.startswith('__'):  # Skip introspection types
                continue

            all_schemas[product][type_name] = type_data

            # Write individual .graphql file
            graphql_content = type_to_graphql(type_data, product)
            output_file = output_dir / f"{product}_{type_name}.graphql"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(graphql_content)

        print(f"  Extracted {len(all_schemas[product])} types from {product}")

    # 2. Process proto-derived JSON schemas
    proto_schema_file = nl2cypher_dir / 'data' / 'parsed_proto_schema.json'
    if proto_schema_file.exists():
        print(f"Processing proto schema: {proto_schema_file.name}")
        schema_data = load_proto_schema(proto_schema_file)

        product = schema_data.get('product', 'infragraph')
        types = schema_data.get('types', [])

        # Group by message_source if available
        for type_data in types:
            type_name = type_data.get('name', '')
            if not type_name:
                continue

            # Determine product from message_source or type prefix
            message_source = (type_data.get('message_source') or '').lower()
            if message_source in ('aws', 'azure', 'gcp'):
                sub_product = message_source
            elif type_name.startswith('Aws'):
                sub_product = 'aws'
            elif type_name.startswith('Azure'):
                sub_product = 'azure'
            elif type_name.startswith('Gcp'):
                sub_product = 'gcp'
            elif type_name.startswith('Hcp'):
                sub_product = 'hcp'
            else:
                sub_product = 'infragraph'

            if sub_product not in all_schemas:
                all_schemas[sub_product] = {}

            all_schemas[sub_product][type_name] = type_data

            # Write individual .graphql file
            graphql_content = type_to_graphql(type_data, sub_product)
            output_file = output_dir / f"{sub_product}_{type_name}.graphql"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(graphql_content)

        print(f"  Extracted types into sub-products from proto schema")

    # 3. Process individual proto JSON schemas
    proto_json_dir = nl2cypher_dir / 'schemas' / 'json'
    if proto_json_dir.exists():
        for schema_file in proto_json_dir.glob('*_schema.json'):
            print(f"Processing proto JSON: {schema_file.name}")
            schema_data = load_proto_schema(schema_file)

            product = schema_data.get('product', schema_file.stem.replace('_schema', ''))
            types = schema_data.get('types', [])

            if product not in all_schemas:
                all_schemas[product] = {}

            for type_data in types:
                type_name = type_data.get('name', '')
                if not type_name:
                    continue

                all_schemas[product][type_name] = type_data

                # Write individual .graphql file
                graphql_content = type_to_graphql(type_data, product)
                output_file = output_dir / f"{product}_{type_name}.graphql"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(graphql_content)

            print(f"  Extracted {len(types)} types from {product}")

    return all_schemas


def create_api_lookup(
    all_schemas: Dict[str, Dict[str, Any]],
    output_file: Path
) -> Dict[str, Dict[str, Any]]:
    """
    Create API lookup JSON file.

    Format: {
        "product_TypeName.fieldName": {
            "product_name": "product",
            "type_name": "TypeName",
            "field_name": "fieldName",
            "description": "...",
            "field_type": "String"
        }
    }
    """
    api_lookup = {}

    for product, types in all_schemas.items():
        for type_name, type_data in types.items():
            fields = type_data.get('fields', [])
            type_description = type_data.get('description', '')

            for field in fields:
                field_name = field.get('name', '')
                if not field_name:
                    continue

                # Create lookup key
                key = f"{product}_{type_name}.{field_name}"

                api_lookup[key] = {
                    "product_name": product,
                    "type_name": type_name,
                    "field_name": field_name,
                    "full_name": f"{type_name}.{field_name}",
                    "description": field.get('description', ''),
                    "field_type": field.get('type', 'String'),
                    "is_list": field.get('is_list', False),
                    "is_required": field.get('is_required', False),
                    "type_description": type_description
                }

    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(api_lookup, f, indent=2)

    print(f"Created API lookup with {len(api_lookup)} entries: {output_file}")
    return api_lookup


def create_hierarchy(
    all_schemas: Dict[str, Dict[str, Any]],
    output_file: Path
) -> Dict[str, Any]:
    """
    Create hierarchy JSON file.

    Format: {
        "product": {
            "types": ["Type1", "Type2"],
            "description": "...",
            "type_count": 10,
            "field_count": 100
        }
    }
    """
    hierarchy = {}

    # Product descriptions
    product_descriptions = {
        "cloudability": "FinOps cost optimization and cloud spend management",
        "concert": "Concert IT Operations management platform",
        "instana": "Application performance monitoring (APM)",
        "ns1": "DNS and traffic management",
        "turbonomic": "Application resource management and optimization",
        "watsonx-orchestrate": "IBM Watson AI orchestration and automation",
        "aws": "Amazon Web Services infrastructure entities",
        "azure": "Microsoft Azure cloud resources",
        "gcp": "Google Cloud Platform resources",
        "hcp": "HashiCorp Cloud Platform",
        "hcp_terraform": "HashiCorp Terraform configuration types",
        "hcp_packer": "HashiCorp Packer image building",
        "infragraph": "Infrastructure graph base types",
        "base": "Base infrastructure types",
        "edge": "Edge computing resources",
        "model": "Data model base types",
        "resource": "Generic resource types",
        "packer": "HashiCorp Packer types"
    }

    for product, types in sorted(all_schemas.items()):
        type_names = sorted(types.keys())
        field_count = sum(len(t.get('fields', [])) for t in types.values())

        hierarchy[product] = {
            "types": type_names,
            "description": product_descriptions.get(product, f"Schema for {product}"),
            "type_count": len(type_names),
            "field_count": field_count
        }

    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(hierarchy, f, indent=2)

    print(f"Created hierarchy with {len(hierarchy)} products: {output_file}")
    return hierarchy


def main():
    parser = argparse.ArgumentParser(
        description='Create ITOpsGraph dataset files from nl2cypher repository'
    )
    parser.add_argument(
        '--nl2cypher-dir',
        type=str,
        default='/Users/balaji/code/solis-dal-dataset/nl2cypher',
        help='Path to nl2cypher repository'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Output directory for generated files'
    )

    args = parser.parse_args()

    nl2cypher_dir = Path(args.nl2cypher_dir)
    output_dir = Path(args.output_dir)
    graphql_dir = output_dir / 'itopsgraph_graphql'

    print(f"Source: {nl2cypher_dir}")
    print(f"Output: {output_dir}")
    print()

    # Step 1: Convert schemas to .graphql files
    print("=" * 60)
    print("Phase 1.1: Converting schemas to GraphQL files")
    print("=" * 60)
    all_schemas = convert_schemas_to_graphql(nl2cypher_dir, graphql_dir)

    total_types = sum(len(types) for types in all_schemas.values())
    print(f"\nTotal: {len(all_schemas)} products, {total_types} types")
    print(f"GraphQL files written to: {graphql_dir}")
    print()

    # Step 2: Create API lookup
    print("=" * 60)
    print("Phase 1.2: Creating API lookup file")
    print("=" * 60)
    api_lookup_file = output_dir / 'itopsgraph_api_lookup.json'
    api_lookup = create_api_lookup(all_schemas, api_lookup_file)
    print()

    # Step 3: Create hierarchy
    print("=" * 60)
    print("Phase 1.3: Creating hierarchy file")
    print("=" * 60)
    hierarchy_file = output_dir / 'itopsgraph_hierarchy.json'
    hierarchy = create_hierarchy(all_schemas, hierarchy_file)
    print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Products: {len(all_schemas)}")
    for product, types in sorted(all_schemas.items()):
        field_count = sum(len(t.get('fields', [])) for t in types.values())
        print(f"  {product}: {len(types)} types, {field_count} fields")
    print(f"\nTotal API entries: {len(api_lookup)}")
    print(f"\nFiles created:")
    print(f"  - {graphql_dir}/ ({total_types} .graphql files)")
    print(f"  - {api_lookup_file}")
    print(f"  - {hierarchy_file}")


if __name__ == '__main__':
    main()
