#!/usr/bin/env python3
"""
Convert OMOP hierarchy JSON files to GraphQL schema format for SMT/ReAct.

This script reads the OMOP hierarchy JSON files and generates GraphQL schema
files that can be used by SMT and ReAct methods.
"""

import json
import os
import re
from pathlib import Path


def sanitize_name(name: str) -> str:
    """Convert name to valid GraphQL identifier."""
    # Remove dots and replace with underscore
    name = name.replace(".", "_")
    # Remove any invalid characters
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # Ensure doesn't start with number
    if name and name[0].isdigit():
        name = "_" + name
    return name


def escape_description(desc: str) -> str:
    """Escape description for GraphQL comments."""
    if not desc:
        return ""
    # Replace quotes and newlines
    desc = desc.replace('"', '\\"').replace('\n', ' ').strip()
    # Truncate if too long
    if len(desc) > 500:
        desc = desc[:497] + "..."
    return desc


def convert_hierarchy_to_graphql(hierarchy: dict, source_db: str) -> str:
    """Convert OMOP hierarchy structure to GraphQL schema string."""
    lines = []
    query_fields = []

    # Process each database (usually 'omop' and source like 'cms', 'mimic', 'synthea')
    for db_name, tables in hierarchy.items():
        for table_name, fields in tables.items():
            # Create type name: db_tablename (e.g., omop_PERSON, cms_BENEFICIARY)
            type_name = f"{db_name}_{sanitize_name(table_name)}"

            # Get table description from first field if available
            table_desc = ""
            if fields and len(fields) > 0:
                table_desc = fields[0].get("table_description", "")

            # Start type definition
            if table_desc:
                lines.append(f'"""{escape_description(table_desc)}"""')
            lines.append(f"type {type_name} {{")

            # Add fields
            for field in fields:
                field_name = field.get("name", "")
                # Extract just the field part (after the dot)
                if "." in field_name:
                    field_name = field_name.split(".")[-1]
                field_name = sanitize_name(field_name)

                if not field_name:
                    continue

                field_desc = field.get("field_description", field.get("description", ""))
                field_type = field.get("type", "String") or "String"

                # Map common types to GraphQL types
                type_mapping = {
                    "": "String",
                    "int": "Int",
                    "integer": "Int",
                    "float": "Float",
                    "double": "Float",
                    "boolean": "Boolean",
                    "bool": "Boolean",
                    "date": "String",
                    "datetime": "String",
                    "timestamp": "String",
                }
                graphql_type = type_mapping.get(field_type.lower(), "String")

                if field_desc:
                    lines.append(f'  "{escape_description(field_desc)}"')
                lines.append(f"  {field_name}: {graphql_type}")

            lines.append("}")
            lines.append("")

            # Add query field for this type
            query_fields.append(f"  {table_name.lower()}List: [{type_name}]")
            query_fields.append(f"  {table_name.lower()}(id: ID!): {type_name}")

    # Add Query type
    lines.append("type Query {")
    lines.extend(query_fields)
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def process_omop_dataset(data_dir: Path, dataset_name: str, output_dir: Path):
    """Process a single OMOP dataset and generate GraphQL schema."""
    hierarchy_file = data_dir / f"{dataset_name}_hierarchy.json"

    if not hierarchy_file.exists():
        print(f"Hierarchy file not found: {hierarchy_file}")
        return False

    print(f"Processing {dataset_name}...")

    with open(hierarchy_file, 'r') as f:
        hierarchy = json.load(f)

    # Extract source database name (e.g., 'cms', 'mimic', 'synthea')
    source_db = dataset_name.replace("omop_", "")

    # Convert to GraphQL
    graphql_schema = convert_hierarchy_to_graphql(hierarchy, source_db)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write single combined schema file
    output_file = output_dir / f"{dataset_name}.graphql"
    with open(output_file, 'w') as f:
        f.write(graphql_schema)

    print(f"  Created: {output_file}")

    # Also create per-table files for better indexing
    for db_name, tables in hierarchy.items():
        for table_name, fields in tables.items():
            type_name = f"{db_name}_{sanitize_name(table_name)}"

            # Create individual schema for this table
            table_lines = []
            table_desc = fields[0].get("table_description", "") if fields else ""

            if table_desc:
                table_lines.append(f'"""{escape_description(table_desc)}"""')
            table_lines.append(f"type {type_name} {{")

            for field in fields:
                field_name = field.get("name", "")
                if "." in field_name:
                    field_name = field_name.split(".")[-1]
                field_name = sanitize_name(field_name)

                if not field_name:
                    continue

                field_desc = field.get("field_description", field.get("description", ""))

                if field_desc:
                    table_lines.append(f'  "{escape_description(field_desc)}"')
                table_lines.append(f"  {field_name}: String")

            table_lines.append("}")
            table_lines.append("")
            table_lines.append("type Query {")
            table_lines.append(f"  {table_name.lower()}List: [{type_name}]")
            table_lines.append(f"  {table_name.lower()}(id: ID!): {type_name}")
            table_lines.append("}")

            table_file = output_dir / f"{db_name}_{table_name}.graphql"
            with open(table_file, 'w') as f:
                f.write("\n".join(table_lines))

    return True


def main():
    """Main entry point."""
    # Paths - use api_mapping OMAP_data which has the detailed hierarchy format
    script_dir = Path(__file__).parent
    source_dir = Path("/Users/balaji/code/api_mapping/OMAP_data")
    output_base = script_dir

    # Process each OMOP dataset
    datasets = ["omop_cms", "omop_mimic", "omop_synthea"]

    for dataset in datasets:
        output_dir = output_base / f"{dataset}_graphql"
        success = process_omop_dataset(source_dir, dataset, output_dir)
        if success:
            print(f"Successfully converted {dataset} to GraphQL format")
            print(f"  Output directory: {output_dir}")
        else:
            print(f"Failed to convert {dataset}")

    print("\nDone! GraphQL schemas created for OMOP datasets.")
    print("You can now run SMT/ReAct experiments with these schemas.")


if __name__ == "__main__":
    main()
