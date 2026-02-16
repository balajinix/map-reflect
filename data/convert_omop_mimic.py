#!/usr/bin/env python3
"""
Convert OMOP-MIMIC dataset from api_mapping format to graphql-schema-mapping evaluation format.

Input files (from api_mapping/OMAP_data/):
- omop_mimic_property_lookup.json
- omop_mimic_ground_truth.json
- omop_mimic_hierarchy.json

Output files:
- omop_mimic_api_lookup.json (property lookup in ITOps format)
- omop_mimic_hierarchy.json (schema hierarchy)
- OMOP_MIMIC.json (evaluation dataset with utterances)
"""

import json
import os
from pathlib import Path
from collections import defaultdict

# Paths
API_MAPPING_DIR = Path("/Users/balaji/code/api_mapping/OMAP_data")
OUTPUT_DIR = Path("/Users/balaji/code/graphql-schema-mapping/data")

def load_source_files():
    """Load the source files from api_mapping."""
    with open(API_MAPPING_DIR / "omop_mimic_property_lookup.json") as f:
        property_lookup = json.load(f)

    with open(API_MAPPING_DIR / "omop_mimic_ground_truth.json") as f:
        ground_truth = json.load(f)

    with open(API_MAPPING_DIR / "omop_mimic_hierarchy.json") as f:
        hierarchy = json.load(f)

    return property_lookup, ground_truth, hierarchy


def convert_property_lookup(property_lookup):
    """
    Convert property lookup to ITOps format.

    Input format:
    {
        "omop.PERSON.person_id": {
            "database_name": "omop",
            "table_name": "PERSON",
            "field_name": "PERSON.person_id",
            "description": "...",
            ...
        }
    }

    Output format (ITOps-style):
    {
        "omop_PERSON.person_id": {
            "product_name": "OMOP",
            "type_name": "PERSON",
            "field_name": "person_id",
            "description": "..."
        }
    }
    """
    converted = {}

    for key, value in property_lookup.items():
        # Parse the key: "omop.PERSON.person_id" -> "omop_PERSON.person_id"
        parts = key.split(".")
        if len(parts) >= 3:
            db_name = parts[0]  # omop or mimic
            table_name = parts[1]  # PERSON, VISIT_OCCURRENCE, etc.
            field_name = ".".join(parts[2:])  # person_id or nested.field

            # Create new key in ITOps format
            new_key = f"{db_name}_{table_name}.{field_name}"

            converted[new_key] = {
                "product_name": db_name.upper(),  # OMOP or MIMIC
                "type_name": table_name,
                "field_name": field_name,
                "description": value.get("description", ""),
                "table_description": value.get("table_description", ""),
                "original_key": key
            }

    return converted


def convert_hierarchy(hierarchy):
    """
    Convert hierarchy to graphql-schema-mapping format.

    Input format:
    {
        "omop": {
            "PERSON": [{"name": "PERSON.person_id", "description": "..."}],
            ...
        },
        "mimic": {...}
    }

    Output format:
    {
        "OMOP": {
            "PERSON": {
                "fields": ["person_id", "gender_concept_id", ...],
                "description": "..."
            }
        },
        "MIMIC": {...}
    }
    """
    converted = {}

    for db_name, tables in hierarchy.items():
        db_key = db_name.upper()
        converted[db_key] = {}

        for table_name, fields in tables.items():
            field_names = []
            table_desc = ""

            for field in fields:
                field_full_name = field.get("name", "")
                # Extract just the field name (after the table prefix)
                if "." in field_full_name:
                    field_name = field_full_name.split(".", 1)[1]
                else:
                    field_name = field_full_name
                field_names.append(field_name)

                # Get table description from first field
                if not table_desc:
                    table_desc = field.get("table_description", "")

            converted[db_key][table_name] = {
                "fields": field_names,
                "description": table_desc
            }

    return converted


def create_evaluation_dataset(property_lookup, ground_truth, converted_lookup):
    """
    Create evaluation dataset with utterances.

    Each sample represents a mapping task:
    - Source: OMOP table
    - Target: MIMIC table
    - Utterance: "map OMOP [table] with MIMIC [table]"
    - Mapping: ground truth field mappings
    """
    samples = []

    # Group ground truth by source table
    table_mappings = defaultdict(list)
    for source_field, target_fields in ground_truth.items():
        # Parse source: "omop.PERSON.person_id"
        parts = source_field.split(".")
        if len(parts) >= 3:
            source_db = parts[0]
            source_table = parts[1]
            source_field_name = ".".join(parts[2:])

            for target_field in target_fields:
                # Parse target: "mimic.PATIENTS.mimic_id"
                target_parts = target_field.split(".")
                if len(target_parts) >= 3:
                    target_db = target_parts[0]
                    target_table = target_parts[1]
                    target_field_name = ".".join(target_parts[2:])

                    key = (source_db, source_table, target_db, target_table)
                    table_mappings[key].append({
                        "source": f"{source_db}_{source_table}.{source_field_name}",
                        "target": f"{target_db}_{target_table}.{target_field_name}",
                        "source_original": source_field,
                        "target_original": target_field
                    })

    # Create samples for each table pair
    sample_id = 1
    for (src_db, src_table, tgt_db, tgt_table), mappings in table_mappings.items():
        # Create retrieval-style utterance - model must find matching OMOP tables
        utterance = f"find OMOP tables that map to {src_db} {src_table.lower().replace('_', ' ')}"

        # Get source properties (all fields from source table)
        source_properties = []
        for key, value in converted_lookup.items():
            if key.startswith(f"{src_db}_{src_table}."):
                source_properties.append({
                    "name": key,
                    "description": value.get("description", f"Field {key}"),
                    "metadata": {}
                })

        # Get target properties (all fields from target table)
        target_properties = []
        for key, value in converted_lookup.items():
            if key.startswith(f"{tgt_db}_{tgt_table}."):
                target_properties.append({
                    "name": key,
                    "description": value.get("description", f"Field {key}"),
                    "metadata": {}
                })

        # Create ground truth mapping dict
        mapping_dict = {}
        for m in mappings:
            mapping_dict[m["source"]] = m["target"]

        sample = {
            "id": f"omop_mimic_{sample_id:03d}",
            "utterance": utterance,
            "source_schema": {
                "id": f"{src_db}_{src_table}",
                "name": f"{src_db.upper()} {src_table}",
                "description": f"Schema for {src_db.upper()} {src_table}",
                "metadata": {},
                "properties": source_properties
            },
            "target_schema": {
                "id": f"{tgt_db}_{tgt_table}",
                "name": f"{tgt_db.upper()} {tgt_table}",
                "description": f"Schema for {tgt_db.upper()} {tgt_table}",
                "metadata": {},
                "properties": target_properties
            },
            "mapping": mapping_dict
        }

        samples.append(sample)
        sample_id += 1

    return samples


def main():
    print("Loading source files from api_mapping...")
    property_lookup, ground_truth, hierarchy = load_source_files()

    print(f"  Property lookup: {len(property_lookup)} properties")
    print(f"  Ground truth: {len(ground_truth)} mappings")
    print(f"  Hierarchy: {len(hierarchy)} databases")

    print("\nConverting property lookup...")
    converted_lookup = convert_property_lookup(property_lookup)
    print(f"  Converted: {len(converted_lookup)} properties")

    print("\nConverting hierarchy...")
    converted_hierarchy = convert_hierarchy(hierarchy)
    print(f"  Converted: {sum(len(tables) for tables in converted_hierarchy.values())} tables")

    print("\nCreating evaluation dataset...")
    samples = create_evaluation_dataset(property_lookup, ground_truth, converted_lookup)
    print(f"  Created: {len(samples)} samples")

    # Save outputs
    print("\nSaving output files...")

    # Property lookup
    lookup_path = OUTPUT_DIR / "omop_mimic_api_lookup.json"
    with open(lookup_path, "w") as f:
        json.dump(converted_lookup, f, indent=2)
    print(f"  Saved: {lookup_path}")

    # Hierarchy
    hier_path = OUTPUT_DIR / "omop_mimic_hierarchy.json"
    with open(hier_path, "w") as f:
        json.dump(converted_hierarchy, f, indent=2)
    print(f"  Saved: {hier_path}")

    # Evaluation dataset (JSONL format)
    dataset_path = OUTPUT_DIR / "omop_mimic_dataset.jsonl"
    with open(dataset_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print(f"  Saved: {dataset_path}")

    # Evaluation dataset (JSON format for evaluation script)
    eval_path = OUTPUT_DIR / "OMOP_MIMIC.json"
    with open(eval_path, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"  Saved: {eval_path}")

    # Print sample statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total samples (table pairs): {len(samples)}")
    total_mappings = sum(len(s['mapping']) for s in samples)
    print(f"Total field mappings: {total_mappings}")

    # Show sample
    print("\n=== Sample Entry ===")
    if samples:
        sample = samples[0]
        print(f"ID: {sample['id']}")
        print(f"Utterance: {sample['utterance']}")
        print(f"Source schema: {sample['source_schema']['id']} ({len(sample['source_schema']['properties'])} properties)")
        print(f"Target schema: {sample['target_schema']['id']} ({len(sample['target_schema']['properties'])} properties)")
        print(f"Mappings: {len(sample['mapping'])}")
        for src, tgt in list(sample['mapping'].items())[:3]:
            print(f"  {src} -> {tgt}")


if __name__ == "__main__":
    main()
