#!/usr/bin/env python3
"""
Generic OMOP Dataset Converter for Schema Mapping Evaluation.

Converts OMAP datasets (OMOP-MIMIC, OMOP-CMS, OMOP-Synthea) from api_mapping
format to graphql-schema-mapping evaluation format.

Usage:
    python convert_omop_generic.py [mimic|cms|synthea|all]
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Paths
API_MAPPING_DIR = Path("/Users/balaji/code/api_mapping/OMAP_data")
OUTPUT_DIR = Path("/Users/balaji/code/graphql-schema-mapping/data")

# Dataset configurations
DATASET_CONFIGS = {
    'mimic': {
        'source_db': 'omop',
        'target_db': 'mimic',
        'property_lookup': 'omop_mimic_property_lookup.json',
        'ground_truth': 'omop_mimic_ground_truth.json',
        'hierarchy': 'omop_mimic_hierarchy.json',
        'output_prefix': 'omop_mimic'
    },
    'cms': {
        'source_db': 'omop',
        'target_db': 'cms',
        'property_lookup': 'omop_cms_property_lookup.json',
        'ground_truth': 'omop_cms_ground_truth.json',
        'hierarchy': 'omop_cms_hierarchy.json',
        'output_prefix': 'omop_cms'
    },
    'synthea': {
        'source_db': 'omop',
        'target_db': 'synthea',
        'property_lookup': 'omop_synthea_property_lookup.json',
        'ground_truth': 'omop_synthea_ground_truth.json',
        'hierarchy': 'omop_synthea_hierarchy.json',
        'output_prefix': 'omop_synthea'
    }
}


def load_source_files(config):
    """Load the source files from api_mapping."""
    with open(API_MAPPING_DIR / config['property_lookup']) as f:
        property_lookup = json.load(f)

    with open(API_MAPPING_DIR / config['ground_truth']) as f:
        ground_truth = json.load(f)

    with open(API_MAPPING_DIR / config['hierarchy']) as f:
        hierarchy = json.load(f)

    return property_lookup, ground_truth, hierarchy


def convert_property_lookup(property_lookup):
    """Convert property lookup to ITOps format."""
    converted = {}

    for key, value in property_lookup.items():
        parts = key.split(".")
        if len(parts) >= 3:
            db_name = parts[0]
            table_name = parts[1]
            field_name = ".".join(parts[2:])

            new_key = f"{db_name}_{table_name}.{field_name}"

            converted[new_key] = {
                "product_name": db_name.upper(),
                "type_name": table_name,
                "field_name": field_name,
                "description": value.get("description", ""),
                "table_description": value.get("table_description", ""),
                "original_key": key
            }

    return converted


def convert_hierarchy(hierarchy):
    """Convert hierarchy to graphql-schema-mapping format."""
    converted = {}

    for db_name, tables in hierarchy.items():
        db_key = db_name.upper()
        converted[db_key] = {}

        for table_name, fields in tables.items():
            field_names = []
            table_desc = ""

            for field in fields:
                field_full_name = field.get("name", "")
                if "." in field_full_name:
                    field_name = field_full_name.split(".", 1)[1]
                else:
                    field_name = field_full_name
                field_names.append(field_name)

                if not table_desc:
                    table_desc = field.get("table_description", "")

            converted[db_key][table_name] = {
                "fields": field_names,
                "description": table_desc
            }

    return converted


def create_evaluation_dataset(property_lookup, ground_truth, converted_lookup, config):
    """Create evaluation dataset with utterances."""
    samples = []
    src_db = config['source_db']
    tgt_db = config['target_db']

    # Group ground truth by source table
    table_mappings = defaultdict(list)
    for source_field, target_fields in ground_truth.items():
        parts = source_field.split(".")
        if len(parts) >= 3:
            source_db = parts[0]
            source_table = parts[1]
            source_field_name = ".".join(parts[2:])

            for target_field in target_fields:
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
    prefix = config['output_prefix'].replace('_', '_')

    for (src_db_name, src_table, tgt_db_name, tgt_table), mappings in table_mappings.items():
        # Create utterance
        utterance = f"map {src_db_name} {src_table.lower().replace('_', ' ')} with {tgt_db_name} {tgt_table.lower().replace('_', ' ')}"

        # Get source properties
        source_properties = []
        for key, value in converted_lookup.items():
            if key.startswith(f"{src_db_name}_{src_table}."):
                source_properties.append({
                    "name": key,
                    "description": value.get("description", f"Field {key}"),
                    "metadata": {}
                })

        # Get target properties
        target_properties = []
        for key, value in converted_lookup.items():
            if key.startswith(f"{tgt_db_name}_{tgt_table}."):
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
            "id": f"{prefix}_{sample_id:03d}",
            "utterance": utterance,
            "source_schema": {
                "id": f"{src_db_name}_{src_table}",
                "name": f"{src_db_name.upper()} {src_table}",
                "description": f"Schema for {src_db_name.upper()} {src_table}",
                "metadata": {},
                "properties": source_properties
            },
            "target_schema": {
                "id": f"{tgt_db_name}_{tgt_table}",
                "name": f"{tgt_db_name.upper()} {tgt_table}",
                "description": f"Schema for {tgt_db_name.upper()} {tgt_table}",
                "metadata": {},
                "properties": target_properties
            },
            "mapping": mapping_dict
        }

        samples.append(sample)
        sample_id += 1

    return samples


def convert_dataset(dataset_name):
    """Convert a single OMOP dataset."""
    if dataset_name not in DATASET_CONFIGS:
        print(f"Unknown dataset: {dataset_name}")
        return False

    config = DATASET_CONFIGS[dataset_name]
    print(f"\n{'='*60}")
    print(f"Converting {dataset_name.upper()} dataset")
    print(f"{'='*60}")

    print("Loading source files from api_mapping...")
    try:
        property_lookup, ground_truth, hierarchy = load_source_files(config)
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        return False

    print(f"  Property lookup: {len(property_lookup)} properties")
    print(f"  Ground truth: {len(ground_truth)} mappings")
    print(f"  Hierarchy: {len(hierarchy)} databases")

    print("Converting property lookup...")
    converted_lookup = convert_property_lookup(property_lookup)
    print(f"  Converted: {len(converted_lookup)} properties")

    print("Converting hierarchy...")
    converted_hierarchy = convert_hierarchy(hierarchy)
    print(f"  Converted: {sum(len(tables) for tables in converted_hierarchy.values())} tables")

    print("Creating evaluation dataset...")
    samples = create_evaluation_dataset(property_lookup, ground_truth, converted_lookup, config)
    print(f"  Created: {len(samples)} samples")

    # Save outputs
    print("Saving output files...")
    prefix = config['output_prefix']

    # Property lookup
    lookup_path = OUTPUT_DIR / f"{prefix}_api_lookup.json"
    with open(lookup_path, "w") as f:
        json.dump(converted_lookup, f, indent=2)
    print(f"  Saved: {lookup_path}")

    # Hierarchy
    hier_path = OUTPUT_DIR / f"{prefix}_hierarchy.json"
    with open(hier_path, "w") as f:
        json.dump(converted_hierarchy, f, indent=2)
    print(f"  Saved: {hier_path}")

    # Evaluation dataset (JSONL format)
    dataset_path = OUTPUT_DIR / f"{prefix}_dataset.jsonl"
    with open(dataset_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print(f"  Saved: {dataset_path}")

    # Evaluation dataset (JSON format)
    dataset_name_upper = f"OMOP_{dataset_name.upper()}"
    eval_path = OUTPUT_DIR / f"{dataset_name_upper}.json"
    with open(eval_path, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"  Saved: {eval_path}")

    # Print statistics
    print(f"\n=== {dataset_name.upper()} Statistics ===")
    print(f"Total samples (table pairs): {len(samples)}")
    total_mappings = sum(len(s['mapping']) for s in samples)
    print(f"Total field mappings: {total_mappings}")

    # Show sample entry
    if samples:
        sample = samples[0]
        print(f"\nSample entry:")
        print(f"  ID: {sample['id']}")
        print(f"  Utterance: {sample['utterance']}")
        print(f"  Source: {sample['source_schema']['id']} ({len(sample['source_schema']['properties'])} properties)")
        print(f"  Target: {sample['target_schema']['id']} ({len(sample['target_schema']['properties'])} properties)")
        print(f"  Mappings: {len(sample['mapping'])}")

    return True


def main():
    if len(sys.argv) < 2:
        dataset = 'all'
    else:
        dataset = sys.argv[1].lower()

    if dataset == 'all':
        datasets = ['mimic', 'cms', 'synthea']
    else:
        datasets = [dataset]

    results = {}
    for ds in datasets:
        results[ds] = convert_dataset(ds)

    print("\n" + "="*60)
    print("Conversion Summary")
    print("="*60)
    for ds, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {ds.upper()}: {status}")


if __name__ == "__main__":
    main()
