#!/usr/bin/env python3
"""
Create mapping files for GNN training from OMOP ground truth data.
Converts OMOP ground truth format to the format expected by populate_nx.py.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

API_MAPPING_DIR = Path("/Users/balaji/code/api_mapping/OMAP_data")
OUTPUT_DIR = Path("/Users/balaji/code/graphql-schema-mapping/data")

DATASET_CONFIGS = {
    'mimic': {
        'ground_truth': 'omop_mimic_ground_truth.json',
        'output_dir': 'omop_mimic_mappings'
    },
    'cms': {
        'ground_truth': 'omop_cms_ground_truth.json',
        'output_dir': 'omop_cms_mappings'
    },
    'synthea': {
        'ground_truth': 'omop_synthea_ground_truth.json',
        'output_dir': 'omop_synthea_mappings'
    }
}


def create_mapping_files(dataset_name):
    """Create mapping files from ground truth for GNN training."""
    if dataset_name not in DATASET_CONFIGS:
        print(f"Unknown dataset: {dataset_name}")
        return False

    config = DATASET_CONFIGS[dataset_name]

    # Load ground truth
    gt_file = API_MAPPING_DIR / config['ground_truth']
    print(f"Loading ground truth from: {gt_file}")

    with open(gt_file) as f:
        ground_truth = json.load(f)

    print(f"  Loaded {len(ground_truth)} mappings")

    # Group by source table
    # Format: {source_table: {source_field: [(target_table, target_field)]}}
    table_mappings = defaultdict(lambda: defaultdict(list))

    for source_key, target_keys in ground_truth.items():
        # source_key: "omop.PERSON.person_id"
        parts = source_key.split(".")
        if len(parts) >= 3:
            source_db = parts[0]  # omop
            source_table = parts[1]  # PERSON
            source_field = ".".join(parts[2:])  # person_id

            for target_key in target_keys:
                # target_key: "mimic.PATIENTS.mimic_id"
                target_parts = target_key.split(".")
                if len(target_parts) >= 3:
                    target_db = target_parts[0]  # mimic
                    target_table = target_parts[1]  # PATIENTS
                    target_field = ".".join(target_parts[2:])  # mimic_id

                    # Store mapping
                    table_key = f"{source_db}_{source_table}"
                    field_key = f"{source_table}.{source_field}"
                    target_table_key = f"{target_db}_{target_table}"
                    target_field_key = f"{target_table}.{target_field}"

                    table_mappings[table_key][field_key].append({
                        "mapTable": target_table_key,
                        "mapProperty": target_field_key,
                        "justification": f"Field mapping: {source_field} maps to {target_field}"
                    })

    # Create output directory
    output_dir = OUTPUT_DIR / config['output_dir']
    output_dir.mkdir(exist_ok=True)

    # Write mapping files (one per source table)
    files_created = 0
    for table_name, field_mappings in table_mappings.items():
        mapping_data = {
            "name": table_name,
            "properties": dict(field_mappings)
        }

        output_file = output_dir / f"{table_name}_mapping.json"
        with open(output_file, 'w') as f:
            json.dump(mapping_data, f, indent=2)

        files_created += 1
        print(f"  Created: {output_file.name} ({len(field_mappings)} fields)")

    print(f"\nCreated {files_created} mapping files in {output_dir}")
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

    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"Creating mapping files for {ds.upper()}")
        print(f"{'='*60}")
        create_mapping_files(ds)


if __name__ == "__main__":
    main()
