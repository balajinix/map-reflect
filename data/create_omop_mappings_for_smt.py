#!/usr/bin/env python3
"""
Create OMOP mapping files for SMT from ground truth.

SMT expects mapping files in data/mappings_output/ with format:
{
    "name": "source_schema",
    "properties": {
        "SourceType.field": [
            {
                "mapTable": "target_schema",
                "mapProperty": "TargetType.field",
                "justification": "explanation"
            }
        ]
    }
}

Filename: source_schema-target_schema.json

This script also creates dataset-specific mapping directories for cleaner SMT indexing.
"""

import json
from pathlib import Path
from collections import defaultdict


def load_ground_truth(filepath: Path) -> dict:
    """Load ground truth mapping file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def create_smt_mappings(ground_truth: dict, dataset_name: str) -> dict:
    """
    Convert ground truth to SMT mapping format.

    Ground truth format: "omop.TABLE.field": ["source.TABLE.field", ...]

    Returns: dict mapping (source_schema, target_schema) -> mapping_content
    """
    # Group mappings by source table -> target table
    # In OMOP ground truth: key is target (OMOP), value is source (CMS/MIMIC/Synthea)
    table_mappings = defaultdict(lambda: defaultdict(list))

    for target_field, source_fields in ground_truth.items():
        # Parse target field: "omop.TABLE.field"
        target_parts = target_field.split(".")
        if len(target_parts) != 3:
            continue
        target_db, target_table, target_field_name = target_parts
        target_schema = f"{target_db}_{target_table}"

        for source_field in source_fields:
            # Parse source field: "cms.TABLE.field" or "mimic.TABLE.field"
            source_parts = source_field.split(".")
            if len(source_parts) != 3:
                continue
            source_db, source_table, source_field_name = source_parts
            source_schema = f"{source_db}_{source_table}"

            # Create mapping entry
            mapping_entry = {
                "mapTable": target_schema,
                "mapProperty": f"{target_schema}.{target_field_name}",
                "justification": f"OMOP CDM mapping: {source_field} maps to {target_field} for medical data standardization."
            }

            # Group by (source_schema, target_schema) and source_field
            key = (source_schema, target_schema)
            full_source_field = f"{source_schema}.{source_field_name}"
            table_mappings[key][full_source_field].append(mapping_entry)

    return table_mappings


def write_mapping_files(table_mappings: dict, output_dir: Path, dataset_name: str):
    """Write SMT mapping files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    files_created = 0
    for (source_schema, target_schema), field_mappings in table_mappings.items():
        mapping_content = {
            "name": source_schema,
            "properties": dict(field_mappings)
        }

        # Filename: source_schema-target_schema.json
        filename = f"{source_schema}-{target_schema}.json"
        filepath = output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(mapping_content, f, indent=4)

        files_created += 1

    return files_created


def process_omop_dataset(source_dir: Path, output_dir: Path, dataset_name: str,
                         also_write_to_shared: bool = True):
    """Process a single OMOP dataset."""
    ground_truth_file = source_dir / f"{dataset_name}_ground_truth.json"

    if not ground_truth_file.exists():
        print(f"Ground truth file not found: {ground_truth_file}")
        return 0

    print(f"Processing {dataset_name}...")

    # Load ground truth
    ground_truth = load_ground_truth(ground_truth_file)
    print(f"  Loaded {len(ground_truth)} ground truth mappings")

    # Convert to SMT format
    table_mappings = create_smt_mappings(ground_truth, dataset_name)
    print(f"  Created {len(table_mappings)} table-level mappings")

    # Write to dataset-specific directory
    dataset_output_dir = output_dir / f"mappings_{dataset_name}"
    files_created = write_mapping_files(table_mappings, dataset_output_dir, dataset_name)
    print(f"  Wrote {files_created} mapping files to {dataset_output_dir}")

    # Also write to shared mappings_output directory
    if also_write_to_shared:
        shared_output_dir = output_dir / "mappings_output"
        write_mapping_files(table_mappings, shared_output_dir, dataset_name)
        print(f"  Also wrote to shared directory: {shared_output_dir}")

    return files_created


def main():
    """Main entry point."""
    # Source directory with ground truth files
    source_dir = Path("/Users/balaji/code/api_mapping/OMAP_data")

    # Output base directory
    output_base = Path(__file__).parent

    # Process each OMOP dataset
    datasets = ["omop_cms", "omop_mimic", "omop_synthea"]

    total_files = 0
    for dataset in datasets:
        files = process_omop_dataset(source_dir, output_base, dataset)
        total_files += files

    print(f"\nDone! Created mapping files for OMOP datasets.")
    print(f"\nDataset-specific mapping directories:")
    for dataset in datasets:
        print(f"  {output_base / f'mappings_{dataset}'}")
    print(f"\nTo index for SMT, run:")
    print(f"  cd smt && python smt_mappings_indexer.py --mappings-dir ../data/mappings_omop_cms --collection-name smt_mappings_omop_cms --force-reindex")


if __name__ == "__main__":
    main()
