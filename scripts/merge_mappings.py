#!/usr/bin/env python3
"""
Merge individual mapping files into compact dataset-level files.

This script consolidates mapping files like:
  omop_PERSON-cms_BENEFICIARYSUMMARY.json
  omop_VISIT_OCCURRENCE-cms_INPATIENTCLAIMS.json

Into single files per dataset pair:
  mappings_omop_cms.json
  mappings_omop_mimic.json
  mappings_omop_synthea.json
  mappings_itops.json
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def clean_schema_name(name: str) -> str:
    """
    Remove timestamps from schema names.

    E.g., 'cms_BENEFICIARYSUMMARY-1768896967301-6182170624' -> 'cms_BENEFICIARYSUMMARY'
    """
    # Pattern: name-timestamp-id where timestamp and id are numeric
    parts = name.split('-')
    if len(parts) >= 3:
        # Check if last two parts are numeric (timestamp and id)
        if parts[-1].isdigit() and parts[-2].isdigit():
            return '-'.join(parts[:-2])
    return name


def identify_dataset_pair(filename: str) -> tuple:
    """
    Identify the dataset pair from a mapping filename.

    Returns (dataset_name, source_schema, target_schema)
    """
    # Remove .json extension
    base = filename.replace('.json', '')

    # Split by hyphen to get source-target
    parts = base.split('-')
    if len(parts) < 2:
        return None, None, None

    source = parts[0]
    target = '-'.join(parts[1:])  # Handle cases with multiple hyphens

    # Clean timestamps from names
    source = clean_schema_name(source)
    target = clean_schema_name(target)

    # Identify dataset based on prefixes
    source_lower = source.lower()
    target_lower = target.lower()

    # OMOP datasets
    if 'omop' in source_lower and 'cms' in target_lower:
        return 'omop_cms', source, target
    if 'cms' in source_lower and 'omop' in target_lower:
        return 'omop_cms', source, target
    if 'omop' in source_lower and 'mimic' in target_lower:
        return 'omop_mimic', source, target
    if 'mimic' in source_lower and 'omop' in target_lower:
        return 'omop_mimic', source, target
    if 'omop' in source_lower and 'synthea' in target_lower:
        return 'omop_synthea', source, target
    if 'synthea' in source_lower and 'omop' in target_lower:
        return 'omop_synthea', source, target

    # ITOps datasets (concert, instana, turbo/turbonomic)
    itops_prefixes = ['concert', 'instana', 'turbo']
    if any(p in source_lower for p in itops_prefixes) and any(p in target_lower for p in itops_prefixes):
        return 'itops', source, target

    # Default: unknown
    return 'other', source, target


def merge_mappings(input_dir: str, output_dir: str):
    """
    Merge individual mapping files into dataset-level files.
    Deduplicates by keeping only one mapping per source-target pair.
    Prefers files without timestamps.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Group files by dataset
    dataset_mappings = defaultdict(dict)
    file_counts = defaultdict(int)
    duplicates_skipped = 0

    # Sort files to process non-timestamped files first (they come first alphabetically)
    json_files = sorted(input_path.glob('*.json'))

    for json_file in json_files:
        dataset, source, target = identify_dataset_pair(json_file.name)
        if dataset is None:
            print(f"  Skipping unrecognized file: {json_file.name}")
            continue

        try:
            # Create a key for this mapping pair (cleaned names)
            mapping_key = f"{source}_to_{target}"

            # Skip if we already have this mapping (prefer first/non-timestamped version)
            if mapping_key in dataset_mappings[dataset]:
                duplicates_skipped += 1
                continue

            with open(json_file, 'r') as f:
                mapping_data = json.load(f)

            # Store the mapping with cleaned schema names
            dataset_mappings[dataset][mapping_key] = {
                'source_schema': source,
                'target_schema': target,
                'mapping': mapping_data
            }
            file_counts[dataset] += 1

        except json.JSONDecodeError as e:
            print(f"  Error parsing {json_file.name}: {e}")
        except Exception as e:
            print(f"  Error processing {json_file.name}: {e}")

    print(f"\nSkipped {duplicates_skipped} duplicate mappings")

    # Write merged files
    print("\nWriting merged files:")
    for dataset, mappings in sorted(dataset_mappings.items()):
        output_file = output_path / f"mappings_{dataset}.json"

        merged_data = {
            'metadata': {
                'dataset': dataset,
                'num_mappings': len(mappings),
                'created_at': datetime.now().isoformat(),
                'source_files': file_counts[dataset]
            },
            'mappings': mappings
        }

        with open(output_file, 'w') as f:
            json.dump(merged_data, f, indent=2)

        print(f"  {output_file.name}: {len(mappings)} mappings from {file_counts[dataset]} files")

    return dataset_mappings


def create_summary(dataset_mappings: dict, output_dir: str):
    """Create a summary file with statistics."""
    output_path = Path(output_dir)

    summary = {
        'total_mappings': sum(len(m) for m in dataset_mappings.values()),
        'datasets': {}
    }

    for dataset, mappings in sorted(dataset_mappings.items()):
        # Count unique source and target schemas
        sources = set()
        targets = set()
        total_field_mappings = 0

        for key, data in mappings.items():
            sources.add(data['source_schema'])
            targets.add(data['target_schema'])

            # Count field mappings
            mapping = data.get('mapping', {})
            properties = mapping.get('properties', {})
            total_field_mappings += len(properties)

        summary['datasets'][dataset] = {
            'num_schema_pairs': len(mappings),
            'unique_sources': len(sources),
            'unique_targets': len(targets),
            'total_field_mappings': total_field_mappings,
            'source_schemas': sorted(sources),
            'target_schemas': sorted(targets)
        }

    summary_file = output_path / 'mappings_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary written to: {summary_file}")
    print(f"Total mappings across all datasets: {summary['total_mappings']}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Merge individual mapping files into dataset-level files')
    parser.add_argument('--input', '-i', default='data/mappings_raw',
                        help='Input directory with individual mapping files')
    parser.add_argument('--output', '-o', default='data/mappings_output',
                        help='Output directory for merged files')

    args = parser.parse_args()

    print(f"Merging mapping files from: {args.input}")
    print(f"Output directory: {args.output}")

    dataset_mappings = merge_mappings(args.input, args.output)
    create_summary(dataset_mappings, args.output)

    print("\nDone!")


if __name__ == '__main__':
    main()
