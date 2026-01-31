#!/usr/bin/env python3
"""
Standardize result file names across all methods.

Standard format: {Dataset}_{model}.json
- Dataset: ITOps, ITOpsGraph, NL2GQL, OMOP_CMS, OMOP_MIMIC, OMOP_Synthea, ToolBench
- Model: granite, llama, mistral

Checkpoint files are archived to an 'archive' subdirectory.
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime

# Model name mappings
MODEL_MAPPINGS = {
    # Granite variants
    'granite': 'granite',
    'granite-4': 'granite',
    'ibm_granite-4-h-small': 'granite',
    'ibm_granite-3-3-8b-instruct': 'granite',  # Old granite
    'watsonx_ibm_granite-4-h-small': 'granite',
    'watsonx_ibm_granite-3-3-8b-instruct': 'granite',

    # Llama variants
    'llama': 'llama',
    'llama-3-70b-instruct': 'llama',
    'meta-llama_llama-3-3-70b-instruct': 'llama',
    'watsonx_meta-llama_llama-3-3-70b-instruct': 'llama',

    # Mistral variants
    'mistral': 'mistral',
    'mistral-medium': 'mistral',
    'mistral-large': 'mistral-large',  # Keep mistral-large separate
    'mistral-large-2512': 'mistral-large',
    'mistralai_mistral-medium-2505': 'mistral',
    'mistralai_mistral-large-2': 'mistral-large',
    'watsonx_mistralai_mistral-medium-2505': 'mistral',
    'watsonx_mistralai_mistral-large-2': 'mistral-large',
    'watsonx_mistral-large-2512': 'mistral-large',

    # Qwen (keep as is)
    'qwen': 'qwen',
}

# Dataset name mappings (normalize case)
DATASET_MAPPINGS = {
    'itops': 'ITOps',
    'ITOps': 'ITOps',
    'itopsgraph': 'ITOpsGraph',
    'ITOpsGraph': 'ITOpsGraph',
    'nl2gql': 'NL2GQL',
    'NL2GQL': 'NL2GQL',
    'omop_cms': 'OMOP_CMS',
    'OMOP_CMS': 'OMOP_CMS',
    'omop_mimic': 'OMOP_MIMIC',
    'OMOP_MIMIC': 'OMOP_MIMIC',
    'omop_synthea': 'OMOP_Synthea',
    'OMOP_Synthea': 'OMOP_Synthea',
    'toolbench': 'ToolBench',
    'ToolBench': 'ToolBench',
    'graphqlrestbench': 'GraphQLRestBench',
    'GraphQLRestBench': 'GraphQLRestBench',
}

def parse_filename(filename):
    """Parse a result filename to extract dataset and model."""
    # Remove extension
    name = filename.replace('.json', '')

    # Handle special prefixes
    prefixes_to_remove = ['dual_context_', 'ue_reflect_']
    prefix = None
    for p in prefixes_to_remove:
        if name.startswith(p):
            prefix = p.rstrip('_')
            name = name[len(p):]
            break

    # Remove suffixes
    is_checkpoint = '_checkpoint' in name
    is_full = '_full' in name
    name = name.replace('_checkpoint', '').replace('_full', '')

    # Try to find dataset
    dataset = None
    model_part = None

    # Check for known datasets - try longer matches first (ITOpsGraph before ITOps)
    sorted_datasets = sorted(DATASET_MAPPINGS.keys(), key=len, reverse=True)
    for ds_key in sorted_datasets:
        ds_normalized = DATASET_MAPPINGS[ds_key]
        if name.lower().startswith(ds_key.lower() + '_'):
            dataset = ds_normalized
            model_part = name[len(ds_key) + 1:]
            break
        elif name.lower() == ds_key.lower():
            # Exact match with no model part
            dataset = ds_normalized
            model_part = ''
            break

    if not dataset:
        return None, None, prefix, is_checkpoint, is_full

    # Normalize model name
    model = None
    for model_key, model_normalized in MODEL_MAPPINGS.items():
        if model_part.lower() == model_key.lower():
            model = model_normalized
            break
        elif model_part.lower().startswith(model_key.lower()):
            model = model_normalized
            break

    if not model:
        # Try to extract just the first part before any underscore
        simple_model = model_part.split('_')[0].lower()
        model = MODEL_MAPPINGS.get(simple_model, simple_model)

    return dataset, model, prefix, is_checkpoint, is_full


def get_file_info(filepath):
    """Get file size and modification time."""
    stat = os.stat(filepath)
    return {
        'size': stat.st_size,
        'mtime': stat.st_mtime,
        'mtime_str': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    }


def count_results(filepath):
    """Count results in a JSON file."""
    try:
        with open(filepath) as f:
            data = json.load(f)
        if isinstance(data, list):
            return len(data)
        elif isinstance(data, dict) and 'results' in data:
            return len(data['results'])
        return -1
    except:
        return -1


def standardize_directory(results_dir, dry_run=True):
    """Standardize file names in a results directory."""
    if not os.path.isdir(results_dir):
        return []

    archive_dir = os.path.join(results_dir, 'archive')

    actions = []
    files_by_key = {}  # (dataset, model, prefix) -> list of files

    # Collect all files
    for filename in os.listdir(results_dir):
        if not filename.endswith('.json'):
            continue

        filepath = os.path.join(results_dir, filename)
        if os.path.isdir(filepath):
            continue

        dataset, model, prefix, is_checkpoint, is_full = parse_filename(filename)

        if not dataset or not model:
            actions.append({
                'action': 'skip',
                'file': filename,
                'reason': f'Could not parse: dataset={dataset}, model={model}'
            })
            continue

        key = (dataset, model, prefix)
        if key not in files_by_key:
            files_by_key[key] = []

        file_info = get_file_info(filepath)
        file_info['filename'] = filename
        file_info['filepath'] = filepath
        file_info['is_checkpoint'] = is_checkpoint
        file_info['is_full'] = is_full
        file_info['result_count'] = count_results(filepath)

        files_by_key[key].append(file_info)

    # Process each group
    for (dataset, model, prefix), files in files_by_key.items():
        if prefix:
            standard_name = f"{prefix}_{dataset}_{model}.json"
        else:
            standard_name = f"{dataset}_{model}.json"

        standard_path = os.path.join(results_dir, standard_name)

        # Sort files: prefer non-checkpoint, larger files, newer files
        files.sort(key=lambda x: (
            not x['is_checkpoint'],  # Non-checkpoint first
            x['result_count'],       # More results
            x['size'],               # Larger file
            x['mtime']               # Newer file
        ), reverse=True)

        best_file = files[0]

        # Check if best file already has standard name
        if best_file['filename'] == standard_name:
            # Archive the rest
            for f in files[1:]:
                actions.append({
                    'action': 'archive',
                    'file': f['filename'],
                    'dest': os.path.join('archive', f['filename']),
                    'reason': f"Duplicate of {standard_name} (results: {f['result_count']})"
                })
        else:
            # Rename best file to standard name
            if os.path.exists(standard_path) and best_file['filepath'] != standard_path:
                # Standard name exists but isn't best - archive the current standard
                actions.append({
                    'action': 'archive',
                    'file': standard_name,
                    'dest': os.path.join('archive', f"old_{standard_name}"),
                    'reason': f"Replaced by better file: {best_file['filename']}"
                })

            actions.append({
                'action': 'rename',
                'file': best_file['filename'],
                'dest': standard_name,
                'reason': f"Standardize name (results: {best_file['result_count']}, {best_file['mtime_str']})"
            })

            # Archive the rest
            for f in files[1:]:
                if f['filename'] != standard_name:  # Don't archive if it's already the standard name
                    actions.append({
                        'action': 'archive',
                        'file': f['filename'],
                        'dest': os.path.join('archive', f['filename']),
                        'reason': f"Duplicate/old version (results: {f['result_count']})"
                    })

    # Execute actions if not dry run
    if not dry_run:
        os.makedirs(archive_dir, exist_ok=True)

        for action in actions:
            src = os.path.join(results_dir, action['file'])
            if action['action'] == 'rename':
                dst = os.path.join(results_dir, action['dest'])
                if os.path.exists(src) and src != dst:
                    if os.path.exists(dst):
                        # Move existing to archive first
                        shutil.move(dst, os.path.join(archive_dir, f"replaced_{os.path.basename(dst)}"))
                    shutil.move(src, dst)
            elif action['action'] == 'archive':
                dst = os.path.join(results_dir, action['dest'])
                if os.path.exists(src):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.move(src, dst)

    return actions


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Standardize result file names')
    parser.add_argument('--execute', action='store_true', help='Actually perform the changes (default is dry run)')
    parser.add_argument('--dir', type=str, help='Process only this directory')
    args = parser.parse_args()

    dry_run = not args.execute

    if dry_run:
        print("=" * 60)
        print("DRY RUN - No changes will be made")
        print("Run with --execute to apply changes")
        print("=" * 60)
        print()

    result_dirs = [
        'smt/results',
        'gnn/results',
        'react/results',
        'anytool/results',
        'map_reflect/results',
        'match_maker/results',
        'rematch/results',
    ]

    if args.dir:
        result_dirs = [args.dir]

    for results_dir in result_dirs:
        if not os.path.isdir(results_dir):
            continue

        print(f"\n{'=' * 60}")
        print(f"Processing: {results_dir}")
        print("=" * 60)

        actions = standardize_directory(results_dir, dry_run=dry_run)

        if not actions:
            print("  No changes needed")
            continue

        for action in actions:
            if action['action'] == 'skip':
                print(f"  SKIP: {action['file']}")
                print(f"         Reason: {action['reason']}")
            elif action['action'] == 'rename':
                print(f"  RENAME: {action['file']}")
                print(f"      ->  {action['dest']}")
                print(f"          ({action['reason']})")
            elif action['action'] == 'archive':
                print(f"  ARCHIVE: {action['file']}")
                print(f"       ->  {action['dest']}")
                print(f"           ({action['reason']})")


if __name__ == '__main__':
    main()
