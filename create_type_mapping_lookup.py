#!/usr/bin/env python3
"""
Create a lookup file showing which types map to which types across all methods and models.

For the ITOpsGraph dataset, creates a JSON file where:
- Each key is a source type
- Each value contains target types and which (method, model) combinations produced the mapping
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

BASE_DIR = Path(__file__).parent

# Method configurations (matching print_all_results.py)
METHODS = {
    'SMT': {
        'dirs': ['smt/results', 'remote_results/smt'],
        'models': [('Granite', 'granite'), ('Llama', 'llama')],
    },
    'GNN': {
        'dirs': ['gnn/results', 'remote_results/gnn'],
        'models': [('(no LLM)', '')],
    },
    'ReAct': {
        'dirs': ['react/results', 'remote_results/react'],
        'models': [('Granite', 'granite'), ('Llama', 'llama')],
    },
    'AnyTool': {
        'dirs': ['anytool/results', 'remote_results/anytool'],
        'models': [('Granite', 'granite'), ('Llama', 'llama')],
    },
    'Map-Reflect': {
        'dirs': ['map_reflect/results', 'remote_results/map_reflect', 'results'],
        'models': [('Granite', 'granite'), ('Llama', 'llama')],
    },
    'MatchMaker': {
        'dirs': ['match_maker/results', 'remote_results/match_maker'],
        'models': [('Granite', r'^(?!dual_context).*granite'), ('Llama', r'^(?!dual_context).*llama')],
    },
    'ReMatch': {
        'dirs': ['rematch/results', 'remote_results/rematch'],
        'models': [('Granite', 'granite'), ('Llama', 'llama')],
    },
    'UE-Reflect': {
        'dirs': ['map_reflect/results', 'remote_results/map_reflect'],
        'models': [('Granite', r'ue_reflect.*granite'), ('Llama', r'ue_reflect.*llama')],
    },
}


def extract_type_from_field(field_name: str) -> str:
    """Extract type name from field name like 'concert_Application.id' -> 'concert_Application'."""
    field_name = str(field_name).strip()
    if '.' in field_name:
        return field_name.split('.')[0]
    # If no dot, try to extract up to last underscore (e.g., product_Type_field -> product_Type)
    parts = field_name.rsplit('_', 1)
    if len(parts) == 2 and parts[1].islower():
        return parts[0]
    return field_name


def load_results(file_path: Path) -> List[Dict]:
    """Load results from JSON or JSONL file."""
    with open(file_path, 'r') as f:
        content = f.read().strip()

    if file_path.suffix == '.jsonl' or '\n{' in content:
        results = []
        for line in content.split('\n'):
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return results

    data = json.loads(content)
    if isinstance(data, dict) and 'results' in data:
        results = data['results']
        # Handle UE-Reflect format
        if results and 'predictions' in results[0] and 'predicted_mapping' not in results[0]:
            results = [convert_ue_reflect_result(r) for r in results]
        return results
    return data if isinstance(data, list) else [data]


def convert_ue_reflect_result(result: Dict) -> Dict:
    """Convert UE-Reflect result format to standard format."""
    converted = {
        'sample_id': result.get('sample_id', ''),
        'predicted_mapping': {},
        'ground_truth_mapping': result.get('ground_truth_mapping', {}),
    }

    predictions = result.get('predictions', {})
    for src_field, pred_data in predictions.items():
        reflection = pred_data.get('reflection', {})
        revised_ranking = reflection.get('revised_ranking', [])

        if revised_ranking and isinstance(revised_ranking[0], dict):
            top_pred = revised_ranking[0]
            converted['predicted_mapping'][src_field] = {
                'target_property': top_pred.get('id', ''),
            }

    return converted


def find_result_files(method_dirs: List[str], dataset: str, model_pattern: str) -> List[Path]:
    """Find result files matching dataset and model pattern."""
    files = []
    ds_lower = dataset.lower()
    ds_patterns = [dataset, ds_lower, dataset.replace('_', '-'), ds_lower.replace('_', '-')]

    for dir_path_str in method_dirs:
        method_dir = BASE_DIR / dir_path_str
        if not method_dir.exists():
            continue

        for pattern in ['*.json', '*.jsonl']:
            for f in method_dir.glob(pattern):
                if 'checkpoint' in f.name.lower():
                    continue
                if 'coverage_eval' in f.name.lower():
                    continue
                if 'batch_' in f.name.lower():
                    continue
                if 'discovery_valid' in f.name.lower():
                    continue
                if 'discovery_' in f.name.lower() and 'discovery_valid' not in f.name.lower():
                    continue
                # Allow mapreflect_ files (they use predictions format)
                # if f.name.lower().startswith('mapreflect_'):
                #     continue

                fname_lower = f.name.lower()

                # Skip prefixed files unless model_pattern specifically requests them
                if fname_lower.startswith('ue_reflect_') and 'ue_reflect' not in model_pattern.lower():
                    continue
                if fname_lower.startswith('dual_context_') and 'dual_context' not in model_pattern.lower():
                    continue

                # Check dataset match with word boundary
                dataset_match = False
                for p in ds_patterns:
                    p_lower = p.lower()
                    idx = fname_lower.find(p_lower)
                    if idx >= 0:
                        end_idx = idx + len(p_lower)
                        preceded_ok = idx == 0 or not fname_lower[idx-1].isalpha()
                        followed_ok = end_idx >= len(fname_lower) or not fname_lower[end_idx].isalpha()
                        if preceded_ok and followed_ok:
                            dataset_match = True
                            break

                # Check model match
                if not model_pattern:
                    model_match = True
                elif '.*' in model_pattern or model_pattern.startswith('^'):
                    model_match = bool(re.search(model_pattern.lower(), fname_lower))
                else:
                    model_match = model_pattern.lower() in fname_lower

                if dataset_match and model_match:
                    files.append(f)

    return files


def extract_type_mappings(results: List[Dict]) -> Set[Tuple[str, str]]:
    """Extract all (source_type, target_type) pairs from results."""
    mappings = set()

    for result in results:
        predicted = result.get('predicted_mapping', {})
        if not predicted:
            continue

        for src_field, tgt_info in predicted.items():
            if isinstance(tgt_info, dict):
                tgt_field = tgt_info.get('target_property', '')
            else:
                tgt_field = str(tgt_info)

            if not tgt_field:
                continue

            src_type = extract_type_from_field(src_field)
            tgt_type = extract_type_from_field(tgt_field)

            if src_type and tgt_type:
                mappings.add((src_type, tgt_type))

    return mappings


def extract_type_mappings_mapreflect(results: List[Dict]) -> Set[Tuple[str, str]]:
    """Extract type mappings from Map-Reflect format (uses 'predictions' key)."""
    mappings = set()
    for result in results:
        predictions = result.get('predictions', {})
        if not predictions:
            continue
        for src_field, pred_data in predictions.items():
            if not isinstance(pred_data, dict):
                continue
            ranking = pred_data.get('initial_ranking', [])
            if ranking and isinstance(ranking[0], dict):
                tgt_field = ranking[0].get('id', '')
                if tgt_field and tgt_field != 'NO_MATCH':
                    src_type = extract_type_from_field(src_field)
                    tgt_type = extract_type_from_field(tgt_field)
                    if src_type and tgt_type:
                        mappings.add((src_type, tgt_type))
    return mappings


def main():
    print("Creating type mapping lookup...\n")

    # Structure: {source_type: {target_type: [(method, model), ...]}}
    type_lookup = defaultdict(lambda: defaultdict(list))

    datasets = ['ITOpsGraph', 'ITOM_Hashicorp']
    total_mappings = 0

    for dataset in datasets:
        print(f"Processing {dataset}...")

        for method_name, method_config in METHODS.items():
            method_dirs = method_config['dirs']

            for model_short, model_pattern in method_config['models']:
                files = find_result_files(method_dirs, dataset, model_pattern)

                if files:
                    # Use file with most samples
                    best_file = None
                    best_count = 0

                    for f in files:
                        try:
                            results = load_results(f)
                            if len(results) > best_count:
                                best_count = len(results)
                                best_file = f
                        except Exception:
                            pass

                    if best_file:
                        try:
                            results = load_results(best_file)
                            mappings = extract_type_mappings(results)
                            # Also try Map-Reflect format (predictions key)
                            if not mappings:
                                mappings = extract_type_mappings_mapreflect(results)

                            for src_type, tgt_type in mappings:
                                method_model = f"{method_name} ({model_short})"
                                if method_model not in type_lookup[src_type][tgt_type]:
                                    type_lookup[src_type][tgt_type].append(method_model)
                                    total_mappings += 1

                            print(f"  {method_name} {model_short}: {len(mappings)} type mappings from {best_file.name}")
                        except Exception as e:
                            print(f"  {method_name} {model_short}: Error - {e}")

    # Convert to regular dict for JSON serialization
    output = {}
    for src_type in sorted(type_lookup.keys()):
        output[src_type] = {}
        for tgt_type in sorted(type_lookup[src_type].keys()):
            output[src_type][tgt_type] = sorted(type_lookup[src_type][tgt_type])

    # Write output file
    output_file = BASE_DIR / 'data' / 'itopsgraph_type_mapping_lookup.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Created lookup file: {output_file}")
    print(f"Total source types: {len(output)}")
    print(f"Total type mappings: {total_mappings}")

    # Print summary of top mapped types
    print(f"\nTop 10 most mapped source types:")
    sorted_sources = sorted(output.items(), key=lambda x: sum(len(v) for v in x[1].values()), reverse=True)
    for src, targets in sorted_sources[:10]:
        total_methods = sum(len(methods) for methods in targets.values())
        print(f"  {src}: maps to {len(targets)} target types ({total_methods} method/model combinations)")


if __name__ == "__main__":
    main()
