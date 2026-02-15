#!/usr/bin/env python3
"""Print all results from all methods, models, and datasets in a formatted table.

Uses the same accuracy calculation as eval/calc_accuracy.py to match vldb.tex numbers.
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# Base directory
BASE_DIR = Path(__file__).parent

# =============================================================================
# Configuration - matching vldb.tex naming
# =============================================================================

# Datasets in display order
DATASETS = ["ITOpsGraph", "NL2GQL", "OMOP_CMS", "OMOP_MIMIC", "OMOP_Synthea"]

# Methods configuration with directory and model patterns
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
        'dirs': ['map_reflect/results', 'remote_results/map_reflect'],
        'models': [('Granite', 'granite-4'), ('Llama', 'llama-3')],  # Prefer granite-4 and llama-3
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

# Method display order matching vldb.tex
METHOD_ORDER = ['SMT', 'GNN', 'ReAct', 'AnyTool', 'Map-Reflect', 'MatchMaker', 'ReMatch', 'UE-Reflect']

# =============================================================================
# Helper Functions (from calc_accuracy.py)
# =============================================================================

def load_results(file_path: Path) -> List[Dict]:
    """Load results from JSON or JSONL file."""
    with open(file_path, 'r') as f:
        content = f.read().strip()

    # Try JSONL format
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

    # Standard JSON format
    data = json.loads(content)
    if isinstance(data, dict) and 'results' in data:
        results = data['results']
        # Check if this is UE-Reflect format
        if results and 'predictions' in results[0] and 'predicted_mapping' not in results[0]:
            results = [convert_ue_reflect_result(r) for r in results]
        return results
    return data if isinstance(data, list) else [data]


def convert_ue_reflect_result(result: Dict) -> Dict:
    """Convert UE-Reflect result format to standard format."""
    converted = {
        'sample_id': result.get('sample_id', ''),
        'utterance': result.get('utterance', ''),
        'expected_source': result.get('expected_source', {}),
        'expected_target': result.get('expected_target', {}),
        'ground_truth_mapping': result.get('ground_truth_mapping', {}),
        'predicted_mapping': {},
    }

    predictions = result.get('predictions', {})
    for src_field, pred_data in predictions.items():
        reflection = pred_data.get('reflection', {})
        revised_ranking = reflection.get('revised_ranking', [])

        if revised_ranking and isinstance(revised_ranking[0], dict):
            top_pred = revised_ranking[0]
            converted['predicted_mapping'][src_field] = {
                'target_property': top_pred.get('id', ''),
                'score': top_pred.get('score', 0),
                'justification': top_pred.get('reason', ''),
            }

    return converted


def get_schema_name(schema_info) -> str:
    """Extract schema name from various formats."""
    if isinstance(schema_info, dict):
        return schema_info.get('id', schema_info.get('name', ''))
    return str(schema_info) if schema_info else ''


def extract_product(field_or_schema: str) -> str:
    """Extract product name from field or schema name."""
    name = str(field_or_schema).lower()
    name = re.sub(r'^(nl2gql_|omop_)', '', name)
    if '.' in name:
        name = name.split('.')[0]
    if '_' in name:
        name = name.split('_')[0]
    return name


def extract_field_parts(field_name: str) -> Tuple[str, str]:
    """Extract (table, field) from field name."""
    field_name = str(field_name).lower().strip()
    field_name = re.sub(r'^(nl2gql_|omop_)', '', field_name)

    if '.' in field_name:
        parts = field_name.split('.', 1)
        return (parts[0], parts[1] if len(parts) > 1 else '')

    parts = field_name.split('_')
    if len(parts) >= 2:
        return (parts[0], '_'.join(parts[1:]))
    return ('', field_name)


def fields_match(gt_field: str, pred_field: str) -> bool:
    """Check if two field names match."""
    gt_table, gt_name = extract_field_parts(gt_field)
    pred_table, pred_name = extract_field_parts(pred_field)

    # Exact match
    if gt_table == pred_table and gt_name == pred_name:
        return True

    # Field name only match
    if gt_name and pred_name and gt_name == pred_name:
        return True

    # Partial match
    if gt_name and pred_name:
        gt_last = gt_name.split('_')[-1]
        pred_last = pred_name.split('_')[-1]
        if gt_last == pred_last and len(gt_last) > 2:
            gt_first = gt_name.split('_')[0] if '_' in gt_name else gt_name
            pred_first = pred_name.split('_')[0] if '_' in pred_name else pred_name
            if gt_first == pred_first or gt_table == pred_table:
                return True

    return False


def calculate_accuracies(results: List[Dict]) -> Dict:
    """Calculate API-level and field-level accuracy (same as calc_accuracy.py)."""
    api_correct_samples = 0
    total_samples = len(results)
    field_correct = 0
    field_total = 0
    api_correct_with_gt = 0

    for result in results:
        expected_source = get_schema_name(result.get('expected_source', ''))
        expected_target = get_schema_name(result.get('expected_target', ''))
        predicted_mapping = result.get('predicted_mapping', {})
        ground_truth = result.get('ground_truth_mapping', {})

        expected_src_product = extract_product(expected_source)
        expected_tgt_product = extract_product(expected_target)

        source_match = False
        target_match = False

        if predicted_mapping:
            for src_field, tgt_info in predicted_mapping.items():
                src_product = extract_product(src_field)
                if expected_src_product and src_product == expected_src_product:
                    source_match = True

                tgt_field = tgt_info.get('target_property', tgt_info) if isinstance(tgt_info, dict) else str(tgt_info)
                tgt_product = extract_product(tgt_field)
                if expected_tgt_product and tgt_product == expected_tgt_product:
                    target_match = True

        api_correct = source_match and target_match
        if api_correct:
            api_correct_samples += 1

        # Field-level accuracy: ONLY if API was correct
        if api_correct and ground_truth:
            api_correct_with_gt += 1
            field_total += len(ground_truth)

            for gt_src, gt_tgt in ground_truth.items():
                for pred_src, pred_info in predicted_mapping.items():
                    pred_tgt = pred_info.get('target_property', pred_info) if isinstance(pred_info, dict) else str(pred_info)
                    if fields_match(gt_src, pred_src) and fields_match(gt_tgt, pred_tgt):
                        field_correct += 1
                        break

    return {
        'api_accuracy': api_correct_samples / total_samples * 100 if total_samples > 0 else 0,
        'api_correct': api_correct_samples,
        'field_accuracy': field_correct / field_total * 100 if field_total > 0 else None,
        'field_correct': field_correct,
        'field_total': field_total,
        'api_correct_with_gt': api_correct_with_gt,
        'samples': total_samples
    }


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

                fname_lower = f.name.lower()

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


def print_table(title: str, headers: List[str], rows: List[List[str]], col_widths: List[int] = None):
    """Print a formatted table."""
    if col_widths is None:
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2 for i in range(len(headers))]

    # Border
    border = "+" + "+".join("-" * w for w in col_widths) + "+"

    print(border)
    total_width = sum(col_widths) + len(col_widths) - 1
    print("|" + title.center(total_width) + "|")
    print(border)

    # Headers
    header_row = "|" + "|".join(str(headers[i]).center(col_widths[i]) for i in range(len(headers))) + "|"
    print(header_row)
    print(border)

    # Data rows with method separators
    current_method = None
    for row in rows:
        if row[0] != current_method and current_method is not None:
            print(border)
        current_method = row[0]
        row_str = "|" + "|".join(str(row[i]).center(col_widths[i]) for i in range(len(row))) + "|"
        print(row_str)

    print(border)


def main():
    print("\nCollecting results from all methods...\n")

    # Collect results for each (method, model, dataset)
    all_results = {}

    for method_name in METHOD_ORDER:
        if method_name not in METHODS:
            continue

        method_config = METHODS[method_name]
        method_dirs = method_config['dirs']

        for model_short, model_pattern in method_config['models']:
            for dataset in DATASETS:
                files = find_result_files(method_dirs, dataset, model_pattern)

                if files:
                    # Use the file with most samples
                    best_acc = None

                    for f in files:
                        try:
                            results = load_results(f)
                            acc = calculate_accuracies(results)
                            if best_acc is None or acc['samples'] > best_acc.get('samples', 0):
                                best_acc = acc
                        except Exception as e:
                            pass

                    if best_acc and best_acc['samples'] > 0:
                        key = (method_name, model_short, dataset)
                        all_results[key] = best_acc

    # Build output
    print("=" * 100)
    print("COMPREHENSIVE RESULTS: ALL METHODS, MODELS, AND DATASETS")
    print("(Numbers should match vldb.tex tables)")
    print("=" * 100)

    headers = ["Method", "Model"] + DATASETS
    col_widths = [14, 12] + [12] * len(DATASETS)

    # Table 1: API-Level Accuracy
    print("\n" + "=" * 100)
    print("TABLE 1: API-LEVEL (PRODUCT) ACCURACY")
    print("Correct when predicted mapping contains fields from expected source AND target products")
    print("=" * 100)

    rows = []
    for method in METHOD_ORDER:
        if method not in METHODS:
            continue
        for model_short, _ in METHODS[method]['models']:
            row = [method, model_short]
            has_data = False
            for dataset in DATASETS:
                r = all_results.get((method, model_short, dataset))
                if r and r['samples'] > 0:
                    row.append(f"{r['api_accuracy']:.1f}%")
                    has_data = True
                else:
                    row.append("--")
            if has_data:
                rows.append(row)

    if rows:
        print_table("API-LEVEL ACCURACY", headers, rows, col_widths)
    else:
        print("(No API-level accuracy data available)")

    # Table 2: Field-Level Accuracy
    print("\n" + "=" * 100)
    print("TABLE 2: FIELD-LEVEL ACCURACY (Given Correct API)")
    print("Field mapping accuracy calculated ONLY on samples where API selection was correct")
    print("=" * 100)

    rows = []
    for method in METHOD_ORDER:
        if method not in METHODS:
            continue
        for model_short, _ in METHODS[method]['models']:
            row = [method, model_short]
            has_data = False
            for dataset in DATASETS:
                r = all_results.get((method, model_short, dataset))
                if r and r['api_correct_with_gt'] > 0 and r['field_accuracy'] is not None:
                    row.append(f"{r['field_accuracy']:.1f}%")
                    has_data = True
                else:
                    row.append("N/A")
            if has_data:
                rows.append(row)

    if rows:
        print_table("FIELD-LEVEL ACCURACY", headers, rows, col_widths)
    else:
        print("(No field-level accuracy data available)")

    # Table 3: Sample Counts
    print("\n" + "=" * 100)
    print("TABLE 3: SAMPLE COUNTS (API Correct / Total Samples)")
    print("=" * 100)

    rows = []
    for method in METHOD_ORDER:
        if method not in METHODS:
            continue
        for model_short, _ in METHODS[method]['models']:
            row = [method, model_short]
            has_data = False
            for dataset in DATASETS:
                r = all_results.get((method, model_short, dataset))
                if r and r['samples'] > 0:
                    row.append(f"{r['api_correct']}/{r['samples']}")
                    has_data = True
                else:
                    row.append("--")
            if has_data:
                rows.append(row)

    if rows:
        print_table("SAMPLE COUNTS", headers, rows, col_widths)

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    methods_found = set(k[0] for k in all_results.keys())
    models_found = set(k[1] for k in all_results.keys())
    datasets_found = set(k[2] for k in all_results.keys())

    print(f"\nMethods: {', '.join(sorted(methods_found))}")
    print(f"Models: {', '.join(sorted(models_found))}")
    print(f"Datasets: {', '.join(sorted(datasets_found))}")

    print("\nResults count by method:")
    for method in METHOD_ORDER:
        count = sum(1 for k in all_results if k[0] == method)
        if count > 0:
            print(f"  {method}: {count} result entries")


if __name__ == "__main__":
    main()
