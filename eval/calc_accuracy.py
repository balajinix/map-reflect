#!/usr/bin/env python3
"""
Calculate accuracy for all schema mapping methods.

This script reads results from each method's results/ directory and calculates:
- API-level accuracy (correct source/target product identification)
- Field-level accuracy (correct field mappings, conditional on correct API)

Two evaluation modes:
1. Default: Per-sample accuracy using embedded ground_truth_mapping in results
2. Consensus: Evaluate against external consensus ground truth (precision/recall/F1)

Usage:
    # Default mode (per-sample accuracy)
    python eval/calc_accuracy.py
    python eval/calc_accuracy.py --methods map_reflect,gnn --datasets ITOps,NL2GQL
    python eval/calc_accuracy.py --output markdown  # Output as markdown table
    python eval/calc_accuracy.py --output csv       # Output as CSV

    # Consensus ground truth mode (precision/recall/F1)
    python eval/calc_accuracy.py --ground-truth data/itops_consensus_ground_truth.jsonl
    python eval/calc_accuracy.py --ground-truth data/itops_consensus_ground_truth.jsonl --methods SMT,ReAct,GNN
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set

# =============================================================================
# Configuration
# =============================================================================

# Script is in eval/, so parent.parent gets to project root
BASE_DIR = Path(__file__).parent.parent
DATASETS = ['ITOps', 'ITOpsGraph', 'NL2GQL', 'OMOP_CMS', 'OMOP_MIMIC', 'OMOP_Synthea']

# Method directories and model patterns
# Note: Patterns are matched against lowercased filenames
# Some methods use "granite" (no version), others use "granite-4-h-small"
METHODS = {
    'SMT': {
        'dir': 'smt/results',
        'models': [
            ('Granite', 'granite'),  # matches granite.jsonl files
            ('Llama', 'llama'),
        ],
    },
    'GNN': {
        'dir': 'gnn/results',
        'models': [('(no LLM)', '')],  # GNN doesn't use LLM for inference
    },
    'ReAct': {
        'dir': 'react/results',
        'models': [
            ('Granite', 'granite'),  # matches granite.jsonl files
            ('Llama', 'llama'),
        ],
    },
    'AnyTool': {
        'dir': 'anytool/results',
        'models': [
            ('Granite', 'granite'),  # matches both granite and granite-4
            ('Llama', 'llama'),
        ],
    },
    'Map-Reflect': {
        'dir': 'map_reflect/results',
        'models': [
            ('Granite', 'granite-4'),  # uses full model name granite-4-h-small
            ('Llama', 'llama-3-3'),
        ],
    },
    'MatchMaker': {
        'dir': 'match_maker/results',
        'models': [
            ('Granite', '^(?!dual_context).*granite'),  # exclude dual_context files
            ('Llama', '^(?!dual_context).*llama'),
        ],
    },
    'ReMatch': {
        'dir': 'rematch/results',
        'models': [
            ('Granite', 'granite'),
            ('Llama', 'llama'),
        ],
    },
    'DualContext': {
        'dir': 'match_maker/results',
        'models': [
            ('Granite', 'dual_context_.*granite'),
            ('Llama', 'dual_context_.*llama'),
        ],
    },
    'UE-Reflect': {
        'dir': 'map_reflect/results',
        'models': [
            ('Granite', 'ue_reflect_.*granite'),
            ('Llama', 'ue_reflect_.*llama'),
        ],
    },
}

# =============================================================================
# Helper Functions
# =============================================================================

def load_results(file_path: Path) -> List[Dict]:
    """Load results from JSON or JSONL file."""
    with open(file_path, 'r') as f:
        content = f.read().strip()

    # Try JSONL format (one JSON object per line)
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
        # Check if this is UE-Reflect format (has 'predictions' instead of 'predicted_mapping')
        if results and 'predictions' in results[0] and 'predicted_mapping' not in results[0]:
            results = [convert_ue_reflect_result(r) for r in results]
        return results
    return data if isinstance(data, list) else [data]


def convert_ue_reflect_result(result: Dict) -> Dict:
    """Convert UE-Reflect result format to standard format.

    UE-Reflect format:
        predictions: {source_field: {reflection: {revised_ranking: [{id, score, reason}]}}}

    Standard format:
        predicted_mapping: {source_field: {target_property: target_field}}
    """
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
            # Get top-1 prediction
            top_pred = revised_ranking[0]
            converted['predicted_mapping'][src_field] = {
                'target_property': top_pred.get('id', ''),
                'score': top_pred.get('score', 0),
                'justification': top_pred.get('reason', ''),
                'uncertainty': pred_data.get('uncertainty', 0),
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
    # Remove common prefixes
    name = re.sub(r'^(nl2gql_|omop_)', '', name)
    # Get first part before dot or underscore
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

    # Partial match (last part of field name)
    if gt_name and pred_name:
        gt_last = gt_name.split('_')[-1]
        pred_last = pred_name.split('_')[-1]
        if gt_last == pred_last and len(gt_last) > 2:
            gt_first = gt_name.split('_')[0] if '_' in gt_name else gt_name
            pred_first = pred_name.split('_')[0] if '_' in pred_name else pred_name
            if gt_first == pred_first or gt_table == pred_table:
                return True

    return False


def calculate_accuracies(results: List[Dict], dataset: str = '') -> Dict:
    """Calculate API-level and field-level accuracy."""
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

        # Extract products
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
        'api_accuracy': api_correct_samples / total_samples if total_samples > 0 else 0,
        'api_correct': api_correct_samples,
        'field_accuracy': field_correct / field_total if field_total > 0 else 0,
        'field_correct': field_correct,
        'field_total': field_total,
        'api_correct_with_gt': api_correct_with_gt,
        'samples': total_samples
    }


# =============================================================================
# Consensus Ground Truth Functions
# =============================================================================

def normalize_field(field: str) -> str:
    """Normalize field name for comparison in consensus mode."""
    name = field.lower().strip()
    name = re.sub(r'^(nl2gql_|omop_)', '', name)
    return name


def load_consensus_ground_truth(gt_path: Path) -> Tuple[Set[Tuple[str, str]], Dict]:
    """
    Load consensus ground truth file.

    Returns:
        Tuple of:
        - Set of (source_field, target_field) tuples (normalized)
        - Dict with metadata (product pairs, counts, etc.)
    """
    ground_truth = set()
    product_pairs = defaultdict(set)

    with open(gt_path) as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)

            src_product = sample['source_schema']['id']
            tgt_product = sample['target_schema']['id']

            for src_field, tgt_field in sample['mapping'].items():
                src_norm = normalize_field(src_field)
                tgt_norm = normalize_field(tgt_field)
                ground_truth.add((src_norm, tgt_norm))
                product_pairs[(src_product, tgt_product)].add((src_norm, tgt_norm))

    metadata = {
        'total_mappings': len(ground_truth),
        'product_pairs': len(product_pairs),
        'by_pair': {f"{k[0]}->{k[1]}": len(v) for k, v in product_pairs.items()},
    }

    return ground_truth, metadata


def extract_all_predictions(results: List[Dict]) -> Set[Tuple[str, str]]:
    """Extract all (source_field, target_field) pairs from results."""
    predictions = set()

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

            src_norm = normalize_field(src_field)
            tgt_norm = normalize_field(tgt_field)
            predictions.add((src_norm, tgt_norm))

    return predictions


def calculate_consensus_metrics(
    predicted: Set[Tuple[str, str]],
    ground_truth: Set[Tuple[str, str]]
) -> Dict:
    """Calculate precision, recall, and F1 against consensus ground truth."""
    if not predicted:
        return {
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
            'tp': 0, 'fp': 0, 'fn': 0,
            'predicted_count': 0, 'gt_count': len(ground_truth)
        }

    true_positives = predicted & ground_truth
    false_positives = predicted - ground_truth
    false_negatives = ground_truth - predicted

    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'predicted_count': len(predicted),
        'gt_count': len(ground_truth),
    }


def find_result_files(method_dir: Path, dataset: str, model_pattern: str) -> List[Path]:
    """Find result files matching dataset and model pattern."""
    if not method_dir.exists():
        return []

    files = []
    ds_lower = dataset.lower()
    ds_patterns = [dataset, ds_lower, dataset.replace('_', '-'), ds_lower.replace('_', '-')]

    # Look for both .json and .jsonl files
    for pattern in ['*.json', '*.jsonl']:
        for f in method_dir.glob(pattern):
            if 'checkpoint' in f.name.lower():
                continue

            fname_lower = f.name.lower()

            # Check dataset match - must be an exact word boundary match
            # to avoid "ITOps" matching "ITOpsGraph"
            dataset_match = False
            for p in ds_patterns:
                p_lower = p.lower()
                idx = fname_lower.find(p_lower)
                if idx >= 0:
                    # Check that the match is at a word boundary
                    # (preceded by non-alpha or start, followed by non-alpha or end)
                    end_idx = idx + len(p_lower)
                    preceded_ok = idx == 0 or not fname_lower[idx-1].isalpha()
                    followed_ok = end_idx >= len(fname_lower) or not fname_lower[end_idx].isalpha()
                    if preceded_ok and followed_ok:
                        dataset_match = True
                        break

            # Check model match (empty pattern matches all)
            # Support both simple string patterns and regex patterns
            if not model_pattern:
                model_match = True
            elif '.*' in model_pattern:
                # It's a regex pattern
                model_match = bool(re.search(model_pattern.lower(), fname_lower))
            else:
                model_match = model_pattern.lower() in fname_lower

            if dataset_match and model_match:
                files.append(f)

    return files


# =============================================================================
# Main
# =============================================================================

def run_consensus_evaluation(args, base_dir: Path, selected_methods: List[str]):
    """Run evaluation against consensus ground truth."""
    gt_path = Path(args.ground_truth)
    if not gt_path.exists():
        print(f"Error: Ground truth file not found: {gt_path}")
        print("Run generate_ground_truth.py first to create it.")
        return

    print(f"Loading consensus ground truth from {gt_path}...")
    ground_truth, gt_metadata = load_consensus_ground_truth(gt_path)
    print(f"Ground truth: {gt_metadata['total_mappings']} mappings across {gt_metadata['product_pairs']} product pairs")
    print()

    # Collect results for each method
    all_results = {}

    for method_name in selected_methods:
        if method_name not in METHODS:
            continue

        method_config = METHODS[method_name]
        method_dir = base_dir / method_config['dir']

        # Load all ITOps results for this method
        results = []
        for f in method_dir.glob('*.json'):
            if 'itops' in f.name.lower() and 'checkpoint' not in f.name.lower():
                try:
                    results.extend(load_results(f))
                except Exception:
                    pass
        for f in method_dir.glob('*.jsonl'):
            if 'itops' in f.name.lower() and 'checkpoint' not in f.name.lower():
                try:
                    results.extend(load_results(f))
                except Exception:
                    pass

        if not results:
            continue

        predictions = extract_all_predictions(results)
        metrics = calculate_consensus_metrics(predictions, ground_truth)
        all_results[method_name] = metrics

    # Output results
    if args.output == 'json':
        print(json.dumps(all_results, indent=2))
        return

    print("=" * 95)
    print("EVALUATION AGAINST CONSENSUS GROUND TRUTH")
    print(f"Ground truth: {gt_metadata['total_mappings']} mappings from inter-annotator agreement")
    print("=" * 95)
    print()

    if args.output == 'csv':
        print("Method,Precision,Recall,F1,TP,FP,FN,Predicted,GT")
    else:
        print("| Method       | Precision | Recall |    F1 |  TP |   FP |  FN | Predicted |  GT |")
        print("|--------------|-----------|--------|-------|-----|------|-----|-----------|-----|")

    for method in selected_methods:
        if method not in all_results:
            continue

        m = all_results[method]
        if args.output == 'csv':
            print(f"{method},{m['precision']:.3f},{m['recall']:.3f},{m['f1']:.3f},"
                  f"{m['tp']},{m['fp']},{m['fn']},{m['predicted_count']},{m['gt_count']}")
        else:
            print(f"| {method:12} | {m['precision']*100:8.1f}% | {m['recall']*100:5.1f}% | "
                  f"{m['f1']*100:5.1f}% | {m['tp']:3} | {m['fp']:4} | {m['fn']:3} | "
                  f"{m['predicted_count']:9} | {m['gt_count']:3} |")

    print()
    print("Legend:")
    print("  Precision = TP / (TP + FP) - How many predicted mappings are correct")
    print("  Recall = TP / (TP + FN) - How many ground truth mappings were found")
    print("  F1 = Harmonic mean of precision and recall")


def main():
    parser = argparse.ArgumentParser(description='Calculate accuracy for schema mapping methods')
    parser.add_argument('--methods', type=str, default='all',
                        help='Comma-separated methods (default: all)')
    parser.add_argument('--datasets', type=str, default='all',
                        help='Comma-separated datasets (default: all)')
    parser.add_argument('--output', type=str, choices=['table', 'markdown', 'csv', 'json'],
                        default='markdown', help='Output format')
    parser.add_argument('--base-dir', type=str, default=str(BASE_DIR),
                        help='Base directory for results')
    parser.add_argument('--ground-truth', type=str, default=None,
                        help='Path to consensus ground truth JSONL file (enables consensus evaluation mode)')
    args = parser.parse_args()

    base_dir = Path(args.base_dir)

    # Parse method selection
    if args.methods == 'all':
        selected_methods = list(METHODS.keys())
    else:
        selected_methods = [m.strip() for m in args.methods.split(',')]

    # If ground truth file specified, run consensus evaluation mode
    if args.ground_truth:
        run_consensus_evaluation(args, base_dir, selected_methods)
        return

    # Parse dataset selection for default mode
    if args.datasets == 'all':
        selected_datasets = DATASETS
    else:
        selected_datasets = [d.strip() for d in args.datasets.split(',')]

    # Collect results
    all_results = {}

    for method_name in selected_methods:
        if method_name not in METHODS:
            print(f"Warning: Unknown method '{method_name}'")
            continue

        method_config = METHODS[method_name]
        method_dir = base_dir / method_config['dir']

        for model_short, model_pattern in method_config['models']:
            for dataset in selected_datasets:
                files = find_result_files(method_dir, dataset, model_pattern)

                if files:
                    # Use the file with most samples
                    best_file = None
                    best_acc = None

                    for f in files:
                        try:
                            results = load_results(f)
                            acc = calculate_accuracies(results, dataset)
                            if best_acc is None or acc['samples'] > best_acc.get('samples', 0):
                                best_acc = acc
                                best_file = f
                        except Exception as e:
                            print(f"Warning: Error loading {f}: {e}")

                    if best_acc:
                        key = (method_name, model_short, dataset)
                        all_results[key] = best_acc

    # Output results
    if args.output == 'json':
        output = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in all_results.items()}
        print(json.dumps(output, indent=2))
        return

    # Build method-model pairs for table
    method_models = []
    for method_name in selected_methods:
        if method_name in METHODS:
            for model_short, _ in METHODS[method_name]['models']:
                method_models.append((method_name, model_short))

    # Print tables
    print("\n" + "=" * 100)
    print("## TABLE 1: API-LEVEL (PRODUCT) ACCURACY")
    print("*Correct when predicted mapping contains fields from expected source AND target products*\n")

    if args.output == 'csv':
        print("Method,Model," + ",".join(selected_datasets))
    else:
        header = "| Method | Model | " + " | ".join(selected_datasets) + " |"
        print(header)
        print("|" + "--------|" * 2 + "--------|" * len(selected_datasets))

    for method, model in method_models:
        row = [method, model]
        for ds in selected_datasets:
            r = all_results.get((method, model, ds))
            if r:
                row.append(f"{r['api_accuracy']*100:.1f}%")
            else:
                row.append('--')

        if args.output == 'csv':
            print(",".join(row))
        else:
            print(f"| {' | '.join(row)} |")

    print("\n## TABLE 2: FIELD-LEVEL ACCURACY (Given Correct API)")
    print("*Field mapping accuracy calculated ONLY on samples where API selection was correct*\n")

    if args.output == 'csv':
        print("Method,Model," + ",".join(selected_datasets))
    else:
        header = "| Method | Model | " + " | ".join(selected_datasets) + " |"
        print(header)
        print("|" + "--------|" * 2 + "--------|" * len(selected_datasets))

    for method, model in method_models:
        row = [method, model]
        for ds in selected_datasets:
            r = all_results.get((method, model, ds))
            if r and r['api_correct_with_gt'] > 0:
                row.append(f"{r['field_accuracy']*100:.1f}%")
            else:
                row.append('N/A')

        if args.output == 'csv':
            print(",".join(row))
        else:
            print(f"| {' | '.join(row)} |")

    print("\n## Sample Counts (API Correct / Total)")

    if args.output == 'csv':
        print("Method,Model," + ",".join(selected_datasets))
    else:
        header = "| Method | Model | " + " | ".join(selected_datasets) + " |"
        print(header)
        print("|" + "--------|" * 2 + "--------|" * len(selected_datasets))

    for method, model in method_models:
        row = [method, model]
        for ds in selected_datasets:
            r = all_results.get((method, model, ds))
            if r:
                row.append(f"{r['api_correct']}/{r['samples']}")
            else:
                row.append('--')

        if args.output == 'csv':
            print(",".join(row))
        else:
            print(f"| {' | '.join(row)} |")


if __name__ == '__main__':
    main()
