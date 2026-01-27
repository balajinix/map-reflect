#!/usr/bin/env python3
"""
Evaluate schema mapping methods against consensus ground truth.

This script:
1. Loads consensus ground truth generated from inter-annotator agreement
2. Evaluates each method's field mappings against the consensus
3. Reports precision, recall, and F1 for each method

Usage:
    python eval/eval_against_consensus.py
    python eval/eval_against_consensus.py --ground-truth data/itops_consensus_ground_truth.jsonl
    python eval/eval_against_consensus.py --methods SMT,ReAct,GNN
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(__file__).parent.parent
DEFAULT_GROUND_TRUTH = BASE_DIR / "data" / "itops_consensus_ground_truth.jsonl"

# Method directories
METHODS = {
    'SMT': 'smt/results',
    'ReAct': 'react/results',
    'GNN': 'gnn/results',
    'MapReflect': 'map_reflect/results',
    'AnyTool': 'anytool/results',
    'ReMatch': 'rematch/results',
    'MatchMaker': 'match_maker/results',
}


# =============================================================================
# Helper Functions
# =============================================================================

def normalize_field(field: str) -> str:
    """Normalize field name for comparison."""
    name = field.lower().strip()
    name = re.sub(r'^(nl2gql_|omop_)', '', name)
    return name


def extract_product(field: str) -> str:
    """Extract product from field name."""
    name = normalize_field(field)
    if '.' in name:
        name = name.split('.')[0]
    if '_' in name:
        parts = name.split('_')
        known = ['concert', 'instana', 'turbonomic', 'turbo', 'aws', 'azure', 'gcp']
        if parts[0] in known:
            return 'turbonomic' if parts[0] == 'turbo' else parts[0]
    return name.split('_')[0] if '_' in name else name


def load_ground_truth(gt_path: Path) -> Dict[Tuple[str, str], Set[Tuple[str, str]]]:
    """
    Load consensus ground truth file.

    Returns:
        Dict mapping (source_product, target_product) to set of (source_field, target_field) tuples
    """
    ground_truth = defaultdict(set)

    with open(gt_path) as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)

            src_product = sample['source_schema']['id']
            tgt_product = sample['target_schema']['id']
            pair_key = (src_product, tgt_product)

            for src_field, tgt_field in sample['mapping'].items():
                # Normalize for comparison
                src_norm = normalize_field(src_field)
                tgt_norm = normalize_field(tgt_field)
                ground_truth[pair_key].add((src_norm, tgt_norm))

    return ground_truth


def load_method_results(results_dir: Path) -> List[Dict]:
    """Load all ITOps results from a method's results directory."""
    results = []

    if not results_dir.exists():
        return results

    for f in results_dir.iterdir():
        if f.name.startswith('.') or 'itops' not in f.name.lower():
            continue
        if f.suffix not in ('.json', '.jsonl'):
            continue

        try:
            if f.suffix == '.jsonl':
                with open(f) as fp:
                    for line in fp:
                        if line.strip():
                            results.append(json.loads(line))
            else:
                with open(f) as fp:
                    data = json.load(fp)
                    if isinstance(data, dict) and 'results' in data:
                        results.extend(data['results'])
                    elif isinstance(data, list):
                        results.extend(data)
        except Exception as e:
            print(f"Warning: Error loading {f}: {e}")

    return results


def extract_mappings_from_results(results: List[Dict]) -> Set[Tuple[str, str]]:
    """Extract all (source_field, target_field) pairs from results."""
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

            src_norm = normalize_field(src_field)
            tgt_norm = normalize_field(tgt_field)
            mappings.add((src_norm, tgt_norm))

    return mappings


def calculate_metrics(
    predicted: Set[Tuple[str, str]],
    ground_truth: Set[Tuple[str, str]]
) -> Dict:
    """Calculate precision, recall, and F1."""
    if not predicted:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': 0}

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


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate methods against consensus ground truth'
    )
    parser.add_argument(
        '--ground-truth',
        type=Path,
        default=DEFAULT_GROUND_TRUTH,
        help='Path to consensus ground truth JSONL'
    )
    parser.add_argument(
        '--methods',
        type=str,
        default='all',
        help='Comma-separated methods to evaluate (default: all)'
    )
    parser.add_argument(
        '--base-dir',
        type=Path,
        default=BASE_DIR,
        help='Base directory for method results'
    )
    parser.add_argument(
        '--output',
        choices=['markdown', 'csv', 'json'],
        default='markdown',
        help='Output format'
    )
    parser.add_argument(
        '--by-product-pair',
        action='store_true',
        help='Show results broken down by product pair'
    )

    args = parser.parse_args()

    # Load ground truth
    if not args.ground_truth.exists():
        print(f"Error: Ground truth file not found: {args.ground_truth}")
        print("Run generate_ground_truth.py first to create it.")
        return 1

    print(f"Loading ground truth from {args.ground_truth}...")
    ground_truth_by_pair = load_ground_truth(args.ground_truth)

    # Flatten ground truth for overall evaluation
    all_ground_truth = set()
    for mappings in ground_truth_by_pair.values():
        all_ground_truth.update(mappings)

    print(f"Ground truth: {len(all_ground_truth)} field mappings across {len(ground_truth_by_pair)} product pairs")
    print()

    # Select methods
    if args.methods == 'all':
        selected_methods = list(METHODS.keys())
    else:
        selected_methods = [m.strip() for m in args.methods.split(',')]

    # Evaluate each method
    all_results = {}

    for method_name in selected_methods:
        if method_name not in METHODS:
            print(f"Warning: Unknown method '{method_name}'")
            continue

        method_dir = args.base_dir / METHODS[method_name]
        results = load_method_results(method_dir)

        if not results:
            continue

        predicted = extract_mappings_from_results(results)
        metrics = calculate_metrics(predicted, all_ground_truth)
        all_results[method_name] = metrics

    # Output results
    if args.output == 'json':
        print(json.dumps(all_results, indent=2))
        return 0

    # Print header
    print("=" * 90)
    print("EVALUATION AGAINST CONSENSUS GROUND TRUTH")
    print(f"Ground truth: {len(all_ground_truth)} mappings from inter-annotator agreement")
    print("=" * 90)
    print()

    # Table header
    if args.output == 'csv':
        print("Method,Precision,Recall,F1,TP,FP,FN,Predicted,GroundTruth")
    else:
        print("| Method | Precision | Recall | F1 | TP | FP | FN | Predicted | GT |")
        print("|--------|-----------|--------|----|----|----|----|-----------|-----|")

    # Table rows
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

    # Per-product-pair breakdown if requested
    if args.by_product_pair:
        print("\n" + "=" * 90)
        print("BREAKDOWN BY PRODUCT PAIR")
        print("=" * 90)

        for (src_prod, tgt_prod), gt_mappings in sorted(ground_truth_by_pair.items()):
            print(f"\n### {src_prod} -> {tgt_prod} ({len(gt_mappings)} GT mappings)")

            if args.output != 'csv':
                print("| Method | Precision | Recall | F1 | TP |")
                print("|--------|-----------|--------|----|----|")

            for method in selected_methods:
                if method not in METHODS:
                    continue

                method_dir = args.base_dir / METHODS[method]
                results = load_method_results(method_dir)
                if not results:
                    continue

                # Filter predictions by product pair
                predicted = set()
                for result in results:
                    pred_mapping = result.get('predicted_mapping', {})
                    for src_f, tgt_info in pred_mapping.items():
                        if isinstance(tgt_info, dict):
                            tgt_f = tgt_info.get('target_property', '')
                        else:
                            tgt_f = str(tgt_info)

                        if not tgt_f:
                            continue

                        if extract_product(src_f) == src_prod and extract_product(tgt_f) == tgt_prod:
                            predicted.add((normalize_field(src_f), normalize_field(tgt_f)))

                metrics = calculate_metrics(predicted, gt_mappings)
                if args.output != 'csv':
                    print(f"| {method:12} | {metrics['precision']*100:8.1f}% | "
                          f"{metrics['recall']*100:5.1f}% | {metrics['f1']*100:5.1f}% | {metrics['tp']:3} |")

    return 0


if __name__ == '__main__':
    exit(main())
