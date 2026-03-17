#!/usr/bin/env python3
"""
Analyze UQ Method Comparison Results

Computes error detection metrics for each UQ method:
- Precision: Of predictions flagged as uncertain, how many are actually wrong?
- Recall: Of wrong predictions, how many were flagged as uncertain?
- F1 Score: Harmonic mean of precision and recall
- AUROC: Area under ROC curve

Usage:
    python scripts/analyze_uq_methods.py results/uq_method_comparison
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def load_results(results_dir: Path) -> dict:
    """Load results from all UQ methods."""
    methods = {}
    for method_dir in results_dir.iterdir():
        if not method_dir.is_dir():
            continue
        method_name = method_dir.name

        # Find the JSON results file
        json_files = list(method_dir.glob("*.json"))
        if not json_files:
            continue

        with open(json_files[0]) as f:
            methods[method_name] = json.load(f)

    return methods


def extract_predictions_with_uncertainty(results: dict) -> list:
    """Extract all predictions with their uncertainty and correctness."""
    predictions = []

    for sample in results.get("results", []):
        ground_truth = sample.get("ground_truth_mapping", {})

        for source_field, pred_info in sample.get("predictions", {}).items():
            # Get predicted target (top-1)
            ranking = pred_info.get("initial_ranking", [])
            if not ranking:
                continue

            predicted_target = ranking[0].get("id")
            expected_target = ground_truth.get(source_field)

            # Check correctness
            is_correct = predicted_target == expected_target

            # Get uncertainty
            uncertainty = pred_info.get("uncertainty", 0.0)
            uncertainty_info = pred_info.get("uncertainty_info", {})

            predictions.append({
                "source": source_field,
                "predicted": predicted_target,
                "expected": expected_target,
                "is_correct": is_correct,
                "uncertainty": uncertainty,
                "uncertainty_info": uncertainty_info
            })

    return predictions


def compute_error_detection_metrics(predictions: list, threshold: float = None) -> dict:
    """Compute error detection metrics at a given threshold."""
    if not predictions:
        return {}

    uncertainties = [p["uncertainty"] for p in predictions]
    is_correct = [p["is_correct"] for p in predictions]

    # Use median threshold if not specified
    if threshold is None:
        threshold = np.median(uncertainties)

    # Predictions flagged as uncertain (high uncertainty = potential error)
    flagged = [u > threshold for u in uncertainties]

    # True errors
    errors = [not c for c in is_correct]

    # Compute metrics
    tp = sum(1 for f, e in zip(flagged, errors) if f and e)  # Flagged and actually wrong
    fp = sum(1 for f, e in zip(flagged, errors) if f and not e)  # Flagged but correct
    fn = sum(1 for f, e in zip(flagged, errors) if not f and e)  # Not flagged but wrong
    tn = sum(1 for f, e in zip(flagged, errors) if not f and not e)  # Not flagged and correct

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "threshold": threshold,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_predictions": len(predictions),
        "n_errors": sum(errors),
        "n_flagged": sum(flagged)
    }


def compute_auroc(predictions: list) -> float:
    """Compute AUROC for error detection."""
    if not HAS_SKLEARN:
        return None

    if not predictions:
        return None

    uncertainties = np.array([p["uncertainty"] for p in predictions])
    errors = np.array([not p["is_correct"] for p in predictions])

    # Need at least one of each class
    if errors.sum() == 0 or errors.sum() == len(errors):
        return None

    try:
        return roc_auc_score(errors, uncertainties)
    except:
        return None


def compute_aupr(predictions: list) -> float:
    """Compute Area Under Precision-Recall curve."""
    if not HAS_SKLEARN:
        return None

    if not predictions:
        return None

    uncertainties = np.array([p["uncertainty"] for p in predictions])
    errors = np.array([not p["is_correct"] for p in predictions])

    if errors.sum() == 0 or errors.sum() == len(errors):
        return None

    try:
        precision, recall, _ = precision_recall_curve(errors, uncertainties)
        return auc(recall, precision)
    except:
        return None


def analyze_method(name: str, results: dict) -> dict:
    """Analyze a single UQ method."""
    predictions = extract_predictions_with_uncertainty(results)

    if not predictions:
        return {"name": name, "error": "No predictions found"}

    # Basic stats
    n_total = len(predictions)
    n_correct = sum(1 for p in predictions if p["is_correct"])
    n_errors = n_total - n_correct
    accuracy = n_correct / n_total if n_total > 0 else 0

    uncertainties = [p["uncertainty"] for p in predictions]
    mean_uq = np.mean(uncertainties)
    std_uq = np.std(uncertainties)
    median_uq = np.median(uncertainties)

    # Error detection metrics at various thresholds
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    threshold_metrics = {}
    for t in thresholds:
        threshold_metrics[f"t_{t:.2f}"] = compute_error_detection_metrics(predictions, t)

    # Median threshold metrics
    median_metrics = compute_error_detection_metrics(predictions, median_uq)

    # AUROC and AUPR
    auroc = compute_auroc(predictions)
    aupr = compute_aupr(predictions)

    # Uncertainty distribution for correct vs incorrect
    correct_uq = [p["uncertainty"] for p in predictions if p["is_correct"]]
    incorrect_uq = [p["uncertainty"] for p in predictions if not p["is_correct"]]

    return {
        "name": name,
        "n_predictions": n_total,
        "n_correct": n_correct,
        "n_errors": n_errors,
        "accuracy": accuracy,
        "mean_uncertainty": mean_uq,
        "std_uncertainty": std_uq,
        "median_uncertainty": median_uq,
        "min_uncertainty": min(uncertainties),
        "max_uncertainty": max(uncertainties),
        "mean_uq_correct": np.mean(correct_uq) if correct_uq else None,
        "mean_uq_incorrect": np.mean(incorrect_uq) if incorrect_uq else None,
        "auroc": auroc,
        "aupr": aupr,
        "median_threshold_metrics": median_metrics,
        "threshold_metrics": threshold_metrics
    }


def print_comparison_table(analyses: list):
    """Print comparison table of all methods."""
    print("\n" + "=" * 80)
    print("UQ METHOD COMPARISON RESULTS")
    print("=" * 80)

    # Basic stats
    print("\n1. BASIC STATISTICS")
    print("-" * 80)
    print(f"{'Method':<20} {'Accuracy':<12} {'Mean UQ':<12} {'Std UQ':<12} {'Errors':<10}")
    print("-" * 80)
    for a in analyses:
        if "error" in a:
            print(f"{a['name']:<20} {a['error']}")
        else:
            print(f"{a['name']:<20} {a['accuracy']*100:.1f}%{'':<7} "
                  f"{a['mean_uncertainty']:.4f}{'':<6} "
                  f"{a['std_uncertainty']:.4f}{'':<6} "
                  f"{a['n_errors']}")

    # Uncertainty separation
    print("\n2. UNCERTAINTY SEPARATION (higher is better)")
    print("-" * 80)
    print(f"{'Method':<20} {'Mean UQ (Correct)':<20} {'Mean UQ (Wrong)':<20} {'Gap':<10}")
    print("-" * 80)
    for a in analyses:
        if "error" in a:
            continue
        c = a.get('mean_uq_correct', 0) or 0
        w = a.get('mean_uq_incorrect', 0) or 0
        gap = w - c
        print(f"{a['name']:<20} {c:.4f}{'':<14} {w:.4f}{'':<14} {gap:+.4f}")

    # AUROC/AUPR
    print("\n3. ERROR DETECTION (AUROC/AUPR)")
    print("-" * 80)
    print(f"{'Method':<20} {'AUROC':<15} {'AUPR':<15}")
    print("-" * 80)
    for a in analyses:
        if "error" in a:
            continue
        auroc = a.get('auroc')
        aupr = a.get('aupr')
        auroc_str = f"{auroc:.4f}" if auroc is not None else "N/A"
        aupr_str = f"{aupr:.4f}" if aupr is not None else "N/A"
        print(f"{a['name']:<20} {auroc_str:<15} {aupr_str:<15}")

    # Error detection at median threshold
    print("\n4. ERROR DETECTION AT MEDIAN THRESHOLD")
    print("-" * 80)
    print(f"{'Method':<20} {'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<10}")
    print("-" * 80)
    for a in analyses:
        if "error" in a:
            continue
        m = a.get('median_threshold_metrics', {})
        print(f"{a['name']:<20} {m.get('threshold', 0):.4f}{'':<6} "
              f"{m.get('precision', 0):.4f}{'':<6} "
              f"{m.get('recall', 0):.4f}{'':<6} "
              f"{m.get('f1', 0):.4f}")

    # Best threshold for each method
    print("\n5. BEST THRESHOLD (Max F1)")
    print("-" * 80)
    print(f"{'Method':<20} {'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<10}")
    print("-" * 80)
    for a in analyses:
        if "error" in a:
            continue
        best_t = None
        best_f1 = 0
        for t_name, t_metrics in a.get('threshold_metrics', {}).items():
            if t_metrics.get('f1', 0) > best_f1:
                best_f1 = t_metrics['f1']
                best_t = t_metrics
        if best_t:
            print(f"{a['name']:<20} {best_t.get('threshold', 0):.4f}{'':<6} "
                  f"{best_t.get('precision', 0):.4f}{'':<6} "
                  f"{best_t.get('recall', 0):.4f}{'':<6} "
                  f"{best_t.get('f1', 0):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze UQ method comparison results")
    parser.add_argument("results_dir", type=Path, help="Directory containing UQ method results")
    parser.add_argument("--output", "-o", type=Path, help="Save analysis to JSON file")
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Error: {args.results_dir} does not exist")
        return

    print(f"Loading results from: {args.results_dir}")
    methods = load_results(args.results_dir)

    if not methods:
        print("No results found!")
        return

    print(f"Found {len(methods)} methods: {list(methods.keys())}")

    # Analyze each method
    analyses = []
    for name, results in methods.items():
        print(f"Analyzing: {name}...")
        analysis = analyze_method(name, results)
        analyses.append(analysis)

    # Print comparison
    print_comparison_table(analyses)

    # Save to JSON if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(analyses, f, indent=2, default=float)
        print(f"\nAnalysis saved to: {args.output}")


if __name__ == "__main__":
    main()
