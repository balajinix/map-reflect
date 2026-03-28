#!/usr/bin/env python3
"""
Fix AnyTool evaluation by handling format mismatches.

Issues:
1. predicted_mapping: {src_field: {target_property: tgt_field}}
2. ground_truth_mapping: {src_field: tgt_field} (ITOpsGraph) or reversed
3. Need to try both directions and flexible field name matching

Also recomputes accuracy for all datasets.
"""
import json
import os
import re
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent.parent


def normalize_field(field):
    """Normalize a field name for comparison."""
    return field.lower().strip().replace(" ", "")


def extract_type_and_field(full_key):
    """Extract (type, field) from keys like 'product_Type.field'."""
    if "." in full_key:
        parts = full_key.rsplit(".", 1)
        return parts[0], parts[1]
    return "", full_key


def match_predicted_to_gt(predicted_mapping, ground_truth_mapping):
    """Try to match predictions to ground truth, handling format variations.

    Returns (correct, total, details).
    """
    if not predicted_mapping or not ground_truth_mapping:
        return 0, 0, []

    correct = 0
    total = 0
    details = []

    # Normalize ground truth into all possible formats
    gt_pairs = set()  # (src_normalized, tgt_normalized)
    gt_by_src = {}
    gt_by_tgt = {}

    for k, v in ground_truth_mapping.items():
        if isinstance(v, str):
            gt_pairs.add((normalize_field(k), normalize_field(v)))
            gt_by_src[normalize_field(k)] = normalize_field(v)
            gt_by_tgt[normalize_field(v)] = normalize_field(k)
        elif isinstance(v, dict):
            tgt = v.get("target_property", v.get("mapProperty", ""))
            if tgt:
                gt_pairs.add((normalize_field(k), normalize_field(tgt)))
                gt_by_src[normalize_field(k)] = normalize_field(tgt)

    for src_field, pred_info in predicted_mapping.items():
        if isinstance(pred_info, dict):
            tgt_field = pred_info.get("target_property", "")
        elif isinstance(pred_info, str):
            tgt_field = pred_info
        else:
            continue

        if not tgt_field:
            continue

        total += 1
        src_norm = normalize_field(src_field)
        tgt_norm = normalize_field(tgt_field)

        # Try direct match
        if (src_norm, tgt_norm) in gt_pairs:
            correct += 1
            details.append(("match", src_field, tgt_field))
            continue

        # Try reversed match (GT might have inverted direction)
        if (tgt_norm, src_norm) in gt_pairs:
            correct += 1
            details.append(("reverse_match", src_field, tgt_field))
            continue

        # Try field-name-only match (ignore product/type prefix)
        _, src_fname = extract_type_and_field(src_field)
        _, tgt_fname = extract_type_and_field(tgt_field)

        # Check if src_fname -> tgt_fname appears anywhere in GT
        for gt_src, gt_tgt in gt_pairs:
            _, gt_src_f = extract_type_and_field(gt_src)
            _, gt_tgt_f = extract_type_and_field(gt_tgt)
            if (normalize_field(src_fname) == normalize_field(gt_src_f) and
                normalize_field(tgt_fname) == normalize_field(gt_tgt_f)):
                correct += 1
                details.append(("field_match", src_field, tgt_field))
                break
            if (normalize_field(src_fname) == normalize_field(gt_tgt_f) and
                normalize_field(tgt_fname) == normalize_field(gt_src_f)):
                correct += 1
                details.append(("field_reverse", src_field, tgt_field))
                break
        else:
            details.append(("miss", src_field, tgt_field))

    return correct, total, details


def evaluate_dataset(result_path, label):
    """Evaluate an AnyTool result file."""
    with open(result_path) as f:
        d = json.load(f)

    data = d if isinstance(d, list) else d.get("results", [])

    total_correct = 0
    total_preds = 0
    type_correct = 0
    type_total = 0
    match_types = defaultdict(int)

    for r in data:
        pm = r.get("predicted_mapping", {})
        gt = r.get("ground_truth_mapping", r.get("ground_truth", {}))

        if not pm:
            continue

        c, t, details = match_predicted_to_gt(pm, gt)
        total_correct += c
        total_preds += t

        # Type accuracy
        if t > 0:
            type_total += 1
            if c > 0:
                type_correct += 1

        for dtype, _, _ in details:
            match_types[dtype] += 1

    facc = total_correct / total_preds * 100 if total_preds else 0
    tacc = type_correct / type_total * 100 if type_total else 0

    print(f"  {label:55s} type={tacc:5.1f}%  field={total_correct}/{total_preds} ({facc:5.1f}%)  samples={len(data)}")
    if match_types:
        for mt, count in sorted(match_types.items()):
            print(f"    {mt}: {count}")

    return tacc, facc, total_correct, total_preds


def main():
    print("=" * 70)
    print("AnyTool Re-evaluation (with format fix)")
    print("=" * 70)

    results_dir = BASE / "anytool" / "results"
    datasets = [
        ("ITOpsGraph (Llama)", "ITOpsGraph_meta-llama_llama-3-3-70b-instruct.json"),
        ("ITOpsGraph (Granite)", "ITOpsGraph_ibm_granite-4-h-small.json"),
        ("NL2GQL (Llama)", "NL2GQL_meta-llama_llama-3-3-70b-instruct.json"),
        ("NL2GQL (Granite)", "NL2GQL_ibm_granite-4-h-small.json"),
        ("NL2GQL (Mistral)", "NL2GQL_mistral.json"),
        ("OMOP-CMS (Llama)", "OMOP_CMS_meta-llama_llama-3-3-70b-instruct.json"),
        ("OMOP-CMS (Granite)", "OMOP_CMS_ibm_granite-4-h-small.json"),
        ("OMOP-MIMIC (Llama)", "OMOP_MIMIC_meta-llama_llama-3-3-70b-instruct.json"),
        ("OMOP-MIMIC (Granite)", "OMOP_MIMIC_ibm_granite-4-h-small.json"),
        ("OMOP-Synthea (Llama)", "OMOP_Synthea_meta-llama_llama-3-3-70b-instruct.json"),
        ("OMOP-Synthea (Granite)", "OMOP_Synthea_ibm_granite-4-h-small.json"),
    ]

    print()
    for label, fname in datasets:
        path = results_dir / fname
        if path.exists():
            evaluate_dataset(path, label)
        else:
            print(f"  {label:55s} NOT FOUND")

    # Also show a few example predictions for debugging
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTIONS (ITOpsGraph Llama)")
    print("=" * 70)

    path = results_dir / "ITOpsGraph_meta-llama_llama-3-3-70b-instruct.json"
    with open(path) as f:
        d = json.load(f)
    data = d if isinstance(d, list) else d.get("results", [])

    shown = 0
    for r in data:
        pm = r.get("predicted_mapping", {})
        gt = r.get("ground_truth_mapping", {})
        if pm and gt:
            sid = r.get("sample_id", "?")
            print(f"\n  Sample: {sid}")
            print(f"  Predicted ({len(pm)} fields):")
            for k, v in list(pm.items())[:3]:
                tgt = v.get("target_property", "") if isinstance(v, dict) else v
                print(f"    {k} -> {tgt}")
            print(f"  Ground truth ({len(gt)} fields):")
            for k, v in list(gt.items())[:3]:
                print(f"    {k} -> {v}")
            shown += 1
            if shown >= 3:
                break


if __name__ == "__main__":
    main()
