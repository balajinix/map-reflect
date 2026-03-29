#!/usr/bin/env python3
"""Debug OMOP ablation: answer 3 questions."""
import json
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# Use the base file (old ue_reflect format)
base_path = BASE / "results/omop_ablation/OMOP_CMS/base/ue_reflect_omop_cms_meta-llama_llama-3-3-70b-instruct.json"
# And a new config for comparison
new_path = BASE / "results/omop_ablation/OMOP_CMS/self_refl/mapreflect_omop_cms_meta-llama_llama-3-3-70b-instruct.json"
# Main results (oracle mode)
main_path = BASE / "map_reflect/results/mapreflect_omop_cms_meta-llama_llama-3-3-70b-instruct.json"

for label, path in [("Base (ablation)", base_path), ("Self-refl (ablation)", new_path), ("Main oracle", main_path)]:
    if not path.exists():
        print(f"\n{label}: NOT FOUND ({path})")
        continue

    with open(path) as f:
        d = json.load(f)
    results = d.get("results", [])

    print(f"\n{'='*60}")
    print(f"{label}: {path.name}")
    print(f"{'='*60}")
    print(f"Samples: {len(results)}")

    # Q1: Is evaluation conditioned on API correctness?
    total_gt = sum(len(r.get("ground_truth_mapping", {})) for r in results)
    total_preds = 0
    total_correct = 0
    type_correct_count = 0

    for r in results:
        preds = r.get("predictions", {})
        gt = r.get("ground_truth_mapping", {})
        tc = r.get("type_correct_count", r.get("correct_count", 0))
        if tc > 0:
            type_correct_count += 1

        for src, pd in preds.items():
            if not isinstance(pd, dict):
                continue
            rnk = pd.get("initial_ranking", [])
            if not rnk:
                continue
            pred = rnk[0].get("id", "")
            if pred and pred != "NO_MATCH":
                total_preds += 1
                gt_t = gt.get(src, "")
                if gt_t and pred.lower() == gt_t.lower():
                    total_correct += 1

    print(f"\nQ1: API conditioning")
    print(f"  Type-correct samples: {type_correct_count}/{len(results)}")
    print(f"  Total GT fields: {total_gt}")
    print(f"  Total predicted (non-NO_MATCH): {total_preds}")
    print(f"  Total correct: {total_correct}")
    print(f"  Field accuracy (correct/predicted): {total_correct/total_preds*100:.1f}%" if total_preds else "  No predictions")
    print(f"  Field recall (correct/GT): {total_correct/total_gt*100:.1f}%" if total_gt else "  No GT")

    # Q2: Field count
    field_counts = [len(r.get("ground_truth_mapping", {})) for r in results]
    print(f"\nQ2: Field counts")
    print(f"  GT fields total: {total_gt}")
    print(f"  Mean GT fields/sample: {np.mean(field_counts):.1f}")
    print(f"  Min/Max: {min(field_counts)}/{max(field_counts)}")

    # Q3: Check config
    meta = d.get("metadata", {})
    config = d.get("config", {})
    print(f"\nQ3: Config")
    if meta:
        print(f"  Metadata: {json.dumps(meta, indent=2)[:200]}")
    if config:
        print(f"  Config: {json.dumps(config, indent=2)[:200]}")
    # Check mean_uncertainty to infer if UQ was on
    uncs = [r.get("mean_uncertainty", 0) for r in results]
    print(f"  Mean uncertainty: {np.mean(uncs):.4f}")
