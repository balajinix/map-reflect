#!/usr/bin/env python3
"""Debug OMOP ablation base value discrepancy.
Old Table 8: CMS base=41.4%, MIMIC base=41.2%
New ablation: CMS base=16.6%, MIMIC base=23.5%
"""
import json
import os
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# Find all base result files for OMOP
for ds in ["OMOP_CMS", "OMOP_MIMIC", "OMOP_Synthea"]:
    ds_lower = ds.lower().replace("_", "_")

    # Old base (ue_reflect format)
    old_dir = BASE / f"results/omop_ablation/{ds}/base"
    # New base (mapreflect format)
    new_dir = BASE / f"results/omop_ablation/{ds}/base"

    print(f"\n{'='*60}")
    print(f"{ds}")
    print(f"{'='*60}")

    if old_dir.exists():
        for f in sorted(old_dir.glob("*.json")):
            if "checkpoint" in f.name:
                continue
            with open(f) as fh:
                d = json.load(fh)
            results = d.get("results", [])
            print(f"\n  File: {f.name}")
            print(f"  Samples: {len(results)}")

            if not results:
                continue

            r0 = results[0]
            print(f"  Top-level keys: {sorted(r0.keys())}")

            # Method 1: field accuracy (correct/predicted)
            c1 = t1 = 0
            for r in results:
                preds = r.get("predictions", {})
                gt = r.get("ground_truth_mapping", {})
                for src, pd in preds.items():
                    if not isinstance(pd, dict):
                        continue
                    rnk = pd.get("initial_ranking", [])
                    if not rnk:
                        continue
                    pred = rnk[0].get("id", "")
                    if pred and pred != "NO_MATCH":
                        t1 += 1
                        gt_t = gt.get(src, "")
                        if gt_t and pred.lower() == gt_t.lower():
                            c1 += 1

            # Method 2: field recall (correct/GT fields)
            total_gt = sum(len(r.get("ground_truth_mapping", {})) for r in results)

            # Method 3: per-sample F1 (how main tables compute it)
            sample_f1s = []
            for r in results:
                preds = r.get("predictions", {})
                gt = r.get("ground_truth_mapping", {})
                pred_set = set()
                gt_set = set()
                for src, pd in preds.items():
                    if not isinstance(pd, dict):
                        continue
                    rnk = pd.get("initial_ranking", [])
                    if not rnk:
                        continue
                    pred = rnk[0].get("id", "")
                    if pred and pred != "NO_MATCH":
                        pred_set.add((src.lower(), pred.lower()))
                for src, tgt in gt.items():
                    gt_set.add((src.lower(), tgt.lower()))

                tp = len(pred_set & gt_set)
                p = tp / len(pred_set) if pred_set else 0
                rec = tp / len(gt_set) if gt_set else 0
                f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0
                sample_f1s.append(f1)

            # Method 4: use the stored accuracy/field_accuracy
            stored_accs = [r.get("accuracy", r.get("field_accuracy", None)) for r in results]
            stored_accs = [a for a in stored_accs if a is not None]

            print(f"  Method 1 (correct/predicted): {c1}/{t1} = {c1/t1*100:.1f}%" if t1 else "  Method 1: no predictions")
            print(f"  Method 2 (correct/GT): {c1}/{total_gt} = {c1/total_gt*100:.1f}%" if total_gt else "  Method 2: no GT")
            print(f"  Method 3 (mean per-sample F1): {np.mean(sample_f1s)*100:.1f}%")
            if stored_accs:
                print(f"  Method 4 (stored accuracy): mean={np.mean(stored_accs)*100:.1f}%")

            # Check mean_uncertainty
            uncs = [r.get("mean_uncertainty", 0) for r in results]
            print(f"  Mean uncertainty: {np.mean(uncs):.4f}")
