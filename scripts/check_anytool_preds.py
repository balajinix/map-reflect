#!/usr/bin/env python3
import json
from pathlib import Path

path = Path(__file__).resolve().parent.parent / "anytool/results/faithful_itopsgraph_llama_test.json"
with open(path) as f:
    results = json.load(f)

for r in results[:3]:
    sid = r["sample_id"]
    pm = r["predicted_mapping"]
    gt = r["ground_truth_mapping"]
    print(f"Sample: {sid}")
    print(f"  Predicted ({len(pm)}):")
    for k, v in list(pm.items())[:4]:
        tgt = v.get("target_property", "") if isinstance(v, dict) else v
        print(f"    {k} -> {tgt}")
    print(f"  Ground truth ({len(gt)}):")
    for k, v in list(gt.items())[:4]:
        print(f"    {k} -> {v}")
    print()
