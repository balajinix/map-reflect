#!/usr/bin/env python3
"""Compare model behavior on ITGraph_HC."""
import json
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
ALL_PRODUCTS = ["instana","concert","turbonomic","hcp_terraform","infragraph","packer","aws","azure","gcp","hcp"]

def get_product(field):
    for p in sorted(ALL_PRODUCTS, key=len, reverse=True):
        if field.startswith(p + "_"):
            return p
    return field.split("_")[0]

files = {
    "Mistral": BASE / "map_reflect/results/mapreflect_itgraph_hc_mistralai_mistral-medium-2505.json",
    "Llama": BASE / "map_reflect/results/mapreflect_itgraph_hc_meta-llama_llama-3-3-70b-instruct.json",
    "Granite": BASE / "map_reflect/results/mapreflect_itgraph_hc_ibm_granite-4-h-small.json",
}

for model, path in files.items():
    with open(path) as f:
        data = json.load(f)
    results = data.get("results", [])

    scores = []
    pair_stats = defaultdict(lambda: [0, 0])
    no_match = wrong_product = total = correct = 0

    for r in results:
        preds = r.get("predictions", {})
        gt = r.get("ground_truth_mapping", {})
        for src, pd in preds.items():
            if not isinstance(pd, dict): continue
            rnk = pd.get("initial_ranking", [])
            if not rnk: continue
            pred = rnk[0].get("id", "")
            score = rnk[0].get("score", 0)
            if pred == "NO_MATCH":
                no_match += 1; continue
            if not pred: continue
            total += 1
            scores.append(score)
            sp, tp = get_product(src), get_product(pred)
            pair_stats[f"{sp} -> {tp}"][0] += 1
            gt_t = gt.get(src, "")
            if gt_t:
                if pred.lower() == gt_t.lower():
                    correct += 1
                    pair_stats[f"{sp} -> {tp}"][1] += 1
                if get_product(pred) != get_product(gt_t):
                    wrong_product += 1

    scores.sort()
    n = len(scores)
    print(f"\n{'='*60}")
    print(f"{model}")
    print(f"{'='*60}")
    print(f"  Correct:       {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"  NO_MATCH:      {no_match}")
    print(f"  Wrong product: {wrong_product}/{total} ({wrong_product/total*100:.1f}%)")
    if n > 0:
        print(f"  Scores: min={scores[0]:.2f} p25={scores[n//4]:.2f} "
              f"median={scores[n//2]:.2f} p75={scores[3*n//4]:.2f} max={scores[-1]:.2f}")
    print(f"  Per product pair:")
    for pair in sorted(pair_stats, key=lambda x: -pair_stats[x][0])[:10]:
        t, c = pair_stats[pair]
        print(f"    {pair:30s} {c:3d}/{t:3d} ({c/t*100:5.1f}%)")
