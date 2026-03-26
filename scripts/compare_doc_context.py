#!/usr/bin/env python3
"""Compare Map-Reflect results with and without doc context."""
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

def analyze(path, label):
    with open(path) as f:
        data = json.load(f)
    results = data.get("results", [])

    correct = total = no_match = 0
    scores = []
    pair_stats = defaultdict(lambda: [0, 0])

    for r in results:
        preds = r.get("predictions", {})
        gt = r.get("ground_truth_mapping", {})
        for src, pd in preds.items():
            if not isinstance(pd, dict):
                continue
            ranking = pd.get("initial_ranking", [])
            if not ranking:
                continue
            pred = ranking[0].get("id", "")
            score = ranking[0].get("score", 0)
            if pred == "NO_MATCH":
                no_match += 1
                total += 1
                continue
            if pred:
                total += 1
                scores.append(score)
                sp = get_product(src)
                tp = get_product(pred)
                pair_stats[f"{sp} -> {tp}"][0] += 1
                gt_t = gt.get(src, "")
                if gt_t and pred.lower() == gt_t.lower():
                    correct += 1
                    pair_stats[f"{sp} -> {tp}"][1] += 1

    acc = correct / total * 100 if total else 0
    mean_score = sum(scores) / len(scores) if scores else 0
    metadata = data.get("metadata", {})
    mean_unc = metadata.get("mean_uncertainty", "n/a")

    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    print(f"  Field accuracy:   {correct}/{total} ({acc:.1f}%)")
    print(f"  NO_MATCH:         {no_match}")
    print(f"  Mean score:       {mean_score:.3f}")
    print(f"  Mean uncertainty: {mean_unc}")
    print(f"  Samples:          {len(results)}")
    print(f"\n  Per product pair (top 10):")
    for pair in sorted(pair_stats, key=lambda x: -pair_stats[x][0])[:10]:
        t, c = pair_stats[pair]
        print(f"    {pair:35s} {c:3d}/{t:3d} ({c/t*100:5.1f}%)")


if __name__ == "__main__":
    baseline = BASE / "map_reflect/results/mapreflect_itgraph_hc_meta-llama_llama-3-3-70b-instruct.json"
    doc_ctx = BASE / "map_reflect/results/doc_context/mapreflect_itgraph_hc_meta-llama_llama-3-3-70b-instruct.json"

    if baseline.exists():
        analyze(baseline, "BASELINE (no doc context)")
    if doc_ctx.exists():
        analyze(doc_ctx, "WITH DOC CONTEXT (RAG-enriched)")
