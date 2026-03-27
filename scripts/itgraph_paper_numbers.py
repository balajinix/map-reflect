#!/usr/bin/env python3
"""Collect all ITGraph/ITOM-Hashicorp experiment results for the paper."""
import json
import os
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent.parent

ALL_PRODUCTS = ["instana","concert","turbonomic","hcp_terraform","infragraph","packer","aws","azure","gcp","hcp"]

def get_product(field):
    for p in sorted(ALL_PRODUCTS, key=len, reverse=True):
        if field.startswith(p + "_"):
            return p
    return field.split("_")[0]

def analyze_mr(path):
    """Analyze Map-Reflect format (predictions with initial_ranking)."""
    with open(path) as f:
        data = json.load(f)
    results = data.get("results", [])
    correct = total = no_match = 0
    scores = []
    for r in results:
        preds = r.get("predictions", {})
        gt = r.get("ground_truth_mapping", {})
        for src, pd in preds.items():
            if not isinstance(pd, dict): continue
            ranking = pd.get("initial_ranking", [])
            if not ranking: continue
            pred = ranking[0].get("id", "")
            score = ranking[0].get("score", 0)
            if pred == "NO_MATCH":
                no_match += 1; total += 1; continue
            if pred:
                total += 1; scores.append(score)
                gt_t = gt.get(src, "")
                if gt_t and pred.lower() == gt_t.lower():
                    correct += 1
    return correct, total, no_match, scores, len(results)

def analyze_mm(path):
    """Analyze MatchMaker format (predicted_mapping)."""
    with open(path) as f:
        data = json.load(f)
    results = data if isinstance(data, list) else data.get("results", [])
    correct = total = 0
    scores = []
    for r in results:
        pm = r.get("predicted_mapping", {})
        gt = r.get("ground_truth_mapping", {})
        for src, ti in pm.items():
            tgt = ti.get("target_property", "") if isinstance(ti, dict) else str(ti)
            if tgt:
                total += 1
                gt_t = gt.get(src, "")
                if gt_t and tgt.lower() == gt_t.lower():
                    correct += 1
    return correct, total, 0, scores, len(results)

def analyze_generic(path):
    """Try both formats."""
    with open(path) as f:
        data = json.load(f)
    results = data.get("results", []) if isinstance(data, dict) else data
    if not results:
        return 0, 0, 0, [], 0
    # Check format
    r0 = results[0]
    if "predictions" in r0 and isinstance(r0.get("predictions"), dict):
        return analyze_mr(path)
    elif "predicted_mapping" in r0:
        return analyze_mm(path)
    return 0, 0, 0, [], len(results)

def main():
    print("=" * 70)
    print("ITGraph / ITOM-Hashicorp — Complete Experiment Results for Paper")
    print("=" * 70)

    # === Dataset stats ===
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    for name, path in [
        ("ITGraph (all, >=1 method)", "data/itgraph_dataset.jsonl"),
        ("ITGraph_HC (>=2 methods)", "data/itgraph_dataset_high_confidence.jsonl"),
        ("ITOM_Hashicorp (original)", "data/itom_hashicorp_dataset.jsonl"),
    ]:
        fpath = BASE / path
        if not fpath.exists(): continue
        with open(fpath) as f:
            samples = [json.loads(l) for l in f if l.strip()]
        products = set()
        cross = same = 0
        total_fields = sum(len(s["mapping"]) for s in samples)
        for s in samples:
            products.add(s["source_schema"]); products.add(s["target_schema"])
            if s["source_schema"] != s["target_schema"]: cross += 1
            else: same += 1
        print(f"\n  {name}:")
        print(f"    Type pairs: {len(samples)} ({cross} cross-product, {same} same-product)")
        print(f"    Field mappings: {total_fields}")
        print(f"    Products: {len(products)}")

    # === Method results ===
    experiments = [
        # (label, path, analyzer)
        ("Map-Reflect (Llama) on ITOM_Hashicorp",
         "map_reflect/results/mapreflect_itom_hashicorp_meta-llama_llama-3-3-70b-instruct.json", "mr"),
        ("Map-Reflect (Granite) on ITOM_Hashicorp",
         "map_reflect/results/mapreflect_itom_hashicorp_ibm_granite-4-h-small.json", "mr"),
        ("Map-Reflect (Mistral) on ITOM_Hashicorp",
         "map_reflect/results/mapreflect_itom_hashicorp_mistralai_mistral-medium-2505.json", "mr"),
        ("MatchMaker (Llama) on ITOM_Hashicorp",
         "match_maker/results/ITOM_Hashicorp_llama_full.json", "mm"),
        ("MatchMaker (Granite) on ITOM_Hashicorp",
         "match_maker/results/ITOM_Hashicorp_granite_full.json", "mm"),
        ("ReMatch (Llama) on ITOM_Hashicorp",
         "rematch/results/ITOM_Hashicorp_llama_full.json", "mm"),
        ("Map-Reflect (Llama) on ITGraph_HC",
         "map_reflect/results/mapreflect_itgraph_hc_meta-llama_llama-3-3-70b-instruct.json", "mr"),
        ("Map-Reflect (Granite) on ITGraph_HC",
         "map_reflect/results/mapreflect_itgraph_hc_ibm_granite-4-h-small.json", "mr"),
        ("Map-Reflect+DocRAG (Llama) on ITGraph_HC",
         "map_reflect/results/doc_context/mapreflect_itgraph_hc_meta-llama_llama-3-3-70b-instruct.json", "mr"),
        # Constraint ablation
        ("Ablation: base (Llama) ITOM_Hashicorp",
         "results/constraint_ablation/itom_hashi_base/mapreflect_itom_hashicorp_meta-llama_llama-3-3-70b-instruct.json", "mr"),
        ("Ablation: constrained (Llama) ITOM_Hashicorp",
         "results/constraint_ablation/itom_hashi_constrained/mapreflect_itom_hashicorp_meta-llama_llama-3-3-70b-instruct.json", "mr"),
        ("Ablation: UQ (Llama) ITOM_Hashicorp",
         "results/constraint_ablation/itom_hashi_uq/mapreflect_itom_hashicorp_meta-llama_llama-3-3-70b-instruct.json", "mr"),
        ("Ablation: constrained+UQ (Llama) ITOM_Hashicorp",
         "results/constraint_ablation/itom_hashi_constrained_uq/mapreflect_itom_hashicorp_meta-llama_llama-3-3-70b-instruct.json", "mr"),
        # Generic results dir
        ("Map-Reflect (Llama) results/ ITOM_Hashicorp",
         "results/ITOM_Hashicorp_meta-llama_llama-3-3-70b-instruct.json", "generic"),
        ("Map-Reflect (Granite) results/ ITGraph_HC",
         "results/ITGraph_HC_ibm_granite-4-h-small.json", "generic"),
    ]

    print("\n" + "=" * 70)
    print("METHOD RESULTS")
    print("=" * 70)
    print(f"\n  {'Method':<50s} {'Correct':>7s} {'Total':>7s} {'Acc%':>7s} {'Samples':>7s}")
    print(f"  {'-'*50} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    for label, rel_path, fmt in experiments:
        fpath = BASE / rel_path
        if not fpath.exists():
            continue
        if fmt == "mr":
            c, t, nm, sc, n = analyze_mr(fpath)
        elif fmt == "mm":
            c, t, nm, sc, n = analyze_mm(fpath)
        else:
            c, t, nm, sc, n = analyze_generic(fpath)
        acc = c / t * 100 if t else 0
        print(f"  {label:<50s} {c:>7d} {t:>7d} {acc:>6.1f}% {n:>7d}")

    # === Doc Context comparison ===
    print("\n" + "=" * 70)
    print("DOC CONTEXT ABLATION (ITGraph_HC, Llama)")
    print("=" * 70)

    baseline = BASE / "map_reflect/results/mapreflect_itgraph_hc_meta-llama_llama-3-3-70b-instruct.json"
    doc_ctx = BASE / "map_reflect/results/doc_context/mapreflect_itgraph_hc_meta-llama_llama-3-3-70b-instruct.json"

    if baseline.exists() and doc_ctx.exists():
        bc, bt, bnm, bsc, bn = analyze_mr(baseline)
        dc, dt, dnm, dsc, dn = analyze_mr(doc_ctx)
        bacc = bc / bt * 100 if bt else 0
        dacc = dc / dt * 100 if dt else 0
        bscore = sum(bsc) / len(bsc) if bsc else 0
        dscore = sum(dsc) / len(dsc) if dsc else 0

        print(f"\n  {'Variant':<35s} {'Acc%':>8s} {'Mean Score':>12s} {'Correct/Total':>15s}")
        print(f"  {'-'*35} {'-'*8} {'-'*12} {'-'*15}")
        print(f"  {'Baseline (no docs)':<35s} {bacc:>7.1f}% {bscore:>11.3f} {bc:>6d}/{bt:<6d}")
        print(f"  {'+ Doc Context (RAG)':<35s} {dacc:>7.1f}% {dscore:>11.3f} {dc:>6d}/{dt:<6d}")
        print(f"  {'Delta':<35s} {dacc-bacc:>+7.1f}% {dscore-bscore:>+11.3f} {dc-bc:>+6d}")

    # === Per-product-pair breakdown for doc context ===
    if doc_ctx.exists():
        print("\n  Per product pair (doc context):")
        with open(doc_ctx) as f:
            data = json.load(f)
        pair_stats = defaultdict(lambda: [0, 0])
        for r in data.get("results", []):
            preds = r.get("predictions", {})
            gt = r.get("ground_truth_mapping", {})
            for src, pd in preds.items():
                if not isinstance(pd, dict): continue
                ranking = pd.get("initial_ranking", [])
                if not ranking: continue
                pred = ranking[0].get("id", "")
                if not pred or pred == "NO_MATCH": continue
                sp = get_product(src)
                tp = get_product(pred)
                pair_stats[f"{sp} -> {tp}"][0] += 1
                gt_t = gt.get(src, "")
                if gt_t and pred.lower() == gt_t.lower():
                    pair_stats[f"{sp} -> {tp}"][1] += 1

        for pair in sorted(pair_stats, key=lambda x: -pair_stats[x][0])[:12]:
            t, c = pair_stats[pair]
            print(f"    {pair:35s} {c:3d}/{t:3d} ({c/t*100:5.1f}%)")

    # === Summary table for paper ===
    print("\n" + "=" * 70)
    print("PAPER TABLE: ITGraph_HC Field Accuracy (%)")
    print("=" * 70)
    print("""
  | Method               | Model   | Field Acc | Notes                    |
  |----------------------|---------|-----------|--------------------------|""")

    paper_rows = [
        ("Map-Reflect", "Llama", "map_reflect/results/mapreflect_itgraph_hc_meta-llama_llama-3-3-70b-instruct.json", "mr", "baseline"),
        ("Map-Reflect", "Granite", "map_reflect/results/mapreflect_itgraph_hc_ibm_granite-4-h-small.json", "mr", ""),
        ("Map-Reflect+DocRAG", "Llama", "map_reflect/results/doc_context/mapreflect_itgraph_hc_meta-llama_llama-3-3-70b-instruct.json", "mr", "+doc context"),
        ("MatchMaker", "Llama", "match_maker/results/ITOM_Hashicorp_llama_full.json", "mm", "on ITOM_Hashicorp"),
        ("MatchMaker", "Granite", "match_maker/results/ITOM_Hashicorp_granite_full.json", "mm", "on ITOM_Hashicorp"),
        ("ReMatch", "Llama", "rematch/results/ITOM_Hashicorp_llama_full.json", "mm", "on ITOM_Hashicorp"),
    ]
    for method, model, rel_path, fmt, notes in paper_rows:
        fpath = BASE / rel_path
        if not fpath.exists():
            print(f"  | {method:<20s} | {model:<7s} | {'N/A':>9s} | {notes + ' (missing)':24s} |")
            continue
        if fmt == "mr":
            c, t, nm, sc, n = analyze_mr(fpath)
        else:
            c, t, nm, sc, n = analyze_mm(fpath)
        acc = c / t * 100 if t else 0
        print(f"  | {method:<20s} | {model:<7s} | {acc:>8.1f}% | {notes:24s} |")

    print("""
  Dataset: ITGraph_HC — 246 type pairs, 561 bipartite field mappings,
  10 products, many-to-many type-level, bipartite field-level.
  Ground truth built from method consensus (>=2 methods agree).
""")

if __name__ == "__main__":
    main()
