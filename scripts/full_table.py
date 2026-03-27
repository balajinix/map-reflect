#!/usr/bin/env python3
"""Print full type + field accuracy tables for paper."""
import json, os
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

def analyze(path):
    with open(path) as f:
        if str(path).endswith('.jsonl'):
            data = [json.loads(l) for l in f if l.strip()]
        else:
            d = json.load(f)
            data = d.get('results', []) if isinstance(d, dict) else d

    fc = ft = tc = tt = 0
    for r in data:
        preds = r.get('predictions', {})
        pm = r.get('predicted_mapping', {})
        gt = r.get('ground_truth_mapping', {})
        pred_types = set()
        gt_types = set()

        if isinstance(preds, dict) and preds:
            for src, pd in preds.items():
                if not isinstance(pd, dict): continue
                rnk = pd.get('initial_ranking', [])
                if not rnk: continue
                pred = rnk[0].get('id', '')
                if pred and pred != 'NO_MATCH':
                    ft += 1
                    pp = pred.rsplit('.', 1)
                    if len(pp) == 2: pred_types.add(pp[0])
                    gt_t = gt.get(src, '')
                    if gt_t:
                        gp = gt_t.rsplit('.', 1)
                        if len(gp) == 2: gt_types.add(gp[0])
                        if pred.lower() == gt_t.lower(): fc += 1
        elif pm:
            for src, ti in pm.items():
                tgt = ti.get('target_property', '') if isinstance(ti, dict) else str(ti)
                if tgt:
                    ft += 1
                    pp = tgt.rsplit('.', 1)
                    if len(pp) == 2: pred_types.add(pp[0])
                    gt_t = gt.get(src, '')
                    if gt_t:
                        gp = gt_t.rsplit('.', 1)
                        if len(gp) == 2: gt_types.add(gp[0])
                        if tgt.lower() == gt_t.lower(): fc += 1

        tt += 1
        if pred_types and gt_types and pred_types & gt_types:
            tc += 1

    return tc, tt, fc, ft

configs = [
    ('Map-Reflect', 'Llama', 'map_reflect/results/mapreflect_itgraph_hc_meta-llama_llama-3-3-70b-instruct.json'),
    ('Map-Reflect', 'Granite', 'map_reflect/results/mapreflect_itgraph_hc_ibm_granite-4-h-small.json'),
    ('Map-Reflect', 'Mistral', 'map_reflect/results/mapreflect_itgraph_hc_mistralai_mistral-medium-2505.json'),
    ('MR+DocRAG', 'Llama', 'map_reflect/results/doc_context/mapreflect_itgraph_hc_meta-llama_llama-3-3-70b-instruct.json'),
    ('MR+DocRAG', 'Granite', 'map_reflect/results/doc_context/mapreflect_itgraph_hc_ibm_granite-4-h-small.json'),
    ('MR+DocRAG', 'Mistral', 'map_reflect/results/doc_context/mapreflect_itgraph_hc_mistralai_mistral-medium-2505.json'),
    ('MatchMaker', 'Llama', 'match_maker/results/ITGraph_HC_llama_full.json'),
    ('MatchMaker', 'Granite', 'match_maker/results/ITGraph_HC_granite_full.json'),
    ('MatchMaker', 'Mistral', 'match_maker/results/ITGraph_HC_mistral_full.json'),
    ('AnyTool', 'Llama', 'anytool/results/ITGraph_HC_llama.json'),
    ('AnyTool', 'Granite', 'anytool/results/ITGraph_HC_granite.json'),
    ('AnyTool', 'Mistral', 'anytool/results/ITGraph_HC_mistral.json'),
]

grid = {}
for method, model, rel_path in configs:
    path = BASE / rel_path
    if method not in grid:
        grid[method] = {}
    if path.exists():
        tc, tt, fc, ft = analyze(path)
        tacc = tc/tt*100 if tt else 0
        facc = fc/ft*100 if ft else 0
        grid[method][model] = (tacc, facc, tc, tt, fc, ft)

methods = ['Map-Reflect', 'MR+DocRAG', 'MatchMaker', 'AnyTool']
models = ['Llama', 'Granite', 'Mistral']

def cell(method, model, idx):
    r = grid.get(method, {}).get(model)
    if r:
        return f"{r[idx]:.1f}%"
    return "TBA"

print("=" * 80)
print("TABLE 1: ITGraph_HC — Type-Level API Accuracy (%) [many-to-many]")
print("Dataset: 246 type pairs, 10 products, many-to-many type mappings")
print("=" * 80)
print()
header = f"  {'Method':<22s} | {'Llama':>10s} | {'Granite':>10s} | {'Mistral':>10s}"
print(header)
print("  " + "-" * (len(header) - 2))
for m in methods:
    print(f"  {m:<22s} | {cell(m,'Llama',0):>10s} | {cell(m,'Granite',0):>10s} | {cell(m,'Mistral',0):>10s}")

print()
print("=" * 80)
print("TABLE 2: ITGraph_HC — Field-Level Accuracy (%) [bipartite]")
print("Dataset: 561 bipartite field mappings across 246 type pairs")
print("=" * 80)
print()
print(header)
print("  " + "-" * (len(header) - 2))
for m in methods:
    print(f"  {m:<22s} | {cell(m,'Llama',1):>10s} | {cell(m,'Granite',1):>10s} | {cell(m,'Mistral',1):>10s}")

# Also print raw numbers
print()
print("=" * 80)
print("RAW NUMBERS (correct/total)")
print("=" * 80)
print()
print(f"  {'Method':<22s} | {'Metric':<6s} | {'Llama':>14s} | {'Granite':>14s} | {'Mistral':>14s}")
print("  " + "-" * 76)
for m in methods:
    for label, ci, ti in [("Type", 2, 3), ("Field", 4, 5)]:
        vals = []
        for model in models:
            r = grid.get(m, {}).get(model)
            if r:
                vals.append(f"{r[ci]}/{r[ti]}")
            else:
                vals.append("TBA")
        print(f"  {m:<22s} | {label:<6s} | {vals[0]:>14s} | {vals[1]:>14s} | {vals[2]:>14s}")
