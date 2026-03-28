#!/usr/bin/env python3
"""Spider GNN: oracle type eval + field eval."""
import json
import pickle
import random
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gnn"))
from gnn_model import load_model_safely

BASE = Path(__file__).resolve().parent.parent


def main():
    model = load_model_safely(str(BASE / "data/gnn_results/spider_mapping_gnn_results.pkl"))
    model.eval()

    with open(BASE / "data/spider_graph_data.pkl", "rb") as f:
        saved = pickle.load(f)

    data = saved["data"]
    node_list = saved["node_list"]
    node_to_idx = saved.get("node_to_idx", {n: i for i, n in enumerate(node_list)})
    idx_to_node = {i: n for n, i in node_to_idx.items()}
    graph = saved["graph"]

    # Build indices
    db_to_types = defaultdict(list)
    type_to_fields = defaultdict(list)
    for node, attrs in graph.nodes(data=True):
        if attrs.get("node_type") == "type" and node in node_to_idx:
            db_to_types[attrs.get("product", "")].append(node)
        elif attrs.get("node_type") == "field" and node in node_to_idx:
            product = attrs.get("product", "")
            type_name = attrs.get("type", "")
            type_to_fields[product + "_" + type_name].append(node)

    with open(BASE / "data/spider_dataset.jsonl") as f:
        samples = [json.loads(l) for l in f if l.strip()]

    print(f"Nodes: {len(node_list)}, Samples: {len(samples)}")

    # === Oracle type: score > 0.5 for GT pair ===
    oracle_correct = oracle_total = 0
    rank_correct = rank_total = 0
    field_correct = field_total = 0

    for s in samples:
        db = s["database"]
        result = s["result"]
        props = result.get("properties", {})
        if not props:
            continue

        src_table = result["name"]
        src_type = db + "_" + src_table

        gt_tgt_tables = set()
        for tgt_list in props.values():
            if isinstance(tgt_list, list):
                for ti in tgt_list:
                    tbl = ti.get("mapTable", "")
                    if tbl:
                        gt_tgt_tables.add(db + "_" + tbl)

        if not gt_tgt_tables or src_type not in node_to_idx:
            continue

        # Oracle: does GNN score the GT pair > 0.5?
        gt_tgt = list(gt_tgt_tables)[0]
        if gt_tgt in node_to_idx:
            src_idx = node_to_idx[src_type]
            tgt_idx = node_to_idx[gt_tgt]
            query = torch.tensor([[src_idx], [tgt_idx]], dtype=torch.long)
            with torch.no_grad():
                score = model.predict_links(data.x, data.edge_index, query).item()
            oracle_total += 1
            if score > 0.5:
                oracle_correct += 1

        # Rank: is GT the top-ranked type in the same DB?
        cands = [t for t in db_to_types.get(db, [])
                 if t != src_type and t in node_to_idx]
        if cands:
            src_idx = node_to_idx[src_type]
            cand_indices = [node_to_idx[t] for t in cands]
            query = torch.tensor(
                [[src_idx] * len(cand_indices), cand_indices], dtype=torch.long
            )
            with torch.no_grad():
                scores = model.predict_links(data.x, data.edge_index, query).cpu().numpy()
            best = cands[np.argmax(scores)]
            rank_total += 1
            if best in gt_tgt_tables:
                rank_correct += 1

        # Field accuracy
        for src_field_key, tgt_list in props.items():
            if not isinstance(tgt_list, list):
                continue
            for tgt_info in tgt_list:
                tgt_table = tgt_info.get("mapTable", "")
                tgt_field = tgt_info.get("mapProperty", "")
                if not tgt_table or not tgt_field:
                    continue

                if src_field_key not in node_to_idx:
                    continue

                tgt_type_key = db + "_" + tgt_table
                candidates = type_to_fields.get(tgt_type_key, [])
                candidate_indices = [node_to_idx[c] for c in candidates if c in node_to_idx]
                if not candidate_indices:
                    continue

                src_idx = node_to_idx[src_field_key]
                query = torch.tensor(
                    [[src_idx] * len(candidate_indices), candidate_indices],
                    dtype=torch.long,
                )
                with torch.no_grad():
                    sc = model.predict_links(data.x, data.edge_index, query)
                best_idx = candidate_indices[np.argmax(sc.cpu().numpy())]
                predicted = idx_to_node[best_idx]

                field_total += 1
                if predicted == tgt_field:
                    field_correct += 1

    print(f"\n{'='*50}")
    print(f"GNN on Spider")
    print(f"{'='*50}")
    print(f"  Oracle type (score > 0.5):  {oracle_correct}/{oracle_total} ({oracle_correct/oracle_total*100:.1f}%)")
    print(f"  Rank type (top-1 in DB):    {rank_correct}/{rank_total} ({rank_correct/rank_total*100:.1f}%)")
    print(f"  Field accuracy:             {field_correct}/{field_total} ({field_correct/field_total*100:.1f}%)")


if __name__ == "__main__":
    main()
