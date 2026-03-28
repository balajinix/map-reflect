#!/usr/bin/env python3
"""Evaluate GNN on Spider at both type and field level."""
import json
import pickle
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gnn"))
from gnn_model import MappingKGGNN, load_model_safely

BASE = Path(__file__).resolve().parent.parent


def main():
    model_path = BASE / "data" / "gnn_results" / "spider_mapping_gnn_results.pkl"
    graph_path = BASE / "data" / "spider_graph_data.pkl"

    print("Loading GNN model...")
    model = load_model_safely(str(model_path))
    model.eval()

    with open(graph_path, "rb") as f:
        saved = pickle.load(f)

    data = saved["data"]
    node_list = saved["node_list"]
    node_to_idx = saved.get("node_to_idx", {n: i for i, n in enumerate(node_list)})
    idx_to_node = {i: n for n, i in node_to_idx.items()}
    graph = saved["graph"]

    print(f"  Nodes: {len(node_list)}, Edges: {data.edge_index.size(1)}")

    # Build indices
    type_to_fields = defaultdict(list)
    db_to_types = defaultdict(list)
    for node, attrs in graph.nodes(data=True):
        if attrs.get("node_type") == "field" and node in node_to_idx:
            product = attrs.get("product", "")
            type_name = attrs.get("type", "")
            type_key = f"{product}_{type_name}"
            type_to_fields[type_key].append(node)
        elif attrs.get("node_type") == "type" and node in node_to_idx:
            product = attrs.get("product", "")
            db_to_types[product].append(node)

    with open(BASE / "data/spider_dataset.jsonl") as f:
        samples = [json.loads(l) for l in f if l.strip()]
    print(f"Spider samples: {len(samples)}")

    train_edge_index = data.edge_index

    # === Type-level evaluation ===
    type_correct = type_total = type_skipped = 0

    for s in samples:
        result = s.get("result", {})
        props = result.get("properties", {})
        if not props:
            continue

        db = s["database"]
        src_table = result["name"]
        src_type_node = f"{db}_{src_table}"

        # Get ground truth target table(s)
        gt_tgt_tables = set()
        for tgt_list in props.values():
            if isinstance(tgt_list, list):
                for ti in tgt_list:
                    gt_tgt_tables.add(f"{db}_{ti.get('mapTable', '')}")

        if not gt_tgt_tables:
            continue

        if src_type_node not in node_to_idx:
            type_skipped += 1
            continue

        # Get all candidate target types in same database
        candidate_types = [t for t in db_to_types.get(db, [])
                          if t != src_type_node and t in node_to_idx]

        if not candidate_types:
            type_skipped += 1
            continue

        src_idx = node_to_idx[src_type_node]
        cand_indices = [node_to_idx[t] for t in candidate_types]

        query_edges = torch.tensor(
            [[src_idx] * len(cand_indices), cand_indices],
            dtype=torch.long,
        )

        with torch.no_grad():
            scores = model.predict_links(data.x, train_edge_index, query_edges)

        best_idx = cand_indices[np.argmax(scores.cpu().numpy())]
        predicted_type = idx_to_node[best_idx]

        type_total += 1
        if predicted_type in gt_tgt_tables:
            type_correct += 1

    type_acc = type_correct / type_total * 100 if type_total else 0

    # === Field-level evaluation ===
    field_correct = field_total = field_skipped = 0

    for s in samples:
        result = s.get("result", {})
        props = result.get("properties", {})
        if not props:
            continue

        db = s["database"]

        for src_field_key, tgt_list in props.items():
            if not tgt_list or not isinstance(tgt_list, list):
                continue

            for tgt_info in tgt_list:
                tgt_table = tgt_info.get("mapTable", "")
                tgt_field = tgt_info.get("mapProperty", "")

                if not tgt_table or not tgt_field:
                    continue

                src_node = src_field_key
                tgt_node = tgt_field

                if src_node not in node_to_idx:
                    field_skipped += 1
                    continue

                tgt_type_key = f"{db}_{tgt_table}"
                candidates = type_to_fields.get(tgt_type_key, [])

                if not candidates:
                    field_skipped += 1
                    continue

                candidate_indices = [node_to_idx[c] for c in candidates if c in node_to_idx]
                if not candidate_indices:
                    field_skipped += 1
                    continue

                src_idx = node_to_idx[src_node]
                query_edges = torch.tensor(
                    [[src_idx] * len(candidate_indices), candidate_indices],
                    dtype=torch.long,
                )

                with torch.no_grad():
                    scores = model.predict_links(data.x, train_edge_index, query_edges)

                best_idx = candidate_indices[np.argmax(scores.cpu().numpy())]
                predicted = idx_to_node[best_idx]

                field_total += 1
                if predicted == tgt_node:
                    field_correct += 1

    field_acc = field_correct / field_total * 100 if field_total else 0

    print(f"\n{'='*50}")
    print(f"GNN on Spider")
    print(f"{'='*50}")
    print(f"  Type accuracy:  {type_correct}/{type_total} ({type_acc:.1f}%)  skipped={type_skipped}")
    print(f"  Field accuracy: {field_correct}/{field_total} ({field_acc:.1f}%)  skipped={field_skipped}")


if __name__ == "__main__":
    main()
