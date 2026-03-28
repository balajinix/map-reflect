#!/usr/bin/env python3
"""Evaluate GNN on Spider using field accuracy (same as Map-Reflect/MatchMaker)."""
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

    print(f"Loading GNN model...")
    model = load_model_safely(str(model_path))
    model.eval()

    print(f"Loading graph...")
    with open(graph_path, "rb") as f:
        saved = pickle.load(f)

    data = saved["data"]
    node_list = saved["node_list"]
    node_to_idx = saved.get("node_to_idx", {n: i for i, n in enumerate(node_list)})
    idx_to_node = {i: n for n, i in node_to_idx.items()}
    graph = saved["graph"]

    print(f"  Nodes: {len(node_list)}, Edges: {data.edge_index.size(1)}")

    # Build type -> fields index
    type_to_fields = defaultdict(list)
    for node, attrs in graph.nodes(data=True):
        if attrs.get("node_type") == "field" and node in node_to_idx:
            product = attrs.get("product", "")
            type_name = attrs.get("type", "")
            type_key = f"{product}_{type_name}"
            type_to_fields[type_key].append(node)

    # Load Spider dataset
    with open(BASE / "data/spider_dataset.jsonl") as f:
        samples = [json.loads(l) for l in f if l.strip()]
    print(f"Spider samples: {len(samples)}")

    # Load API lookup for field name resolution
    with open(BASE / "data/spider_api_lookup.json") as f:
        api_lookup = json.load(f)

    correct = total = skipped = 0
    train_edge_index = data.edge_index

    for s in samples:
        result = s.get("result", {})
        props = result.get("properties", {})
        if not props:
            continue

        db = s["database"]
        src_table = result["name"]

        for src_field_key, tgt_list in props.items():
            if not tgt_list or not isinstance(tgt_list, list):
                continue

            for tgt_info in tgt_list:
                tgt_table = tgt_info.get("mapTable", "")
                tgt_field = tgt_info.get("mapProperty", "")

                if not tgt_table or not tgt_field:
                    continue

                # Resolve source field in graph
                # Spider uses: {table}_{field} as node ID in api_lookup
                src_node = src_field_key  # e.g., "cite_citing"
                tgt_node = tgt_field      # e.g., "publication_pid"

                if src_node not in node_to_idx:
                    skipped += 1
                    continue

                # Get all candidate target fields in the target table
                tgt_type_key = f"{db}_{tgt_table}"
                candidates = type_to_fields.get(tgt_type_key, [])

                if not candidates:
                    skipped += 1
                    continue

                candidate_indices = [node_to_idx[c] for c in candidates if c in node_to_idx]
                if not candidate_indices:
                    skipped += 1
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

                total += 1
                if predicted == tgt_node:
                    correct += 1

    acc = correct / total * 100 if total else 0
    print(f"\n{'='*50}")
    print(f"GNN Field Accuracy on Spider")
    print(f"{'='*50}")
    print(f"  Correct: {correct}/{total} ({acc:.1f}%)")
    print(f"  Skipped: {skipped}")


if __name__ == "__main__":
    main()
