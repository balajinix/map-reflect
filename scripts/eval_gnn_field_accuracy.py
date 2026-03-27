#!/usr/bin/env python3
"""
Evaluate GNN on ITGraph_HC using field accuracy (same metric as Map-Reflect/MatchMaker).

For each source field in ITGraph_HC, rank all target fields in the
target type by GNN edge score, pick the top-1, and check if it matches
ground truth. Reports accuracy = correct / total.
"""

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

ALL_PRODUCTS = [
    "instana", "concert", "turbonomic", "hcp_terraform",
    "infragraph", "packer", "aws", "azure", "gcp", "hcp",
    "cloudability", "ns1", "watsonx-orchestrate",
]


def get_product(field):
    for p in sorted(ALL_PRODUCTS, key=len, reverse=True):
        if field.startswith(p + "_"):
            return p
    return field.split("_")[0]


def main():
    # Load model and graph
    model_path = BASE / "data" / "gnn_results" / "itgraph_mapping_gnn_results.pkl"
    print(f"Loading GNN model from {model_path}...")
    model = load_model_safely(str(model_path))
    model.eval()

    graph_path = BASE / "data" / "itgraph_graph_data.pkl"
    with open(graph_path, "rb") as f:
        saved = pickle.load(f)

    data = saved["data"]
    node_to_idx = saved["node_to_idx"]
    idx_to_node = saved["idx_to_node"]
    graph = saved["graph"]

    # Build type -> field index from graph
    type_to_fields = defaultdict(list)
    for node, attrs in graph.nodes(data=True):
        if attrs.get("node_type") == "field" and node in node_to_idx:
            product = attrs.get("product", "")
            type_name = attrs.get("type", "")
            type_key = f"{product}_{type_name}"
            type_to_fields[type_key].append(node)

    print(f"Types with fields: {len(type_to_fields)}")

    # Load ITGraph_HC
    hc_path = BASE / "data" / "itgraph_dataset_high_confidence.jsonl"
    with open(hc_path) as f:
        hc_samples = [json.loads(l) for l in f if l.strip()]
    print(f"ITGraph_HC samples: {len(hc_samples)}")

    # Evaluate: for each source field, rank target fields and pick top-1
    correct = 0
    total = 0
    skipped = 0
    pair_stats = defaultdict(lambda: [0, 0])  # [total, correct]

    train_edge_index = data.edge_index

    for s in hc_samples:
        mapping = s.get("mapping", {})
        if not mapping:
            continue

        # Get target type from first mapping value
        tgt_fields_gt = list(mapping.values())
        if not tgt_fields_gt:
            continue

        # Determine target type
        first_tgt = tgt_fields_gt[0]
        tgt_product = get_product(first_tgt)
        tgt_type_prefix = first_tgt.split(".")[0]
        tgt_type_name = tgt_type_prefix[len(tgt_product) + 1:] if tgt_type_prefix.startswith(tgt_product + "_") else tgt_type_prefix
        tgt_type_key = f"{tgt_product}_{tgt_type_name}"

        # Get all candidate target fields for this type
        candidate_fields = type_to_fields.get(tgt_type_key, [])
        if not candidate_fields:
            skipped += len(mapping)
            continue

        candidate_indices = [node_to_idx[f] for f in candidate_fields if f in node_to_idx]
        if not candidate_indices:
            skipped += len(mapping)
            continue

        for src_field, gt_target in mapping.items():
            if src_field not in node_to_idx:
                skipped += 1
                continue

            src_idx = node_to_idx[src_field]

            # Score all candidate target fields
            query_edges = torch.tensor(
                [[src_idx] * len(candidate_indices), candidate_indices],
                dtype=torch.long,
            )

            with torch.no_grad():
                scores = model.predict_links(data.x, train_edge_index, query_edges)

            scores_np = scores.cpu().numpy()
            best_idx = candidate_indices[np.argmax(scores_np)]
            predicted = idx_to_node[best_idx]

            total += 1
            sp = get_product(src_field)
            tp = get_product(predicted)
            pair_key = f"{sp} -> {tp}"
            pair_stats[pair_key][0] += 1

            if predicted.lower() == gt_target.lower():
                correct += 1
                pair_stats[pair_key][1] += 1

    acc = correct / total * 100 if total else 0
    print(f"\n{'='*60}")
    print(f"GNN Field Accuracy on ITGraph_HC")
    print(f"{'='*60}")
    print(f"  Correct:  {correct}/{total} ({acc:.1f}%)")
    print(f"  Skipped:  {skipped}")
    print(f"\n  Per product pair:")
    for pair in sorted(pair_stats, key=lambda x: -pair_stats[x][0])[:12]:
        t, c = pair_stats[pair]
        print(f"    {pair:30s} {c:3d}/{t:3d} ({c/t*100:5.1f}%)")


if __name__ == "__main__":
    main()
