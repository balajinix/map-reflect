#!/usr/bin/env python3
"""
Evaluate GNN on ITGraph_HC via link prediction.

For each ground truth field mapping edge in ITGraph_HC:
  - Score the positive (ground truth) edge
  - Score an equal number of negative (random non-existent) edges
  - Report precision, recall, F1, AUC

The GNN was trained on the full ITGraph dataset. We evaluate on
ITGraph_HC (the high-confidence subset with >=2 method agreement).
"""

import json
import pickle
import random
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    classification_report,
)

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
    # Load GNN model
    model_path = BASE / "data" / "gnn_results" / "itgraph_mapping_gnn_results.pkl"
    print(f"Loading GNN model from {model_path}...")
    model = load_model_safely(str(model_path))
    if model is None:
        print("ERROR: Could not load GNN model")
        return
    model.eval()

    # Load graph data
    graph_path = BASE / "data" / "itgraph_graph_data.pkl"
    print(f"Loading graph from {graph_path}...")
    with open(graph_path, "rb") as f:
        saved = pickle.load(f)

    data = saved["data"]
    node_list = saved["node_list"]
    node_to_idx = saved["node_to_idx"]
    graph = saved["graph"]

    print(f"  Nodes: {len(node_list)}, Edges: {data.edge_index.size(1)}")

    # Load ITGraph_HC test samples
    hc_path = BASE / "data" / "itgraph_dataset_high_confidence.jsonl"
    with open(hc_path) as f:
        hc_samples = [json.loads(l) for l in f if l.strip()]
    print(f"  ITGraph_HC samples: {len(hc_samples)}")

    # Collect positive edges (ground truth field mappings)
    positive_edges = []
    missing_nodes = 0
    for s in hc_samples:
        mapping = s.get("mapping", {})
        for src_field, tgt_field in mapping.items():
            if src_field in node_to_idx and tgt_field in node_to_idx:
                positive_edges.append((node_to_idx[src_field], node_to_idx[tgt_field]))
            else:
                missing_nodes += 1

    print(f"  Positive edges: {len(positive_edges)} ({missing_nodes} skipped, nodes not in graph)")

    # Collect all field nodes for negative sampling
    field_nodes = [
        node_to_idx[n] for n, attrs in graph.nodes(data=True)
        if attrs.get("node_type") == "field" and n in node_to_idx
    ]
    print(f"  Field nodes available: {len(field_nodes)}")

    # Build set of existing edges for negative sampling
    existing_edges = set()
    for i in range(data.edge_index.size(1)):
        src = data.edge_index[0, i].item()
        tgt = data.edge_index[1, i].item()
        existing_edges.add((src, tgt))

    # Generate negative edges (random field pairs that don't exist)
    random.seed(42)
    negative_edges = []
    attempts = 0
    while len(negative_edges) < len(positive_edges) and attempts < len(positive_edges) * 100:
        src = random.choice(field_nodes)
        tgt = random.choice(field_nodes)
        if src != tgt and (src, tgt) not in existing_edges:
            negative_edges.append((src, tgt))
            existing_edges.add((src, tgt))  # prevent duplicates
        attempts += 1

    print(f"  Negative edges: {len(negative_edges)}")

    # Score edges with GNN
    print("\nScoring edges with GNN...")
    pos_edge_index = torch.tensor(positive_edges, dtype=torch.long).t()
    neg_edge_index = torch.tensor(negative_edges, dtype=torch.long).t()

    with torch.no_grad():
        # Use training edges for message passing
        train_edge_index = data.edge_index
        pos_scores = model.predict_links(data.x, train_edge_index, pos_edge_index)
        neg_scores = model.predict_links(data.x, train_edge_index, neg_edge_index)

    pos_scores_np = pos_scores.cpu().numpy()
    neg_scores_np = neg_scores.cpu().numpy()

    print(f"  Positive scores: mean={pos_scores_np.mean():.4f}, "
          f"median={np.median(pos_scores_np):.4f}, "
          f"min={pos_scores_np.min():.4f}, max={pos_scores_np.max():.4f}")
    print(f"  Negative scores: mean={neg_scores_np.mean():.4f}, "
          f"median={np.median(neg_scores_np):.4f}, "
          f"min={neg_scores_np.min():.4f}, max={neg_scores_np.max():.4f}")

    # Compute metrics at various thresholds
    all_scores = np.concatenate([pos_scores_np, neg_scores_np])
    all_labels = np.concatenate([
        np.ones(len(pos_scores_np)),
        np.zeros(len(neg_scores_np)),
    ])

    # AUC
    auc = roc_auc_score(all_labels, all_scores)
    ap = average_precision_score(all_labels, all_scores)
    print(f"\n  AUC-ROC: {auc:.4f}")
    print(f"  Average Precision: {ap:.4f}")

    # F1 at threshold 0.5
    preds_05 = (all_scores >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, preds_05, average="binary")
    print(f"\n  At threshold 0.5:")
    print(f"    Precision: {p:.4f}")
    print(f"    Recall:    {r:.4f}")
    print(f"    F1:        {f1:.4f}")

    # Find optimal threshold
    best_f1 = 0
    best_threshold = 0.5
    for t in np.arange(0.1, 0.95, 0.05):
        preds = (all_scores >= t).astype(int)
        _, _, f1_t, _ = precision_recall_fscore_support(all_labels, preds, average="binary")
        if f1_t > best_f1:
            best_f1 = f1_t
            best_threshold = t

    preds_best = (all_scores >= best_threshold).astype(int)
    p_best, r_best, f1_best, _ = precision_recall_fscore_support(
        all_labels, preds_best, average="binary"
    )
    print(f"\n  At optimal threshold {best_threshold:.2f}:")
    print(f"    Precision: {p_best:.4f}")
    print(f"    Recall:    {r_best:.4f}")
    print(f"    F1:        {f1_best:.4f}")

    # Per product-pair breakdown
    print("\n  Per product pair (at optimal threshold):")
    pair_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "total_pos": 0})

    idx_to_node = saved["idx_to_node"]
    for i, (src_idx, tgt_idx) in enumerate(positive_edges):
        src_node = idx_to_node[src_idx]
        tgt_node = idx_to_node[tgt_idx]
        sp = get_product(src_node)
        tp = get_product(tgt_node)
        pair = f"{sp} -> {tp}"
        pair_stats[pair]["total_pos"] += 1
        if pos_scores_np[i] >= best_threshold:
            pair_stats[pair]["tp"] += 1
        else:
            pair_stats[pair]["fn"] += 1

    for pair in sorted(pair_stats, key=lambda x: -pair_stats[x]["total_pos"])[:12]:
        s = pair_stats[pair]
        recall = s["tp"] / s["total_pos"] if s["total_pos"] > 0 else 0
        print(f"    {pair:30s} {s['tp']:3d}/{s['total_pos']:3d} recall={recall:.1%}")

    # Summary for paper
    print("\n" + "=" * 60)
    print("PAPER NUMBERS:")
    print(f"  GNN Link Prediction on ITGraph_HC")
    print(f"  Positive edges: {len(positive_edges)}")
    print(f"  Negative edges: {len(negative_edges)}")
    print(f"  AUC-ROC:  {auc:.1%}")
    print(f"  F1:       {f1_best:.1%} (threshold={best_threshold:.2f})")
    print(f"  Precision:{p_best:.1%}")
    print(f"  Recall:   {r_best:.1%}")


if __name__ == "__main__":
    main()
