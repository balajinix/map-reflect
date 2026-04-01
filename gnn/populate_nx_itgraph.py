#!/usr/bin/env python3
"""
Build a NetworkX graph from the ITGraph dataset for GNN training.

Unlike populate_nx.py (which processes one-to-one mapping files),
this script enforces:
  - Many-to-many at type level: one source type can have edges to
    multiple target types
  - Bipartite at field level: within each type pair, each target
    field has at most one incoming mapping edge

Graph structure:
  Product nodes  --has_type-->  Type nodes  --has_field-->  Field nodes
  Type nodes     --maps_to-->   Type nodes   (many-to-many)
  Field nodes    --field_mapping-->  Field nodes  (bipartite per type pair)
"""

import json
import os
import argparse
import pickle
import random
from pathlib import Path
from collections import defaultdict

import networkx as nx
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    from torch_geometric.utils import from_networkx
    from torch_geometric.transforms import RandomLinkSplit
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    print("Warning: torch_geometric not available")
    TORCH_GEOMETRIC_AVAILABLE = False


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(52)


ALL_PRODUCTS = [
    "instana", "concert", "turbonomic", "hcp_terraform",
    "infragraph", "packer", "aws", "azure", "gcp", "hcp",
    "cloudability", "ns1", "watsonx-orchestrate",
]


def get_product(field_key):
    for p in sorted(ALL_PRODUCTS, key=len, reverse=True):
        if field_key.startswith(p + "_"):
            return p
    return field_key.split("_")[0]


class ITGraphBuilder:
    """Build a graph from ITGraph dataset with many-to-many type mappings
    and bipartite field mappings."""

    def __init__(self, dataset_path, api_lookup_path):
        self.graph = nx.DiGraph()
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        with open(dataset_path) as f:
            self.samples = [json.loads(l) for l in f if l.strip()]
        print(f"Loaded {len(self.samples)} samples from {dataset_path}")

        with open(api_lookup_path) as f:
            self.api_lookup = json.load(f)
        print(f"Loaded {len(self.api_lookup)} fields from {api_lookup_path}")

    def _add_product(self, product):
        if product not in self.graph:
            self.graph.add_node(product, node_type="product", name=product)

    def _add_type(self, product, type_name):
        full = f"{product}_{type_name}"
        if full not in self.graph:
            self.graph.add_node(full, node_type="type", name=type_name,
                                product=product, full_name=full)
            self._add_product(product)
            self.graph.add_edge(product, full, relationship="has_type")
        return full

    def _add_field(self, field_key):
        """Add a field node from a key like 'concert_Application.id'."""
        if field_key in self.graph:
            return field_key

        product = get_product(field_key)
        parts = field_key.split(".", 1)
        if len(parts) < 2:
            return field_key

        type_prefix = parts[0]  # e.g., 'concert_Application'
        field_name = parts[1]   # e.g., 'id'
        type_name = type_prefix[len(product) + 1:] if type_prefix.startswith(product + "_") else type_prefix

        meta = self.api_lookup.get(field_key, {})
        description = meta.get("description", "")

        self.graph.add_node(field_key, node_type="field", name=field_name,
                            full_name=field_key, type=type_name,
                            product=product, description=description)

        type_full = self._add_type(product, type_name)
        self.graph.add_edge(type_full, field_key, relationship="has_field")
        return field_key

    def build_from_samples(self):
        """Build graph from ITGraph dataset samples.

        Handles human annotations:
        - Samples with human_judgment='incorrect' are skipped for positive edges
          but their type nodes are still added (used for negative edges).
        - Samples with human_judgment='correct'/'plausible' are treated as
          high-confidence ground truth.
        """
        print("\nStep 1: Adding nodes from dataset samples...")

        type_pair_count = 0
        field_mapping_count = 0
        type_pairs_seen = set()
        self.negative_type_pairs = set()  # For GNN negative sampling

        for s in tqdm(self.samples, desc="Processing samples"):
            meta = s.get("metadata", {})
            judgment = meta.get("human_judgment", "")
            src_product = s.get("source_schema", "")
            tgt_product = s.get("target_schema", "")
            src_type = meta.get("source_type", "")
            tgt_type = meta.get("target_type", "")

            # Always add type nodes (even for incorrect mappings)
            if src_product and src_type:
                self._add_type(src_product, src_type)
            if tgt_product and tgt_type:
                self._add_type(tgt_product, tgt_type)

            # If human marked incorrect, record as negative and skip positive edges
            if judgment == "incorrect":
                src_full = f"{src_product}_{src_type}"
                tgt_full = f"{tgt_product}_{tgt_type}"
                self.negative_type_pairs.add((src_full, tgt_full))
                continue

            mapping = s.get("mapping", {})
            if not mapping:
                continue

            # Collect type pairs from this sample
            src_types = set()
            tgt_types = set()

            for src_field, tgt_field in mapping.items():
                src_node = self._add_field(src_field)
                tgt_node = self._add_field(tgt_field)

                src_f_product = get_product(src_field)
                tgt_f_product = get_product(tgt_field)
                src_type_prefix = src_field.split(".")[0]
                tgt_type_prefix = tgt_field.split(".")[0]
                src_type_name = src_type_prefix[len(src_f_product) + 1:]
                tgt_type_name = tgt_type_prefix[len(tgt_f_product) + 1:]

                src_types.add((src_f_product, src_type_name))
                tgt_types.add((tgt_f_product, tgt_type_name))

                # Add bipartite field mapping edge (unidirectional src -> tgt)
                existing_mappers = [
                    u for u, v, d in self.graph.in_edges(tgt_node, data=True)
                    if d.get("relationship") == "field_mapping"
                ]
                if not existing_mappers:
                    self.graph.add_edge(src_node, tgt_node,
                                        relationship="field_mapping")
                    self.graph.add_edge(tgt_node, src_node,
                                        relationship="field_mapping_reverse")
                    field_mapping_count += 1

            # Add many-to-many type-level edges
            for sp, st in src_types:
                for tp, tt in tgt_types:
                    src_full = f"{sp}_{st}"
                    tgt_full = f"{tp}_{tt}"
                    pair = (src_full, tgt_full)
                    if pair not in type_pairs_seen:
                        type_pairs_seen.add(pair)
                        n_methods = meta.get("n_methods", 1)
                        is_human = meta.get("annotation_source") == "human"
                        self.graph.add_edge(src_full, tgt_full,
                                            relationship="maps_to",
                                            n_methods=n_methods,
                                            human_verified=is_human)
                        type_pair_count += 1

        print(f"  Type-level edges (many-to-many): {type_pair_count}")
        print(f"  Field-level edges (bipartite): {field_mapping_count}")
        print(f"  Negative type pairs (human-rejected): {len(self.negative_type_pairs)}")

    def add_all_fields_from_lookup(self):
        """Add all fields from API lookup to ensure full coverage."""
        print("\nStep 2: Adding all fields from API lookup...")
        added = 0
        for field_key, meta in tqdm(self.api_lookup.items(), desc="Adding fields"):
            product = meta.get("product_name", "")
            if not product:
                continue
            if field_key not in self.graph:
                self._add_field(field_key)
                added += 1
        print(f"  Added {added} additional fields from lookup")

    def add_embeddings(self):
        """Compute sentence-transformer embeddings for all nodes."""
        print("\nStep 3: Computing node embeddings...")
        nodes = list(self.graph.nodes(data=True))

        texts = []
        for node_id, attrs in nodes:
            node_type = attrs.get("node_type", "")
            if node_type == "field":
                desc = attrs.get("description", "")
                name = attrs.get("name", "")
                product = attrs.get("product", "")
                type_name = attrs.get("type", "")
                text = f"{product} {type_name} {name} {desc}".strip()
            elif node_type == "type":
                text = f"{attrs.get('product', '')} {attrs.get('name', '')}"
            elif node_type == "product":
                text = attrs.get("name", node_id)
            else:
                text = str(node_id)
            texts.append(text)

        embeddings = self.embedding_model.encode(texts, show_progress_bar=True,
                                                  batch_size=256)

        for i, (node_id, _) in enumerate(nodes):
            self.graph.nodes[node_id]["embedding"] = embeddings[i].tolist()

        print(f"  Computed {len(embeddings)}-dim embeddings for {len(nodes)} nodes")

    def get_stats(self):
        """Print graph statistics."""
        node_types = defaultdict(int)
        edge_types = defaultdict(int)

        for _, attrs in self.graph.nodes(data=True):
            node_types[attrs.get("node_type", "unknown")] += 1
        for _, _, attrs in self.graph.edges(data=True):
            edge_types[attrs.get("relationship", "unknown")] += 1

        # Check bipartite constraint
        field_nodes = [n for n, d in self.graph.nodes(data=True)
                       if d.get("node_type") == "field"]
        violations = 0
        for fn in field_nodes:
            mappers = [u for u, _, d in self.graph.in_edges(fn, data=True)
                       if d.get("relationship") == "field_mapping"]
            if len(mappers) > 1:
                violations += 1

        # Check many-to-many
        type_nodes = [n for n, d in self.graph.nodes(data=True)
                      if d.get("node_type") == "type"]
        m2m_count = 0
        for tn in type_nodes:
            targets = [v for _, v, d in self.graph.out_edges(tn, data=True)
                       if d.get("relationship") == "maps_to"]
            if len(targets) > 1:
                m2m_count += 1

        print(f"\nGraph Statistics:")
        print(f"  Nodes: {self.graph.number_of_nodes()}")
        for nt, c in sorted(node_types.items()):
            print(f"    {nt}: {c}")
        print(f"  Edges: {self.graph.number_of_edges()}")
        for et, c in sorted(edge_types.items()):
            print(f"    {et}: {c}")
        print(f"  Many-to-many type nodes: {m2m_count}")
        print(f"  Bipartite violations: {violations}")

    def save(self, output_path):
        """Save graph and prepare PyG data.

        Saves in the format expected by train_gnn.py:
          saved_data['data'] = PyG Data object with x, edge_index
          saved_data['node_list'] = list of node IDs
          saved_data['type_to_idx'] = mapping of type names to indices
          saved_data['graph'] = NetworkX graph
        """
        print(f"\nSaving graph to {output_path}...")

        nodes = list(self.graph.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}

        # Feature matrix from embeddings
        features = []
        for n in nodes:
            emb = self.graph.nodes[n].get("embedding", [0.0] * 384)
            features.append(emb)
        x = torch.tensor(features, dtype=torch.float)

        # All edges for the graph structure (used for message passing)
        all_src = []
        all_tgt = []
        for u, v, d in self.graph.edges(data=True):
            if u in node_to_idx and v in node_to_idx:
                all_src.append(node_to_idx[u])
                all_tgt.append(node_to_idx[v])

        edge_index = torch.tensor([all_src, all_tgt], dtype=torch.long)

        # Build type_to_idx for type nodes
        type_to_idx = {}
        for n, attrs in self.graph.nodes(data=True):
            if attrs.get("node_type") == "type":
                type_to_idx[n] = node_to_idx[n]

        # Create PyG-compatible Data object
        if TORCH_GEOMETRIC_AVAILABLE:
            from torch_geometric.data import Data
            from torch_geometric.transforms import RandomLinkSplit

            data = Data(x=x, edge_index=edge_index)

            # Split edges for training/val/test
            try:
                transform = RandomLinkSplit(
                    num_val=0.1,
                    num_test=0.2,
                    is_undirected=False,
                    add_negative_train_samples=True,
                    neg_sampling_ratio=5.0,
                )
                train_data, val_data, test_data = transform(data)

                # Use the split data — train_gnn.py expects these attributes
                data = train_data
                data.test_pos_edge_index = test_data.pos_edge_label_index if hasattr(test_data, 'pos_edge_label_index') else test_data.edge_index
                data.test_neg_edge_index = test_data.neg_edge_label_index if hasattr(test_data, 'neg_edge_label_index') else torch.zeros(2, 0, dtype=torch.long)
                data.train_pos_edge_index = train_data.pos_edge_label_index if hasattr(train_data, 'pos_edge_label_index') else train_data.edge_index

                print(f"  Train edges: {data.train_pos_edge_index.size(1)}")
                print(f"  Test edges: {data.test_pos_edge_index.size(1)}")
            except Exception as e:
                print(f"  Warning: RandomLinkSplit failed ({e}), using all edges for training")
                data.train_pos_edge_index = edge_index
                data.test_pos_edge_index = edge_index[:, :edge_index.size(1)//5]
                data.test_neg_edge_index = torch.zeros(2, 0, dtype=torch.long)
        else:
            # Fallback: simple dict
            data = type("Data", (), {
                "x": x,
                "edge_index": edge_index,
                "train_pos_edge_index": edge_index,
                "test_pos_edge_index": edge_index[:, :edge_index.size(1)//5],
                "test_neg_edge_index": torch.zeros(2, 0, dtype=torch.long),
            })()

        # Build explicit negative type edges from human annotations
        neg_type_src = []
        neg_type_tgt = []
        for src_full, tgt_full in getattr(self, "negative_type_pairs", set()):
            if src_full in node_to_idx and tgt_full in node_to_idx:
                neg_type_src.append(node_to_idx[src_full])
                neg_type_tgt.append(node_to_idx[tgt_full])
        human_neg_edge_index = torch.tensor(
            [neg_type_src, neg_type_tgt], dtype=torch.long
        ) if neg_type_src else torch.zeros(2, 0, dtype=torch.long)
        print(f"  Human-annotated negative type edges: {human_neg_edge_index.size(1)}")

        save_data = {
            "data": data,
            "node_list": nodes,
            "type_to_idx": type_to_idx,
            "graph": self.graph,
            "node_to_idx": node_to_idx,
            "idx_to_node": {i: n for n, i in node_to_idx.items()},
            "human_neg_edge_index": human_neg_edge_index,
        }

        with open(output_path, "wb") as f:
            pickle.dump(save_data, f)

        print(f"  Saved: {output_path}")
        print(f"  Nodes: {len(nodes)}, Features: {x.size(1)}, Edges: {edge_index.size(1)}")
        print(f"  Type nodes: {len(type_to_idx)}")


def main():
    parser = argparse.ArgumentParser(description="Build ITGraph GNN training graph")
    parser.add_argument("--dataset-file", type=str,
                        default="../data/itgraph_dataset.jsonl",
                        help="ITGraph dataset JSONL")
    parser.add_argument("--api-lookup-file", type=str,
                        default="../data/itom_hashicorp_api_lookup_v2.json",
                        help="API lookup JSON")
    parser.add_argument("--output-dir", type=str, default="../data",
                        help="Output directory")
    parser.add_argument("--dataset", type=str, default="itgraph",
                        help="Dataset name for output files")
    args = parser.parse_args()

    builder = ITGraphBuilder(args.dataset_file, args.api_lookup_file)
    builder.build_from_samples()
    builder.add_all_fields_from_lookup()
    builder.add_embeddings()
    builder.get_stats()

    output_path = os.path.join(args.output_dir, f"{args.dataset}_graph_data.pkl")
    builder.save(output_path)


if __name__ == "__main__":
    main()
