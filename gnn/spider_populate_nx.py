#!/usr/bin/env python3
"""
Build GNN graph from Spider dataset.

Spider uses flat api_lookup (product_name=database, type_name=table,
field_name=column) and dataset samples with foreign key mappings.
Reuses the SchemaNet graph building infrastructure.
"""

import torch
import numpy as np
import networkx as nx
import json
import os
import argparse
import pickle
from pathlib import Path
from tqdm import tqdm

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from sentence_transformers import SentenceTransformer

try:
    from torch_geometric.utils import from_networkx
except ImportError:
    print("PyTorch Geometric not available")
    from_networkx = None

# Reuse SchemaNet graph preparation functions
from schemanet_populate_nx import prepare_graph_data, create_train_test_split


class SpiderGraphBuilder:
    """Build graph from Spider dataset format."""

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.graph = nx.DiGraph()
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Load data
        self.api_lookup = self._load_api_lookup()
        self.samples = self._load_dataset()

    def _load_api_lookup(self):
        """Load flat Spider API lookup."""
        with open(self.data_dir / "spider_api_lookup.json") as f:
            return json.load(f)

    def _load_dataset(self):
        """Load Spider dataset samples."""
        samples = []
        with open(self.data_dir / "spider_dataset.jsonl") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples

    def _standardize_node_attrs(self, attrs):
        for key in ["node_type", "name", "full_name", "product", "type", "description"]:
            if key not in attrs:
                attrs[key] = ''
        return attrs

    def _standardize_edge_attrs(self, attrs):
        for key in ["relationship", "justification"]:
            if key not in attrs:
                attrs[key] = ''
        return attrs

    def _add_product_node(self, product_name):
        if product_name in self.graph:
            return
        attrs = {"node_type": "product", "name": product_name}
        self.graph.add_node(product_name, **self._standardize_node_attrs(attrs))

    def _add_type_node(self, product_name, type_name):
        full_type_name = f"{product_name}_{type_name}"
        if full_type_name in self.graph:
            return full_type_name
        attrs = {
            "node_type": "type",
            "name": type_name,
            "product": product_name,
            "full_name": full_type_name
        }
        self.graph.add_node(full_type_name, **self._standardize_node_attrs(attrs))
        self.graph.add_edge(product_name, full_type_name,
                            **self._standardize_edge_attrs({'relationship': 'has_type'}))
        return full_type_name

    def _add_field_node(self, product_name, type_name, field_name, description=''):
        field_id = f"{type_name}_{field_name}"
        full_type_name = f"{product_name}_{type_name}"
        if field_id in self.graph:
            return field_id
        attrs = {
            "node_type": "field",
            "name": field_name,
            "full_name": f"{type_name}.{field_name}",
            "type": type_name,
            "product": product_name,
            "description": description or f'{field_name} in {type_name}'
        }
        self.graph.add_node(field_id, **self._standardize_node_attrs(attrs))
        self.graph.add_edge(full_type_name, field_id,
                            **self._standardize_edge_attrs({'relationship': 'has_field'}))
        return field_id

    def build_graph(self):
        """Build the complete graph from Spider data."""
        print(f"Building Spider graph...")

        # Step 1: Add all fields from API lookup
        print("Step 1: Adding all schema fields from API lookup...")
        for field_key, meta in self.api_lookup.items():
            product = meta['product_name']  # database name
            type_name = meta['type_name']  # table name
            field_name = meta['field_name']  # column name
            description = meta.get('description', '')

            self._add_product_node(product)
            self._add_type_node(product, type_name)
            self._add_field_node(product, type_name, field_name, description)

        base_nodes = self.graph.number_of_nodes()
        print(f"  Schema: {base_nodes} nodes")

        # Step 2: Add field mapping edges from dataset (FK relationships)
        print("Step 2: Adding FK mapping edges from dataset...")
        mapping_count = 0

        for sample in self.samples:
            result = sample.get('result', {})
            src_table = result.get('name', '')
            db = sample.get('database', '')

            for src_field, mappings in result.get('properties', {}).items():
                for m in mappings:
                    tgt_table = m['mapTable']
                    tgt_field_key = m['mapProperty']

                    # src_field format: "table_column" (e.g., "concert_stadium_id")
                    # tgt_field_key format: "table_column" (e.g., "stadium_stadium_id")
                    src_field_id = src_field  # Already in table_column format
                    tgt_field_id = tgt_field_key

                    # Ensure both nodes exist
                    if src_field_id not in self.graph:
                        # Parse and add
                        src_col = src_field.replace(f"{src_table}_", "", 1)
                        self._add_field_node(db, src_table, src_col)
                        src_field_id = f"{src_table}_{src_col}"

                    if tgt_field_id not in self.graph:
                        tgt_col = tgt_field_key.replace(f"{tgt_table}_", "", 1)
                        self._add_field_node(db, tgt_table, tgt_col)
                        tgt_field_id = f"{tgt_table}_{tgt_col}"

                    # Add bidirectional mapping edge
                    if src_field_id in self.graph and tgt_field_id in self.graph:
                        edge_attrs = self._standardize_edge_attrs({
                            'relationship': 'field_mapping',
                            'justification': 'Foreign key relationship'
                        })
                        self.graph.add_edge(src_field_id, tgt_field_id, **edge_attrs)
                        self.graph.add_edge(tgt_field_id, src_field_id, **edge_attrs)
                        mapping_count += 1

        total_nodes = self.graph.number_of_nodes()
        total_edges = self.graph.number_of_edges()
        print(f"  Added {mapping_count} FK mapping edges")
        print(f"Graph built: {total_nodes} nodes, {total_edges} edges")

        return self.graph


def main():
    parser = argparse.ArgumentParser(description='Build GNN graph from Spider')
    parser.add_argument('--data-dir', type=str, default='../data',
                        help='Directory containing spider_api_lookup.json and spider_dataset.jsonl')
    parser.add_argument('--output-dir', type=str, default='../data',
                        help='Output directory for graph data')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    builder = SpiderGraphBuilder(args.data_dir)
    graph = builder.build_graph()

    # Statistics
    stats = {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges()}
    node_types = {}
    for _, data in graph.nodes(data=True):
        nt = data.get('node_type', 'unknown')
        node_types[nt] = node_types.get(nt, 0) + 1
    edge_types = {}
    for _, _, data in graph.edges(data=True):
        et = data.get('relationship', 'unknown')
        edge_types[et] = edge_types.get(et, 0) + 1

    print(f"\nStatistics: {stats['nodes']} nodes, {stats['edges']} edges")
    print(f"  Node types: {node_types}")
    print(f"  Edge types: {edge_types}")

    # Prepare PyTorch Geometric data
    data, node_list, type_to_idx = prepare_graph_data(graph, builder.model)
    data = create_train_test_split(data, seed=args.seed)

    # Save
    output_file = Path(args.output_dir) / "spider_graph_data.pkl"
    saved_data = {
        'data': data,
        'node_list': node_list,
        'type_to_idx': type_to_idx,
        'graph': graph
    }

    with open(output_file, 'wb') as f:
        pickle.dump(saved_data, f)

    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
