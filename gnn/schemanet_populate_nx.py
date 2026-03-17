#!/usr/bin/env python3
"""
Build GNN graph from SchemaNet datasets.
Converts SchemaNet format to the graph format expected by train_gnn.py
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


class SchemaNetGraphBuilder:
    """Build graph from SchemaNet dataset format."""

    def __init__(self, dataset_name, schemanet_dir):
        self.dataset_name = dataset_name
        self.schemanet_dir = Path(schemanet_dir)
        self.graph = nx.DiGraph()
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Load data
        self.target_lookup = self._load_target_lookup()
        self.samples = self._load_dataset()

    def _load_target_lookup(self):
        """Load and flatten target API lookup."""
        lookup_file = self.schemanet_dir / f"{self.dataset_name}_api_lookup.json"
        with open(lookup_file) as f:
            type_lookup = json.load(f)

        # Flatten to field-level
        field_lookup = {}
        for type_id, type_data in type_lookup.items():
            # type_id format: "bank2_branch"
            parts = type_id.split('_', 1)
            product = parts[0]
            type_name = type_data.get('name', parts[1] if len(parts) > 1 else type_id)

            for field_name, field_data in type_data.get('fields', {}).items():
                field_key = f"{type_name}.{field_name}"
                field_lookup[field_key] = {
                    'product_name': product,
                    'type_name': type_name,
                    'type_id': type_id,
                    'field_name': field_name,
                    'description': field_data.get('description', f'{field_name} field'),
                    'data_type': field_data.get('type', 'unknown')
                }

        return field_lookup

    def _load_dataset(self):
        """Load dataset samples."""
        dataset_file = self.schemanet_dir / f"{self.dataset_name}_dataset.jsonl"
        samples = []
        with open(dataset_file) as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples

    def _standardize_node_attrs(self, attrs):
        """Ensure consistent node attributes."""
        standard_keys = ["node_type", "name", "full_name", "product", "type", "description"]
        for key in standard_keys:
            if key not in attrs:
                attrs[key] = ''
        return attrs

    def _standardize_edge_attrs(self, attrs):
        """Ensure consistent edge attributes."""
        standard_keys = ["relationship", "justification"]
        for key in standard_keys:
            if key not in attrs:
                attrs[key] = ''
        return attrs

    def _add_product_node(self, product_name):
        """Add a product (schema) node."""
        if product_name in self.graph:
            return
        attrs = {
            "node_type": "product",
            "name": product_name
        }
        self.graph.add_node(product_name, **self._standardize_node_attrs(attrs))

    def _add_type_node(self, product_name, type_name):
        """Add a type (table) node."""
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

        # Edge from product to type
        self.graph.add_edge(product_name, full_type_name,
                           **self._standardize_edge_attrs({'relationship': 'has_type'}))
        return full_type_name

    def _add_field_node(self, product_name, type_name, field_name, description=''):
        """Add a field (column) node."""
        full_field_name = f"{type_name}.{field_name}"
        field_id = f"{product_name}_{full_field_name}"
        full_type_name = f"{product_name}_{type_name}"

        if field_id in self.graph:
            return field_id

        attrs = {
            "node_type": "field",
            "name": field_name,
            "full_name": full_field_name,
            "type": type_name,
            "product": product_name,
            "description": description or f'{field_name} in {type_name}'
        }
        self.graph.add_node(field_id, **self._standardize_node_attrs(attrs))

        # Edge from type to field
        self.graph.add_edge(full_type_name, field_id,
                           **self._standardize_edge_attrs({'relationship': 'has_field'}))
        return field_id

    def _add_field_mapping(self, src_field_id, tgt_field_id, justification=''):
        """Add bidirectional field mapping edge."""
        edge_attrs = self._standardize_edge_attrs({
            'relationship': 'field_mapping',
            'justification': justification or 'Ground truth mapping'
        })
        self.graph.add_edge(src_field_id, tgt_field_id, **edge_attrs)
        self.graph.add_edge(tgt_field_id, src_field_id, **edge_attrs)

    def build_graph(self):
        """Build the complete graph from SchemaNet data."""
        print(f"Building graph for {self.dataset_name}...")

        # Step 1: Add all target schema fields from API lookup
        print("Step 1: Adding target schema fields from API lookup...")
        for field_key, field_meta in self.target_lookup.items():
            product = field_meta['product_name']
            type_name = field_meta['type_name']
            field_name = field_meta['field_name']
            description = field_meta.get('description', '')

            self._add_product_node(product)
            self._add_type_node(product, type_name)
            self._add_field_node(product, type_name, field_name, description)

        target_nodes = self.graph.number_of_nodes()
        print(f"  Target schema: {target_nodes} nodes")

        # Step 2: Add source schema fields and mappings from dataset
        print("Step 2: Adding source schema fields and mappings...")
        mapping_count = 0

        for sample in self.samples:
            source_schema = sample['source_schema']
            target_schema = sample['target_schema']

            self._add_product_node(source_schema)

            for source_field, target_field in sample.get('mapping', {}).items():
                # Parse source field: "bank1_branch.branch_id"
                src_parts = source_field.split('.')
                src_type_full = src_parts[0]  # "bank1_branch"
                src_field_name = src_parts[1] if len(src_parts) > 1 else src_parts[0]

                # Extract type name from "bank1_branch" -> "branch"
                src_type_name = src_type_full.replace(f'{source_schema}_', '')

                # Add source nodes
                self._add_type_node(source_schema, src_type_name)
                src_field_id = self._add_field_node(
                    source_schema, src_type_name, src_field_name,
                    f'{src_field_name} in {src_type_name}'
                )

                # Parse target field: "bank2_branch.branch_id"
                tgt_parts = target_field.split('.')
                tgt_type_full = tgt_parts[0]  # "bank2_branch"
                tgt_field_name = tgt_parts[1] if len(tgt_parts) > 1 else tgt_parts[0]

                # Extract type name
                tgt_type_name = tgt_type_full.replace(f'{target_schema}_', '')

                # Add target nodes (may already exist)
                self._add_type_node(target_schema, tgt_type_name)
                tgt_field_id = self._add_field_node(
                    target_schema, tgt_type_name, tgt_field_name,
                    self.target_lookup.get(f'{tgt_type_name}.{tgt_field_name}', {}).get('description', '')
                )

                # Add mapping edge
                self._add_field_mapping(src_field_id, tgt_field_id)
                mapping_count += 1

        total_nodes = self.graph.number_of_nodes()
        total_edges = self.graph.number_of_edges()
        print(f"  Added {total_nodes - target_nodes} source schema nodes")
        print(f"  Added {mapping_count} field mappings")
        print(f"Graph built: {total_nodes} nodes, {total_edges} edges")

        return self.graph

    def get_statistics(self):
        """Get graph statistics."""
        stats = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'node_types': {},
            'edge_types': {}
        }

        for node, data in self.graph.nodes(data=True):
            node_type = data.get('node_type', 'unknown')
            stats['node_types'][node_type] = stats['node_types'].get(node_type, 0) + 1

        for src, tgt, data in self.graph.edges(data=True):
            rel_type = data.get('relationship', 'unknown')
            stats['edge_types'][rel_type] = stats['edge_types'].get(rel_type, 0) + 1

        return stats


def prepare_graph_data(graph, model):
    """Convert NetworkX graph to PyTorch Geometric format."""
    print("Preparing graph data for GNN...")

    node_features = []
    node_list = list(graph.nodes())

    for node in tqdm(node_list, desc="Creating node embeddings"):
        node_data = graph.nodes[node]
        node_type = node_data.get('node_type', 'unknown')

        if node_type == 'product':
            text_repr = f"Schema: {node_data.get('name', node)}"
        elif node_type == 'type':
            text_repr = f"Table: {node_data.get('name', node)} in {node_data.get('product', 'unknown')}"
        elif node_type == 'field':
            description = node_data.get('description', 'No description')
            text_repr = f"Column: {node_data.get('full_name', node)} - {description[:100]}"
        else:
            text_repr = f"{node_type}: {node}"

        embedding = model.encode([text_repr])[0]
        node_features.append(embedding)

    # Convert to PyTorch Geometric
    data = from_networkx(graph)

    # Add node features
    data.x = torch.FloatTensor(np.array(node_features))

    # Add node type one-hot encoding
    node_types = [graph.nodes[node].get('node_type', 'unknown') for node in node_list]
    unique_types = list(set(node_types))
    type_to_idx = {t: i for i, t in enumerate(unique_types)}

    type_features = torch.zeros(len(node_list), len(unique_types))
    for i, node_type in enumerate(node_types):
        type_features[i, type_to_idx[node_type]] = 1

    data.x = torch.cat([data.x, type_features], dim=-1)

    return data, node_list, type_to_idx


def create_train_test_split(data, test_ratio=0.2, val_ratio=0.1, seed=42):
    """Create train/val/test split for edges."""
    print("Creating train/test split...")

    if data.edge_index is None or data.edge_index.size(1) == 0:
        print("Warning: No edges found")
        return data

    edge_index = data.edge_index
    num_edges = edge_index.size(1)

    torch.manual_seed(seed)
    perm = torch.randperm(num_edges)

    test_size = int(num_edges * test_ratio)
    val_size = int(num_edges * val_ratio)
    train_size = num_edges - test_size - val_size

    train_edges = perm[:train_size]
    val_edges = perm[train_size:train_size + val_size]
    test_edges = perm[train_size + val_size:]

    data.train_pos_edge_index = edge_index[:, train_edges]
    data.val_pos_edge_index = edge_index[:, val_edges]
    data.test_pos_edge_index = edge_index[:, test_edges]

    # Negative sampling
    def simple_negative_sampling(pos_edges, num_nodes, num_samples):
        all_edges = set()
        for i in range(pos_edges.size(1)):
            src, tgt = pos_edges[0, i].item(), pos_edges[1, i].item()
            all_edges.add((src, tgt))

        neg_edges = []
        attempts = 0
        while len(neg_edges) < num_samples and attempts < num_samples * 10:
            src = torch.randint(0, num_nodes, (1,)).item()
            tgt = torch.randint(0, num_nodes, (1,)).item()
            if src != tgt and (src, tgt) not in all_edges:
                neg_edges.append([src, tgt])
            attempts += 1

        while len(neg_edges) < num_samples:
            neg_edges.append([0, 1])

        return torch.tensor(neg_edges[:num_samples]).t()

    data.val_neg_edge_index = simple_negative_sampling(
        data.train_pos_edge_index, data.num_nodes, data.val_pos_edge_index.size(1)
    )
    data.test_neg_edge_index = simple_negative_sampling(
        torch.cat([data.train_pos_edge_index, data.val_pos_edge_index], dim=1),
        data.num_nodes, data.test_pos_edge_index.size(1)
    )

    print(f"  Train edges: {data.train_pos_edge_index.size(1)}")
    print(f"  Val edges: {data.val_pos_edge_index.size(1)}")
    print(f"  Test edges: {data.test_pos_edge_index.size(1)}")

    return data


def main():
    parser = argparse.ArgumentParser(description='Build GNN graph from SchemaNet')
    parser.add_argument('--dataset', type=str, required=True,
                        help='SchemaNet dataset name (e.g., schemanet_bank1_bank2)')
    parser.add_argument('--schemanet-dir', type=str, default='../data/schemanet',
                        help='Directory containing SchemaNet data')
    parser.add_argument('--output-dir', type=str, default='../data',
                        help='Output directory for graph data')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Build graph
    builder = SchemaNetGraphBuilder(args.dataset, args.schemanet_dir)
    graph = builder.build_graph()

    # Print statistics
    stats = builder.get_statistics()
    print(f"\nGraph Statistics:")
    print(f"  Nodes: {stats['total_nodes']}")
    print(f"  Edges: {stats['total_edges']}")
    print(f"  Node types: {stats['node_types']}")
    print(f"  Edge types: {stats['edge_types']}")

    # Prepare PyTorch Geometric data
    data, node_list, type_to_idx = prepare_graph_data(graph, builder.model)
    data = create_train_test_split(data, seed=args.seed)

    # Save
    output_file = Path(args.output_dir) / f"{args.dataset}_graph_data.pkl"
    saved_data = {
        'data': data,
        'node_list': node_list,
        'type_to_idx': type_to_idx,
        'graph': graph
    }

    with open(output_file, 'wb') as f:
        pickle.dump(saved_data, f)

    print(f"\nGraph data saved to: {output_file}")


if __name__ == "__main__":
    main()
