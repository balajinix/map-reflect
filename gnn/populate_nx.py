import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import from_networkx, negative_sampling
from torch_geometric.transforms import RandomLinkSplit
import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import json
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(52)

class MappingGraphBuilder:
    def __init__(self, mapping_dir, api_lookup_file):
        self.mapping_dir = mapping_dir
        self.api_lookup_file = api_lookup_file
        self.graph = nx.DiGraph()
        self.api_lookup = self.load_api_lookup()
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
    def load_api_lookup(self):
        """Load API lookup data for field descriptions"""
        try:
            with open(self.api_lookup_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"API lookup file not found: {self.api_lookup_file}")
            return {}
    
    def standardize_node_attrs(self, node_attrs):
        """
        Fill in missing node attributes with empty defaults to ensure consistency.
        """
        standard_keys = ["node_type", "name", "full_name", "product", "type", "description"]
        for key in standard_keys:
            if key not in node_attrs:
                node_attrs[key] = ''
        return node_attrs

    def standardize_edge_attrs(self, edge_attrs):
        """
        Ensure every edge has the same attributes, filling missing ones.
        """
        standard_keys = ["relationship", "justification"]
        for key in standard_keys:
            if key not in edge_attrs:
                edge_attrs[key] = ''
        return edge_attrs


    def add_product_node(self, product_name):
        node_attrs = {
            "node_type": "product",
            "name": product_name
        }
        self.graph.add_node(product_name, **self.standardize_node_attrs(node_attrs))

    def add_type_node(self, product_name, type_name):
        full_type_name = f"{product_name}_{type_name}"
        node_attrs = {
            "node_type": "type",
            "name": type_name,
            "product": product_name,
            "full_name": full_type_name
        }
        self.graph.add_node(full_type_name, **self.standardize_node_attrs(node_attrs))
        
        # Add directed edge from product to type
        self.graph.add_edge(product_name, full_type_name, **self.standardize_edge_attrs({'relationship': 'has_type'}))

    def add_field_node(self, product_name, type_name, field_name):
        full_field_name = f"{type_name}.{field_name}"
        field_id = f"{product_name}_{full_field_name}"
        full_type_name = f"{product_name}_{type_name}"

        description = self.api_lookup.get(full_field_name, {}).get('description', 'No description available')

        node_attrs = {
            "node_type": "field",
            "name": field_name,
            "full_name": full_field_name,
            "type": type_name,
            "product": product_name,
            "description": description
        }
        self.graph.add_node(field_id, **self.standardize_node_attrs(node_attrs))

        # Add directed edge from type to field
        self.graph.add_edge(full_type_name, field_id, **self.standardize_edge_attrs({'relationship': 'has_field'}))
    
    def add_field_mapping(self, src_product, src_field_full, tgt_product, tgt_field_full, justification):
        """Add bidirectional field mapping"""
        src_field_id = f"{src_product}_{src_field_full}"
        tgt_field_id = f"{tgt_product}_{tgt_field_full}"
        
        # Add bidirectional edges with justification
        edge_attrs = self.standardize_edge_attrs({'relationship': 'field_mapping', 'justification': justification})
        self.graph.add_edge(src_field_id, tgt_field_id, **edge_attrs)
        self.graph.add_edge(tgt_field_id, src_field_id, **edge_attrs)

    def process_mapping_file(self, file_path):
        """Process a single mapping JSON file"""
        try:
            with open(file_path, 'r') as f:
                mapping_data = json.load(f)
            
            product_name = mapping_data["name"].split("_")[0]
            self.add_product_node(product_name)
            
            properties = mapping_data["properties"]
            
            for src_field_full, mappings in properties.items():
                # Parse source field
                src_type, src_field = src_field_full.split(".", 1)
                
                # Add source nodes
                self.add_type_node(product_name, src_type)
                self.add_field_node(product_name, src_type, src_field)
                
                # Process mappings
                for mapping in mappings:
                    tgt_schema = mapping["mapTable"]
                    tgt_product = tgt_schema.split("_")[0]
                    tgt_type, tgt_field = mapping["mapProperty"].split(".", 1)
                    justification = mapping["justification"]
                    
                    # Add target nodes
                    self.add_product_node(tgt_product)
                    self.add_type_node(tgt_product, tgt_type)
                    self.add_field_node(tgt_product, tgt_type, tgt_field)
                    
                    # Add field mapping
                    self.add_field_mapping(
                        product_name, src_field_full,
                        tgt_product, f"{tgt_type}.{tgt_field}",
                        justification
                    )
                    
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    def build_graph(self):
        """Build the complete graph from all mapping files"""
        print("Building mapping graph...")
        
        # Process all JSON files in the mapping directory
        for filename in os.listdir(self.mapping_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.mapping_dir, filename)
                self.process_mapping_file(file_path)
        
        # Clean all edge attributes except for 'relationship' and 'justification'
        for u, v, edge_attrs in list(self.graph.edges(data=True)):
            cleaned_attrs = {k: edge_attrs.get(k, '') for k in ['relationship', 'justification']}
            self.graph[u][v].clear()
            self.graph[u][v].update(cleaned_attrs)

        print(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        return self.graph
    
    
    def get_graph_statistics(self):
        """Get basic graph statistics"""
        stats = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'node_types': {}
        }
        
        # Count nodes by type
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('node_type', 'unknown')
            stats['node_types'][node_type] = stats['node_types'].get(node_type, 0) + 1
        
        # Count edges by relationship
        edge_types = {}
        for src, tgt, data in self.graph.edges(data=True):
            rel_type = data.get('relationship', 'unknown')
            edge_types[rel_type] = edge_types.get(rel_type, 0) + 1
        
        stats['edge_types'] = edge_types
        return stats

def populate_mapping_graph(mapping_dir, api_lookup_file):
    """
    Main function to populate the mapping graph
    """
    builder = MappingGraphBuilder(mapping_dir, api_lookup_file)
    graph = builder.build_graph()
    
    # Print statistics
    stats = builder.get_graph_statistics()
    print(f"\nGraph Statistics:")
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Total edges: {stats['total_edges']}")
    
    print(f"\nNode types:")
    for node_type, count in stats['node_types'].items():
        print(f"  {node_type}: {count}")
    
    print(f"\nEdge types:")
    for edge_type, count in stats['edge_types'].items():
        print(f"  {edge_type}: {count}")
    
    return graph, builder

def test_graph_queries(graph):
    """
    Test queries to explore the graph structure
    """
    print("\n=== Graph Test Queries ===")
    
    # Query 1: Find all products and their types
    print("\nQuery 1: Products and their types")
    products = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'product']
    for product in products[:3]:  # Show first 3
        types = [n for n in graph.successors(product) if graph.nodes[n].get('node_type') == 'type']
        print(f"  {product}: {len(types)} types")
    
    # Query 2: Find field mappings with justifications
    print("\nQuery 2: Field mappings with justifications")
    field_mappings = [(u, v, d) for u, v, d in graph.edges(data=True) 
                     if d.get('relationship') == 'field_mapping']
    for src, tgt, data in field_mappings[:5]:  # Show first 5
        src_field = graph.nodes[src].get('full_name', src)
        tgt_field = graph.nodes[tgt].get('full_name', tgt)
        justification = data.get('justification', 'No justification')
        print(f"  {src_field} -> {tgt_field}")
        print(f"    Justification: {justification}...")
    
    # Query 3: Find fields with descriptions
    print("\nQuery 3: Fields with descriptions")
    fields_with_desc = [(n, d) for n, d in graph.nodes(data=True) 
                       if d.get('node_type') == 'field' and 'description' in d]
    for node, data in fields_with_desc[:3]:  # Show first 3
        print(f"  {data.get('full_name', node)}")
        print(f"    Description: {data.get('description', 'No description')}...")

def prepare_mapping_graph_data(graph, model):
    """
    Convert NetworkX mapping graph to PyTorch Geometric format
    """
    print("Preparing mapping graph data...")
    
    # Create node features
    node_features = []
    node_list = list(graph.nodes())
    
    for node in tqdm(node_list, desc="Creating node features"):
        node_data = graph.nodes[node]
        node_type = node_data.get('node_type', 'unknown')
        
        # Create text representation based on node type
        if node_type == 'product':
            text_repr = f"Product: {node_data.get('name', node)}"
        elif node_type == 'type':
            text_repr = f"Type: {node_data.get('name', node)} in {node_data.get('product', 'unknown')}"
        elif node_type == 'field':
            description = node_data.get('description', 'No description')
            text_repr = f"Field: {node_data.get('full_name', node)} - {description[:100]}"
        else:
            text_repr = f"{node_type}: {node}"
        
        # Get embedding
        embedding = model.encode([text_repr])[0]
        node_features.append(embedding)
    
    # Convert to PyTorch Geometric format
    data = from_networkx(graph)
    
    # Add node features
    node_features_array = np.array(node_features)
    data.x = torch.FloatTensor(node_features_array)
    
    # Add node type one-hot encoding
    node_types = [graph.nodes[node].get('node_type', 'unknown') for node in node_list]
    unique_types = list(set(node_types))
    type_to_idx = {t: i for i, t in enumerate(unique_types)}
    
    type_features = torch.zeros(len(node_list), len(unique_types))
    for i, node_type in enumerate(node_types):
        type_features[i, type_to_idx[node_type]] = 1
    
    # Concatenate embeddings with type features
    data.x = torch.cat([data.x, type_features], dim=-1)
    
    return data, node_list, type_to_idx

def create_train_test_split_mapping(data, test_ratio=0.2, val_ratio=0.1, seed=42):
    """
    * Randomly permutes edge indices
    * Splits them into train/val/test based on the ratios
    * Uses negative sampling to create negative edges for validation and test
    * Avoids tensor boolean indexing errors
    """
    print("Creating train/test split...")
    
    if data.edge_index is None or data.edge_index.size(1) == 0:
        print("Warning: No edges found in the graph.")
        return data
    
    try:
        # Get all edges
        edge_index = data.edge_index
        num_edges = edge_index.size(1)
        
        # Create random permutation of edge indices with seed
        torch.manual_seed(seed)  # Set seed before random permutation
        perm = torch.randperm(num_edges)
                
        # Calculate split sizes
        test_size = int(num_edges * test_ratio)
        val_size = int(num_edges * val_ratio)
        train_size = num_edges - test_size - val_size
        
        # Split edge indices
        train_edges = perm[:train_size]
        val_edges = perm[train_size:train_size + val_size]
        test_edges = perm[train_size + val_size:]
        
        # Create positive edge sets
        data.train_pos_edge_index = edge_index[:, train_edges]
        data.val_pos_edge_index = edge_index[:, val_edges]
        data.test_pos_edge_index = edge_index[:, test_edges]
        
        # Generate negative edges for validation and test
        # For validation negative edges
        val_neg_edges = negative_sampling(
            edge_index=data.train_pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.val_pos_edge_index.size(1)
        )
        data.val_neg_edge_index = val_neg_edges
        
        # For test negative edges  
        test_neg_edges = negative_sampling(
            edge_index=torch.cat([data.train_pos_edge_index, data.val_pos_edge_index], dim=1),
            num_nodes=data.num_nodes,
            num_neg_samples=data.test_pos_edge_index.size(1)
        )
        data.test_neg_edge_index = test_neg_edges
        
        print(f"Train edges: {data.train_pos_edge_index.size(1)}")
        print(f"Val positive edges: {data.val_pos_edge_index.size(1)}")
        print(f"Val negative edges: {data.val_neg_edge_index.size(1)}")
        print(f"Test positive edges: {data.test_pos_edge_index.size(1)}")
        print(f"Test negative edges: {data.test_neg_edge_index.size(1)}")
        
    except Exception as e:
        print(f"Error during edge splitting: {e}")
        # Fallback: use all edges for training
        data.train_pos_edge_index = data.edge_index.clone()
        data.val_pos_edge_index = torch.empty((2, 0), dtype=torch.long)
        data.val_neg_edge_index = torch.empty((2, 0), dtype=torch.long)
        data.test_pos_edge_index = torch.empty((2, 0), dtype=torch.long)
        data.test_neg_edge_index = torch.empty((2, 0), dtype=torch.long)
    
    return data

if __name__ == "__main__":
    SEED = 52
    set_seed(SEED)
    
    mapping_dir = "../data/mappings_output"
    api_lookup_file = "api_lookup.json"
    
    # Build mapping graph
    graph, builder = populate_mapping_graph(mapping_dir, api_lookup_file)
    test_graph_queries(graph)
    
    # Prepare data for GNN
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    data, node_list, type_to_idx = prepare_mapping_graph_data(graph, model)
    data = create_train_test_split_mapping(data, seed=SEED)
    
    # Save everything needed for training
    import pickle
    with open('graph_data.pkl', 'wb') as f:
        pickle.dump({
            'data': data,
            'node_list': node_list,
            'type_to_idx': type_to_idx,
            'graph': graph
        }, f)
    print("Graph data saved to 'graph_data.pkl'")
