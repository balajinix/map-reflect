import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool
from torch_geometric.utils import from_networkx, negative_sampling
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import matplotlib.pyplot as plt
from networkx.readwrite import json_graph
from spider import Spider
from functools import lru_cache
from milvus_helper import MilvusHelper
from pymilvus import connections, Collection
import os

# Original code setup with collection creation handling
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

def setup_milvus_collection(collection_name="spider_wikidata_nary_facts", dim=384):
    """
    Setup Milvus collection, create if it doesn't exist
    """
    connections.connect("default", host="127.0.0.1", port="19530")
    
    # Check if collection exists
    if utility.has_collection(collection_name):
        print(f"Collection '{collection_name}' exists, loading...")
        collection = Collection(name=collection_name)
        collection.load()
        return collection
    else:
        print(f"Collection '{collection_name}' doesn't exist, creating...")
        
        # Define collection schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="entity", dtype=DataType.VARCHAR, max_length=512)
        ]
        
        schema = CollectionSchema(fields, description="Spider Wikidata N-ary Facts Collection")
        
        # Create collection
        collection = Collection(name=collection_name, schema=schema)
        
        # Create index for vector field
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="vector", index_params=index_params)
        
        print(f"Collection '{collection_name}' created successfully!")
        print("Note: The collection is empty. You'll need to populate it with your data.")
        
        return collection

# Setup collection
collection_name = "spider_wikidata_nary_facts"
collection = setup_milvus_collection(collection_name)

def populate_milvus_collection(collection, entities, embeddings):
    """
    Populate the Milvus collection with entities and their embeddings
    """
    print("Populating Milvus collection...")
    
    # Prepare data for insertion
    data = [
        list(range(len(entities))),  # IDs (will be auto-generated)
        embeddings.tolist(),         # Vector embeddings
        entities.tolist()            # Entity names
    ]
    
    # Insert data
    mr = collection.insert(data)
    collection.flush()
    
    print(f"Inserted {len(entities)} entities into collection")
    return mr

def load_or_create_entities_and_embeddings():
    """
    Load entities and embeddings, create them if they don't exist
    """
    try:
        # Try to load existing entities and embeddings
        wikidata_entities = np.load('./data/wikidata_nary_facts_entities.npy', allow_pickle=True)
        spider_entities = np.load('./data/spider_nary_facts_entities.npy', allow_pickle=True)
        entities = np.concatenate([wikidata_entities, spider_entities])
        
        print(f"Loaded {len(entities)} entities from existing files")
        return entities
        
    except FileNotFoundError as e:
        print(f"Entity files not found: {e}")
        print("Creating dummy entities for demonstration...")
        
        # Create some dummy entities for demonstration
        dummy_entities = np.array([
            "database_1", "database_2", "table_users", "table_orders", 
            "column_id", "column_name", "concept_person", "concept_product"
        ])
        
        # You would replace this with your actual entity creation logic
        print("Note: Using dummy entities. Replace with your actual entity loading/creation logic.")
        return dummy_entities

def load_or_create_entity_mappings():
    """
    Load entity fact mappings, create empty ones if they don't exist
    """
    try:
        with open('./data/wikidata_entity_nary_facts_mapping.json', 'r') as f:
            wikidata_entity_fact_mapping = json.load(f)
    except FileNotFoundError:
        print("Wikidata entity mapping not found, using empty mapping")
        wikidata_entity_fact_mapping = {}
    
    try:
        with open('./data/spider_entity_nary_facts_mapping.json', 'r') as f:
            spider_entity_fact_mapping = json.load(f)
    except FileNotFoundError:
        print("Spider entity mapping not found, using empty mapping")
        spider_entity_fact_mapping = {}
    
    # Handle conflicts gracefully
    entity_fact_mapping = {**wikidata_entity_fact_mapping, **spider_entity_fact_mapping}
    print(f"Loaded {len(entity_fact_mapping)} entity fact mappings")
    
    return entity_fact_mapping

# Load entities and mappings with error handling
entities = load_or_create_entities_and_embeddings()
entity_fact_mapping = load_or_create_entity_mappings()

milvus_helper = MilvusHelper(model_name='sentence-transformers/all-MiniLM-L6-v2',
                             collection_name='n_ary_facts',
                             dim=384)

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
metadataKG = nx.Graph()

# Original utility functions
@lru_cache(maxsize=10000)
def get_query_embedding(query):
    global model
    try:
        return model.encode([query])[0].tolist()
    except Exception as e:
        print(f"Failed to encode query: {e}")
        return None

@lru_cache(maxsize=10000)
def search_in_milvus_cached(query, top_k=5):
    try:
        query_embedding = get_query_embedding(query)
        if query_embedding is None:
            return []
        
        # Check if collection has data
        if collection.num_entities == 0:
            print("Warning: Milvus collection is empty. Returning empty results.")
            return []
            
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search([query_embedding], "vector", search_params, limit=top_k, expr=None)
        
        # Handle results safely
        related_entities = []
        for hits in results:
            for hit in hits:
                if hit.id < len(entities):
                    related_entities.append(entities[hit.id])
        
        return related_entities
    except Exception as e:
        print(f"Failed to search in Milvus: {e}")
        return []

def search_in_milvus(query, top_k=5):
    return search_in_milvus_cached(query, top_k)

def process_related_entities_and_update_graph(query, node_type):
    global metadataKG, entity_fact_mapping
    related_entities = search_in_milvus_cached(query)
    for entity in related_entities:
        nary_facts = entity_fact_mapping.get(entity)
        if nary_facts is None:
            continue
        for nary_fact in nary_facts:
            process_fact_and_update_graph(query, node_type, nary_fact)
            if 'attributes' in nary_fact:
                for attribute in nary_fact['attributes']:
                    # not connecting attributes to query by sending node type as concept
                    process_fact_and_update_graph(query, "concept", attribute)

def process_fact_and_update_graph(query, subject_type, fact):
    global metadataKG
    if fact['subject'] is not None and 'None' not in fact['subject']:
        metadataKG.add_node(fact['subject'], type='concept')
    if fact['object'] is not None and 'None' not in fact['object']:
        metadataKG.add_node(fact['object'], type='concept')
    if fact['subject'] is not None and 'None' not in fact['subject'] and fact['object'] is not None and 'None' not in fact['object'] and 'RNone' not in fact['predicate']:
        metadataKG.add_edge(fact['subject'], fact['object'], predicate=fact['predicate'])
    # add the link between metadata node and knowledge graph node
    if fact['subject'] is not None and 'None' not in fact['subject'] and subject_type != "concept":
        metadataKG.add_edge(query, fact['subject'], predicate='has_concept')

def populate_metadata_knowledge_graph(split_name):
    global metadataKG
    spider = Spider(split=split_name)
    database_names = spider.get_database_names()

    for db_id in database_names:
        db_details = spider.get_db_details(db_id)
        if not db_details:
            continue

        metadataKG.add_node(db_id, type='database')

        process_related_entities_and_update_graph(db_id, 'database')
        for table in db_details['tables']:
            metadataKG.add_node(table, type='table')
            metadataKG.add_edge(db_id, table, predicate="has_table")

            process_related_entities_and_update_graph(table, 'table')

            columns = db_details['columns'][table]
            for column in columns:
                metadataKG.add_node(column, type='column')
                metadataKG.add_edge(table, column, predicate="has_column")

                query_string = f"{db_id} {table} {column}"
                process_related_entities_and_update_graph(query_string, 'column')

    return metadataKG

def save_metadata_knowledge_graph(metadataKG, file_path):
    graph_data = json_graph.node_link_data(metadataKG)
    with open(file_path, 'w') as f:
        json.dump(graph_data, f)

def load_metadata_knowledge_graph(file_path):
    try:
        with open(file_path, 'r') as f:
            graph_data = json.load(f)
        # Fix NetworkX warning by explicitly setting edges parameter
        metadataKG = json_graph.node_link_graph(graph_data, edges="links")
        print(f"Metadata knowledge graph loaded successfully from {file_path}.")
        return metadataKG
    except FileNotFoundError:
        print(f"No existing metadata knowledge graph found at {file_path}.")
        return None
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON from {file_path}: {e}")
        return None

def print_metadata_knowledge_graph_stats(metadataKG):
    print(f"Number of nodes: {metadataKG.number_of_nodes()}")
    print(f"Number of edges: {metadataKG.number_of_edges()}")

    average_degree = sum(dict(metadataKG.degree()).values()) / metadataKG.number_of_nodes()
    print(f"Average degree: {average_degree:.2f}")

    density = nx.density(metadataKG)
    print(f"Density: {density:.4f}")

    if nx.is_connected(metadataKG):
        print("The graph is connected")
    else:
        print("The graph is not connected")

    connected_components = nx.number_connected_components(metadataKG)
    print(f"Number of connected components: {connected_components}")

    degree_centrality = nx.degree_centrality(metadataKG)
    top_5_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Top 5 nodes by degree centrality:", top_5_degree)

    betweenness_centrality = nx.betweenness_centrality(metadataKG)
    top_5_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Top 5 nodes by betweenness centrality:", top_5_betweenness)

    node_types = [data.get('type', 'undefined') for node, data in metadataKG.nodes(data=True)]
    type_counts = {}
    for node_type in node_types:
        if node_type in type_counts:
            type_counts[node_type] += 1
        else:
            type_counts[node_type] = 1

    print("Node type counts:")
    for node_type, count in type_counts.items():
        print(f"  {node_type}: {count}")

# GNN Classes and Functions
class MetadataKGGNN(nn.Module):
    """
    GNN model for link prediction on metadata knowledge graphs
    Supports different GNN architectures: GCN, GraphSAGE, GAT
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, 
                 gnn_type='GCN', dropout=0.2, heads=1):
        super(MetadataKGGNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type
        
        # Create GNN layers
        self.convs = nn.ModuleList()
        
        if gnn_type == 'GCN':
            self.convs.append(GCNConv(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.convs.append(GCNConv(hidden_dim, output_dim))
            
        elif gnn_type == 'SAGE':
            self.convs.append(SAGEConv(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.convs.append(SAGEConv(hidden_dim, output_dim))
            
        elif gnn_type == 'GAT':
            self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
            self.convs.append(GATConv(hidden_dim * heads, output_dim, heads=1, dropout=dropout))
        
        # Link prediction head
        self.link_predictor = nn.Sequential(
            nn.Linear(2 * output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x, edge_index):
        # Apply GNN layers
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        return x
    
    def predict_links(self, x, edge_index, query_edges):
        """
        Predict links for given edge pairs
        """
        # Get node embeddings
        node_embeddings = self.forward(x, edge_index)
        
        # Get embeddings for source and target nodes
        source_embeddings = node_embeddings[query_edges[0]]
        target_embeddings = node_embeddings[query_edges[1]]
        
        # Concatenate embeddings
        edge_embeddings = torch.cat([source_embeddings, target_embeddings], dim=-1)
        
        # Predict link probability
        return torch.sigmoid(self.link_predictor(edge_embeddings))

def prepare_metadata_graph_data(metadataKG, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """
    Convert NetworkX metadata graph to PyTorch Geometric format with node features
    """
    print("Preparing metadata graph data...")
    
    # Use the existing sentence transformer model
    global model
    sentence_model = model
    
    # Create node features based on node names and types
    node_features = []
    node_list = list(metadataKG.nodes())
    
    for node in tqdm(node_list, desc="Creating node features"):
        node_data = metadataKG.nodes[node]
        node_type = node_data.get('type', 'unknown')
        
        # Create text representation of node
        text_repr = f"{node} {node_type}"
        
        # Get embedding
        embedding = sentence_model.encode([text_repr])[0]
        node_features.append(embedding)
    
    # Convert to PyTorch Geometric format
    data = from_networkx(metadataKG)
    
    # Fix tensor creation warning by converting to numpy array first
    node_features_array = np.array(node_features)
    data.x = torch.FloatTensor(node_features_array)
    
    # Create node type mapping for additional features
    node_types = [metadataKG.nodes[node].get('type', 'unknown') for node in node_list]
    unique_types = list(set(node_types))
    type_to_idx = {t: i for i, t in enumerate(unique_types)}
    
    # Add one-hot encoded node types
    type_features = torch.zeros(len(node_list), len(unique_types))
    for i, node_type in enumerate(node_types):
        type_features[i, type_to_idx[node_type]] = 1
    
    # Concatenate embeddings with type features
    data.x = torch.cat([data.x, type_features], dim=-1)
    
    return data, node_list, type_to_idx

from torch_geometric.transforms import RandomLinkSplit

def create_train_test_split(data, test_ratio=0.2, val_ratio=0.1):
    """
    Split edges into train/validation/test sets
    """
    print("Creating train/test split...")
    
    # Check if we have edges to split
    if data.edge_index is None or data.edge_index.size(1) == 0:
        print("Warning: No edges found in the graph. Cannot create train/test split.")
        # Create dummy edge indices for compatibility
        data.train_pos_edge_index = torch.empty((2, 0), dtype=torch.long)
        data.test_pos_edge_index = torch.empty((2, 0), dtype=torch.long)
        data.test_neg_edge_index = torch.empty((2, 0), dtype=torch.long)
        data.val_pos_edge_index = torch.empty((2, 0), dtype=torch.long)
        data.val_neg_edge_index = torch.empty((2, 0), dtype=torch.long)
        return data
    
    try:
        # Use modern PyG RandomLinkSplit transform instead of deprecated function
        transform = RandomLinkSplit(
            num_val=val_ratio,
            num_test=test_ratio,
            is_undirected=True,
            add_negative_train_samples=False,
            neg_sampling_ratio=1.0,
        )
        
        train_data, val_data, test_data = transform(data)
        
        # Copy the split data back to original data object for compatibility
        data.train_pos_edge_index = train_data.edge_index
        data.val_pos_edge_index = val_data.edge_label_index[:, val_data.edge_label == 1]
        data.val_neg_edge_index = val_data.edge_label_index[:, val_data.edge_label == 0]
        data.test_pos_edge_index = test_data.edge_label_index[:, test_data.edge_label == 1]
        data.test_neg_edge_index = test_data.edge_label_index[:, test_data.edge_label == 0]
        
        print(f"Train edges: {data.train_pos_edge_index.size(1)}")
        print(f"Val positive edges: {data.val_pos_edge_index.size(1)}")
        print(f"Val negative edges: {data.val_neg_edge_index.size(1)}")
        print(f"Test positive edges: {data.test_pos_edge_index.size(1)}")
        print(f"Test negative edges: {data.test_neg_edge_index.size(1)}")
        
    except Exception as e:
        print(f"Error during edge splitting: {e}")
        # Create empty edge indices as fallback
        data.train_pos_edge_index = data.edge_index.clone()
        data.test_pos_edge_index = torch.empty((2, 0), dtype=torch.long)
        data.test_neg_edge_index = torch.empty((2, 0), dtype=torch.long)
        data.val_pos_edge_index = torch.empty((2, 0), dtype=torch.long) 
        data.val_neg_edge_index = torch.empty((2, 0), dtype=torch.long)
    
    return data

def train_metadata_kg_model(model, data, num_epochs=200, lr=0.01, weight_decay=5e-4):
    """
    Train the GNN model for link prediction on metadata KG
    """
    print("Training metadata KG model...")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()
    
    model.train()
    train_losses = []
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        optimizer.zero_grad()
        
        # Forward pass
        node_embeddings = model(data.x, data.train_pos_edge_index)
        
        # Positive edges
        pos_edges = data.train_pos_edge_index
        pos_pred = model.predict_links(data.x, data.train_pos_edge_index, pos_edges)
        pos_labels = torch.ones(pos_edges.size(1), 1)
        
        # Negative edges (sample them)
        neg_edges = negative_sampling(
            edge_index=data.train_pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=pos_edges.size(1)
        )
        neg_pred = model.predict_links(data.x, data.train_pos_edge_index, neg_edges)
        neg_labels = torch.zeros(neg_edges.size(1), 1)
        
        # Combine predictions and labels
        all_pred = torch.cat([pos_pred, neg_pred])
        all_labels = torch.cat([pos_labels, neg_labels])
        
        # Calculate loss
        loss = criterion(all_pred, all_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        if epoch % 50 == 0:
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')
    
    return train_losses

def evaluate_metadata_kg_model(model, data):
    """
    Evaluate the model on test set
    """
    print("Evaluating metadata KG model...")
    
    # Check if we have test data
    if (not hasattr(data, 'test_pos_edge_index') or 
        data.test_pos_edge_index is None or 
        data.test_pos_edge_index.size(1) == 0):
        print("Warning: No test edges available for evaluation.")
        return 0.0, 0.0
    
    if data.train_pos_edge_index is None or data.train_pos_edge_index.size(1) == 0:
        print("Warning: No training edges available for evaluation.")
        return 0.0, 0.0
    
    model.eval()
    with torch.no_grad():
        try:
            # Test positive edges
            pos_pred = model.predict_links(data.x, data.train_pos_edge_index, data.test_pos_edge_index)
            pos_labels = torch.ones(data.test_pos_edge_index.size(1))
            
            # Test negative edges
            if hasattr(data, 'test_neg_edge_index') and data.test_neg_edge_index is not None:
                neg_pred = model.predict_links(data.x, data.train_pos_edge_index, data.test_neg_edge_index)
                neg_labels = torch.zeros(data.test_neg_edge_index.size(1))
            else:
                print("Warning: No negative test edges found. Using random negative sampling.")
                # Generate negative edges
                neg_edges = negative_sampling(
                    edge_index=data.train_pos_edge_index,
                    num_nodes=data.num_nodes,
                    num_neg_samples=data.test_pos_edge_index.size(1)
                )
                neg_pred = model.predict_links(data.x, data.train_pos_edge_index, neg_edges)
                neg_labels = torch.zeros(neg_edges.size(1))
            
            # Combine predictions
            all_pred = torch.cat([pos_pred.squeeze(), neg_pred.squeeze()])
            all_labels = torch.cat([pos_labels, neg_labels])
            
            # Calculate metrics
            if len(all_pred) > 0 and len(all_labels) > 0:
                auc = roc_auc_score(all_labels.cpu(), all_pred.cpu())
                ap = average_precision_score(all_labels.cpu(), all_pred.cpu())
            else:
                print("Warning: No predictions available for evaluation.")
                auc, ap = 0.0, 0.0
                
        except Exception as e:
            print(f"Error during evaluation: {e}")
            auc, ap = 0.0, 0.0
        
    return auc, ap

def predict_new_metadata_links(model, data, node_list, top_k=10):
    """
    Predict new links for metadata KG
    """
    print("Predicting new metadata links...")
    
    # Check if we have valid data
    if data.edge_index is None or data.edge_index.size(1) == 0:
        print("Warning: No edges found in the graph. Cannot predict new links.")
        return []
    
    if data.train_pos_edge_index is None or data.train_pos_edge_index.size(1) == 0:
        print("Warning: No training edges found. Cannot predict new links.")
        return []
    
    model.eval()
    with torch.no_grad():
        # Get all possible edges (excluding existing ones)
        all_edges = []
        existing_edges = set()
        
        # Add existing edges to set (use all edges, not just training)
        edge_index_to_use = data.edge_index if data.edge_index is not None else data.train_pos_edge_index
        
        for i in range(edge_index_to_use.size(1)):
            edge = (edge_index_to_use[0, i].item(), edge_index_to_use[1, i].item())
            existing_edges.add(edge)
            existing_edges.add((edge[1], edge[0]))  # Add reverse edge
        
        # Add test edges if they exist
        if hasattr(data, 'test_pos_edge_index') and data.test_pos_edge_index is not None:
            for i in range(data.test_pos_edge_index.size(1)):
                edge = (data.test_pos_edge_index[0, i].item(), data.test_pos_edge_index[1, i].item())
                existing_edges.add(edge)
                existing_edges.add((edge[1], edge[0]))
        
        # Add validation edges if they exist
        if hasattr(data, 'val_pos_edge_index') and data.val_pos_edge_index is not None:
            for i in range(data.val_pos_edge_index.size(1)):
                edge = (data.val_pos_edge_index[0, i].item(), data.val_pos_edge_index[1, i].item())
                existing_edges.add(edge)
                existing_edges.add((edge[1], edge[0]))
        
        # Generate all possible edges
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                if (i, j) not in existing_edges:
                    all_edges.append((i, j))
        
        if not all_edges:
            print("No new edges to predict - all possible edges already exist")
            return []
        
        print(f"Predicting probabilities for {len(all_edges)} potential edges...")
        
        # Convert to tensor
        query_edges = torch.tensor(all_edges).t()
        
        # Predict probabilities
        try:
            probs = model.predict_links(data.x, data.train_pos_edge_index, query_edges)
            
            # Get top-k predictions
            top_indices = torch.topk(probs.squeeze(), min(top_k, len(all_edges))).indices
            
            predictions = []
            for idx in top_indices:
                edge_idx = all_edges[idx]
                prob = probs[idx].item()
                source_node = node_list[edge_idx[0]]
                target_node = node_list[edge_idx[1]]
                predictions.append((source_node, target_node, prob))
            
        except Exception as e:
            print(f"Error during link prediction: {e}")
            return []
        
    return predictions

# Main execution and configuration
splits = ['dev', 'train', 'test']

def build_or_load_metadata_kg(metadata_kg_file_path="./data/spider_metadata_kg.json"):
    """
    Build or load the metadata knowledge graph
    """
    global metadataKG
    
    if os.path.exists(metadata_kg_file_path):
        metadataKG = load_metadata_knowledge_graph(metadata_kg_file_path)
    else:
        metadataKG = nx.Graph()
        for split_name in splits:
            print(f"Processing split: {split_name}")
            populate_metadata_knowledge_graph(split_name)
        save_metadata_knowledge_graph(metadataKG, metadata_kg_file_path)
        print_metadata_knowledge_graph_stats(metadataKG)
    
    return metadataKG

def run_gnn_link_prediction(metadata_kg_file_path="./data/spider_metadata_kg.json"):
    """
    Main function to run GNN link prediction on metadata KG
    """
    # Build or load metadata KG
    metadataKG = build_or_load_metadata_kg(metadata_kg_file_path)
    
    print(f"Metadata KG loaded: {metadataKG.number_of_nodes()} nodes, {metadataKG.number_of_edges()} edges")
    
    # Prepare data for GNN
    data, node_list, type_to_idx = prepare_metadata_graph_data(metadataKG)
    data = create_train_test_split(data)
    
    # Initialize model
    input_dim = data.x.size(1)
    hidden_dim = 128
    output_dim = 64
    
    gnn_model = MetadataKGGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=3,
        gnn_type='GCN',  # Can be 'GCN', 'SAGE', or 'GAT'
        dropout=0.2
    )
    
    print(f"Model initialized with {sum(p.numel() for p in gnn_model.parameters())} parameters")
    
    # Train model
    train_losses = train_metadata_kg_model(gnn_model, data, num_epochs=200, lr=0.01)
    
    # Evaluate model
    auc, ap = evaluate_metadata_kg_model(gnn_model, data)
    print(f"Test AUC: {auc:.4f}")
    print(f"Test AP: {ap:.4f}")
    
    # Predict new links
    predictions = predict_new_metadata_links(gnn_model, data, node_list, top_k=20)
    
    print("\nTop predicted metadata links:")
    for source, target, prob in predictions[:10]:
        print(f"{source} -> {target}: {prob:.4f}")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title('Metadata KG Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    
    return gnn_model, data, node_list, predictions, metadataKG

if __name__ == "__main__":
    # Run the complete pipeline
    gnn_model, data, node_list, predictions, metadataKG = run_gnn_link_prediction()
    
    # Print final statistics
    print("\n=== Final Metadata KG Statistics ===")
    print_metadata_knowledge_graph_stats(metadataKG)
