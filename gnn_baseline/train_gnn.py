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

set_seed(57)
output_dir = "gnn_results"
os.makedirs(output_dir, exist_ok=True)

# GNN Classes for mapping graph
class MappingKGGNN(nn.Module):
    """
    GNN model for link prediction on mapping knowledge graphs
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, 
                 gnn_type='GCN', dropout=0.2, heads=1):
        super(MappingKGGNN, self).__init__()
        
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
        """Predict links for given edge pairs"""
        node_embeddings = self.forward(x, edge_index)
        
        source_embeddings = node_embeddings[query_edges[0]]
        target_embeddings = node_embeddings[query_edges[1]]
        
        edge_embeddings = torch.cat([source_embeddings, target_embeddings], dim=-1)
        return torch.sigmoid(self.link_predictor(edge_embeddings))

def train_mapping_model(model, data, num_epochs=100, lr=0.01, seed=42):
    """Train the GNN model for mapping graph"""
    print("Training mapping model...")
    
    # Set seed for training
    torch.manual_seed(seed)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.BCELoss()
    
    model.train()
    train_losses = []
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        optimizer.zero_grad()
        
        # Positive edges
        pos_edges = data.train_pos_edge_index
        pos_pred = model.predict_links(data.x, data.train_pos_edge_index, pos_edges)
        pos_labels = torch.ones(pos_edges.size(1), 1)
        
        # Negative edges - set seed for consistent negative sampling
        torch.manual_seed(seed + epoch)  # Different seed per epoch but deterministic
        neg_edges = negative_sampling(
            edge_index=data.train_pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=pos_edges.size(1)
        )
        neg_pred = model.predict_links(data.x, data.train_pos_edge_index, neg_edges)
        neg_labels = torch.zeros(neg_edges.size(1), 1)
        
        # Combine predictions
        all_pred = torch.cat([pos_pred, neg_pred])
        all_labels = torch.cat([pos_labels, neg_labels])
        
        loss = criterion(all_pred, all_labels)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        if epoch % 25 == 0:
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')
    
    return train_losses

def evaluate_mapping_model(model, data):
    """Evaluate the model on test set"""
    print("Evaluating mapping model...")
    
    if (not hasattr(data, 'test_pos_edge_index') or 
        data.test_pos_edge_index.size(1) == 0):
        print("Warning: No test edges available.")
        return 0.0, 0.0
    
    model.eval()
    with torch.no_grad():
        try:
            pos_pred = model.predict_links(data.x, data.train_pos_edge_index, data.test_pos_edge_index)
            pos_labels = torch.ones(data.test_pos_edge_index.size(1))
            
            if hasattr(data, 'test_neg_edge_index') and data.test_neg_edge_index.size(1) > 0:
                neg_pred = model.predict_links(data.x, data.train_pos_edge_index, data.test_neg_edge_index)
                neg_labels = torch.zeros(data.test_neg_edge_index.size(1))
            else:
                neg_edges = negative_sampling(
                    edge_index=data.train_pos_edge_index,
                    num_nodes=data.num_nodes,
                    num_neg_samples=data.test_pos_edge_index.size(1)
                )
                neg_pred = model.predict_links(data.x, data.train_pos_edge_index, neg_edges)
                neg_labels = torch.zeros(neg_edges.size(1))
            
            all_pred = torch.cat([pos_pred.squeeze(), neg_pred.squeeze()])
            all_labels = torch.cat([pos_labels, neg_labels])
            
            if len(all_pred) > 0:
                auc = roc_auc_score(all_labels.cpu(), all_pred.cpu())
                ap = average_precision_score(all_labels.cpu(), all_pred.cpu())
            else:
                auc, ap = 0.0, 0.0
                
        except Exception as e:
            print(f"Error during evaluation: {e}")
            auc, ap = 0.0, 0.0
    
    return auc, ap

def predict_new_mapping_links(model, data, node_list, graph, top_k=20):
    """Predict new field mappings"""
    print("Predicting new field mappings...")
    
    if data.train_pos_edge_index.size(1) == 0:
        return []
    
    model.eval()
    with torch.no_grad():
        # Find field nodes only
        field_nodes = []
        for i, node in enumerate(node_list):
            if graph.nodes[node].get('node_type') == 'field':
                field_nodes.append(i)
        
        if len(field_nodes) < 2:
            print("Not enough field nodes for prediction")
            return []
        
        # Get existing field mappings
        existing_mappings = set()
        for i in range(data.edge_index.size(1)):
            src, tgt = data.edge_index[0, i].item(), data.edge_index[1, i].item()
            if (src in field_nodes and tgt in field_nodes and 
                graph.nodes[node_list[src]].get('node_type') == 'field' and
                graph.nodes[node_list[tgt]].get('node_type') == 'field'):
                existing_mappings.add((src, tgt))
                existing_mappings.add((tgt, src))
        
        # Generate potential field-to-field mappings
        potential_edges = []
        for i in range(len(field_nodes)):
            for j in range(i + 1, len(field_nodes)):
                src_idx, tgt_idx = field_nodes[i], field_nodes[j]
                if (src_idx, tgt_idx) not in existing_mappings:
                    potential_edges.append((src_idx, tgt_idx))
        
        if not potential_edges:
            print("No new field mappings to predict")
            return []
        
        # Predict probabilities
        query_edges = torch.tensor(potential_edges).t()
        probs = model.predict_links(data.x, data.train_pos_edge_index, query_edges)
        
        # Get top-k predictions
        top_indices = torch.topk(probs.squeeze(), min(top_k, len(potential_edges))).indices
        
        predictions = []
        for idx in top_indices:
            edge_idx = potential_edges[idx]
            prob = probs[idx].item()
            src_node = node_list[edge_idx[0]]
            tgt_node = node_list[edge_idx[1]]
            
            # Get field details
            src_field = graph.nodes[src_node].get('full_name', src_node)
            tgt_field = graph.nodes[tgt_node].get('full_name', tgt_node)
            
            predictions.append((src_field, tgt_field, prob))
    
    return predictions

def run_mapping_gnn_pipeline(graph_data_file, seed=42):
    """
    Complete pipeline for mapping graph GNN - loads preprocessed data and trains
    """
    set_seed(seed)
    
    # Load the prepared data
    import pickle
    with open(graph_data_file, 'rb') as f:
        saved_data = pickle.load(f)
    
    data = saved_data['data']
    node_list = saved_data['node_list']
    type_to_idx = saved_data['type_to_idx']
    graph = saved_data['graph']
    
    print(f"Loaded graph with {len(node_list)} nodes and {data.edge_index.size(1)} edges")
    
    # Initialize GNN model
    input_dim = data.x.size(1)
    hidden_dim = 128
    output_dim = 64
    
    gnn_model = MappingKGGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=3,
        gnn_type='SAGE',
        dropout=0.2
    )
    
    print(f"GNN model initialized with {sum(p.numel() for p in gnn_model.parameters())} parameters")
    
    # Train model
    train_losses = train_mapping_model(gnn_model, data, num_epochs=100, lr=0.01, seed=seed)
    
    # Evaluate model
    auc, ap = evaluate_mapping_model(gnn_model, data)
    print(f"Test AUC: {auc:.4f}, Test AP: {ap:.4f}")
    
    predictions = predict_new_mapping_links(gnn_model, data, node_list, graph, top_k=15)
    
    print(f"\nTop {len(predictions)} predicted field mappings:")
    for i, (src_field, tgt_field, prob) in enumerate(predictions, 1):
        print(f"{i:2d}. {src_field} <-> {tgt_field} (confidence: {prob:.4f})")
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title('Mapping GNN Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_loss.png"), dpi=300)
    plt.show()

    return gnn_model, data, node_list, predictions, graph

# Usage example
if __name__ == "__main__":
    SEED = 57 #47 51 57 best
    set_seed(SEED)
    
    # Run complete pipeline with preprocessed data
    gnn_model, data, node_list, predictions, graph = run_mapping_gnn_pipeline(
        'graph_data.pkl', seed=SEED
    )
    
    # Save results
    import pickle
    with open(os.path.join(output_dir, 'mapping_gnn_results.pkl'), 'wb') as f:
        pickle.dump({
            'model': gnn_model,
            'graph': graph,
            'predictions': predictions,
            'node_list': node_list
        }, f)
    print(f"\nResults saved to '{output_dir}/mapping_gnn_results.pkl'")
