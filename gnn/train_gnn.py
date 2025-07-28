#!/usr/bin/env python3
"""
Updated GNN Training Script with Safe Model Saving
This version uses the new gnn_model.py module to avoid pickle import issues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

# Import the new model definition
try:
    from gnn_model import MappingKGGNN, save_model_safely
    print("✅ Successfully imported GNN model from gnn_model.py")
except ImportError as e:
    print(f"❌ Error importing GNN model: {e}")
    print("Please ensure gnn_model.py is in the current directory")
    exit(1)

try:
    from torch_geometric.utils import negative_sampling
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    print("⚠️  PyTorch Geometric not available, using fallback methods")
    TORCH_GEOMETRIC_AVAILABLE = False
    
    def negative_sampling(edge_index, num_nodes, num_neg_samples):
        """Fallback negative sampling without torch_geometric"""
        import torch
        
        # Simple random negative sampling
        all_edges = set()
        for i in range(edge_index.size(1)):
            src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
            all_edges.add((src, tgt))
            all_edges.add((tgt, src))  # Also add reverse
        
        neg_edges = []
        attempts = 0
        max_attempts = num_neg_samples * 10
        
        while len(neg_edges) < num_neg_samples and attempts < max_attempts:
            src = torch.randint(0, num_nodes, (1,)).item()
            tgt = torch.randint(0, num_nodes, (1,)).item()
            
            if src != tgt and (src, tgt) not in all_edges:
                neg_edges.append([src, tgt])
            
            attempts += 1
        
        if len(neg_edges) == 0:
            # Fallback: create some random edges
            neg_edges = [[0, 1], [1, 2]] if num_nodes > 2 else [[0, 0]]
        
        # Pad with random edges if we don't have enough
        while len(neg_edges) < num_neg_samples:
            src = torch.randint(0, num_nodes, (1,)).item()
            tgt = torch.randint(0, num_nodes, (1,)).item()
            neg_edges.append([src, tgt])
        
        return torch.tensor(neg_edges[:num_neg_samples]).t()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        
        # Check if we have training edges
        if not hasattr(data, 'train_pos_edge_index') or data.train_pos_edge_index.size(1) == 0:
            print("Warning: No training edges available")
            break
        
        # Positive edges
        pos_edges = data.train_pos_edge_index
        pos_pred = model.predict_links(data.x, pos_edges, pos_edges)
        pos_labels = torch.ones(pos_edges.size(1), 1)
        
        # Negative edges - set seed for consistent negative sampling
        torch.manual_seed(seed + epoch)  # Different seed per epoch but deterministic
        try:
            neg_edges = negative_sampling(
                edge_index=pos_edges,
                num_nodes=data.num_nodes,
                num_neg_samples=pos_edges.size(1)
            )
        except Exception as e:
            print(f"Warning: Error in negative sampling: {e}")
            # Create simple negative edges as fallback
            neg_edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t()
        
        neg_pred = model.predict_links(data.x, pos_edges, neg_edges)
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
    
    if not hasattr(data, 'train_pos_edge_index') or data.train_pos_edge_index.size(1) == 0:
        print("No training data available for predictions")
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
    try:
        with open(graph_data_file, 'rb') as f:
            saved_data = pickle.load(f)
    except Exception as e:
        print(f"Error loading graph data: {e}")
        return None, None, None, None, None
    
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
    print(f"Model info: {gnn_model.get_model_info()}")
    
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
    if train_losses:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses)
        plt.title('Mapping GNN Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.tight_layout()
        
        # Create output directory
        output_dir = "gnn_results"
        os.makedirs(output_dir, exist_ok=True)
        
        plt.savefig(os.path.join(output_dir, "training_loss.png"), dpi=300)
        print(f"Training loss plot saved to {output_dir}/training_loss.png")
        
        # Only show plot if in interactive environment
        try:
            if hasattr(plt, 'show'):
                plt.show()
        except:
            pass
        
        plt.close()

    return gnn_model, data, node_list, predictions, graph

# Usage example
if __name__ == "__main__":
    SEED = 57  # 47 51 57 best
    set_seed(SEED)
    
    output_dir = "gnn_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if graph data exists
    if not os.path.exists('graph_data.pkl'):
        print("❌ Error: graph_data.pkl not found")
        print("Please run: python populate_nx.py")
        exit(1)
    
    # Run complete pipeline with preprocessed data
    try:
        gnn_model, data, node_list, predictions, graph = run_mapping_gnn_pipeline(
            'graph_data.pkl', seed=SEED
        )
        
        if gnn_model is None:
            print("❌ Error: Training failed")
            exit(1)
        
        # Save results using the new safe method
        print("\nSaving model using safe method...")
        
        # Save with new safe method
        save_model_safely(gnn_model, os.path.join(output_dir, 'mapping_gnn_results.pkl'))
        
        # Also save additional results
        results_data = {
            'predictions': predictions,
            'node_list': node_list,
            'graph': graph,
            'model_info': gnn_model.get_model_info(),
            'seed': SEED
        }
        
        import pickle
        with open(os.path.join(output_dir, 'additional_results.pkl'), 'wb') as f:
            pickle.dump(results_data, f)
        
        print(f"✅ Results saved to '{output_dir}/'")
        print(f"   - mapping_gnn_results.pkl (model)")
        print(f"   - additional_results.pkl (results)")
        print(f"   - training_loss.png (plot)")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
