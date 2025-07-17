# GNN Baseline for API Mapping Evaluation

This project trains a Graph Neural Network (GNN) to evaluate the correctness of API field mappings using a graph constructed from mapping files.

---

## 📁 Structure
```
gnn_baseline/
├── gnn_results/      # Stores outputs like loss plots, predictions
├── api_lookup.json   # Field metadata (used for node features)
├── graph_data.pkl    # Serialized NetworkX graph
├── populate_nx.py    # Builds graph from mappings
└── train_gnn.py      # Trains and evaluates GNN
```
---

## 🔄 Flow

1. **Graph Creation** (`populate_nx.py`)
   - Parses mapping files and builds a NetworkX graph.
   - Nodes: product, type, field
   - Edges: hierarchy + field-to-field `maps_to`
   - Saves `graph_data.pkl`.

2. **GNN Training** (`train_gnn.py`)
   - Loads graph, applies train/val/test split.
   - Trains a GNN (GCN, GraphSAGE, or GAT).
   - Evaluates AUC and Average Precision.
   - Saves outputs to `gnn_results/`.

---

## 📦 Outputs

- `training_loss.png`: Loss curve  
- `mapping_gnn_results.pkl`: GNN predictions and metadata

