#!/bin/bash
# Train GNN on ITGraph dataset
# Converts ITGraph mappings to the format expected by populate_nx.py,
# then trains the GNN model.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BASE_DIR"

if [[ -f "${BASE_DIR}/.venv/bin/activate" ]]; then
    source "${BASE_DIR}/.venv/bin/activate"
fi

DATASET="itgraph"
API_LOOKUP="data/itom_hashicorp_api_lookup_v2.json"
MAPPINGS_DIR="data/itgraph_mappings_output"
OUTPUT_DIR="data"

echo "=== Step 1: Build ITGraph graph (many-to-many + bipartite) ==="
cd gnn
python3 populate_nx_itgraph.py \
    --dataset "$DATASET" \
    --dataset-file "../data/itgraph_dataset.jsonl" \
    --api-lookup-file "../$API_LOOKUP" \
    --output-dir "../$OUTPUT_DIR"

echo ""
echo "=== Step 2: Train GNN model ==="
python3 train_gnn.py \
    --dataset "$DATASET" \
    --output-dir "../$OUTPUT_DIR"

echo ""
echo "=== Step 3: Verify ==="
ls -la "../$OUTPUT_DIR/${DATASET}_graph_data.pkl" 2>/dev/null
ls -la "../$OUTPUT_DIR/gnn_results/${DATASET}_mapping_gnn_results.pkl" 2>/dev/null

echo ""
echo "=== Done! ==="
echo "GNN model trained on ITGraph (many-to-many types, bipartite fields)."
echo "Run evaluation with:"
echo "  cd gnn && python3 gnn_runner.py --datasets ITGraph_HC --gnn-model-path ../data/gnn_results/${DATASET}_mapping_gnn_results.pkl --graph-data-path ../data/${DATASET}_graph_data.pkl --api-lookup-file ../data/itom_hashicorp_api_lookup_v2.json --data-dir ../data"
