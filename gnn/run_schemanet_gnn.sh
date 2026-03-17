#!/bin/bash
# Run GNN on all SchemaNet datasets

set -e

cd /Users/balaji/code/graphql-schema-mapping/gnn
source .venv/bin/activate

DATASETS=(
    "schemanet_bank1_bank2"
    "schemanet_imdb_sakila"
    "schemanet_cms_omop"
    "schemanet_mimic_iii_omop"
    "schemanet_synthea_omop"
    "schemanet_cprd_gold_omop"
    "schemanet_cprd_aurum_omop"
)

echo "=== GNN SchemaNet Experiments ==="
echo "Start: $(date)"
echo ""

for dataset in "${DATASETS[@]}"; do
    echo "========================================"
    echo "Dataset: $dataset"
    echo "========================================"

    # Step 1: Build graph
    echo "Building graph..."
    python schemanet_populate_nx.py \
        --dataset "$dataset" \
        --schemanet-dir ../data/schemanet \
        --output-dir ../data \
        2>&1 | tail -10

    # Step 2: Train GNN
    echo "Training GNN..."
    python train_gnn.py \
        --dataset "$dataset" \
        --output-dir ../data \
        --seed 42 \
        --epochs 100 \
        2>&1 | grep -E "(AUC|AP|Epoch|completed|Error)"

    echo ""
done

echo "=== All GNN experiments completed ==="
echo "End: $(date)"
