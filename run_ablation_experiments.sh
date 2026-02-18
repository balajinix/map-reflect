#!/bin/bash
# Ablation experiments for Map-Reflect on ITOpsGraph
# Runs 8 variants: 2^3 factorial design (neighborhood, reflection, UQ)

set -e

# Configuration
DATASET="ITOpsGraph"
MODEL="watsonx/meta-llama/llama-3-3-70b-instruct"
LOOKUP_FILE="data/itopsgraph_api_lookup.json"
DATA_DIR="data"
OUTPUT_DIR="results/ablation"
MAX_SAMPLES=""  # Empty for full dataset, or set to a number for testing

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Base command
BASE_CMD="python -m map_reflect.map_reflect_runner \
    --datasets $DATASET \
    --models $MODEL \
    --property-lookup-file $LOOKUP_FILE \
    --data-dir $DATA_DIR"

if [ -n "$MAX_SAMPLES" ]; then
    BASE_CMD="$BASE_CMD --max-samples $MAX_SAMPLES"
fi

echo "=========================================="
echo "Map-Reflect Ablation Experiments"
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "=========================================="

# Experiment 1: Base (no neighborhood, no reflection, no UQ)
echo ""
echo "[1/8] Running: Base (no features)"
$BASE_CMD \
    --neighbor-k 0 \
    --output-dir "$OUTPUT_DIR/base" \
    2>&1 | tee "$OUTPUT_DIR/base.log"

# Experiment 2: +Neighborhood only
echo ""
echo "[2/8] Running: +Neighborhood"
$BASE_CMD \
    --neighbor-k 5 \
    --output-dir "$OUTPUT_DIR/neighborhood" \
    2>&1 | tee "$OUTPUT_DIR/neighborhood.log"

# Experiment 3: +Reflection only (includes error analysis)
echo ""
echo "[3/8] Running: +Reflection (with error analysis)"
$BASE_CMD \
    --neighbor-k 0 \
    --use-reflection \
    --output-dir "$OUTPUT_DIR/reflection" \
    2>&1 | tee "$OUTPUT_DIR/reflection.log"

# Experiment 4: +UQ only
echo ""
echo "[4/8] Running: +UQ"
$BASE_CMD \
    --neighbor-k 0 \
    --use-uncertainty \
    --output-dir "$OUTPUT_DIR/uq" \
    2>&1 | tee "$OUTPUT_DIR/uq.log"

# Experiment 5: +Neighborhood +Reflection
echo ""
echo "[5/8] Running: +Neighborhood +Reflection"
$BASE_CMD \
    --neighbor-k 5 \
    --use-reflection \
    --output-dir "$OUTPUT_DIR/neighborhood_reflection" \
    2>&1 | tee "$OUTPUT_DIR/neighborhood_reflection.log"

# Experiment 6: +Neighborhood +UQ
echo ""
echo "[6/8] Running: +Neighborhood +UQ"
$BASE_CMD \
    --neighbor-k 5 \
    --use-uncertainty \
    --output-dir "$OUTPUT_DIR/neighborhood_uq" \
    2>&1 | tee "$OUTPUT_DIR/neighborhood_uq.log"

# Experiment 7: +Reflection +UQ (no neighborhood)
echo ""
echo "[7/8] Running: +Reflection +UQ"
$BASE_CMD \
    --neighbor-k 0 \
    --use-reflection \
    --use-uncertainty \
    --output-dir "$OUTPUT_DIR/reflection_uq" \
    2>&1 | tee "$OUTPUT_DIR/reflection_uq.log"

# Experiment 8: Full (all features)
echo ""
echo "[8/8] Running: Full (Neighborhood + Reflection + UQ)"
$BASE_CMD \
    --neighbor-k 5 \
    --use-reflection \
    --use-uncertainty \
    --output-dir "$OUTPUT_DIR/full" \
    2>&1 | tee "$OUTPUT_DIR/full.log"

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="

# Summary
echo ""
echo "Experiment Summary:"
echo "-------------------"
for exp in base neighborhood reflection uq neighborhood_reflection neighborhood_uq reflection_uq full; do
    if [ -f "$OUTPUT_DIR/$exp.log" ]; then
        acc=$(grep -o "'accuracy': [0-9.]*" "$OUTPUT_DIR/$exp.log" | tail -1 || echo "N/A")
        echo "$exp: $acc"
    fi
done
