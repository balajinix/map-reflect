#!/bin/bash
# Signal-based ablation study - NEW experiments only
# (Existing results already copied from previous runs)

set -e

MODEL="watsonx/meta-llama/llama-3-3-70b-instruct"
DATASET="ITOpsGraph"
LOOKUP="data/itopsgraph_api_lookup.json"
DATA_DIR="data"
OUTPUT_BASE="results/signal_ablation"

echo "=============================================="
echo "Signal-Based Ablation Study (New Experiments)"
echo "=============================================="
echo ""
echo "Existing results (copied from previous runs):"
echo "  base:         88.5%"
echo "  error_refl:   85.1%"
echo "  error_uq:     87.0%"
echo "  error_nbr:    85.7%"
echo "  induced_refl: 86.7%"
echo ""

# 1. Self-reflection only (no external signals) - NEW
echo -e "\n[1/4] Running: self_reflection (always reflect)"
python -m map_reflect.map_reflect_runner \
    --datasets "$DATASET" \
    --models "$MODEL" \
    --property-lookup-file "$LOOKUP" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_BASE/self_refl" \
    --neighbor-k 0 \
    --use-self-reflection

# 2. UQ-triggered reflection only - NEW
echo -e "\n[2/4] Running: uq_reflection (UQ threshold only)"
python -m map_reflect.map_reflect_runner \
    --datasets "$DATASET" \
    --models "$MODEL" \
    --property-lookup-file "$LOOKUP" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_BASE/uq_refl" \
    --neighbor-k 0 \
    --use-uncertainty \
    --use-uq-reflection

# 3. Neighborhood-triggered reflection only - NEW
echo -e "\n[3/4] Running: neighborhood_reflection (score gap trigger)"
python -m map_reflect.map_reflect_runner \
    --datasets "$DATASET" \
    --models "$MODEL" \
    --property-lookup-file "$LOOKUP" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_BASE/nbr_refl" \
    --neighbor-k 5 \
    --use-neighborhood-reflection

# 4. UQ + Neighborhood - NEW
echo -e "\n[4/4] Running: uq + neighborhood"
python -m map_reflect.map_reflect_runner \
    --datasets "$DATASET" \
    --models "$MODEL" \
    --property-lookup-file "$LOOKUP" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_BASE/uq_nbr" \
    --neighbor-k 5 \
    --use-uncertainty \
    --use-uq-reflection \
    --use-neighborhood-reflection

echo -e "\n=============================================="
echo "New ablations complete! Results in $OUTPUT_BASE"
echo "=============================================="

# Print summary
echo -e "\nResults summary:"
for dir in "$OUTPUT_BASE"/*/; do
    name=$(basename "$dir")
    file="$dir/mapreflect_itopsgraph_meta-llama_llama-3-3-70b-instruct.json"
    if [ -f "$file" ]; then
        acc=$(python3 -c "import json; d=json.load(open('$file')); print(f\"{d['metrics']['accuracy']*100:.1f}%\")" 2>/dev/null)
        printf "%-20s %s\n" "$name:" "$acc"
    fi
done
