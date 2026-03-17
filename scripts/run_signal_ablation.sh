#!/bin/bash
# =============================================================================
# Signal Ablation Study (ITOpsGraph, Llama-3.3-70B)
# =============================================================================
# Tests each reflection signal independently and in all pairwise/triple
# combinations on ITOpsGraph (521 samples). Produces Table 5 in the paper.
#
# 9 configurations:
#   1. Base (no reflection)
#   2. Self-reflection (unconditional)
#   3. Error-triggered only
#   4. UQ-triggered only
#   5. Neighbourhood-triggered only
#   6. Error + UQ
#   7. Error + Neighbourhood
#   8. UQ + Neighbourhood
#   9. All three signals (induced reflection)
#
# Usage:
#   ./scripts/run_signal_ablation.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BASE_DIR"

MODEL="watsonx/meta-llama/llama-3-3-70b-instruct"
DATASET="ITOpsGraph"
LOOKUP="data/itopsgraph_api_lookup.json"
DATA_DIR="data"
OUTPUT_BASE="results/signal_ablation"

mkdir -p "$OUTPUT_BASE"

echo "=============================================="
echo "Signal-Based Ablation Study"
echo "=============================================="

# 1. Base (no reflection)
echo -e "\n[1/9] Running: base (no reflection)"
python -m map_reflect.map_reflect_runner \
    --datasets "$DATASET" \
    --models "$MODEL" \
    --property-lookup-file "$LOOKUP" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_BASE/base" \
    --neighbor-k 0

# 2. Self-reflection only (no external signals)
echo -e "\n[2/9] Running: self_reflection"
python -m map_reflect.map_reflect_runner \
    --datasets "$DATASET" \
    --models "$MODEL" \
    --property-lookup-file "$LOOKUP" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_BASE/self_refl" \
    --neighbor-k 0 \
    --use-self-reflection

# 3. Error-triggered reflection only
echo -e "\n[3/9] Running: error_reflection"
python -m map_reflect.map_reflect_runner \
    --datasets "$DATASET" \
    --models "$MODEL" \
    --property-lookup-file "$LOOKUP" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_BASE/error_refl" \
    --neighbor-k 0 \
    --use-error-reflection

# 4. UQ-triggered reflection only
echo -e "\n[4/9] Running: uq_reflection"
python -m map_reflect.map_reflect_runner \
    --datasets "$DATASET" \
    --models "$MODEL" \
    --property-lookup-file "$LOOKUP" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_BASE/uq_refl" \
    --neighbor-k 0 \
    --use-uq-reflection

# 5. Neighborhood-triggered reflection only
echo -e "\n[5/9] Running: neighborhood_reflection"
python -m map_reflect.map_reflect_runner \
    --datasets "$DATASET" \
    --models "$MODEL" \
    --property-lookup-file "$LOOKUP" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_BASE/nbr_refl" \
    --neighbor-k 5 \
    --use-neighborhood-reflection

# 6. Error + UQ
echo -e "\n[6/9] Running: error + uq"
python -m map_reflect.map_reflect_runner \
    --datasets "$DATASET" \
    --models "$MODEL" \
    --property-lookup-file "$LOOKUP" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_BASE/error_uq" \
    --neighbor-k 0 \
    --use-error-reflection \
    --use-uq-reflection

# 7. Error + Neighborhood
echo -e "\n[7/9] Running: error + neighborhood"
python -m map_reflect.map_reflect_runner \
    --datasets "$DATASET" \
    --models "$MODEL" \
    --property-lookup-file "$LOOKUP" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_BASE/error_nbr" \
    --neighbor-k 5 \
    --use-error-reflection \
    --use-neighborhood-reflection

# 8. UQ + Neighborhood
echo -e "\n[8/9] Running: uq + neighborhood"
python -m map_reflect.map_reflect_runner \
    --datasets "$DATASET" \
    --models "$MODEL" \
    --property-lookup-file "$LOOKUP" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_BASE/uq_nbr" \
    --neighbor-k 5 \
    --use-uq-reflection \
    --use-neighborhood-reflection

# 9. Full induced reflection (all 3 signals)
echo -e "\n[9/9] Running: induced_reflection (error + uq + neighborhood)"
python -m map_reflect.map_reflect_runner \
    --datasets "$DATASET" \
    --models "$MODEL" \
    --property-lookup-file "$LOOKUP" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_BASE/induced_refl" \
    --neighbor-k 5 \
    --use-error-reflection \
    --use-uq-reflection \
    --use-neighborhood-reflection

echo -e "\n=============================================="
echo "Ablation complete! Results in $OUTPUT_BASE"
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
