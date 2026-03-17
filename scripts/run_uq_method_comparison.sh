#!/bin/bash
# =============================================================================
# UQ Method Comparison (ITOpsGraph, Llama-3.3-70B)
# =============================================================================
# Compares three uncertainty quantification methods on ITOpsGraph (521 samples):
#   1. Sampling (consistency-based)
#   2. P(True) (self-verification)
#   3. Semantic Entropy
#
# Each method runs sequentially. For parallel execution, use
# run_uq_method_comparison_parallel.sh instead.
#
# Usage:
#   ./scripts/run_uq_method_comparison.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BASE_DIR"

MODEL="watsonx/meta-llama/llama-3-3-70b-instruct"
DATASET="ITOpsGraph"
LOOKUP="data/itopsgraph_api_lookup.json"
DATA_DIR="data"
OUTPUT_BASE="results/uq_method_comparison"
N_SAMPLES=5  # Number of samples for UQ estimation

mkdir -p "$OUTPUT_BASE"

echo "=============================================="
echo "UQ Method Comparison Study"
echo "Dataset: $DATASET (521 samples)"
echo "Model: $MODEL"
echo "N_uncertainty_samples: $N_SAMPLES"
echo "=============================================="
echo ""

# Base command
BASE_CMD="python -m map_reflect.map_reflect_runner \
    --datasets $DATASET \
    --models $MODEL \
    --property-lookup-file $LOOKUP \
    --data-dir $DATA_DIR \
    --neighbor-k 0 \
    --use-uncertainty \
    --n-uncertainty-samples $N_SAMPLES"

# 1. Sampling method (consistency-based)
echo -e "\n[1/3] Running: UQ Method = sampling"
$BASE_CMD \
    --uq-method sampling \
    --output-dir "$OUTPUT_BASE/sampling" \
    2>&1 | tee "$OUTPUT_BASE/sampling.log"

# 2. P(True) method (self-verification)
echo -e "\n[2/3] Running: UQ Method = p_true"
$BASE_CMD \
    --uq-method p_true \
    --output-dir "$OUTPUT_BASE/p_true" \
    2>&1 | tee "$OUTPUT_BASE/p_true.log"

# 3. Semantic Entropy method
echo -e "\n[3/3] Running: UQ Method = semantic_entropy"
$BASE_CMD \
    --uq-method semantic_entropy \
    --output-dir "$OUTPUT_BASE/semantic_entropy" \
    2>&1 | tee "$OUTPUT_BASE/semantic_entropy.log"

echo -e "\n=============================================="
echo "UQ Method Comparison complete!"
echo "=============================================="

# Print summary
echo -e "\nResults summary:"
echo "----------------------------------------"
printf "%-20s %-10s %-15s\n" "Method" "Accuracy" "Mean_UQ"
echo "----------------------------------------"
for method in sampling p_true semantic_entropy; do
    file="$OUTPUT_BASE/$method/mapreflect_itopsgraph_meta-llama_llama-3-3-70b-instruct.json"
    if [ -f "$file" ]; then
        result=$(python3 -c "
import json
d=json.load(open('$file'))
acc = d['metrics']['accuracy']*100
uq = d['metrics'].get('mean_uncertainty', 0)
print(f'{acc:.1f}%\t{uq:.4f}')
" 2>/dev/null)
        printf "%-20s %s\n" "$method" "$result"
    else
        printf "%-20s %s\n" "$method" "Not completed"
    fi
done
