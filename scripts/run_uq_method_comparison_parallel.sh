#!/bin/bash
# =============================================================================
# UQ Method Comparison - Parallel Version (ITOpsGraph, Llama-3.3-70B)
# =============================================================================
# Same as run_uq_method_comparison.sh but launches all three UQ methods
# (sampling, p_true, semantic_entropy) in parallel as background processes.
#
# By default, the script launches all jobs and exits. Pass --wait to block
# until all jobs finish and print a summary table.
#
# Usage:
#   ./scripts/run_uq_method_comparison_parallel.sh           # fire-and-forget
#   ./scripts/run_uq_method_comparison_parallel.sh --wait     # wait for results
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
echo "UQ Method Comparison Study (PARALLEL)"
echo "Dataset: $DATASET (521 samples)"
echo "Model: $MODEL"
echo "N_uncertainty_samples: $N_SAMPLES"
echo "=============================================="
echo ""
echo "Starting all 3 methods in parallel..."
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

# 1. Sampling method (consistency-based) - BACKGROUND
echo "[1/3] Starting: UQ Method = sampling (background)"
$BASE_CMD \
    --uq-method sampling \
    --output-dir "$OUTPUT_BASE/sampling" \
    > "$OUTPUT_BASE/sampling.log" 2>&1 &
PID_SAMPLING=$!
echo "  PID: $PID_SAMPLING"

# 2. P(True) method (self-verification) - BACKGROUND
echo "[2/3] Starting: UQ Method = p_true (background)"
$BASE_CMD \
    --uq-method p_true \
    --output-dir "$OUTPUT_BASE/p_true" \
    > "$OUTPUT_BASE/p_true.log" 2>&1 &
PID_PTRUE=$!
echo "  PID: $PID_PTRUE"

# 3. Semantic Entropy method - BACKGROUND
echo "[3/3] Starting: UQ Method = semantic_entropy (background)"
$BASE_CMD \
    --uq-method semantic_entropy \
    --output-dir "$OUTPUT_BASE/semantic_entropy" \
    > "$OUTPUT_BASE/semantic_entropy.log" 2>&1 &
PID_SEMANTIC=$!
echo "  PID: $PID_SEMANTIC"

echo ""
echo "=============================================="
echo "All 3 experiments started in background!"
echo "=============================================="
echo ""
echo "PIDs:"
echo "  sampling:         $PID_SAMPLING"
echo "  p_true:           $PID_PTRUE"
echo "  semantic_entropy: $PID_SEMANTIC"
echo ""
echo "Monitor with:"
echo "  tail -f $OUTPUT_BASE/sampling.log"
echo "  tail -f $OUTPUT_BASE/p_true.log"
echo "  tail -f $OUTPUT_BASE/semantic_entropy.log"
echo ""
echo "Check status with:"
echo "  ps aux | grep map_reflect"
echo ""
echo "Wait for all to complete:"
echo "  wait $PID_SAMPLING $PID_PTRUE $PID_SEMANTIC"
echo ""

# Save PIDs to file for later reference
echo "$PID_SAMPLING" > "$OUTPUT_BASE/pid_sampling.txt"
echo "$PID_PTRUE" > "$OUTPUT_BASE/pid_p_true.txt"
echo "$PID_SEMANTIC" > "$OUTPUT_BASE/pid_semantic_entropy.txt"

# Optional: wait for all and print summary
if [ "$1" == "--wait" ]; then
    echo "Waiting for all experiments to complete..."
    wait $PID_SAMPLING $PID_PTRUE $PID_SEMANTIC

    echo -e "\n=============================================="
    echo "All experiments complete!"
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
            printf "%-20s %s\n" "$method" "Failed"
        fi
    done
fi
