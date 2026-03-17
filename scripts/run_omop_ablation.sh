#!/bin/bash
# =============================================================================
# OMOP Ablation: Base vs UQ-Triggered Reflection
# =============================================================================
# Runs Map-Reflect (Llama-3.3-70B) on all three OMOP datasets (CMS, MIMIC,
# Synthea) in two conditions: base (no reflection) and UQ-triggered reflection
# with semantic entropy. Produces Table 6 in the paper.
#
# Results are saved to results/omop_ablation_<timestamp>/ with per-dataset
# JSON files and logs. A summary table is printed at the end.
#
# Usage:
#   ./scripts/run_omop_ablation.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BASE_DIR"

if [[ -f "${BASE_DIR}/.venv/bin/activate" ]]; then
    source "${BASE_DIR}/.venv/bin/activate"
    echo "Activated venv: $(which python)"
fi

# Create results directory
RESULTS_DIR="results/omop_ablation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
export RESULTS_DIR

echo "Results will be saved to: $RESULTS_DIR"
echo "Starting OMOP ablation experiments..."

# Common args
MODEL="watsonx/meta-llama/llama-3-3-70b-instruct"
DATA_DIR="data"

# Function to run experiment
run_experiment() {
    local dataset=$1
    local condition=$2
    local lookup_file=$3
    local extra_args=$4

    echo ""
    echo "=========================================="
    echo "Running: $dataset - $condition"
    echo "Time: $(date)"
    echo "=========================================="

    python -m map_reflect.map_reflect_runner \
        --datasets "$dataset" \
        --models "$MODEL" \
        --property-lookup-file "$lookup_file" \
        --data-dir "$DATA_DIR" \
        --output-dir "$RESULTS_DIR" \
        --batch-size 20 \
        $extra_args \
        2>&1 | tee "$RESULTS_DIR/${dataset}_${condition}.log"

    # Copy result file with condition name
    result_file=$(ls -t map_reflect/results/mapreflect_${dataset,,}_*.json 2>/dev/null | head -1)
    if [ -n "$result_file" ]; then
        cp "$result_file" "$RESULTS_DIR/${dataset}_${condition}.json"
        echo "Result copied to: $RESULTS_DIR/${dataset}_${condition}.json"
    fi

    echo "Completed: $dataset - $condition at $(date)"
}

# OMOP-CMS (38 samples)
echo "=== Starting OMOP-CMS experiments ==="
run_experiment "OMOP_CMS" "base" "data/omop_cms_api_lookup.json" ""
run_experiment "OMOP_CMS" "uq_reflect" "data/omop_cms_api_lookup.json" "--use-uncertainty --use-uq-reflection --uq-method semantic_entropy"

# OMOP-MIMIC (46 samples)
echo "=== Starting OMOP-MIMIC experiments ==="
run_experiment "OMOP_MIMIC" "base" "data/omop_mimic_api_lookup.json" ""
run_experiment "OMOP_MIMIC" "uq_reflect" "data/omop_mimic_api_lookup.json" "--use-uncertainty --use-uq-reflection --uq-method semantic_entropy"

# OMOP-Synthea (12 samples)
echo "=== Starting OMOP-Synthea experiments ==="
run_experiment "OMOP_Synthea" "base" "data/omop_synthea_api_lookup.json" ""
run_experiment "OMOP_Synthea" "uq_reflect" "data/omop_synthea_api_lookup.json" "--use-uncertainty --use-uq-reflection --uq-method semantic_entropy"

echo ""
echo "=========================================="
echo "All experiments completed at $(date)!"
echo "Results saved to: $RESULTS_DIR"
echo "=========================================="

# Generate summary table
echo ""
echo "Generating summary..."
python3 << 'SUMMARY_SCRIPT'
import json
import os
from pathlib import Path

results_dir = Path(os.environ.get('RESULTS_DIR', 'results/omop_ablation'))

datasets = ['OMOP_CMS', 'OMOP_MIMIC', 'OMOP_Synthea']
conditions = ['base', 'uq_reflect']

print("\n" + "="*90)
print("OMOP ABLATION RESULTS SUMMARY")
print("="*90)

results = {}
for dataset in datasets:
    results[dataset] = {}
    for condition in conditions:
        result_file = results_dir / f"{dataset}_{condition}.json"
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)
            metrics = data.get('metrics', {})
            total_fields = metrics.get('total_fields', 0)
            correct = metrics.get('correct_predictions', 0)

            # Compute recall and precision
            recall = (correct / total_fields * 100) if total_fields > 0 else 0

            # For F1, we need both precision and recall
            # In this evaluation, precision = recall since we predict exactly one target per source field
            precision = recall
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

            results[dataset][condition] = {
                'recall': recall,
                'f1': f1,
                'correct': correct,
                'total': total_fields,
                'samples': metrics.get('total_samples', 0)
            }
        else:
            print(f"Warning: No results found for {dataset} {condition}")

# Print table header
print(f"\n{'Dataset':<15} | {'Base Recall':<12} | {'UQ Recall':<12} | {'Δ Recall':<10} | {'Base F1':<10} | {'UQ F1':<10} | {'Δ F1':<8}")
print("-" * 95)

for dataset in datasets:
    if dataset in results and 'base' in results[dataset] and 'uq_reflect' in results[dataset]:
        base = results[dataset]['base']
        uq = results[dataset]['uq_reflect']
        delta_recall = uq['recall'] - base['recall']
        delta_f1 = uq['f1'] - base['f1']
        print(f"{dataset:<15} | {base['recall']:<12.1f} | {uq['recall']:<12.1f} | {delta_recall:+.1f}pp     | {base['f1']:<10.1f} | {uq['f1']:<10.1f} | {delta_f1:+.1f}pp")
        print(f"{'  (fields)':<15} | {base['correct']}/{base['total']:<8} | {uq['correct']}/{uq['total']:<8} |")

print("\n" + "="*90)
SUMMARY_SCRIPT
