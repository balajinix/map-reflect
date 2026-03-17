#!/bin/bash
# =============================================================================
# Reflection Ablation: Compare Reflection Strategies on OMOP
# =============================================================================
# Runs Map-Reflect (Llama-3.3-70B) on all three OMOP datasets with five
# reflection conditions: base (none), self-reflection, error-triggered,
# neighbourhood-triggered, and UQ-triggered (sampling).
#
# Produces a summary table comparing accuracy across conditions.
#
# Usage:
#   ./scripts/run_reflection_ablation.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BASE_DIR"

if [[ -f "${BASE_DIR}/.venv/bin/activate" ]]; then
    source "${BASE_DIR}/.venv/bin/activate"
fi

MODEL="watsonx/meta-llama/llama-3-3-70b-instruct"
DATA_DIR="data"

# Create results directory
RESULTS_DIR="results/reflection_ablation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "=== Reflection Ablation Experiments ==="
echo "Results: $RESULTS_DIR"
echo "Model: $MODEL"
echo ""

run_experiment() {
    local dataset=$1
    local condition=$2
    local lookup_file=$3
    local extra_args=$4

    echo "----------------------------------------"
    echo "Running: $dataset - $condition"
    echo "Time: $(date)"
    echo "----------------------------------------"

    python -m map_reflect.map_reflect_runner \
        --datasets "$dataset" \
        --models "$MODEL" \
        --property-lookup-file "$lookup_file" \
        --data-dir "$DATA_DIR" \
        --batch-size 20 \
        $extra_args \
        2>&1 | tee "$RESULTS_DIR/${dataset}_${condition}.log" | tail -5

    # Copy result file
    result_file=$(ls -t map_reflect/results/mapreflect_${dataset,,}_*.json 2>/dev/null | head -1)
    if [ -n "$result_file" ]; then
        cp "$result_file" "$RESULTS_DIR/${dataset}_${condition}.json"
    fi
    echo ""
}

# OMOP_CMS experiments
echo "=== OMOP_CMS Ablation ==="
run_experiment "OMOP_CMS" "base" "data/omop_cms_api_lookup.json" ""
run_experiment "OMOP_CMS" "self_reflect" "data/omop_cms_api_lookup.json" "--use-self-reflection"
run_experiment "OMOP_CMS" "error_reflect" "data/omop_cms_api_lookup.json" "--use-error-reflection"
run_experiment "OMOP_CMS" "neighborhood_reflect" "data/omop_cms_api_lookup.json" "--use-neighborhood-reflection"
run_experiment "OMOP_CMS" "uq_sampling_reflect" "data/omop_cms_api_lookup.json" "--use-uq-reflection --uq-method sampling"

# OMOP_MIMIC experiments
echo "=== OMOP_MIMIC Ablation ==="
run_experiment "OMOP_MIMIC" "base" "data/omop_mimic_api_lookup.json" ""
run_experiment "OMOP_MIMIC" "self_reflect" "data/omop_mimic_api_lookup.json" "--use-self-reflection"
run_experiment "OMOP_MIMIC" "error_reflect" "data/omop_mimic_api_lookup.json" "--use-error-reflection"
run_experiment "OMOP_MIMIC" "neighborhood_reflect" "data/omop_mimic_api_lookup.json" "--use-neighborhood-reflection"
run_experiment "OMOP_MIMIC" "uq_sampling_reflect" "data/omop_mimic_api_lookup.json" "--use-uq-reflection --uq-method sampling"

# OMOP_Synthea experiments
echo "=== OMOP_Synthea Ablation ==="
run_experiment "OMOP_Synthea" "base" "data/omop_synthea_api_lookup.json" ""
run_experiment "OMOP_Synthea" "self_reflect" "data/omop_synthea_api_lookup.json" "--use-self-reflection"
run_experiment "OMOP_Synthea" "error_reflect" "data/omop_synthea_api_lookup.json" "--use-error-reflection"
run_experiment "OMOP_Synthea" "neighborhood_reflect" "data/omop_synthea_api_lookup.json" "--use-neighborhood-reflection"
run_experiment "OMOP_Synthea" "uq_sampling_reflect" "data/omop_synthea_api_lookup.json" "--use-uq-reflection --uq-method sampling"

echo "=== All experiments completed ==="
echo "Results saved to: $RESULTS_DIR"

# Generate summary
python3 << SUMMARY
import json
import os
from pathlib import Path

results_dir = Path("$RESULTS_DIR")
datasets = ['OMOP_CMS', 'OMOP_MIMIC', 'OMOP_Synthea']
conditions = ['base', 'self_reflect', 'error_reflect', 'neighborhood_reflect', 'uq_sampling_reflect']

print("\n" + "="*80)
print("REFLECTION ABLATION SUMMARY")
print("="*80)
print(f"\n{'Dataset':<15} | {'Base':>8} | {'Self':>8} | {'Error':>8} | {'Neigh':>8} | {'UQ':>8}")
print("-"*80)

for dataset in datasets:
    row = f"{dataset:<15}"
    for cond in conditions:
        result_file = results_dir / f"{dataset}_{cond}.json"
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)
            m = data.get('metrics', {})
            acc = m.get('accuracy', 0) * 100
            row += f" | {acc:>7.1f}%"
        else:
            row += f" | {'--':>8}"
    print(row)

print("="*80)
SUMMARY
