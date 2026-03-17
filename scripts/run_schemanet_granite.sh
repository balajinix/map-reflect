#!/bin/bash
# =============================================================================
# Map-Reflect SchemaNet Experiments (Granite-3.3-8B)
# =============================================================================
# Runs Map-Reflect on all 7 SchemaNet schema pairs with the smaller
# Granite-3.3-8B backbone. Companion to run_schemanet_experiments.sh
# which uses Llama-3.3-70B.
#
# Usage:
#   ./scripts/run_schemanet_granite.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BASE_DIR"

if [[ -f "${BASE_DIR}/.venv/bin/activate" ]]; then
    source "${BASE_DIR}/.venv/bin/activate"
fi

MODEL="watsonx/ibm/granite-3-3-8b-instruct"
DATA_DIR="data/schemanet"

DATASETS=(
    "schemanet_bank1_bank2"
    "schemanet_imdb_sakila"
    "schemanet_cms_omop"
    "schemanet_mimic_iii_omop"
    "schemanet_synthea_omop"
    "schemanet_cprd_gold_omop"
    "schemanet_cprd_aurum_omop"
)

echo "=== SchemaNet Experiments (Granite-3.3-8B) ==="
echo "Model: $MODEL"
echo "Start: $(date)"
echo ""

for dataset in "${DATASETS[@]}"; do
    echo "----------------------------------------"
    echo "Running: $dataset"
    echo "Time: $(date)"
    echo "----------------------------------------"

    python -m map_reflect.map_reflect_runner \
        --datasets "$dataset" \
        --models "$MODEL" \
        --property-lookup-file "$DATA_DIR/${dataset}_api_lookup.json" \
        --data-dir "$DATA_DIR" \
        --batch-size 20 \
        --temperature 0 \
        2>&1 | tail -5

    echo ""
done

echo "=== All experiments completed ==="
echo "End: $(date)"
