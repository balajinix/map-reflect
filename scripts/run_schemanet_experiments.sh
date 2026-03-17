#!/bin/bash
# =============================================================================
# Map-Reflect SchemaNet Experiments (Llama-3.3-70B)
# =============================================================================
# Runs Map-Reflect on all 7 SchemaNet schema pairs with Llama-3.3-70B.
# Each pair uses its own api_lookup file from data/schemanet/.
#
# Usage:
#   ./scripts/run_schemanet_experiments.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BASE_DIR"

if [[ -f "${BASE_DIR}/.venv/bin/activate" ]]; then
    source "${BASE_DIR}/.venv/bin/activate"
fi

MODEL="watsonx/meta-llama/llama-3-3-70b-instruct"
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

echo "=== SchemaNet Experiments ==="
echo "Model: $MODEL"
echo ""

for dataset in "${DATASETS[@]}"; do
    echo "----------------------------------------"
    echo "Running: $dataset"
    echo "----------------------------------------"

    python -m map_reflect.map_reflect_runner \
        --datasets "$dataset" \
        --models "$MODEL" \
        --property-lookup-file "$DATA_DIR/${dataset}_api_lookup.json" \
        --data-dir "$DATA_DIR" \
        --batch-size 20 \
        2>&1 | tail -5

    echo ""
done

echo "=== All experiments completed ==="
