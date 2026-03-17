#!/bin/bash
# =============================================================================
# Mistral Map-Reflect SchemaNet Experiments (requires Python >= 3.10)
# =============================================================================
# Runs Map-Reflect on all 7 SchemaNet pairs with Mistral backbone.
# Requires ibm_watsonx_ai >= 1.0 which needs Python >= 3.10.
# Run this on local machine, not the remote (which has Python 3.9).
#
# Usage:
#   ./scripts/run_mistral_mapreflect_schemanet.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BASE_DIR"

if [[ -f "${BASE_DIR}/.venv/bin/activate" ]]; then
    source "${BASE_DIR}/.venv/bin/activate"
fi

if [[ -f "${BASE_DIR}/.env" ]]; then
    set -a && source "${BASE_DIR}/.env" && set +a
fi

MODEL="watsonx/mistralai/mistral-medium-2505"
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

echo "=== Map-Reflect SchemaNet (Mistral) ==="
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
        --checkpoint-interval 10 \
        --resume \
        2>&1 | tail -10

    echo ""
done

echo "=== All experiments completed ==="
echo "End: $(date)"
