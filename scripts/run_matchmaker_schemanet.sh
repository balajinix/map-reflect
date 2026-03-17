#!/bin/bash
# =============================================================================
# MatchMaker SchemaNet Experiments (Llama-3.3-70B)
# =============================================================================
# Runs MatchMaker baseline on all 7 SchemaNet schema pairs using the
# match_maker_runner module with Llama-3.3-70B backbone.
#
# Datasets: bank1_bank2, imdb_sakila, cms_omop, mimic_iii_omop,
#           synthea_omop, cprd_gold_omop, cprd_aurum_omop
#
# Usage:
#   ./scripts/run_matchmaker_schemanet.sh
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

echo "=== MatchMaker SchemaNet Experiments (Llama-3.3-70B) ==="
echo "Model: $MODEL"
echo "Start: $(date)"
echo ""

for dataset in "${DATASETS[@]}"; do
    echo "----------------------------------------"
    echo "Running: $dataset"
    echo "Time: $(date)"
    echo "----------------------------------------"

    python -m match_maker.match_maker_runner \
        --datasets "$dataset" \
        --models "$MODEL" \
        --property-lookup-file "$DATA_DIR/${dataset}_api_lookup.json" \
        --data-dir "$DATA_DIR" \
        --top-k 5 \
        --abstention-mode lenient \
        2>&1 | tail -20

    echo ""
done

echo "=== All experiments completed ==="
echo "End: $(date)"
