#!/bin/bash
# =============================================================================
# MatchMaker SchemaNet Experiments via run_schemanet.py (Llama-3.3-70B)
# =============================================================================
# Runs MatchMaker on all 7 SchemaNet schema pairs using the dedicated
# run_schemanet.py adapter (alternative to match_maker_runner module).
#
# Usage:
#   ./scripts/run_matchmaker_schemanet_all.sh
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

    python match_maker/run_schemanet.py \
        --dataset "$dataset" \
        --data-dir "$DATA_DIR" \
        --model "$MODEL" \
        --top-k 5 \
        2>&1 | tail -10

    echo ""
done

echo "=== All experiments completed ==="
echo "End: $(date)"
