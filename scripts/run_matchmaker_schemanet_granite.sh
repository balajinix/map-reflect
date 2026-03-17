#!/bin/bash
# =============================================================================
# MatchMaker SchemaNet Experiments (Granite)
# =============================================================================
# Runs MatchMaker on all 7 SchemaNet schema pairs using the run_schemanet.py
# adapter with the smaller Granite-3.3-8B backbone.
#
# Usage:
#   ./scripts/run_matchmaker_schemanet_granite.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BASE_DIR"

if [[ -f "${BASE_DIR}/.venv/bin/activate" ]]; then
    source "${BASE_DIR}/.venv/bin/activate"
fi

MODEL="watsonx/ibm/granite-3-3-8b-instruct"

for dataset in schemanet_bank1_bank2 schemanet_imdb_sakila schemanet_cms_omop schemanet_mimic_iii_omop schemanet_synthea_omop schemanet_cprd_gold_omop schemanet_cprd_aurum_omop; do
    echo "Running: $dataset with Granite"
    python match_maker/run_schemanet.py --dataset "$dataset" --data-dir data/schemanet --model "$MODEL" --top-k 5 2>&1 | tail -5
    echo ""
done
echo "Done!"
