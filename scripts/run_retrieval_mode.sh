#!/bin/bash
# =============================================================================
# Retrieval Mode Experiments
# =============================================================================
# Runs Map-Reflect (3 models) and MatchMaker (Llama) in retrieval mode
# on ITOpsGraph and NL2GQL. In retrieval mode the target schema is NOT
# given — the method must discover it from a catalogue.
#
# Usage:
#   ./scripts/run_retrieval_mode.sh
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

LLAMA="watsonx/meta-llama/llama-3-3-70b-instruct"
GRANITE="watsonx/ibm/granite-4-h-small"
MISTRAL="watsonx/mistralai/mistral-medium-2505"
OUTPUT_BASE="results/retrieval_mode"
mkdir -p "$OUTPUT_BASE"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $1"; }

run_mr() {
    local dataset="$1" model="$2" model_name="$3" lookup="$4"
    local output_dir="${OUTPUT_BASE}/map_reflect_${dataset,,}_${model_name}"
    mkdir -p "$output_dir"

    log "Map-Reflect $model_name / $dataset (retrieval)"
    python -m map_reflect.map_reflect_runner \
        --datasets "$dataset" \
        --models "$model" \
        --property-lookup-file "$lookup" \
        --data-dir data/retrieval_mode \
        --batch-size 20 --checkpoint-interval 20 --resume \
        --output-dir "$output_dir" \
        2>&1 | tail -3
    echo
}

run_mm() {
    local dataset="$1" model="$2" model_name="$3" lookup="$4"

    log "MatchMaker $model_name / $dataset (retrieval)"
    python -m match_maker.match_maker_runner \
        --datasets "$dataset" \
        --models "$model" \
        --property-lookup-file "$lookup" \
        --data-dir data/retrieval_mode \
        --top-k 5 --abstention-mode lenient \
        2>&1 | tail -3
    echo
}

log "=========================================="
log "Retrieval Mode Experiments"
log "=========================================="

# --- ITOpsGraph (521 samples, 12 products) ---
ITOPS_LOOKUP="data/itopsgraph_api_lookup.json"

log "=== ITOpsGraph ==="
for MODEL in "$LLAMA" "$GRANITE" "$MISTRAL"; do
    case "$MODEL" in
        *llama*)   NAME="llama" ;;
        *granite*) NAME="granite" ;;
        *mistral*) NAME="mistral" ;;
    esac
    run_mr "ITOpsGraph" "$MODEL" "$NAME" "$ITOPS_LOOKUP"
done

run_mm "ITOpsGraph" "$LLAMA" "llama" "$ITOPS_LOOKUP"

# --- NL2GQL (47 retrieval samples, 54 schemas) ---
NL2GQL_LOOKUP="data/nl2gql_api_lookup.json"

log "=== NL2GQL ==="
for MODEL in "$LLAMA" "$GRANITE" "$MISTRAL"; do
    case "$MODEL" in
        *llama*)   NAME="llama" ;;
        *granite*) NAME="granite" ;;
        *mistral*) NAME="mistral" ;;
    esac
    run_mr "NL2GQL" "$MODEL" "$NAME" "$NL2GQL_LOOKUP"
done

run_mm "NL2GQL" "$LLAMA" "llama" "$NL2GQL_LOOKUP"

log "=========================================="
log "Retrieval mode experiments done"
log "=========================================="
