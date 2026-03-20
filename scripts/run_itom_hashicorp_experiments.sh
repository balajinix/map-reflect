#!/bin/bash
# =============================================================================
# ITOM-Hashicorp Experiments — All Methods x All Models (Sequential)
# =============================================================================
# Runs Map-Reflect, MatchMaker, and AnyTool on the ITOM-Hashicorp dataset
# (61 samples) with Llama, Granite, and Mistral models.
#
# All experiments run sequentially to avoid WatsonX API contention.
# Each runner has checkpoint/resume and tqdm progress.
#
# MatchMaker has known issues with multi-product utterance parsing on
# this dataset — it may fail. AnyTool needs a hierarchy file.
#
# Usage:
#   ./scripts/run_itom_hashicorp_experiments.sh              # run all
#   ./scripts/run_itom_hashicorp_experiments.sh --method mr   # map-reflect only
#   ./scripts/run_itom_hashicorp_experiments.sh --method mm   # matchmaker only
#   ./scripts/run_itom_hashicorp_experiments.sh --model llama  # llama only
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
LOOKUP="data/itom_hashicorp_api_lookup.json"
DATA_DIR="data"
DATASET="ITOM_Hashicorp"

# Parse args
METHOD_FILTER=""
MODEL_FILTER=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --method) METHOD_FILTER="$2"; shift 2 ;;
        --model)  MODEL_FILTER="$2"; shift 2 ;;
        *)        echo "Unknown: $1"; exit 1 ;;
    esac
done

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $1"; }

should_run() {
    local method="$1" model="$2"
    if [[ -n "$METHOD_FILTER" && "$METHOD_FILTER" != "$method" ]]; then
        return 1
    fi
    if [[ -n "$MODEL_FILTER" && "$MODEL_FILTER" != "$model" ]]; then
        return 1
    fi
    return 0
}

# ============================================================================
# Map-Reflect
# ============================================================================

run_map_reflect() {
    local model_id="$1" model_name="$2"
    should_run "mr" "$model_name" || return 0

    log "Map-Reflect / $model_name ($DATASET)"
    python -m map_reflect.map_reflect_runner \
        --datasets "$DATASET" \
        --models "$model_id" \
        --property-lookup-file "$LOOKUP" \
        --data-dir "$DATA_DIR" \
        --batch-size 20 \
        --checkpoint-interval 10 \
        --resume \
        2>&1 | tail -5
    echo
}

# ============================================================================
# MatchMaker
# ============================================================================

run_matchmaker() {
    local model_id="$1" model_name="$2"
    should_run "mm" "$model_name" || return 0

    log "MatchMaker / $model_name ($DATASET)"
    python -m match_maker.match_maker_runner \
        --datasets "$DATASET" \
        --models "$model_id" \
        --property-lookup-file "$LOOKUP" \
        --data-dir "$DATA_DIR" \
        --top-k 5 \
        --abstention-mode lenient \
        2>&1 | tail -5
    echo
}

# ============================================================================
# AnyTool
# ============================================================================

run_anytool() {
    local model_id="$1" model_name="$2"
    should_run "at" "$model_name" || return 0

    # AnyTool needs a hierarchy file — check if one exists
    local hierarchy="${DATA_DIR}/itom_hashicorp_hierarchy.json"
    if [[ ! -f "$hierarchy" ]]; then
        # Try ITOpsGraph hierarchy as fallback (ITOM_Hashicorp is a subset)
        hierarchy="${DATA_DIR}/itopsgraph_hierarchy_anytool.json"
    fi
    if [[ ! -f "$hierarchy" ]]; then
        log "SKIP AnyTool / $model_name: no hierarchy file"
        return 0
    fi

    log "AnyTool / $model_name ($DATASET)"
    python -m anytool.anytool_runner \
        --datasets "$DATASET" \
        --models "$model_id" \
        --hierarchy-file "$hierarchy" \
        --data-dir "$DATA_DIR" \
        --workers 1 \
        2>&1 | tail -5
    echo
}

# ============================================================================
# Main
# ============================================================================

log "=========================================="
log "ITOM-Hashicorp Experiments"
log "Dataset: $DATASET (61 samples)"
log "Method filter: ${METHOD_FILTER:-all}"
log "Model filter: ${MODEL_FILTER:-all}"
log "=========================================="
echo

# Run in order: Llama first (best), then Granite, then Mistral
# Within each model: Map-Reflect, MatchMaker, AnyTool

for MODEL_ID in "$LLAMA" "$GRANITE" "$MISTRAL"; do
    case "$MODEL_ID" in
        *llama*)   MODEL_NAME="llama" ;;
        *granite*) MODEL_NAME="granite" ;;
        *mistral*) MODEL_NAME="mistral" ;;
    esac

    log "--- Model: $MODEL_NAME ---"
    run_map_reflect "$MODEL_ID" "$MODEL_NAME"
    run_matchmaker "$MODEL_ID" "$MODEL_NAME"
    run_anytool "$MODEL_ID" "$MODEL_NAME"
done

log "=========================================="
log "ALL ITOM-Hashicorp experiments done"
log "=========================================="
