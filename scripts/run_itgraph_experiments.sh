#!/bin/bash
# =============================================================================
# ITGraph Experiments — Map-Reflect, MatchMaker, AnyTool on ITGraph dataset
# =============================================================================
# Runs on the consensus-built ITGraph dataset (419 type pairs, 854 bipartite
# field mappings across 10 products).
#
# Usage:
#   ./scripts/run_itgraph_experiments.sh                    # run all
#   ./scripts/run_itgraph_experiments.sh --method mr        # map-reflect only
#   ./scripts/run_itgraph_experiments.sh --method mm        # matchmaker only
#   ./scripts/run_itgraph_experiments.sh --method at        # anytool only
#   ./scripts/run_itgraph_experiments.sh --model llama      # llama only
#   ./scripts/run_itgraph_experiments.sh --dataset hc       # high-confidence only
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
LOOKUP="data/itom_hashicorp_api_lookup_v2.json"
DATA_DIR="data"
DATASET="ITGraph"

# Parse args
METHOD_FILTER=""
MODEL_FILTER=""
DATASET_VARIANT=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --method)  METHOD_FILTER="$2"; shift 2 ;;
        --model)   MODEL_FILTER="$2"; shift 2 ;;
        --dataset) DATASET_VARIANT="$2"; shift 2 ;;
        *)         echo "Unknown: $1"; exit 1 ;;
    esac
done

# Use high-confidence dataset if requested
if [[ "$DATASET_VARIANT" == "hc" ]]; then
    DATASET="ITGraph_HC"
fi

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
# Register the ITGraph dataset in dataset_loader.py if not already there
# ============================================================================

python3 -c "
import re
loader_path = 'data/dataset_loader.py'
with open(loader_path) as f:
    content = f.read()

if 'ITGraph' not in content or 'itgraph_dataset' not in content:
    # Add ITGraph dataset registration
    content = content.replace(
        \"'ITOM_Hashicorp': 'itom_hashicorp_dataset.jsonl',\",
        \"'ITOM_Hashicorp': 'itom_hashicorp_dataset.jsonl',\\n\"
        \"            'ITGraph': 'itgraph_dataset.jsonl',  # ITGraph consensus dataset (419 samples)\\n\"
        \"            'ITGraph_HC': 'itgraph_dataset_high_confidence.jsonl',  # High-confidence (246 samples)\",
    )
    # Add expected counts
    content = content.replace(
        \"'ITOM_Hashicorp': 61,\",
        \"'ITOM_Hashicorp': 61,\\n\"
        \"            'ITGraph': 419,  # ITGraph consensus dataset\\n\"
        \"            'ITGraph_HC': 246,  # ITGraph high-confidence\",
    )
    # Add ITGraph to the ITOpsGraph-style loading branch
    content = content.replace(
        'if dataset.startswith(\"ITOpsGraph\") or dataset == \"ITOM_Hashicorp\":',
        'if dataset.startswith(\"ITOpsGraph\") or dataset == \"ITOM_Hashicorp\" or dataset.startswith(\"ITGraph\"):',
    )
    with open(loader_path, 'w') as f:
        f.write(content)
    print('Registered ITGraph dataset in dataset_loader.py')
else:
    print('ITGraph already registered')
"

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

    local hierarchy="${DATA_DIR}/itopsgraph_hierarchy_anytool.json"
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

# Count samples
SAMPLE_COUNT=$(python3 -c "
with open('data/$(echo $DATASET | tr '[:upper:]' '[:lower:]')_dataset$([ \"$DATASET\" = \"ITGraph_HC\" ] && echo \"_high_confidence\").jsonl') as f:
    print(sum(1 for line in f if line.strip()))
" 2>/dev/null || echo "?")

log "=========================================="
log "ITGraph Experiments"
log "Dataset: $DATASET ($SAMPLE_COUNT samples)"
log "Method filter: ${METHOD_FILTER:-all}"
log "Model filter: ${MODEL_FILTER:-all}"
log "=========================================="
echo

for MODEL_ID in "$LLAMA" "$GRANITE"; do
    case "$MODEL_ID" in
        *llama*)   MODEL_NAME="llama" ;;
        *granite*) MODEL_NAME="granite" ;;
    esac

    log "--- Model: $MODEL_NAME ---"
    run_map_reflect "$MODEL_ID" "$MODEL_NAME"
    run_matchmaker "$MODEL_ID" "$MODEL_NAME"
    run_anytool "$MODEL_ID" "$MODEL_NAME"
done

log "=========================================="
log "ALL ITGraph experiments done"
log "=========================================="
