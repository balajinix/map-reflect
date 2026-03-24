#!/bin/bash
# =============================================================================
# Remaining Experiments — Fill Paper Gaps
# =============================================================================
# 1. MatchMaker Granite ToolBench (with normalization fix)
# 2. Map-Reflect retrieval mode: OMOP (3 datasets) + ToolBench
# 3. MatchMaker retrieval mode: ITOpsGraph + NL2GQL (oracle hints for comparison)
#
# All sequential to avoid WatsonX API contention.
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

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $1"; }

log "=========================================="
log "Remaining Experiments"
log "=========================================="

# --- 1. MatchMaker Granite ToolBench ---
log "=== [1/7] MatchMaker Granite ToolBench ==="
python -m match_maker.match_maker_runner \
    --datasets ToolBench \
    --models "$GRANITE" \
    --property-lookup-file data/toolbench_api_lookup.json \
    --data-dir data \
    --top-k 5 --abstention-mode lenient \
    2>&1 | tail -5
echo

# --- 2-5. Map-Reflect retrieval mode (OMOP + ToolBench) ---
for DS in OMOP_CMS OMOP_MIMIC OMOP_Synthea ToolBench; do
    ds_lower=$(echo "$DS" | tr '[:upper:]' '[:lower:]')
    LOOKUP="data/${ds_lower}_api_lookup.json"
    # ToolBench lookup
    if [[ "$DS" == "ToolBench" ]]; then
        LOOKUP="data/toolbench_api_lookup.json"
    fi
    OUTPUT="results/retrieval_mode/map_reflect_${ds_lower}_llama"
    mkdir -p "$OUTPUT"

    log "=== Map-Reflect Llama $DS (retrieval) ==="
    python -m map_reflect.map_reflect_runner \
        --datasets "$DS" \
        --models "$LLAMA" \
        --property-lookup-file "$LOOKUP" \
        --data-dir data/retrieval_mode \
        --batch-size 20 --checkpoint-interval 50 --resume \
        --output-dir "$OUTPUT" \
        2>&1 | tail -3
    echo
done

# --- 6-7. MatchMaker on ITOpsGraph + NL2GQL (for retrieval comparison) ---
# These run on oracle data but MatchMaker's internal ColBERT does its own retrieval
log "=== [6/7] MatchMaker Llama ITOpsGraph ==="
python -m match_maker.match_maker_runner \
    --datasets ITOpsGraph \
    --models "$LLAMA" \
    --property-lookup-file data/itopsgraph_api_lookup.json \
    --data-dir data \
    --top-k 5 --abstention-mode lenient \
    2>&1 | tail -5
echo

log "=== [7/7] MatchMaker Llama NL2GQL ==="
python -m match_maker.match_maker_runner \
    --datasets NL2GQL \
    --models "$LLAMA" \
    --property-lookup-file data/nl2gql_api_lookup.json \
    --data-dir data \
    --top-k 5 --abstention-mode lenient \
    2>&1 | tail -5
echo

log "=========================================="
log "All remaining experiments done"
log "=========================================="
