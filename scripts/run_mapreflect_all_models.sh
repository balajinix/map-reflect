#!/bin/bash
# =============================================================================
# Map-Reflect Full Re-Run — All Datasets x All Models (Sequential)
# =============================================================================
# Re-runs Map-Reflect with the current map_reflect_runner.py on all 6 main
# datasets with all 3 models (Granite-4, Llama-3.3-70B, Mistral-Medium).
#
# Runs one experiment at a time to avoid WatsonX API throttling.
# Each run has checkpoint support (--resume) and saves results incrementally.
#
# Skips experiments where a mapreflect_* result file already exists with
# the expected number of samples. Use --force to re-run everything.
#
# Usage:
#   ./scripts/run_mapreflect_all_models.sh              # run missing only
#   ./scripts/run_mapreflect_all_models.sh --force       # re-run everything
#   ./scripts/run_mapreflect_all_models.sh --dry-run     # show what would run
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

RESULTS_DIR="map_reflect/results"
DATA_DIR="data"

GRANITE="watsonx/ibm/granite-4-h-small"
LLAMA="watsonx/meta-llama/llama-3-3-70b-instruct"
MISTRAL="watsonx/mistralai/mistral-medium-2505"

FORCE=false
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --force)    FORCE=true; shift ;;
        --dry-run)  DRY_RUN=true; shift ;;
        *)          echo "Unknown: $1"; exit 1 ;;
    esac
done

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $1"; }

# Expected sample counts per dataset
declare -A EXPECTED_SAMPLES
EXPECTED_SAMPLES[ITOpsGraph]=521
EXPECTED_SAMPLES[NL2GQL]=437
EXPECTED_SAMPLES[OMOP_CMS]=38
EXPECTED_SAMPLES[OMOP_MIMIC]=46
EXPECTED_SAMPLES[OMOP_Synthea]=12
EXPECTED_SAMPLES[ToolBench]=3190

get_lookup_file() {
    local ds="$1"
    case "$ds" in
        ITOpsGraph)    echo "${DATA_DIR}/itopsgraph_api_lookup.json" ;;
        NL2GQL)        echo "${DATA_DIR}/nl2gql_api_lookup.json" ;;
        ToolBench)     echo "${DATA_DIR}/toolbench_api_lookup.json" ;;
        OMOP_CMS)      echo "${DATA_DIR}/omop_cms_api_lookup.json" ;;
        OMOP_MIMIC)    echo "${DATA_DIR}/omop_mimic_api_lookup.json" ;;
        OMOP_Synthea)  echo "${DATA_DIR}/omop_synthea_api_lookup.json" ;;
    esac
}

# Check if result already exists with enough samples
# Prints the filename if found, prints nothing if not found
result_exists() {
    local ds="$1" model_id="$2"
    local expected=${EXPECTED_SAMPLES[$ds]}
    local ds_lower=$(echo "$ds" | tr '[:upper:]' '[:lower:]')
    local model_slug="${model_id//\//_}"  # watsonx/ibm/granite-4-h-small -> watsonx_ibm_granite-4-h-small
    model_slug="${model_slug#watsonx_}"   # strip watsonx_ prefix

    # Look for mapreflect_<ds>_<model>.json
    local f
    for f in "${RESULTS_DIR}"/mapreflect_${ds_lower}_${model_slug}*.json; do
        [[ -f "$f" ]] || continue
        [[ "$f" == *checkpoint* ]] && continue
        local n
        n=$(python3 -c "import json; print(len(json.load(open('$f')).get('results',[])))" 2>/dev/null || echo 0)
        if [[ "$n" -ge "$expected" ]]; then
            echo "$f ($n samples)"
            return 0
        fi
    done
    return 0  # Always return 0 to avoid set -e issues; caller checks output
}

run_experiment() {
    local ds="$1" model_id="$2" model_name="$3"
    local lookup=$(get_lookup_file "$ds")
    local expected=${EXPECTED_SAMPLES[$ds]}

    # Check if already done
    if [[ "$FORCE" == "false" ]]; then
        local existing
        existing=$(result_exists "$ds" "$model_id" || true)
        if [[ -n "$existing" ]]; then
            log "SKIP $ds / $model_name — already exists: $existing"
            return 0
        fi
    fi

    log "RUN  $ds / $model_name ($expected samples)"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY] python -m map_reflect.map_reflect_runner --datasets $ds --models $model_id ..."
        return 0
    fi

    python -m map_reflect.map_reflect_runner \
        --datasets "$ds" \
        --models "$model_id" \
        --property-lookup-file "$lookup" \
        --data-dir "$DATA_DIR" \
        --batch-size 20 \
        --checkpoint-interval 20 \
        --resume \
        2>&1 | tail -5

    log "DONE $ds / $model_name"
    echo ""
}

# ============================================================================
# Main
# ============================================================================

log "=========================================="
log "Map-Reflect Full Re-Run (All Models)"
log "Force: $FORCE | Dry-run: $DRY_RUN"
log "=========================================="
echo ""

TOTAL=0
SKIPPED=0
RAN=0

# Run in order: small datasets first (fast feedback), then large ones
# Within each dataset: Llama first (best model), then Granite, then Mistral
DATASETS=(OMOP_Synthea OMOP_CMS OMOP_MIMIC NL2GQL ITOpsGraph ToolBench)

for ds in "${DATASETS[@]}"; do
    log "--- Dataset: $ds (${EXPECTED_SAMPLES[$ds]} samples) ---"

    for model_id in "$LLAMA" "$GRANITE" "$MISTRAL"; do
        case "$model_id" in
            *llama*)   model_name="Llama" ;;
            *granite*) model_name="Granite" ;;
            *mistral*) model_name="Mistral" ;;
        esac

        TOTAL=$((TOTAL + 1))
        run_experiment "$ds" "$model_id" "$model_name"
    done
done

log "=========================================="
log "COMPLETE: $TOTAL total experiments"
log "=========================================="
