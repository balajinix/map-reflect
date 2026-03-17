#!/bin/bash
# =============================================================================
# Mistral Experiments — Sequential Runner (MatchMaker + AnyTool)
# =============================================================================
# Runs all missing Mistral experiments for MatchMaker and AnyTool via litellm.
# Uses watsonx/mistralai/mistral-medium-2505 (WatsonX-hosted Mistral).
#
# Map-Reflect SchemaNet experiments require ibm_watsonx_ai >= 1.0 (Python 3.10+)
# and must be run separately via run_mistral_mapreflect_schemanet.sh on a
# machine with Python >= 3.10.
#
# Each experiment runs sequentially (one at a time) to avoid WatsonX API
# contention. All runners have tqdm progress bars and checkpoint/resume.
#
# Usage:
#   ./scripts/run_mistral_experiments.sh              # dry run + full
#   ./scripts/run_mistral_experiments.sh --dry-run     # 10-sample validation only
#   ./scripts/run_mistral_experiments.sh --skip-dry-run # skip validation, run full
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BASE_DIR"

# Activate venv
if [[ -f "${BASE_DIR}/.venv/bin/activate" ]]; then
    source "${BASE_DIR}/.venv/bin/activate"
fi

# Load env (WatsonX credentials)
if [[ -f "${BASE_DIR}/.env" ]]; then
    set -a && source "${BASE_DIR}/.env" && set +a
fi

MODEL="watsonx/mistralai/mistral-medium-2505"
DATA_DIR="data"
TODO_FILE="scripts/mistral_experiments_todo.md"
DRY_RUN_SAMPLES=10

# Parse args
MODE="full"
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)       MODE="dry-run"; shift ;;
        --skip-dry-run)  MODE="skip-dry-run"; shift ;;
        *)               echo "Unknown: $1"; exit 1 ;;
    esac
done

# ============================================================================
# Helpers
# ============================================================================

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $1"; }

update_todo() {
    local method="$1" dataset="$2" status="$3"
    if [[ -f "$TODO_FILE" ]]; then
        # Mark experiment as done in TODO
        sed -i "s/\(| ${method}.*${dataset}.*|\) pending/\1 ${status}/" "$TODO_FILE" 2>/dev/null || true
    fi
}

get_lookup_file() {
    local dataset="$1"
    case "$dataset" in
        ITOpsGraph)    echo "${DATA_DIR}/itopsgraph_api_lookup.json" ;;
        NL2GQL)        echo "${DATA_DIR}/nl2gql_api_lookup.json" ;;
        ToolBench)     echo "${DATA_DIR}/toolbench_api_lookup.json" ;;
        OMOP_CMS)      echo "${DATA_DIR}/omop_cms_api_lookup.json" ;;
        OMOP_MIMIC)    echo "${DATA_DIR}/omop_mimic_api_lookup.json" ;;
        OMOP_Synthea)  echo "${DATA_DIR}/omop_synthea_api_lookup.json" ;;
        schemanet_*)   echo "${DATA_DIR}/schemanet/${dataset}_api_lookup.json" ;;
    esac
}

get_hierarchy_file() {
    local dataset="$1"
    case "$dataset" in
        ITOpsGraph)    echo "${DATA_DIR}/itopsgraph_hierarchy_anytool.json" ;;
        NL2GQL)        echo "${DATA_DIR}/nl2gql_hierarchy.json" ;;
        OMOP_CMS)      echo "${DATA_DIR}/omop_cms_hierarchy_anytool.json" ;;
        OMOP_MIMIC)    echo "${DATA_DIR}/omop_mimic_hierarchy_anytool.json" ;;
        OMOP_Synthea)  echo "${DATA_DIR}/omop_synthea_hierarchy_anytool.json" ;;
    esac
}

# ============================================================================
# Runner functions
# ============================================================================

run_matchmaker() {
    local dataset="$1"
    local lookup=$(get_lookup_file "$dataset")
    local ddir="$DATA_DIR"
    [[ "$dataset" == schemanet_* ]] && ddir="${DATA_DIR}/schemanet"

    log "MatchMaker: $dataset"
    python -m match_maker.match_maker_runner \
        --datasets "$dataset" \
        --models "$MODEL" \
        --property-lookup-file "$lookup" \
        --data-dir "$ddir" \
        --top-k 5 \
        --abstention-mode lenient
}

run_matchmaker_schemanet() {
    local dataset="$1"
    log "MatchMaker SchemaNet: $dataset"
    python match_maker/run_schemanet.py \
        --dataset "$dataset" \
        --data-dir "${DATA_DIR}/schemanet" \
        --model "$MODEL" \
        --top-k 5
}

run_anytool() {
    local dataset="$1"
    local hierarchy=$(get_hierarchy_file "$dataset")

    if [[ -z "$hierarchy" || ! -f "$hierarchy" ]]; then
        log "SKIP AnyTool $dataset: no hierarchy file at $hierarchy"
        return 1
    fi

    log "AnyTool: $dataset"
    python -m anytool.anytool_runner \
        --datasets "$dataset" \
        --models "$MODEL" \
        --hierarchy-file "$hierarchy" \
        --data-dir "$DATA_DIR" \
        --workers 1
}

run_anytool_schemanet() {
    local dataset="$1"
    local pair="${dataset#schemanet_}"
    local anytool_data="data/schemanet/anytool_format/$pair"
    local queries="$anytool_data/schemanet_${pair}_queries.json"

    if [[ ! -f "$queries" ]]; then
        log "SKIP AnyTool SchemaNet $dataset: no queries at $queries"
        return 1
    fi

    log "AnyTool SchemaNet: $dataset"
    python anytool/run_anytool_schemanet.py \
        --dataset "$dataset" \
        --data-dir "$anytool_data" \
        --queries-file "$queries" \
        --model "mistralai/mistral-medium-2505" \
        --output "anytool/results/schemanet_${pair}_mistral_results.json"
}

# ============================================================================
# Dry run: validate with smallest dataset
# ============================================================================

run_dry_run() {
    log "=========================================="
    log "DRY RUN: Validating Mistral model access"
    log "=========================================="

    # Test MatchMaker on smallest dataset (OMOP-Synthea, 12 samples)
    log "--- Testing MatchMaker (OMOP-Synthea, 12 samples) ---"
    run_matchmaker "OMOP_Synthea" || {
        log "FAILED: MatchMaker dry run"; return 1
    }

    # Check result file exists and has non-zero successful samples
    local result_file=$(ls -t match_maker/results/OMOP_Synthea_*mistral*.jsonl match_maker/results/OMOP_Synthea_*mistral*.json 2>/dev/null | head -1)
    if [[ -z "$result_file" ]]; then
        log "FAILED: No result file found for MatchMaker OMOP_Synthea Mistral"
        return 1
    fi

    local success_count=$(python3 -c "
import json
with open('$result_file') as f:
    if '$result_file'.endswith('.jsonl'):
        results = [json.loads(l) for l in f if l.strip()]
    else:
        results = json.load(f).get('results', [])
ok = sum(1 for r in results if r.get('success', False))
print(ok)
" 2>/dev/null)

    log "MatchMaker result: $success_count successful samples in $result_file"
    if [[ "$success_count" == "0" ]]; then
        log "FAILED: All samples failed — check model access and credentials"
        return 1
    fi

    # Validate eval can read the results
    log "--- Validating eval ---"
    python eval/calc_accuracy.py --methods MatchMaker --datasets OMOP_Synthea --output markdown 2>&1 | tail -8
    if [[ $? -ne 0 ]]; then
        log "FAILED: Eval failed"; return 1
    fi

    log "=========================================="
    log "DRY RUN PASSED"
    log "=========================================="
    return 0
}

# ============================================================================
# Full experiment runs
# ============================================================================

run_all_experiments() {
    log "=========================================="
    log "MISTRAL EXPERIMENTS (MatchMaker + AnyTool)"
    log "Model: $MODEL"
    log "=========================================="

    # --- Priority 1: Core tables ---

    log "=== [1/6] MatchMaker OMOP-CMS (38 samples) ==="
    run_matchmaker "OMOP_CMS" && update_todo "MatchMaker" "OMOP-CMS" "done"
    log ""

    log "=== [2/6] MatchMaker OMOP-MIMIC (46 samples) ==="
    run_matchmaker "OMOP_MIMIC" && update_todo "MatchMaker" "OMOP-MIMIC" "done"
    log ""

    log "=== [3/6] MatchMaker OMOP-Synthea (12 samples) ==="
    # May already exist from dry run — MatchMaker has checkpoint/resume
    run_matchmaker "OMOP_Synthea" && update_todo "MatchMaker" "OMOP-Synthea" "done"
    log ""

    log "=== [4/6] MatchMaker ITOpsGraph (521 samples) ==="
    run_matchmaker "ITOpsGraph" && update_todo "MatchMaker" "ITOpsGraph" "done"
    log ""

    log "=== [5/6] MatchMaker ToolBench (3190 samples) ==="
    run_matchmaker "ToolBench" && update_todo "MatchMaker" "ToolBench" "done"
    log ""

    log "=== [6/6] AnyTool ITOpsGraph (521 samples) ==="
    run_anytool "ITOpsGraph" && update_todo "AnyTool" "ITOpsGraph" "done"
    log ""

    # --- Priority 2: SchemaNet ---

    SCHEMANET_PAIRS=(
        schemanet_bank1_bank2
        schemanet_imdb_sakila
        schemanet_cms_omop
        schemanet_mimic_iii_omop
        schemanet_synthea_omop
        schemanet_cprd_gold_omop
        schemanet_cprd_aurum_omop
    )

    log "=== [P2] AnyTool SchemaNet (7 pairs) ==="
    for ds in "${SCHEMANET_PAIRS[@]}"; do
        log "  Running AnyTool: $ds"
        run_anytool_schemanet "$ds" || log "  WARN: AnyTool $ds failed (continuing)"
    done
    update_todo "AnyTool" "SchemaNet" "done"
    log ""

    log "=== [P2] MatchMaker SchemaNet (7 pairs) ==="
    for ds in "${SCHEMANET_PAIRS[@]}"; do
        log "  Running MatchMaker: $ds"
        run_matchmaker_schemanet "$ds" || log "  WARN: MatchMaker $ds failed (continuing)"
    done
    update_todo "MatchMaker" "SchemaNet" "done"
    log ""

    log "=========================================="
    log "ALL EXPERIMENTS COMPLETED"
    log "=========================================="
}

# ============================================================================
# Main
# ============================================================================

log "Mistral Experiment Runner"
log "Mode: $MODE"
log "Working directory: $BASE_DIR"
log "Python: $(which python)"
log ""

case "$MODE" in
    dry-run)
        run_dry_run
        ;;
    skip-dry-run)
        run_all_experiments
        ;;
    full)
        run_dry_run || { log "Dry run failed — aborting"; exit 1; }
        log ""
        log "Dry run passed. Starting full experiments..."
        log ""
        run_all_experiments
        ;;
esac
