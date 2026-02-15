#!/usr/bin/env bash
# =============================================================================
# Coverage Evaluation Experiments Runner
# =============================================================================
#
# Runs all methods on the ITOpsGraph_Coverage_Eval dataset (521 samples)
# to compare their accuracy on discovering type mappings.
#
# Methods:
#   - Map-Reflect (UE-Reflect)
#   - MatchMaker
#   - ReMatch
#   - SMT
#   - ReAct
#
# Usage:
#   ./run_coverage_eval_experiments.sh [--model MODEL] [--method METHOD]
#
# Options:
#   --model MODEL    Run with specific model (granite, llama, mistral)
#   --method METHOD  Run specific method only (map_reflect, matchmaker, rematch, smt, react)
#   --dry-run        Show what would run without executing
#
# =============================================================================

set -e

# Parse arguments
MODEL_FILTER=""
METHOD_FILTER=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_FILTER="$2"
            shift 2
            ;;
        --method)
            METHOD_FILTER="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--model MODEL] [--method METHOD] [--dry-run]"
            exit 1
            ;;
    esac
done

# Configuration
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${BASE_DIR}/data"
LOG_DIR="${BASE_DIR}/logs"

# Model IDs
LLAMA="watsonx/meta-llama/llama-3-3-70b-instruct"

# Data files
ITOPSGRAPH_LOOKUP="${DATA_DIR}/itopsgraph_api_lookup.json"
DATASET="ITOpsGraph_Coverage_Eval"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

log() {
    echo "[$(timestamp)] $1"
}

log_section() {
    echo ""
    echo "=============================================="
    log "$1"
    echo "=============================================="
}

run_cmd() {
    local description="$1"
    shift
    local cmd="$*"

    log "Running: $description"
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY RUN] $cmd"
    else
        eval "$cmd"
    fi
}

should_run_model() {
    local model="$1"
    [[ -z "$MODEL_FILTER" || "$MODEL_FILTER" == "$model" ]]
}

should_run_method() {
    local method="$1"
    [[ -z "$METHOD_FILTER" || "$METHOD_FILTER" == "$method" ]]
}

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

log_section "ENVIRONMENT SETUP"

echo "Base directory: $BASE_DIR"
echo "Dataset: $DATASET"
echo "Model filter: ${MODEL_FILTER:-all}"
echo "Method filter: ${METHOD_FILTER:-all}"
echo "Dry run: $DRY_RUN"

# Create log directory
mkdir -p "$LOG_DIR"

# Activate virtual environment if available
if [[ -f "${BASE_DIR}/.venv/bin/activate" ]]; then
    log "Activating virtual environment"
    source "${BASE_DIR}/.venv/bin/activate"
fi

# Load environment variables
if [[ -f "${BASE_DIR}/.env" ]]; then
    log "Loading environment variables from .env"
    export $(grep -v '^#' "${BASE_DIR}/.env" | xargs)
fi

cd "$BASE_DIR"

# Verify dataset exists
DATASET_FILE="${DATA_DIR}/itopsgraph_coverage_eval_dataset.jsonl"
if [[ ! -f "$DATASET_FILE" ]]; then
    echo "ERROR: Dataset not found: $DATASET_FILE"
    echo "Run: python scripts/create_coverage_eval_dataset.py"
    exit 1
fi

SAMPLE_COUNT=$(wc -l < "$DATASET_FILE" | tr -d ' ')
log "Dataset has $SAMPLE_COUNT samples"

# =============================================================================
# RUN EXPERIMENTS
# =============================================================================

# Map-Reflect with Llama
if should_run_method "map_reflect" && should_run_model "llama"; then
    log_section "Map-Reflect with Llama"
    run_cmd "Map-Reflect (Llama)" \
        "python -m map_reflect.ue_reflect \
            --datasets '$DATASET' \
            --models '$LLAMA' \
            --property-lookup-file '$ITOPSGRAPH_LOOKUP' \
            --data-dir '$DATA_DIR' \
            --output-dir map_reflect/results \
            --top-k 10"
fi

# MatchMaker with Llama
if should_run_method "matchmaker" && should_run_model "llama"; then
    log_section "MatchMaker with Llama"
    run_cmd "MatchMaker (Llama)" \
        "cd '${BASE_DIR}/match_maker' && python match_maker_runner.py \
            --property-lookup-file '$ITOPSGRAPH_LOOKUP' \
            --data-dir '$DATA_DIR' \
            --datasets '$DATASET' \
            --models '$LLAMA'"
fi

# ReMatch with Llama
if should_run_method "rematch" && should_run_model "llama"; then
    log_section "ReMatch with Llama"
    run_cmd "ReMatch (Llama)" \
        "cd '${BASE_DIR}/rematch' && python rematch_runner.py \
            --property-lookup-file '$ITOPSGRAPH_LOOKUP' \
            --data-dir '$DATA_DIR' \
            --datasets '$DATASET' \
            --models '$LLAMA'"
fi

# SMT with Llama
if should_run_method "smt" && should_run_model "llama"; then
    log_section "SMT with Llama"
    run_cmd "SMT (Llama)" \
        "./run.sh --method smt --dataset $DATASET --model llama --quick"
fi

# ReAct with Llama
if should_run_method "react" && should_run_model "llama"; then
    log_section "ReAct with Llama"
    run_cmd "ReAct (Llama)" \
        "./run.sh --method react --dataset $DATASET --model llama --quick"
fi

# =============================================================================
# SUMMARY
# =============================================================================

log_section "EXPERIMENTS COMPLETED"

echo ""
echo "Results can be found in:"
echo "  - map_reflect/results/"
echo "  - match_maker/results/"
echo "  - rematch/results/"
echo "  - smt/results/"
echo "  - react/results/"
echo ""
log "Finished at $(timestamp)"
