#!/usr/bin/env bash
# =============================================================================
# ITOpsGraph Missing Experiments Runner
# =============================================================================
#
# This script runs all missing ITOpsGraph experiments as identified in the plan.
#
# Model IDs:
#   - Granite-4: watsonx/ibm/granite-4-h-small
#   - Llama: watsonx/meta-llama/llama-3-3-70b-instruct
#   - Mistral-Medium: mistralai/mistral-medium-2505
#
# Usage:
#   Local:  ./run_itopsgraph_experiments.sh
#   Remote: ./run_itopsgraph_experiments.sh --remote
#   Background (remote): nohup ./run_itopsgraph_experiments.sh --remote > logs/itopsgraph_experiments.log 2>&1 &
#
# Options:
#   --remote     Force remote machine paths
#   --reset      Clear checkpoint and start fresh
#   --priority N Only run priority N experiments (1, 2, or 3)
#
# =============================================================================

set -e

# Parse arguments
FORCE_REMOTE=false
RESET_CHECKPOINT=false
PRIORITY_FILTER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --remote)
            FORCE_REMOTE=true
            shift
            ;;
        --reset)
            RESET_CHECKPOINT=true
            shift
            ;;
        --priority)
            PRIORITY_FILTER="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--remote] [--reset] [--priority N]"
            exit 1
            ;;
    esac
done

# Configuration - auto-detect local vs remote
if [[ -d "/home/bganesa1/code/graphql-schema-mapping" ]] || [[ "$FORCE_REMOTE" == "true" ]]; then
    IS_REMOTE=true
    BASE_DIR="/home/bganesa1/code/graphql-schema-mapping"
else
    IS_REMOTE=false
    BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

DATA_DIR="${BASE_DIR}/data"
LOG_DIR="${BASE_DIR}/logs"
DELAY=0  # No delay between experiments

# Model IDs
GRANITE="watsonx/ibm/granite-4-h-small"
LLAMA="watsonx/meta-llama/llama-3-3-70b-instruct"
MISTRAL="mistralai/mistral-medium-2505"

# Short names for filenames
GRANITE_SHORT="granite"
LLAMA_SHORT="llama"
MISTRAL_SHORT="mistral-medium"

# Data files
ITOPSGRAPH_LOOKUP="${DATA_DIR}/itopsgraph_api_lookup.json"
ITOPSGRAPH_HIERARCHY="${DATA_DIR}/itopsgraph_hierarchy.json"
ITOPSGRAPH_SCHEMA_DIR="${DATA_DIR}/itopsgraph_graphql"

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

run_with_delay() {
    log "Waiting ${DELAY}s before next experiment..."
    sleep "$DELAY"
}

# =============================================================================
# CHECKPOINT TRACKING
# =============================================================================

CHECKPOINT_FILE="${LOG_DIR}/itopsgraph_checkpoint.txt"

mark_complete() {
    echo "$1" >> "$CHECKPOINT_FILE"
    log "Marked complete: $1"
}

is_complete() {
    if [[ -f "$CHECKPOINT_FILE" ]]; then
        grep -q "^$1$" "$CHECKPOINT_FILE" 2>/dev/null
        return $?
    fi
    return 1
}

run_experiment() {
    local exp_id="$1"
    local description="$2"
    shift 2
    local cmd="$*"

    if is_complete "$exp_id"; then
        log "[$exp_id] Already complete, skipping..."
        return 0
    fi

    log "[$exp_id] Running: $description"
    if eval "$cmd"; then
        mark_complete "$exp_id"
        log "[$exp_id] SUCCESS"
    else
        log "[$exp_id] FAILED (continuing...)"
    fi
    run_with_delay
}

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

log_section "ENVIRONMENT SETUP"

echo "Running in $(if $IS_REMOTE; then echo 'REMOTE'; else echo 'LOCAL'; fi) mode"
echo "Base directory: $BASE_DIR"

# Create log directory
mkdir -p "$LOG_DIR"

# Reset checkpoint if requested
if [[ "$RESET_CHECKPOINT" == "true" ]]; then
    log "Resetting checkpoint file..."
    rm -f "$CHECKPOINT_FILE"
fi

# Show checkpoint status
if [[ -f "$CHECKPOINT_FILE" ]]; then
    completed_count=$(wc -l < "$CHECKPOINT_FILE" | tr -d ' ')
    log "Checkpoint file found: $completed_count experiments already completed"
else
    log "No checkpoint file found, starting fresh"
fi

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

# Ensure we're in the base directory
cd "$BASE_DIR"

# Verify data files exist
log "Verifying required data files..."
if [[ ! -f "$ITOPSGRAPH_LOOKUP" ]]; then
    echo "ERROR: ITOpsGraph API lookup not found: $ITOPSGRAPH_LOOKUP"
    exit 1
fi
if [[ ! -f "$ITOPSGRAPH_HIERARCHY" ]]; then
    echo "ERROR: ITOpsGraph hierarchy not found: $ITOPSGRAPH_HIERARCHY"
    exit 1
fi
if [[ ! -d "$ITOPSGRAPH_SCHEMA_DIR" ]]; then
    echo "ERROR: ITOpsGraph schema directory not found: $ITOPSGRAPH_SCHEMA_DIR"
    exit 1
fi
log "All required data files present"

# =============================================================================
# PRIORITY 1: CORE METHODS VIA run.sh (10 experiments)
# =============================================================================

if [[ -z "$PRIORITY_FILTER" || "$PRIORITY_FILTER" == "1" ]]; then

log_section "PRIORITY 1: CORE METHODS (10 experiments)"

# SMT - Mistral (only missing model)
run_experiment "smt_mistral" "SMT with Mistral" \
    "./run.sh --method smt --dataset ITOpsGraph --model mistral --quick"

# ReAct - all 3 models
run_experiment "react_granite" "ReAct with Granite" \
    "./run.sh --method react --dataset ITOpsGraph --model granite --quick"

run_experiment "react_llama" "ReAct with Llama" \
    "./run.sh --method react --dataset ITOpsGraph --model llama --quick"

run_experiment "react_mistral" "ReAct with Mistral" \
    "./run.sh --method react --dataset ITOpsGraph --model mistral --quick"

# AnyTool - all 3 models
run_experiment "anytool_granite" "AnyTool with Granite" \
    "./run.sh --method anytool --dataset ITOpsGraph --model granite --quick"

run_experiment "anytool_llama" "AnyTool with Llama" \
    "./run.sh --method anytool --dataset ITOpsGraph --model llama --quick"

run_experiment "anytool_mistral" "AnyTool with Mistral" \
    "./run.sh --method anytool --dataset ITOpsGraph --model mistral --quick"

# Map-Reflect - all 3 models
run_experiment "map_reflect_granite" "Map-Reflect with Granite" \
    "./run.sh --method map_reflect --dataset ITOpsGraph --model granite --quick"

run_experiment "map_reflect_llama" "Map-Reflect with Llama" \
    "./run.sh --method map_reflect --dataset ITOpsGraph --model llama --quick"

run_experiment "map_reflect_mistral" "Map-Reflect with Mistral" \
    "./run.sh --method map_reflect --dataset ITOpsGraph --model mistral --quick"

fi  # End PRIORITY 1

# =============================================================================
# PRIORITY 2: MatchMaker, DualContext, ReMatch (8 experiments)
# =============================================================================

if [[ -z "$PRIORITY_FILTER" || "$PRIORITY_FILTER" == "2" ]]; then

log_section "PRIORITY 2: MatchMaker, DualContext, ReMatch (8 experiments)"

# MatchMaker - all 3 models
run_experiment "matchmaker_granite" "MatchMaker with Granite" \
    "cd '${BASE_DIR}/match_maker' && python match_maker_runner.py \
        --property-lookup-file '$ITOPSGRAPH_LOOKUP' \
        --data-dir '$DATA_DIR' \
        --datasets 'ITOpsGraph' \
        --models '$GRANITE'"

run_experiment "matchmaker_llama" "MatchMaker with Llama" \
    "cd '${BASE_DIR}/match_maker' && python match_maker_runner.py \
        --property-lookup-file '$ITOPSGRAPH_LOOKUP' \
        --data-dir '$DATA_DIR' \
        --datasets 'ITOpsGraph' \
        --models '$LLAMA'"

run_experiment "matchmaker_mistral" "MatchMaker with Mistral" \
    "cd '${BASE_DIR}/match_maker' && python match_maker_runner.py \
        --property-lookup-file '$ITOPSGRAPH_LOOKUP' \
        --data-dir '$DATA_DIR' \
        --datasets 'ITOpsGraph' \
        --models '$MISTRAL'"

# DualContext - all 3 models
run_experiment "dualcontext_granite" "DualContext with Granite" \
    "cd '${BASE_DIR}/match_maker' && python dual_context_runner.py \
        --property-lookup-file '$ITOPSGRAPH_LOOKUP' \
        --data-dir '$DATA_DIR' \
        --datasets 'ITOpsGraph' \
        --models '$GRANITE'"

run_experiment "dualcontext_llama" "DualContext with Llama" \
    "cd '${BASE_DIR}/match_maker' && python dual_context_runner.py \
        --property-lookup-file '$ITOPSGRAPH_LOOKUP' \
        --data-dir '$DATA_DIR' \
        --datasets 'ITOpsGraph' \
        --models '$LLAMA'"

run_experiment "dualcontext_mistral" "DualContext with Mistral" \
    "cd '${BASE_DIR}/match_maker' && python dual_context_runner.py \
        --property-lookup-file '$ITOPSGRAPH_LOOKUP' \
        --data-dir '$DATA_DIR' \
        --datasets 'ITOpsGraph' \
        --models '$MISTRAL'"

# ReMatch - Llama and Mistral (Granite already done)
run_experiment "rematch_llama" "ReMatch with Llama" \
    "cd '${BASE_DIR}/rematch' && python rematch_runner.py \
        --property-lookup-file '$ITOPSGRAPH_LOOKUP' \
        --data-dir '$DATA_DIR' \
        --datasets 'ITOpsGraph' \
        --models '$LLAMA'"

run_experiment "rematch_mistral" "ReMatch with Mistral" \
    "cd '${BASE_DIR}/rematch' && python rematch_runner.py \
        --property-lookup-file '$ITOPSGRAPH_LOOKUP' \
        --data-dir '$DATA_DIR' \
        --datasets 'ITOpsGraph' \
        --models '$MISTRAL'"

fi  # End PRIORITY 2

# =============================================================================
# PRIORITY 3: UE-Reflect (2 experiments)
# =============================================================================

if [[ -z "$PRIORITY_FILTER" || "$PRIORITY_FILTER" == "3" ]]; then

log_section "PRIORITY 3: UE-Reflect (2 experiments)"

# UE-Reflect - Llama and Mistral (Granite already done)
run_experiment "ue_reflect_llama" "UE-Reflect with Llama" \
    "cd '${BASE_DIR}/map_reflect' && python ue_reflect.py \
        --property-lookup-file '$ITOPSGRAPH_LOOKUP' \
        --data-dir '$DATA_DIR' \
        --datasets 'ITOpsGraph' \
        --models '$LLAMA'"

run_experiment "ue_reflect_mistral" "UE-Reflect with Mistral" \
    "cd '${BASE_DIR}/map_reflect' && python ue_reflect.py \
        --property-lookup-file '$ITOPSGRAPH_LOOKUP' \
        --data-dir '$DATA_DIR' \
        --datasets 'ITOpsGraph' \
        --models '$MISTRAL'"

fi  # End PRIORITY 3

# =============================================================================
# VERIFICATION
# =============================================================================

log_section "VERIFICATION"

cd "$BASE_DIR"

log "Updating type mapping lookup..."
python create_type_mapping_lookup.py || log "Failed to update type mapping lookup"

log "Printing results..."
python print_all_results.py || log "Failed to print results"

# Show checkpoint summary
if [[ -f "$CHECKPOINT_FILE" ]]; then
    completed_count=$(wc -l < "$CHECKPOINT_FILE" | tr -d ' ')
    log "Completed experiments: $completed_count / 20"
    echo ""
    echo "Completed experiments:"
    cat "$CHECKPOINT_FILE"
fi

log_section "ALL ITOPSGRAPH EXPERIMENTS COMPLETED"
echo ""
echo "Results can be found in:"
echo "  - map_reflect/results/"
echo "  - react/results/"
echo "  - smt/results/"
echo "  - anytool/results/"
echo "  - match_maker/results/"
echo "  - rematch/results/"
echo ""
log "Finished at $(timestamp)"
