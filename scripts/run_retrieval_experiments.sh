#!/usr/bin/env bash
# =============================================================================
# Retrieval Mode Experiments Runner
# =============================================================================
# Runs UE-Reflect, MatchMaker, and ReMatch on all datasets in retrieval mode,
# where the target schema must be discovered through embedding-based retrieval
# (as opposed to oracle mode where it is given).
#
# Supports filtering by --method, --dataset, and --model. Use --dry-run to
# preview commands without executing.
#
# Usage:
#   ./scripts/run_retrieval_experiments.sh
#   ./scripts/run_retrieval_experiments.sh --method ue_reflect --dataset ITOpsGraph
#   ./scripts/run_retrieval_experiments.sh --model granite --dry-run
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BASE_DIR"

DATA_DIR="${BASE_DIR}/data/retrieval_mode"
OUTPUT_DIR="${BASE_DIR}/results/retrieval_mode"
LOOKUP_DIR="${BASE_DIR}/data"
LOG_DIR="${BASE_DIR}/logs/retrieval_mode"

# Create directories
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Models
LLAMA="watsonx/meta-llama/llama-3-3-70b-instruct"
GRANITE="watsonx/ibm/granite-4-h-small"

# Parse arguments
METHOD_FILTER=""
DATASET_FILTER=""
DRY_RUN=false
MODEL="llama"  # Default to llama

while [[ $# -gt 0 ]]; do
    case $1 in
        --method)
            METHOD_FILTER="$2"
            shift 2
            ;;
        --dataset)
            DATASET_FILTER="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Select model
if [[ "$MODEL" == "llama" ]]; then
    MODEL_ID="$LLAMA"
    MODEL_SHORT="llama"
elif [[ "$MODEL" == "granite" ]]; then
    MODEL_ID="$GRANITE"
    MODEL_SHORT="granite"
else
    echo "Unknown model: $MODEL (use llama or granite)"
    exit 1
fi

echo "=============================================="
echo "Retrieval Mode Experiments"
echo "=============================================="
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Model: $MODEL_ID"
echo "=============================================="

# Activate virtual environment if exists
if [[ -f "${BASE_DIR}/.venv/bin/activate" ]]; then
    source "${BASE_DIR}/.venv/bin/activate"
fi

get_lookup_file() {
    local dataset="$1"
    case "$dataset" in
        ITOpsGraph)
            echo "${LOOKUP_DIR}/itopsgraph_api_lookup.json"
            ;;
        NL2GQL)
            echo "${LOOKUP_DIR}/nl2gql_api_lookup.json"
            ;;
        OMOP_CMS)
            echo "${LOOKUP_DIR}/omop_cms_api_lookup.json"
            ;;
        OMOP_MIMIC)
            echo "${LOOKUP_DIR}/omop_mimic_api_lookup.json"
            ;;
        OMOP_Synthea)
            echo "${LOOKUP_DIR}/omop_synthea_api_lookup.json"
            ;;
        ToolBench)
            echo "${LOOKUP_DIR}/toolbench_api_lookup.json"
            ;;
        *)
            echo "${LOOKUP_DIR}/property_lookup.json"
            ;;
    esac
}

run_experiment() {
    local method="$1"
    local dataset="$2"
    local extra_args="${3:-}"

    local lookup_file=$(get_lookup_file "$dataset")

    # Apply filters
    if [[ -n "$METHOD_FILTER" && "$method" != "$METHOD_FILTER" ]]; then
        return 0
    fi
    if [[ -n "$DATASET_FILTER" && "$dataset" != "$DATASET_FILTER" ]]; then
        return 0
    fi

    echo ""
    echo ">>> Running $method on $dataset (retrieval mode)"
    echo "    Lookup file: $lookup_file"

    local cmd=""
    local output_subdir="${OUTPUT_DIR}/${method}_results"
    mkdir -p "$output_subdir"

    case "$method" in
        ue_reflect)
            cmd="python -m map_reflect.ue_reflect \
                --datasets $dataset \
                --models $MODEL_ID \
                --property-lookup-file $lookup_file \
                --data-dir $DATA_DIR \
                --output-dir $output_subdir \
                --top-k 10 \
                --use-sampling \
                --no-resume \
                $extra_args"
            ;;
        matchmaker)
            cmd="python -m match_maker.match_maker_runner \
                --datasets $dataset \
                --models $MODEL_SHORT \
                --property-lookup-file $lookup_file \
                --data-dir $DATA_DIR \
                $extra_args"
            ;;
        rematch)
            cmd="python -m rematch.rematch_runner \
                --datasets $dataset \
                --models $MODEL_SHORT \
                --property-lookup-file $lookup_file \
                --data-dir $DATA_DIR \
                $extra_args"
            ;;
    esac

    if [[ -z "$cmd" ]]; then
        echo "    [SKIP] Method $method not supported for retrieval mode"
        return 0
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] $cmd"
    else
        local log_file="${LOG_DIR}/${method}_${dataset}_${MODEL_SHORT}.log"
        echo "Logging to: $log_file"
        eval "$cmd" 2>&1 | tee "$log_file"
    fi
}

# =============================================================================
# Run UE-Reflect experiments (primary method)
# =============================================================================

echo ""
echo "=============================================="
echo "UE-Reflect Experiments"
echo "=============================================="

run_experiment "ue_reflect" "ITOpsGraph"
run_experiment "ue_reflect" "NL2GQL"
run_experiment "ue_reflect" "OMOP_CMS"
run_experiment "ue_reflect" "OMOP_MIMIC"
run_experiment "ue_reflect" "OMOP_Synthea"

# =============================================================================
# Run MatchMaker experiments
# =============================================================================

echo ""
echo "=============================================="
echo "MatchMaker Experiments"
echo "=============================================="

run_experiment "matchmaker" "ITOpsGraph"
run_experiment "matchmaker" "NL2GQL"
run_experiment "matchmaker" "OMOP_CMS"
run_experiment "matchmaker" "OMOP_MIMIC"
run_experiment "matchmaker" "OMOP_Synthea"

# =============================================================================
# Run ReMatch experiments
# =============================================================================

echo ""
echo "=============================================="
echo "ReMatch Experiments"
echo "=============================================="

run_experiment "rematch" "ITOpsGraph"
run_experiment "rematch" "NL2GQL"
run_experiment "rematch" "OMOP_CMS"
run_experiment "rematch" "OMOP_MIMIC"
run_experiment "rematch" "OMOP_Synthea"

echo ""
echo "=============================================="
echo "All retrieval mode experiments completed!"
echo "Results in: $OUTPUT_DIR"
echo "=============================================="
