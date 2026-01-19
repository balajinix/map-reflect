#!/bin/bash
# =============================================================================
# Schema Mapping Experiments Runner
# =============================================================================
# Comprehensive script to run all schema mapping experiments for VLDB paper.
# Captures all model/dataset combinations for reproducibility.
#
# Usage:
#   ./run_evaluations.sh                      # Run all experiments
#   ./run_evaluations.sh --method map_reflect # Run only Map-Reflect
#   ./run_evaluations.sh --dataset ITOps      # Run only ITOps dataset
#   ./run_evaluations.sh --model llama        # Run only Llama model
#   ./run_evaluations.sh --dry-run            # Show what would be run
# =============================================================================

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
LOG_DIR="${SCRIPT_DIR}/experiment_logs"

# Models - Full model IDs for different methods
GRANITE4_WATSONX="watsonx/ibm/granite-4-h-small"
LLAMA_WATSONX="watsonx/meta-llama/llama-3-3-70b-instruct"
MISTRAL_DIRECT="mistralai/mistral-medium-2505"

# Short names for ReAct (uses different format)
GRANITE_SHORT="granite"
LLAMA_SHORT="llama"
MISTRAL_SHORT="mistral-medium"

# Datasets
ALL_DATASETS="ITOps NL2GQL OMOP_CMS OMOP_MIMIC OMOP_Synthea"

# Methods
ALL_METHODS="map_reflect anytool react smt gnn"

# Default filters (empty = all)
FILTER_METHOD=""
FILTER_DATASET=""
FILTER_MODEL=""

# Options
DRY_RUN=false
VERBOSE=false
DELAY=30

# =============================================================================
# PARSE ARGUMENTS
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --method)
            FILTER_METHOD="$2"
            shift 2
            ;;
        --dataset)
            FILTER_DATASET="$2"
            shift 2
            ;;
        --model)
            FILTER_MODEL="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --delay)
            DELAY="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --method METHOD    Run only specific method (map_reflect, anytool, react, smt, gnn)"
            echo "  --dataset DATASET  Run only specific dataset (ITOps, NL2GQL, OMOP_CMS, OMOP_MIMIC, OMOP_Synthea)"
            echo "  --model MODEL      Run only specific model (granite, llama, mistral)"
            echo "  --dry-run          Show what would be run without executing"
            echo "  --verbose          Enable verbose logging"
            echo "  --delay SECONDS    Delay between experiments (default: 30)"
            echo "  --help             Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

log() {
    echo "[$(timestamp)] $1"
}

should_run() {
    local method="$1"
    local dataset="$2"
    local model="$3"

    # Check method filter
    if [[ -n "$FILTER_METHOD" && "$FILTER_METHOD" != "$method" ]]; then
        return 1
    fi

    # Check dataset filter
    if [[ -n "$FILTER_DATASET" && "$FILTER_DATASET" != "$dataset" ]]; then
        return 1
    fi

    # Check model filter
    if [[ -n "$FILTER_MODEL" && "$FILTER_MODEL" != "$model" ]]; then
        return 1
    fi

    return 0
}

run_cmd() {
    local description="$1"
    shift
    local cmd="$@"

    log "Running: $description"
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY-RUN] $cmd"
    else
        if [[ "$VERBOSE" == "true" ]]; then
            eval "$cmd"
        else
            eval "$cmd" 2>&1 | tee -a "${LOG_DIR}/experiment_$(date +%Y%m%d).log"
        fi
        sleep "$DELAY"
    fi
}

# =============================================================================
# EXPERIMENT RUNNERS
# =============================================================================

run_map_reflect() {
    local dataset="$1"
    local model_id="$2"
    local model_name="$3"
    local ds_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')

    if ! should_run "map_reflect" "$dataset" "$model_name"; then
        return 0
    fi

    local api_lookup="${DATA_DIR}/${ds_lower}_api_lookup.json"

    if [[ ! -f "$api_lookup" ]]; then
        log "SKIP: API lookup not found for $dataset: $api_lookup"
        return 0
    fi

    run_cmd "Map-Reflect: $dataset with $model_name" \
        "cd ${SCRIPT_DIR}/map_reflect && python map_reflect_runner.py \
            --property-lookup-file '$api_lookup' \
            --data-dir '$DATA_DIR' \
            --datasets '$dataset' \
            --models '$model_id'"
}

run_anytool() {
    local dataset="$1"
    local model_id="$2"
    local model_name="$3"
    local ds_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')

    if ! should_run "anytool" "$dataset" "$model_name"; then
        return 0
    fi

    local hierarchy="${DATA_DIR}/${ds_lower}_hierarchy.json"

    if [[ ! -f "$hierarchy" ]]; then
        log "SKIP: Hierarchy file not found for $dataset: $hierarchy"
        return 0
    fi

    run_cmd "AnyTool: $dataset with $model_name" \
        "cd ${SCRIPT_DIR}/anytool && python anytool_runner.py \
            --datasets '$dataset' \
            --models '$model_id' \
            --data-dir '$DATA_DIR' \
            --hierarchy-file '$hierarchy'"
}

run_react() {
    local dataset="$1"
    local model_short="$2"
    local model_name="$3"
    local ds_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')

    if ! should_run "react" "$dataset" "$model_name"; then
        return 0
    fi

    local schema_dir="${DATA_DIR}/${ds_lower}_graphql"

    run_cmd "ReAct: $dataset with $model_name" \
        "cd ${SCRIPT_DIR}/react && python react_runner.py \
            --datasets '$dataset' \
            --models '$model_short' \
            --schema-dir '$schema_dir'"
}

run_smt() {
    local dataset="$1"
    local model_short="$2"
    local model_name="$3"
    local ds_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')

    if ! should_run "smt" "$dataset" "$model_name"; then
        return 0
    fi

    local schema_dir="${DATA_DIR}/${ds_lower}_graphql"
    local chroma_db="${DATA_DIR}/chroma_db"
    local collection="${ds_lower}_graphql_schemas"

    if [[ ! -d "$schema_dir" ]]; then
        log "SKIP: GraphQL schemas not found for $dataset: $schema_dir"
        return 0
    fi

    run_cmd "SMT: $dataset with $model_name" \
        "cd ${SCRIPT_DIR}/smt && python smt_runner.py \
            --datasets '$dataset' \
            --models '$model_short' \
            --schema-dir '$schema_dir' \
            --chroma-db-path '$chroma_db' \
            --collection-name '$collection'"
}

run_gnn() {
    local dataset="$1"
    local ds_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')

    if ! should_run "gnn" "$dataset" "gnn"; then
        return 0
    fi

    local gnn_model="${DATA_DIR}/gnn_results/${ds_lower}_mapping_gnn_results.pkl"

    if [[ ! -f "$gnn_model" ]]; then
        log "SKIP: GNN model not found for $dataset: $gnn_model"
        return 0
    fi

    run_cmd "GNN: $dataset (inference only)" \
        "cd ${SCRIPT_DIR}/gnn && python gnn_runner.py \
            --dataset '$ds_lower'"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

echo ""
echo "=============================================="
echo "Schema Mapping Experiments Runner"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Method filter: ${FILTER_METHOD:-all}"
echo "  Dataset filter: ${FILTER_DATASET:-all}"
echo "  Model filter: ${FILTER_MODEL:-all}"
echo "  Dry run: $DRY_RUN"
echo "  Delay: ${DELAY}s between experiments"
echo ""

# Create log directory
mkdir -p "$LOG_DIR"

# =============================================================================
# MAP-REFLECT EXPERIMENTS
# All datasets, all models
# =============================================================================
echo ""
echo "############################################"
echo "MAP-REFLECT EXPERIMENTS"
echo "############################################"

for dataset in $ALL_DATASETS; do
    run_map_reflect "$dataset" "$GRANITE4_WATSONX" "granite"
    run_map_reflect "$dataset" "$LLAMA_WATSONX" "llama"
    run_map_reflect "$dataset" "$MISTRAL_DIRECT" "mistral"
done

# =============================================================================
# ANYTOOL EXPERIMENTS
# Only ITOps and NL2GQL (have hierarchy files)
# =============================================================================
echo ""
echo "############################################"
echo "ANYTOOL EXPERIMENTS"
echo "############################################"

for dataset in ITOps NL2GQL; do
    run_anytool "$dataset" "$GRANITE4_WATSONX" "granite"
    run_anytool "$dataset" "$LLAMA_WATSONX" "llama"
    run_anytool "$dataset" "$MISTRAL_DIRECT" "mistral"
done

# =============================================================================
# REACT EXPERIMENTS
# All datasets, all models
# =============================================================================
echo ""
echo "############################################"
echo "REACT EXPERIMENTS"
echo "############################################"

for dataset in $ALL_DATASETS; do
    run_react "$dataset" "$GRANITE_SHORT" "granite"
    run_react "$dataset" "$LLAMA_SHORT" "llama"
    run_react "$dataset" "$MISTRAL_SHORT" "mistral"
done

# =============================================================================
# SMT EXPERIMENTS
# All datasets, all models
# =============================================================================
echo ""
echo "############################################"
echo "SMT EXPERIMENTS"
echo "############################################"

for dataset in $ALL_DATASETS; do
    run_smt "$dataset" "granite" "granite"
    run_smt "$dataset" "llama" "llama"
    run_smt "$dataset" "mistral-large" "mistral"
done

# =============================================================================
# GNN EXPERIMENTS
# All datasets (no LLM required for inference)
# =============================================================================
echo ""
echo "############################################"
echo "GNN EXPERIMENTS"
echo "############################################"

for dataset in $ALL_DATASETS; do
    run_gnn "$dataset"
done

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETED at $(timestamp)"
echo "=============================================="
echo ""
echo "Results locations:"
echo "  Map-Reflect: ${SCRIPT_DIR}/map_reflect/results/"
echo "  AnyTool:     ${SCRIPT_DIR}/anytool/results/"
echo "  ReAct:       ${SCRIPT_DIR}/react/results/"
echo "  SMT:         ${SCRIPT_DIR}/smt/results/"
echo "  GNN:         ${SCRIPT_DIR}/gnn/results/"
echo ""
echo "Logs: ${LOG_DIR}/"
echo ""
echo "To calculate accuracy:"
echo "  python calc_accuracy.py"
echo ""
