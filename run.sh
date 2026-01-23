#!/usr/bin/env bash
# =============================================================================
# Schema Mapping Unified Runner
# =============================================================================
#
# Consolidated script for running experiments, aggregating results, and
# calculating accuracy metrics for the schema mapping project.
#
# This script replaces:
#   - run_evaluations.sh      (experiment runner with filters)
#   - run_all_experiments.sh  (unified experiment runner)
#   - run_remote_experiments.sh (remote machine execution)
#   - aggregate_results.sh    (result aggregation)
#
# =============================================================================
# MODES OF OPERATION
# =============================================================================
#
#   (default)   Run experiments (if no mode flag specified)
#   --aggregate Aggregate results and generate LaTeX tables
#   --accuracy  Calculate and display accuracy metrics
#   --all       Run + Aggregate + Accuracy (full pipeline)
#
# =============================================================================
# USAGE EXAMPLES
# =============================================================================
#
#   # Run all experiments
#   ./run.sh
#
#   # Run specific method/dataset/model
#   ./run.sh --method map_reflect --dataset ITOps --model granite
#
#   # Dry run to see what would execute
#   ./run.sh --dry-run
#
#   # Run on remote machine (uses remote paths)
#   ./run.sh --remote
#
#   # Only aggregate existing results
#   ./run.sh --aggregate
#
#   # Only calculate accuracy
#   ./run.sh --accuracy
#
#   # Full pipeline: run + aggregate + accuracy
#   ./run.sh --all --method map_reflect
#
#   # Quick test with minimal delay
#   ./run.sh --quick --method react --dataset ITOps
#
# =============================================================================
# AVAILABLE METHODS
# =============================================================================
#
#   map_reflect  - Map-Reflect: semantic search + LLM reflection
#   react        - ReAct: agent-based approach with tool use
#   smt          - SMT: schema matching with transformers
#   anytool      - AnyTool: hierarchical tool selection (ITOps/NL2GQL only)
#   gnn          - GNN: graph neural network (inference only, no LLM)
#
# =============================================================================
# AVAILABLE DATASETS
# =============================================================================
#
#   ITOps        - IT Operations (Concert, Instana, Turbonomic)
#   NL2GQL       - Natural Language to GraphQL
#   OMOP_CMS     - OMOP healthcare with CMS data
#   OMOP_MIMIC   - OMOP healthcare with MIMIC data
#   OMOP_Synthea - OMOP healthcare with Synthea data
#
# =============================================================================
# AVAILABLE MODELS
# =============================================================================
#
#   granite      - IBM Granite 4 (watsonx/ibm/granite-4-h-small)
#   llama        - Meta Llama 3.3 70B (watsonx/meta-llama/llama-3-3-70b-instruct)
#   mistral      - Mistral Medium (mistralai/mistral-medium-2505)
#
# =============================================================================

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================

# Detect if running locally or remotely
if [[ -d "/home/bganesa1/code/graphql-schema-mapping" ]]; then
    IS_REMOTE=true
    BASE_DIR="/home/bganesa1/code/graphql-schema-mapping"
    VENV_PATH="${BASE_DIR}/.venv"
else
    IS_REMOTE=false
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    BASE_DIR="$SCRIPT_DIR"
    VENV_PATH="${BASE_DIR}/.venv"
fi

DATA_DIR="${BASE_DIR}/data"
LOG_DIR="${BASE_DIR}/logs"
RESULTS_DIR="${BASE_DIR}/results"

# -----------------------------------------------------------------------------
# Model Configuration
# -----------------------------------------------------------------------------
# Full model IDs for WatsonX and direct API calls
GRANITE_FULL="watsonx/ibm/granite-4-h-small"
LLAMA_FULL="watsonx/meta-llama/llama-3-3-70b-instruct"
MISTRAL_FULL="mistralai/mistral-medium-2505"

# Short names used by some runners (ReAct, SMT)
GRANITE_SHORT="granite"
LLAMA_SHORT="llama"
MISTRAL_SHORT="mistral-medium"

# -----------------------------------------------------------------------------
# Dataset Configuration
# -----------------------------------------------------------------------------
ALL_DATASETS="ITOps NL2GQL OMOP_CMS OMOP_MIMIC OMOP_Synthea"

# Datasets that support AnyTool (require hierarchy files)
ANYTOOL_DATASETS="ITOps NL2GQL"

# -----------------------------------------------------------------------------
# Method Configuration
# -----------------------------------------------------------------------------
ALL_METHODS="map_reflect react smt anytool gnn"

# =============================================================================
# DEFAULT OPTIONS
# =============================================================================

# Modes (can combine multiple)
MODE_RUN=false
MODE_AGGREGATE=false
MODE_ACCURACY=false

# Filters (empty = all)
FILTER_METHOD=""
FILTER_DATASET=""
FILTER_MODEL=""

# Execution options
DRY_RUN=false
VERBOSE=false
QUICK=false
FORCE_REMOTE=false
DELAY=30          # Seconds between experiments (avoids rate limiting)
QUICK_DELAY=5     # Reduced delay for quick mode

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

print_help() {
    cat << 'HELP_EOF'
Schema Mapping Unified Runner
=============================

MODES:
  (default)     Run experiments (if no mode specified)
  --aggregate   Aggregate results and generate LaTeX tables
  --accuracy    Calculate and display accuracy metrics
  --all         Run + Aggregate + Accuracy (full pipeline)

USAGE EXAMPLES:
  ./run.sh                                          # Run all experiments
  ./run.sh --method map_reflect --dataset ITOps    # Run with filters
  ./run.sh --dry-run                                # Preview commands
  ./run.sh --remote                                 # Run on remote machine
  ./run.sh --aggregate                              # Only aggregate results
  ./run.sh --accuracy                               # Only show accuracy
  ./run.sh --all --method react                     # Full pipeline for ReAct

METHODS:
  map_reflect  - Semantic search + LLM reflection
  react        - Agent-based approach with tool use
  smt          - Schema Matching with Transformers
  anytool      - Hierarchical tool selection (ITOps/NL2GQL only)
  gnn          - Graph Neural Network (inference only)

DATASETS:
  ITOps, NL2GQL, OMOP_CMS, OMOP_MIMIC, OMOP_Synthea

MODELS:
  granite (IBM Granite 4), llama (Meta Llama 3.3), mistral (Mistral Medium)

HELP_EOF
    echo "OPTIONS:"
    echo "  --run              Run experiments (default, can be omitted)"
    echo "  --aggregate        Aggregate results and generate LaTeX tables"
    echo "  --accuracy         Calculate accuracy metrics"
    echo "  --all              Run + Aggregate + Accuracy (full pipeline)"
    echo ""
    echo "  --method METHOD    Filter by method (map_reflect, react, smt, anytool, gnn)"
    echo "  --dataset DATASET  Filter by dataset (ITOps, NL2GQL, OMOP_CMS, OMOP_MIMIC, OMOP_Synthea)"
    echo "  --model MODEL      Filter by model (granite, llama, mistral)"
    echo ""
    echo "  --remote           Force remote machine paths (auto-detected normally)"
    echo "  --dry-run          Show what would run without executing"
    echo "  --verbose          Enable verbose output"
    echo "  --quick            Reduce delays between experiments"
    echo "  --delay SECONDS    Custom delay between experiments (default: 30)"
    echo ""
    echo "  --help             Show this help message"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        # Modes
        --run)
            MODE_RUN=true
            shift
            ;;
        --aggregate)
            MODE_AGGREGATE=true
            shift
            ;;
        --accuracy)
            MODE_ACCURACY=true
            shift
            ;;
        --all)
            MODE_RUN=true
            MODE_AGGREGATE=true
            MODE_ACCURACY=true
            shift
            ;;

        # Filters
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

        # Execution options
        --remote)
            FORCE_REMOTE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --quick)
            QUICK=true
            DELAY=$QUICK_DELAY
            shift
            ;;
        --delay)
            DELAY="$2"
            shift 2
            ;;

        --help|-h)
            print_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Default to --run if no mode specified
if [[ "$MODE_RUN" == "false" && "$MODE_AGGREGATE" == "false" && "$MODE_ACCURACY" == "false" ]]; then
    MODE_RUN=true
fi

# Override remote detection if forced
if [[ "$FORCE_REMOTE" == "true" ]]; then
    IS_REMOTE=true
    BASE_DIR="/home/bganesa1/code/graphql-schema-mapping"
    VENV_PATH="${BASE_DIR}/.venv"
    DATA_DIR="${BASE_DIR}/data"
    LOG_DIR="${BASE_DIR}/logs"
    RESULTS_DIR="${BASE_DIR}/results"
fi

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

log_subsection() {
    echo ""
    echo "--------------------------------------------"
    echo "$1"
    echo "--------------------------------------------"
}

# -----------------------------------------------------------------------------
# Check if experiment should run based on filters
# -----------------------------------------------------------------------------
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

    # Check model filter (normalize names)
    if [[ -n "$FILTER_MODEL" ]]; then
        local filter_normalized=$(echo "$FILTER_MODEL" | tr '[:upper:]' '[:lower:]' | sed 's/-.*$//')
        local model_normalized=$(echo "$model" | tr '[:upper:]' '[:lower:]' | sed 's/-.*$//')
        if [[ "$filter_normalized" != "$model_normalized" ]]; then
            return 1
        fi
    fi

    return 0
}

# -----------------------------------------------------------------------------
# Execute command with logging and delay
# -----------------------------------------------------------------------------
run_cmd() {
    local description="$1"
    shift
    local cmd="$*"

    log "Running: $description"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY-RUN] $cmd"
        return 0
    fi

    if [[ "$VERBOSE" == "true" ]]; then
        eval "$cmd"
    else
        eval "$cmd" 2>&1 | tee -a "${LOG_DIR}/experiment_$(date +%Y%m%d).log"
    fi

    # Delay between experiments to avoid rate limiting
    if [[ "$DELAY" -gt 0 ]]; then
        log "Waiting ${DELAY}s before next experiment..."
        sleep "$DELAY"
    fi
}

# -----------------------------------------------------------------------------
# Get full model ID from short name
# -----------------------------------------------------------------------------
get_model_id() {
    local short_name="$1"
    case "$short_name" in
        granite)  echo "$GRANITE_FULL" ;;
        llama)    echo "$LLAMA_FULL" ;;
        mistral)  echo "$MISTRAL_FULL" ;;
        *)        echo "$short_name" ;;
    esac
}

# -----------------------------------------------------------------------------
# Get short model name for methods that require it
# -----------------------------------------------------------------------------
get_model_short() {
    local name="$1"
    case "$name" in
        granite)  echo "$GRANITE_SHORT" ;;
        llama)    echo "$LLAMA_SHORT" ;;
        mistral)  echo "$MISTRAL_SHORT" ;;
        *)        echo "$name" ;;
    esac
}

# =============================================================================
# EXPERIMENT RUNNERS
# =============================================================================

# -----------------------------------------------------------------------------
# Map-Reflect: Semantic search + LLM reflection
# Supports: All datasets, All models
# -----------------------------------------------------------------------------
run_map_reflect() {
    local dataset="$1"
    local model="$2"

    if ! should_run "map_reflect" "$dataset" "$model"; then
        return 0
    fi

    local ds_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
    local model_id=$(get_model_id "$model")
    local api_lookup="${DATA_DIR}/${ds_lower}_api_lookup.json"

    if [[ ! -f "$api_lookup" ]]; then
        log "SKIP: API lookup not found: $api_lookup"
        return 0
    fi

    run_cmd "Map-Reflect: $dataset with $model" \
        "cd ${BASE_DIR}/map_reflect && python map_reflect_runner.py \
            --property-lookup-file '$api_lookup' \
            --data-dir '$DATA_DIR' \
            --datasets '$dataset' \
            --models '$model_id'"
}

# -----------------------------------------------------------------------------
# ReAct: Agent-based approach with tool use
# Supports: All datasets, All models
# -----------------------------------------------------------------------------
run_react() {
    local dataset="$1"
    local model="$2"

    if ! should_run "react" "$dataset" "$model"; then
        return 0
    fi

    local ds_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
    local model_short=$(get_model_short "$model")
    local schema_dir="${DATA_DIR}/${ds_lower}_graphql"

    run_cmd "ReAct: $dataset with $model" \
        "cd ${BASE_DIR}/react && python react_runner.py \
            --datasets '$dataset' \
            --models '$model_short' \
            --schema-dir '$schema_dir'"
}

# -----------------------------------------------------------------------------
# SMT: Schema Matching with Transformers
# Supports: All datasets, All models
# -----------------------------------------------------------------------------
run_smt() {
    local dataset="$1"
    local model="$2"

    if ! should_run "smt" "$dataset" "$model"; then
        return 0
    fi

    local ds_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
    local model_short=$(get_model_short "$model")
    local schema_dir="${DATA_DIR}/${ds_lower}_graphql"
    local chroma_db="${DATA_DIR}/chroma_db"
    local collection="${ds_lower}_graphql_schemas"

    if [[ ! -d "$schema_dir" ]]; then
        log "SKIP: GraphQL schemas not found: $schema_dir"
        return 0
    fi

    run_cmd "SMT: $dataset with $model" \
        "cd ${BASE_DIR}/smt && python smt_runner.py \
            --datasets '$dataset' \
            --models '$model_short' \
            --schema-dir '$schema_dir' \
            --chroma-db-path '$chroma_db' \
            --collection-name '$collection'"
}

# -----------------------------------------------------------------------------
# AnyTool: Hierarchical tool selection
# Supports: ITOps, NL2GQL only (require hierarchy files)
# -----------------------------------------------------------------------------
run_anytool() {
    local dataset="$1"
    local model="$2"

    if ! should_run "anytool" "$dataset" "$model"; then
        return 0
    fi

    local ds_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
    local model_id=$(get_model_id "$model")
    local hierarchy="${DATA_DIR}/${ds_lower}_hierarchy.json"

    if [[ ! -f "$hierarchy" ]]; then
        log "SKIP: Hierarchy file not found: $hierarchy"
        return 0
    fi

    run_cmd "AnyTool: $dataset with $model" \
        "cd ${BASE_DIR}/anytool && python anytool_runner.py \
            --datasets '$dataset' \
            --models '$model_id' \
            --data-dir '$DATA_DIR' \
            --hierarchy-file '$hierarchy'"
}

# -----------------------------------------------------------------------------
# GNN: Graph Neural Network (inference only)
# Supports: All datasets (no LLM required)
# -----------------------------------------------------------------------------
run_gnn() {
    local dataset="$1"

    if ! should_run "gnn" "$dataset" "gnn"; then
        return 0
    fi

    local ds_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
    local gnn_model="${DATA_DIR}/gnn_results/${ds_lower}_mapping_gnn_results.pkl"

    if [[ ! -f "$gnn_model" ]]; then
        log "SKIP: GNN model not found: $gnn_model"
        return 0
    fi

    run_cmd "GNN: $dataset (inference)" \
        "cd ${BASE_DIR}/gnn && python gnn_runner.py \
            --dataset '$ds_lower'"
}

# =============================================================================
# MODE: RUN EXPERIMENTS
# =============================================================================

run_experiments() {
    log_section "RUNNING EXPERIMENTS"

    echo ""
    echo "Configuration:"
    echo "  Base directory: $BASE_DIR"
    echo "  Remote mode:    $IS_REMOTE"
    echo "  Method filter:  ${FILTER_METHOD:-all}"
    echo "  Dataset filter: ${FILTER_DATASET:-all}"
    echo "  Model filter:   ${FILTER_MODEL:-all}"
    echo "  Dry run:        $DRY_RUN"
    echo "  Delay:          ${DELAY}s"
    echo ""

    # Create log directory
    mkdir -p "$LOG_DIR"

    # Activate virtual environment if available
    if [[ -f "${VENV_PATH}/bin/activate" ]]; then
        log "Activating virtual environment: $VENV_PATH"
        source "${VENV_PATH}/bin/activate"
    fi

    # Load environment variables
    if [[ -f "${BASE_DIR}/.env" ]]; then
        log "Loading environment variables from .env"
        export $(grep -v '^#' "${BASE_DIR}/.env" | xargs)
    fi

    local models="granite llama mistral"

    # -------------------------------------------------------------------------
    # Map-Reflect: All datasets, all models
    # -------------------------------------------------------------------------
    log_subsection "MAP-REFLECT EXPERIMENTS"
    for dataset in $ALL_DATASETS; do
        for model in $models; do
            run_map_reflect "$dataset" "$model"
        done
    done

    # -------------------------------------------------------------------------
    # ReAct: All datasets, all models
    # -------------------------------------------------------------------------
    log_subsection "REACT EXPERIMENTS"
    for dataset in $ALL_DATASETS; do
        for model in $models; do
            run_react "$dataset" "$model"
        done
    done

    # -------------------------------------------------------------------------
    # SMT: All datasets, all models
    # -------------------------------------------------------------------------
    log_subsection "SMT EXPERIMENTS"
    for dataset in $ALL_DATASETS; do
        for model in $models; do
            run_smt "$dataset" "$model"
        done
    done

    # -------------------------------------------------------------------------
    # AnyTool: ITOps and NL2GQL only
    # -------------------------------------------------------------------------
    log_subsection "ANYTOOL EXPERIMENTS"
    for dataset in $ANYTOOL_DATASETS; do
        for model in $models; do
            run_anytool "$dataset" "$model"
        done
    done

    # -------------------------------------------------------------------------
    # GNN: All datasets (no LLM)
    # -------------------------------------------------------------------------
    log_subsection "GNN EXPERIMENTS"
    for dataset in $ALL_DATASETS; do
        run_gnn "$dataset"
    done

    log_section "EXPERIMENTS COMPLETED"
    echo ""
    echo "Results saved in:"
    echo "  Map-Reflect: ${BASE_DIR}/map_reflect/results/"
    echo "  ReAct:       ${BASE_DIR}/react/results/"
    echo "  SMT:         ${BASE_DIR}/smt/results/"
    echo "  AnyTool:     ${BASE_DIR}/anytool/results/"
    echo "  GNN:         ${BASE_DIR}/gnn/results/"
    echo ""
    echo "Logs: ${LOG_DIR}/"
}

# =============================================================================
# MODE: AGGREGATE RESULTS
# =============================================================================

aggregate_results() {
    log_section "AGGREGATING RESULTS"

    # Check if evaluation script exists
    if [[ ! -f "${BASE_DIR}/eval/eval_schema_mapping.py" ]]; then
        echo "Error: eval/eval_schema_mapping.py not found"
        exit 1
    fi

    mkdir -p "$RESULTS_DIR"

    # Build dataset list
    local datasets
    if [[ -n "$FILTER_DATASET" ]]; then
        datasets="$FILTER_DATASET"
    else
        datasets="ITOps NL2GQL OMOP_CMS OMOP_MIMIC OMOP_Synthea"
    fi

    # Build method list
    local methods
    if [[ -n "$FILTER_METHOD" ]]; then
        methods="$FILTER_METHOD"
    else
        methods="SMT ReAct GNN AnyTool MapReflect"
    fi

    # Build model list
    local models
    if [[ -n "$FILTER_MODEL" ]]; then
        models="$FILTER_MODEL"
    else
        models="granite llama mistral-large"
    fi

    log "Datasets: $datasets"
    log "Methods:  $methods"
    log "Models:   $models"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] python ${BASE_DIR}/eval/eval_schema_mapping.py \\"
        echo "    --datasets $datasets \\"
        echo "    --methods $methods \\"
        echo "    --models $models \\"
        echo "    --output-dir '$RESULTS_DIR' \\"
        echo "    --base-dir '$BASE_DIR'"
        return 0
    fi

    python "${BASE_DIR}/eval/eval_schema_mapping.py" \
        --datasets $datasets \
        --methods $methods \
        --models $models \
        --output-dir "$RESULTS_DIR" \
        --base-dir "$BASE_DIR"

    log_section "AGGREGATION COMPLETED"
    echo ""
    echo "Results saved in: $RESULTS_DIR"
    echo "  - results_tables.tex (LaTeX tables for paper)"
}

# =============================================================================
# MODE: CALCULATE ACCURACY
# =============================================================================

calculate_accuracy() {
    log_section "CALCULATING ACCURACY"

    # Check if accuracy script exists
    if [[ ! -f "${BASE_DIR}/eval/calc_accuracy.py" ]]; then
        echo "Error: eval/calc_accuracy.py not found"
        exit 1
    fi

    # Build arguments
    local args=""
    if [[ -n "$FILTER_METHOD" ]]; then
        args="$args --methods $FILTER_METHOD"
    fi
    if [[ -n "$FILTER_DATASET" ]]; then
        args="$args --datasets $FILTER_DATASET"
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] python ${BASE_DIR}/eval/calc_accuracy.py $args --base-dir '$BASE_DIR'"
        return 0
    fi

    python "${BASE_DIR}/eval/calc_accuracy.py" $args --base-dir "$BASE_DIR"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

echo ""
echo "=============================================="
echo "Schema Mapping Unified Runner"
echo "=============================================="
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
log "Started at $(timestamp)"
log "Session ID: $TIMESTAMP"

# Run selected modes
if [[ "$MODE_RUN" == "true" ]]; then
    run_experiments
fi

if [[ "$MODE_AGGREGATE" == "true" ]]; then
    aggregate_results
fi

if [[ "$MODE_ACCURACY" == "true" ]]; then
    calculate_accuracy
fi

echo ""
log_section "ALL TASKS COMPLETED"
log "Finished at $(timestamp)"
echo ""
