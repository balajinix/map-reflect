#!/usr/bin/env bash
# =============================================================================
# Unified Schema Mapping Experiment Runner
# =============================================================================
# Run all schema mapping methods (GNN, Map-Reflect, AnyTool, ReAct) across
# multiple datasets and models.
#
# Usage:
#   ./run_all_experiments.sh [OPTIONS]
#
# Examples:
#   ./run_all_experiments.sh --datasets ITOps --methods map_reflect --models granite4
#   ./run_all_experiments.sh --datasets ITOps,NL2GQL --methods all --models granite4,llama
#   ./run_all_experiments.sh --quick  # Quick test with minimal samples
# =============================================================================

set -e

# =============================================================================
# CONFIGURATION - Modify these defaults as needed
# =============================================================================

# Available datasets
ALL_DATASETS="ITOps,NL2GQL,OMOP_CMS,OMOP_MIMIC,OMOP_Synthea"

# Available methods
ALL_METHODS="gnn,map_reflect,anytool,react"

# Available models (short names -> full model IDs)
# Note: Using function instead of associative array for bash 3.x compatibility

# Default selections
DATASETS="ITOps"
METHODS="map_reflect"
MODELS="granite4"

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
LOG_DIR="${SCRIPT_DIR}/experiment_logs"

# Options
QUICK=false
VERBOSE=false
DRY_RUN=false
PARALLEL=false
DELAY_BETWEEN_EXPERIMENTS=30  # seconds

# =============================================================================
# PARSE COMMAND LINE ARGUMENTS
# =============================================================================

print_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Run schema mapping experiments across multiple methods, datasets, and models.

OPTIONS:
  --datasets DATASETS    Comma-separated list of datasets (default: ITOps)
                         Available: ITOps, NL2GQL, OMOP_CMS, OMOP_MIMIC, OMOP_Synthea
                         Use 'all' for all datasets

  --methods METHODS      Comma-separated list of methods (default: map_reflect)
                         Available: gnn, map_reflect, anytool, react
                         Use 'all' for all methods

  --models MODELS        Comma-separated list of models (default: granite4)
                         Available: granite4, llama, mistral
                         Use 'all' for all models

  --quick                Run quick evaluation with minimal samples
  --verbose              Enable verbose logging
  --dry-run              Show what would be run without executing
  --parallel             Run experiments in parallel (careful with rate limits!)
  --delay SECONDS        Delay between experiments (default: 30s)
  --help                 Show this help message

EXAMPLES:
  # Run Map-Reflect on ITOps with Granite 4
  $0 --datasets ITOps --methods map_reflect --models granite4

  # Run all methods on ITOps with Granite 4
  $0 --datasets ITOps --methods all --models granite4

  # Run Map-Reflect on all datasets with all models
  $0 --datasets all --methods map_reflect --models all

  # Quick test
  $0 --quick --datasets ITOps --methods map_reflect

  # Dry run to see what would execute
  $0 --dry-run --datasets all --methods all --models all

METHOD NOTES:
  - GNN: Requires pre-trained model and graph data for each dataset
  - Map-Reflect: Works with all datasets, uses semantic search + LLM
  - AnyTool: Requires hierarchy file, currently only ITOps and NL2GQL
  - ReAct: Requires toolkit package, uses agent-based approach

EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --datasets)
            DATASETS="$2"
            shift 2
            ;;
        --methods)
            METHODS="$2"
            shift 2
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        --quick)
            QUICK=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --delay)
            DELAY_BETWEEN_EXPERIMENTS="$2"
            shift 2
            ;;
        --help)
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

# Expand 'all' to full lists
[[ "$DATASETS" == "all" ]] && DATASETS="$ALL_DATASETS"
[[ "$METHODS" == "all" ]] && METHODS="$ALL_METHODS"
[[ "$MODELS" == "all" ]] && MODELS="granite4,llama,mistral"

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

get_model_id() {
    local short_name="$1"
    case "$short_name" in
        granite4) echo "ibm/granite-4-h-small" ;;
        llama)    echo "watsonx/meta-llama/llama-3-3-70b-instruct" ;;
        mistral)  echo "watsonx/mistralai/mistral-medium-2505" ;;
        *)        echo "$short_name" ;;  # Return as-is if not recognized
    esac
}

get_dataset_files() {
    local dataset="$1"
    local ds_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')

    # Return paths for various dataset-specific files
    echo "api_lookup=${DATA_DIR}/${ds_lower}_api_lookup.json"
    echo "hierarchy=${DATA_DIR}/${ds_lower}_hierarchy.json"
    echo "schema_dir=${DATA_DIR}/${ds_lower}_graphql"
    echo "gnn_model=${DATA_DIR}/gnn_results/${ds_lower}_mapping_gnn_results.pkl"
    echo "graph_data=${DATA_DIR}/${ds_lower}_graph_data.pkl"
}

check_method_dataset_compatibility() {
    local method="$1"
    local dataset="$2"

    case "$method" in
        gnn)
            # GNN needs pre-trained model
            local ds_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
            if [[ ! -f "${DATA_DIR}/gnn_results/${ds_lower}_mapping_gnn_results.pkl" ]]; then
                echo "skip:GNN model not found for $dataset"
                return 1
            fi
            ;;
        anytool)
            # AnyTool needs hierarchy file
            local ds_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
            if [[ ! -f "${DATA_DIR}/${ds_lower}_hierarchy.json" ]]; then
                echo "skip:Hierarchy file not found for $dataset"
                return 1
            fi
            ;;
        react)
            # ReAct needs toolkit package (check Python import)
            if ! python3 -c "from toolkit.solve import solve" 2>/dev/null; then
                echo "skip:toolkit package not installed"
                return 1
            fi
            ;;
    esac

    echo "ok"
    return 0
}

# =============================================================================
# METHOD RUNNERS
# =============================================================================

run_gnn() {
    local dataset="$1"
    local model_short="$2"
    local model_id=$(get_model_id "$model_short")
    local ds_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')

    log "Running GNN: $dataset with $model_short"

    local args="--dataset $ds_lower --models $model_short"
    [[ "$QUICK" == "true" ]] && args="$args --quick"
    [[ "$VERBOSE" == "true" ]] && args="$args --verbose"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY-RUN] cd ${SCRIPT_DIR}/gnn && python3 gnn_runner.py $args"
        return 0
    fi

    cd "${SCRIPT_DIR}/gnn"
    python3 gnn_runner.py $args
}

run_map_reflect() {
    local dataset="$1"
    local model_short="$2"
    local model_id=$(get_model_id "$model_short")
    local ds_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')

    log "Running Map-Reflect: $dataset with $model_short ($model_id)"

    local api_lookup="${DATA_DIR}/${ds_lower}_api_lookup.json"
    local schema_dir="${DATA_DIR}/${ds_lower}_graphql"

    local args="--datasets $dataset --models $model_id"
    args="$args --property-lookup-file $api_lookup"
    args="$args --data-dir $DATA_DIR"
    [[ -d "$schema_dir" ]] && args="$args --schema-dir $schema_dir"
    [[ "$QUICK" == "true" ]] && args="$args --quick"
    [[ "$VERBOSE" == "true" ]] && args="$args --verbose"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY-RUN] cd ${SCRIPT_DIR}/map_reflect && python3 map_reflect_runner.py $args"
        return 0
    fi

    cd "${SCRIPT_DIR}/map_reflect"
    python3 map_reflect_runner.py $args
}

run_anytool() {
    local dataset="$1"
    local model_short="$2"
    local model_id=$(get_model_id "$model_short")
    local ds_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')

    log "Running AnyTool: $dataset with $model_short ($model_id)"

    local hierarchy="${DATA_DIR}/${ds_lower}_hierarchy.json"
    local schema_dir="${DATA_DIR}/${ds_lower}_graphql"

    local args="--datasets $dataset --models $model_id"
    args="$args --hierarchy-file $hierarchy"
    args="$args --data-dir $DATA_DIR"
    [[ -d "$schema_dir" ]] && args="$args --schema-dir $schema_dir"
    [[ "$QUICK" == "true" ]] && args="$args --quick"
    [[ "$VERBOSE" == "true" ]] && args="$args --verbose"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY-RUN] cd ${SCRIPT_DIR}/anytool && python3 anytool_runner.py $args"
        return 0
    fi

    cd "${SCRIPT_DIR}/anytool"
    python3 anytool_runner.py $args
}

run_react() {
    local dataset="$1"
    local model_short="$2"
    local model_id=$(get_model_id "$model_short")
    local ds_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')

    log "Running ReAct: $dataset with $model_short ($model_id)"

    local schema_dir="${DATA_DIR}/${ds_lower}_graphql"

    local args="--datasets $dataset --models $model_id"
    [[ -d "$schema_dir" ]] && args="$args --schema-dir $schema_dir"
    [[ "$QUICK" == "true" ]] && args="$args --quick"
    [[ "$VERBOSE" == "true" ]] && args="$args --verbose"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY-RUN] cd ${SCRIPT_DIR}/react && python3 react_runner.py $args"
        return 0
    fi

    cd "${SCRIPT_DIR}/react"
    python3 react_runner.py $args
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

echo ""
echo "=============================================="
echo "Schema Mapping Unified Experiment Runner"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Datasets: $DATASETS"
echo "  Methods:  $METHODS"
echo "  Models:   $MODELS"
echo "  Quick:    $QUICK"
echo "  Verbose:  $VERBOSE"
echo "  Dry-run:  $DRY_RUN"
echo "  Delay:    ${DELAY_BETWEEN_EXPERIMENTS}s between experiments"
echo ""

# Create log directory
mkdir -p "$LOG_DIR"

# Track results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_LOG="${LOG_DIR}/experiment_summary_${TIMESTAMP}.log"
SUCCESSFUL=0
SKIPPED=0
FAILED=0

# Convert comma-separated lists to arrays
IFS=',' read -ra DATASET_ARRAY <<< "$DATASETS"
IFS=',' read -ra METHOD_ARRAY <<< "$METHODS"
IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"

# Calculate total experiments
TOTAL_EXPERIMENTS=0
for dataset in "${DATASET_ARRAY[@]}"; do
    for method in "${METHOD_ARRAY[@]}"; do
        for model in "${MODEL_ARRAY[@]}"; do
            ((TOTAL_EXPERIMENTS++))
        done
    done
done

log "Total experiments to run: $TOTAL_EXPERIMENTS"
echo ""

# Run experiments
CURRENT=0
for dataset in "${DATASET_ARRAY[@]}"; do
    dataset=$(echo "$dataset" | xargs)  # trim whitespace

    for method in "${METHOD_ARRAY[@]}"; do
        method=$(echo "$method" | xargs)

        for model in "${MODEL_ARRAY[@]}"; do
            model=$(echo "$model" | xargs)
            ((CURRENT++))

            log_section "[$CURRENT/$TOTAL_EXPERIMENTS] $method | $dataset | $model"

            # Check compatibility
            compat=$(check_method_dataset_compatibility "$method" "$dataset")
            if [[ "$compat" == skip:* ]]; then
                reason="${compat#skip:}"
                log "SKIPPING: $reason"
                ((SKIPPED++))
                echo "SKIP | $method | $dataset | $model | $reason" >> "$RESULTS_LOG"
                continue
            fi

            # Run the experiment
            LOG_FILE="${LOG_DIR}/${method}_${dataset}_${model}_${TIMESTAMP}.log"

            if [[ "$DRY_RUN" == "false" ]]; then
                log "Log file: $LOG_FILE"
            fi

            start_time=$(date +%s)

            case "$method" in
                gnn)
                    if run_gnn "$dataset" "$model" 2>&1 | tee "$LOG_FILE"; then
                        ((SUCCESSFUL++))
                        echo "SUCCESS | $method | $dataset | $model" >> "$RESULTS_LOG"
                    else
                        ((FAILED++))
                        echo "FAILED | $method | $dataset | $model" >> "$RESULTS_LOG"
                    fi
                    ;;
                map_reflect)
                    if run_map_reflect "$dataset" "$model" 2>&1 | tee "$LOG_FILE"; then
                        ((SUCCESSFUL++))
                        echo "SUCCESS | $method | $dataset | $model" >> "$RESULTS_LOG"
                    else
                        ((FAILED++))
                        echo "FAILED | $method | $dataset | $model" >> "$RESULTS_LOG"
                    fi
                    ;;
                anytool)
                    if run_anytool "$dataset" "$model" 2>&1 | tee "$LOG_FILE"; then
                        ((SUCCESSFUL++))
                        echo "SUCCESS | $method | $dataset | $model" >> "$RESULTS_LOG"
                    else
                        ((FAILED++))
                        echo "FAILED | $method | $dataset | $model" >> "$RESULTS_LOG"
                    fi
                    ;;
                react)
                    if run_react "$dataset" "$model" 2>&1 | tee "$LOG_FILE"; then
                        ((SUCCESSFUL++))
                        echo "SUCCESS | $method | $dataset | $model" >> "$RESULTS_LOG"
                    else
                        ((FAILED++))
                        echo "FAILED | $method | $dataset | $model" >> "$RESULTS_LOG"
                    fi
                    ;;
                *)
                    log "Unknown method: $method"
                    ((FAILED++))
                    echo "FAILED | $method | $dataset | $model | Unknown method" >> "$RESULTS_LOG"
                    ;;
            esac

            end_time=$(date +%s)
            duration=$((end_time - start_time))
            log "Completed in ${duration}s"

            # Delay between experiments (unless last one or dry-run)
            if [[ "$DRY_RUN" == "false" ]] && [[ $CURRENT -lt $TOTAL_EXPERIMENTS ]]; then
                log "Waiting ${DELAY_BETWEEN_EXPERIMENTS}s before next experiment..."
                sleep "$DELAY_BETWEEN_EXPERIMENTS"
            fi
        done
    done
done

# =============================================================================
# SUMMARY
# =============================================================================

log_section "EXPERIMENT SUMMARY"
echo ""
echo "Total experiments: $TOTAL_EXPERIMENTS"
echo "  Successful: $SUCCESSFUL"
echo "  Skipped:    $SKIPPED"
echo "  Failed:     $FAILED"
echo ""
echo "Results log: $RESULTS_LOG"
echo ""
echo "Result files by method:"
for method in "${METHOD_ARRAY[@]}"; do
    method=$(echo "$method" | xargs)
    result_dir="${SCRIPT_DIR}/${method}/results"
    if [[ -d "$result_dir" ]]; then
        count=$(ls -1 "$result_dir"/*.json 2>/dev/null | wc -l)
        echo "  $method: $count files in $result_dir"
    fi
done
echo ""
log "All experiments completed!"
