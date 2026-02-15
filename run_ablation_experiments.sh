#!/usr/bin/env bash
# =============================================================================
# Ablation Experiments Runner
# =============================================================================
#
# Runs Map-Reflect baseline experiments for UE-Reflect ablation study.
#
# Required experiments:
#   1. Map-Reflect on ITOpsGraph (Llama) - baseline for UE-Reflect ablation
#   2. UE-Reflect on NL2GQL (Llama) - complete coverage (optional)
#
# Model IDs:
#   - Llama: watsonx/meta-llama/llama-3-3-70b-instruct
#
# Usage:
#   Local:  ./run_ablation_experiments.sh
#   Remote: ./run_ablation_experiments.sh --remote
#   Background: nohup ./run_ablation_experiments.sh --remote > logs/ablation.log 2>&1 &
#
# Options:
#   --remote        Force remote machine paths
#   --map-reflect   Only run Map-Reflect experiments
#   --ue-reflect    Only run UE-Reflect experiments
#   --all-datasets  Run on all datasets (not just missing ones)
#
# =============================================================================

set -e

# Parse arguments
FORCE_REMOTE=false
RUN_MAP_REFLECT=true
RUN_UE_REFLECT=true
ALL_DATASETS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --remote)
            FORCE_REMOTE=true
            shift
            ;;
        --map-reflect)
            RUN_MAP_REFLECT=true
            RUN_UE_REFLECT=false
            shift
            ;;
        --ue-reflect)
            RUN_MAP_REFLECT=false
            RUN_UE_REFLECT=true
            shift
            ;;
        --all-datasets)
            ALL_DATASETS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--remote] [--map-reflect] [--ue-reflect] [--all-datasets]"
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
RESULTS_DIR="${BASE_DIR}/map_reflect/results"

# Create directories
mkdir -p "${LOG_DIR}"
mkdir -p "${RESULTS_DIR}"

# Model IDs
LLAMA="watsonx/meta-llama/llama-3-3-70b-instruct"

# Data files
ITOPSGRAPH_LOOKUP="${DATA_DIR}/itopsgraph_api_lookup.json"
NL2GQL_LOOKUP="${DATA_DIR}/property_lookup.json"
OMOP_LOOKUP="${DATA_DIR}/omop_api_lookup.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=============================================="
echo "Ablation Experiments Runner"
echo "=============================================="
echo "Environment: $([ "$IS_REMOTE" = true ] && echo "REMOTE" || echo "LOCAL")"
echo "Base directory: ${BASE_DIR}"
echo "Run Map-Reflect: ${RUN_MAP_REFLECT}"
echo "Run UE-Reflect: ${RUN_UE_REFLECT}"
echo "All datasets: ${ALL_DATASETS}"
echo "=============================================="

cd "${BASE_DIR}"

# Activate virtual environment if it exists
if [[ -f "${BASE_DIR}/.venv/bin/activate" ]]; then
    source "${BASE_DIR}/.venv/bin/activate"
    echo -e "${GREEN}Activated virtual environment${NC}"
fi

# Function to run experiment with logging
run_experiment() {
    local name="$1"
    local cmd="$2"
    local log_file="${LOG_DIR}/${name}_$(date +%Y%m%d_%H%M%S).log"

    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Running: ${name}${NC}"
    echo -e "${BLUE}Log: ${log_file}${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo "Command: ${cmd}"
    echo ""

    # Run with logging
    if eval "${cmd}" 2>&1 | tee "${log_file}"; then
        echo -e "${GREEN}SUCCESS: ${name}${NC}"
        return 0
    else
        echo -e "${RED}FAILED: ${name}${NC}"
        return 1
    fi
}

# =============================================================================
# Map-Reflect Experiments (baseline for UE-Reflect ablation)
# =============================================================================

if [[ "$RUN_MAP_REFLECT" == "true" ]]; then
    echo ""
    echo "=============================================="
    echo "MAP-REFLECT BASELINE EXPERIMENTS"
    echo "=============================================="

    # 1. Map-Reflect on ITOpsGraph (REQUIRED - missing baseline)
    echo -e "${YELLOW}[1/4] Map-Reflect on ITOpsGraph (Llama)${NC}"
    run_experiment "map_reflect_itopsgraph_llama" \
        "python -m map_reflect.map_reflect_runner \
            --datasets ITOpsGraph \
            --models ${LLAMA} \
            --property-lookup-file ${ITOPSGRAPH_LOOKUP} \
            --data-dir ${DATA_DIR}"

    if [[ "$ALL_DATASETS" == "true" ]]; then
        # 2. Map-Reflect on NL2GQL (already have results, but run for completeness)
        echo -e "${YELLOW}[2/4] Map-Reflect on NL2GQL (Llama)${NC}"
        run_experiment "map_reflect_nl2gql_llama" \
            "python -m map_reflect.map_reflect_runner \
                --datasets NL2GQL \
                --models ${LLAMA} \
                --property-lookup-file ${NL2GQL_LOOKUP} \
                --data-dir ${DATA_DIR}"

        # 3. Map-Reflect on OMOP datasets (already have results)
        echo -e "${YELLOW}[3/4] Map-Reflect on OMOP_Synthea (Llama)${NC}"
        run_experiment "map_reflect_omop_synthea_llama" \
            "python -m map_reflect.map_reflect_runner \
                --datasets OMOP_Synthea \
                --models ${LLAMA} \
                --property-lookup-file ${OMOP_LOOKUP} \
                --data-dir ${DATA_DIR}"

        echo -e "${YELLOW}[4/4] Map-Reflect on OMOP_MIMIC,OMOP_CMS (Llama)${NC}"
        run_experiment "map_reflect_omop_mimic_cms_llama" \
            "python -m map_reflect.map_reflect_runner \
                --datasets OMOP_MIMIC,OMOP_CMS \
                --models ${LLAMA} \
                --property-lookup-file ${OMOP_LOOKUP} \
                --data-dir ${DATA_DIR}"
    fi
fi

# =============================================================================
# UE-Reflect Experiments (complete coverage)
# =============================================================================

if [[ "$RUN_UE_REFLECT" == "true" ]]; then
    echo ""
    echo "=============================================="
    echo "UE-REFLECT EXPERIMENTS"
    echo "=============================================="

    # 1. UE-Reflect on NL2GQL (missing)
    echo -e "${YELLOW}[1/2] UE-Reflect on NL2GQL (Llama)${NC}"
    run_experiment "ue_reflect_nl2gql_llama" \
        "python -m map_reflect.ue_reflect \
            --datasets NL2GQL \
            --models ${LLAMA} \
            --property-lookup-file ${NL2GQL_LOOKUP} \
            --data-dir ${DATA_DIR} \
            --output-dir ${RESULTS_DIR} \
            --top-k 10 \
            --use-sampling"

    if [[ "$ALL_DATASETS" == "true" ]]; then
        # 2. UE-Reflect on ITOpsGraph (already have results, but run for completeness)
        echo -e "${YELLOW}[2/2] UE-Reflect on ITOpsGraph (Llama)${NC}"
        run_experiment "ue_reflect_itopsgraph_llama" \
            "python -m map_reflect.ue_reflect \
                --datasets ITOpsGraph \
                --models ${LLAMA} \
                --property-lookup-file ${ITOPSGRAPH_LOOKUP} \
                --data-dir ${DATA_DIR} \
                --output-dir ${RESULTS_DIR} \
                --top-k 10 \
                --use-sampling"
    fi
fi

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=============================================="
echo "ABLATION EXPERIMENTS COMPLETE"
echo "=============================================="
echo ""
echo "Results saved to: ${RESULTS_DIR}"
echo "Logs saved to: ${LOG_DIR}"
echo ""
echo "To evaluate results, run:"
echo "  python -m eval.calc_accuracy --methods Map-Reflect,UE-Reflect"
echo ""
echo "Expected ablation table after experiments:"
echo ""
echo "| Dataset     | Method       | API Acc | Field Acc | Delta |"
echo "|-------------|--------------|---------|-----------|-------|"
echo "| ITOpsGraph  | Map-Reflect  | ???     | ???       | -     |"
echo "| ITOpsGraph  | UE-Reflect   | 98.1%   | 76.6%     | +UQ   |"
echo "| Synthea     | Map-Reflect  | 100%    | 55.4%     | -     |"
echo "| Synthea     | UE-Reflect   | 100%    | 81.2%     | +25.8 |"
echo "| MIMIC       | Map-Reflect  | 97.8%   | 31.4%     | -     |"
echo "| MIMIC       | UE-Reflect   | 100%    | 42.9%     | +11.5 |"
echo "| CMS         | Map-Reflect  | 81.6%   | 30.3%     | -     |"
echo "| CMS         | UE-Reflect   | 100%    | 41.4%     | +11.1 |"
echo ""
