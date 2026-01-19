#!/bin/bash
# Remote experiment runner for schema mapping
# This script runs experiments sequentially to avoid WatsonX rate limiting
# Usage: ./run_remote_experiments.sh [--method METHOD] [--model MODEL] [--dataset DATASET]

set -e

# Configuration
BASE_DIR="/home/bganesa1/code/graphql-schema-mapping"
DATA_DIR="${BASE_DIR}/data"
LOG_DIR="${BASE_DIR}/experiment_logs"
VENV_PATH="${BASE_DIR}/.venv"

# Models to test (name as expected by each runner)
GRANITE4="watsonx/ibm/granite-4-h-small"
LLAMA="watsonx/meta-llama/llama-3-3-70b-instruct"
MISTRAL="mistralai/mistral-medium-2505"

# ReAct uses different model names
REACT_GRANITE="granite"
REACT_LLAMA="llama"
REACT_MISTRAL="mistral"

# Create log directory
mkdir -p "$LOG_DIR"

# Activate virtual environment
source "${VENV_PATH}/bin/activate"

# Load environment variables
if [ -f "${BASE_DIR}/.env" ]; then
    export $(grep -v '^#' "${BASE_DIR}/.env" | xargs)
fi

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "Starting experiments at $(date)"
echo "=========================================="

# Function to run Map-Reflect experiment
run_map_reflect() {
    local dataset=$1
    local model=$2
    local api_lookup=$3
    local model_short=$(echo "$model" | sed 's/watsonx\///; s/\//_/g')
    local log_file="${LOG_DIR}/map_reflect_${dataset}_${model_short}_${TIMESTAMP}.log"

    echo ""
    echo "=========================================="
    echo "[$(date)] Map-Reflect: ${dataset} with ${model}"
    echo "Log: ${log_file}"
    echo "=========================================="

    cd "${BASE_DIR}/map_reflect"
    python map_reflect_runner.py \
        --property-lookup-file "${DATA_DIR}/${api_lookup}" \
        --data-dir "${DATA_DIR}" \
        --datasets "${dataset}" \
        --models "${model}" \
        2>&1 | tee "${log_file}"

    echo "[$(date)] Completed: Map-Reflect ${dataset} with ${model}"
    echo ""

    # Small delay between experiments to avoid rate limiting
    sleep 30
}

# Function to run AnyTool experiment
run_anytool() {
    local dataset=$1
    local model=$2
    local hierarchy=$3
    local model_short=$(echo "$model" | sed 's/watsonx\///; s/\//_/g')
    local log_file="${LOG_DIR}/anytool_${dataset}_${model_short}_${TIMESTAMP}.log"

    echo ""
    echo "=========================================="
    echo "[$(date)] AnyTool: ${dataset} with ${model}"
    echo "Log: ${log_file}"
    echo "=========================================="

    cd "${BASE_DIR}/anytool"
    python anytool_runner.py \
        --datasets "${dataset}" \
        --models "${model}" \
        --data-dir "${DATA_DIR}" \
        --hierarchy-file "${DATA_DIR}/${hierarchy}" \
        2>&1 | tee "${log_file}"

    echo "[$(date)] Completed: AnyTool ${dataset} with ${model}"
    echo ""

    # Small delay between experiments to avoid rate limiting
    sleep 30
}

# ============================================
# MAP-REFLECT EXPERIMENTS
# ============================================
echo ""
echo "############################################"
echo "STARTING MAP-REFLECT EXPERIMENTS"
echo "############################################"

# ITOps
run_map_reflect "ITOps" "$GRANITE4" "itops_api_lookup.json"
run_map_reflect "ITOps" "$LLAMA" "itops_api_lookup.json"
run_map_reflect "ITOps" "$MISTRAL" "itops_api_lookup.json"

# NL2GQL
run_map_reflect "NL2GQL" "$GRANITE4" "nl2gql_api_lookup.json"
run_map_reflect "NL2GQL" "$LLAMA" "nl2gql_api_lookup.json"
run_map_reflect "NL2GQL" "$MISTRAL" "nl2gql_api_lookup.json"

# OMOP datasets
run_map_reflect "OMOP_CMS" "$GRANITE4" "omop_cms_api_lookup.json"
run_map_reflect "OMOP_CMS" "$LLAMA" "omop_cms_api_lookup.json"
run_map_reflect "OMOP_CMS" "$MISTRAL" "omop_cms_api_lookup.json"

run_map_reflect "OMOP_MIMIC" "$GRANITE4" "omop_mimic_api_lookup.json"
run_map_reflect "OMOP_MIMIC" "$LLAMA" "omop_mimic_api_lookup.json"
run_map_reflect "OMOP_MIMIC" "$MISTRAL" "omop_mimic_api_lookup.json"

run_map_reflect "OMOP_Synthea" "$GRANITE4" "omop_synthea_api_lookup.json"
run_map_reflect "OMOP_Synthea" "$LLAMA" "omop_synthea_api_lookup.json"
run_map_reflect "OMOP_Synthea" "$MISTRAL" "omop_synthea_api_lookup.json"

# ============================================
# ANYTOOL EXPERIMENTS
# ============================================
echo ""
echo "############################################"
echo "STARTING ANYTOOL EXPERIMENTS"
echo "############################################"

# ITOps (only these work with AnyTool hierarchy format)
run_anytool "ITOps" "$GRANITE4" "itops_hierarchy.json"
run_anytool "ITOps" "$LLAMA" "itops_hierarchy.json"
run_anytool "ITOps" "$MISTRAL" "itops_hierarchy.json"

# NL2GQL
run_anytool "NL2GQL" "$GRANITE4" "nl2gql_hierarchy.json"
run_anytool "NL2GQL" "$LLAMA" "nl2gql_hierarchy.json"
run_anytool "NL2GQL" "$MISTRAL" "nl2gql_hierarchy.json"

# Function to run ReAct experiment
run_react() {
    local dataset=$1
    local model=$2
    local log_file="${LOG_DIR}/react_${dataset}_${model}_${TIMESTAMP}.log"

    echo ""
    echo "=========================================="
    echo "[$(date)] ReAct: ${dataset} with ${model}"
    echo "Log: ${log_file}"
    echo "=========================================="

    cd "${BASE_DIR}/react"
    python react_runner.py \
        --datasets "${dataset}" \
        --models "${model}" \
        --schema-dir "${DATA_DIR}/graphql" \
        2>&1 | tee "${log_file}"

    echo "[$(date)] Completed: ReAct ${dataset} with ${model}"
    echo ""

    # Small delay between experiments to avoid rate limiting
    sleep 30
}

# ============================================
# REACT EXPERIMENTS
# ============================================
echo ""
echo "############################################"
echo "STARTING REACT EXPERIMENTS"
echo "############################################"

# ITOps
run_react "ITOps" "$REACT_GRANITE"
run_react "ITOps" "$REACT_LLAMA"
run_react "ITOps" "$REACT_MISTRAL"

# NL2GQL
run_react "NL2GQL" "$REACT_GRANITE"
run_react "NL2GQL" "$REACT_LLAMA"
run_react "NL2GQL" "$REACT_MISTRAL"

# ============================================
# SUMMARY
# ============================================
echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETED at $(date)"
echo "=========================================="
echo ""
echo "Results saved in:"
echo "  - Map-Reflect: ${BASE_DIR}/map_reflect/results/"
echo "  - AnyTool: ${BASE_DIR}/anytool/results/"
echo "  - ReAct: ${BASE_DIR}/react/results/"
echo ""
echo "Logs saved in: ${LOG_DIR}/"
