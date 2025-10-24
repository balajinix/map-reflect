#!/bin/bash

# Schema Mapping Evaluation - Result Aggregator
# This script only aggregates existing results from method directories

set -e

echo "Schema Mapping Result Aggregator"
echo "================================"

# Check if evaluation script exists
if [ ! -f "eval_schema_mapping.py" ]; then
    echo "Error: eval_schema_mapping.py not found. Please run this script from the root directory."
    exit 1
fi

# Default parameters
DATASETS="itops,nl2gql"
#DATASETS="itops"
METHODS="smt,react,gnn,anytool,map_reflect"
MODELS="granite,llama,mistral-large"
OUTPUT_DIR="./results"
BASE_DIR="."
CLEAN=false

# Parse command line arguments
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
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --base-dir)
            BASE_DIR="$2"
            shift 2
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --datasets DATASETS  Comma-separated list (default: itops,nl2gql)"
            echo "  --methods METHODS    Comma-separated list (default: smt,react,gnn,anytool,map_reflect)"
            echo "  --models MODELS      Comma-separated list (default: granite,llama,mistral-large)"
            echo "  --output-dir DIR     Output directory (default: ./results)"
            echo "  --base-dir DIR       Base directory containing methods/ folder (default: .)"
            echo "  --clean              Clean output directory before starting"
            echo ""
            echo "This script aggregates existing results from:"
            echo "  smt/results/"
            echo "  react/results/"
            echo "  gnn/results/"
            echo "  anytool/results/"
            echo "  map_reflect/results/"
            echo ""
            echo "Expected file naming: {dataset}_{model}.jsonl or {dataset}_{model}.json"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Convert comma-separated lists to space-separated for Python
DATASET_ARGS=$(echo "$DATASETS" | tr ',' ' ')
METHOD_ARGS=$(echo "$METHODS" | tr ',' ' ')
MODEL_ARGS=$(echo "$MODELS" | tr ',' ' ')

echo "Configuration:"
echo "  Datasets: $DATASETS"
echo "  Methods: $METHODS"
echo "  Models: $MODELS"
echo "  Output directory: $OUTPUT_DIR"
echo "  Base directory: $BASE_DIR"
echo ""

# Clean output directory if requested
if [ "$CLEAN" = true ] && [ -d "$OUTPUT_DIR" ]; then
    echo "Cleaning output directory: $OUTPUT_DIR"
    rm -rf "$OUTPUT_DIR"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the aggregation
echo "Running result aggregation..."
echo "================================"

python3 eval_schema_mapping.py \
    --datasets $DATASET_ARGS \
    --methods $METHOD_ARGS \
    --models $MODEL_ARGS \
    --output-dir "$OUTPUT_DIR" \
    --base-dir "$BASE_DIR"

# Check if results were generated
if [ -f "$OUTPUT_DIR/evaluation_statistics.json" ]; then
    echo ""
    echo "Aggregation completed successfully!"
    echo "Results saved in: $OUTPUT_DIR"
    echo ""
    #echo "Generated files:"
    #ls -la "$OUTPUT_DIR"
else
    echo ""
    echo "Error: No results were generated. Check the log above for errors."
    exit 1
fi

echo ""
echo "Result aggregation complete!"
