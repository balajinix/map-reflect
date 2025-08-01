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
DATASETS="ITOps,NL2GQL"
METHODS="SMT,ReAct,GNN,AnyTool,MapReflect"
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
            echo "  --datasets DATASETS  Comma-separated list (default: ITOps,NL2GQL)"
            echo "  --methods METHODS    Comma-separated list (default: SMT,ReAct,GNN,AnyTool,MapReflect)"
            echo "  --models MODELS      Comma-separated list (default: granite,llama,mistral-large)"
            echo "  --output-dir DIR     Output directory (default: ./results)"
            echo "  --base-dir DIR       Base directory containing method subdirectories (default: .)"
            echo "  --clean              Clean output directory before starting"
            echo ""
            echo "This script aggregates existing results from:"
            echo "  smt/results/"
            echo "  react/results/"
            echo "  gnn/results/"
            echo "  anytool/results/"
            echo "  map_reflect/results/"
            echo ""
            echo "Expected file naming: {dataset}_{model}.json"
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

# Check for existing result files
echo "Checking for existing result files:"

check_method_results() {
    local method=$1
    local method_dir=""
    
    case $method in
        "SMT")
            method_dir="smt/results"
            ;;
        "ReAct")
            method_dir="react/results"
            ;;
        "GNN")
            method_dir="gnn/results"
            ;;
        "AnyTool")
            method_dir="anytool/results"
            ;;
        "MapReflect")
            method_dir="map_reflect/results"
            ;;
    esac
    
    if [ -d "$BASE_DIR/$method_dir" ]; then
        local count=$(find "$BASE_DIR/$method_dir" -name "*.json" 2>/dev/null | wc -l)
        if [ $count -gt 0 ]; then
            echo "  $method: $count result files found in $method_dir"
            # List the files
            find "$BASE_DIR/$method_dir" -name "*.json" 2>/dev/null | sed 's/^/    /'
        else
            echo "  $method: No result files found in $method_dir"
        fi
    else
        echo "  $method: Directory not found: $method_dir"
    fi
}

# Convert comma-separated methods to array for checking
IFS=',' read -ra METHOD_ARRAY <<< "$METHODS"
for method in "${METHOD_ARRAY[@]}"; do
    check_method_results "$method"
done

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
    echo "Generated files:"
    ls -la "$OUTPUT_DIR"
    
    echo ""
    echo "Quick Statistics Summary:"
    echo "========================"
    
    # Extract key statistics using Python
    python3 -c "
import json
import sys

try:
    with open('$OUTPUT_DIR/evaluation_statistics.json', 'r') as f:
        stats = json.load(f)
    
    overall = stats.get('overall', {})
    print(f'Total evaluations: {overall.get(\"total_evaluations\", 0)}')
    print(f'Successful evaluations: {overall.get(\"successful_evaluations\", 0)}')
    print(f'Success rate: {overall.get(\"success_rate\", 0.0):.1%}')
    print(f'Average accuracy: {overall.get(\"avg_accuracy\", 0.0):.3f}')
    print()
    
    print('Method Performance:')
    by_method = stats.get('by_method', {})
    for method in sorted(by_method.keys()):
        metrics = by_method[method]
        acc = metrics.get('avg_accuracy', 0.0)
        sr = metrics.get('success_rate', 0.0)
        count = metrics.get('count', 0)
        print(f'  {method:12}: {count:3d} results, Accuracy={acc:.3f}, Success={sr:.1%}')
    
    print()
    print('Dataset Performance:')
    by_dataset = stats.get('by_dataset', {})
    for dataset in sorted(by_dataset.keys()):
        metrics = by_dataset[dataset]
        acc = metrics.get('avg_accuracy', 0.0)
        count = metrics.get('count', 0)
        print(f'  {dataset:12}: {count:3d} results, Accuracy={acc:.3f}')
    
    print()
    print('Model Performance:')
    by_model = stats.get('by_model', {})
    for model in sorted(by_model.keys()):
        metrics = by_model[model]
        acc = metrics.get('avg_accuracy', 0.0)
        count = metrics.get('count', 0)
        print(f'  {model:12}: {count:3d} results, Accuracy={acc:.3f}')

except Exception as e:
    print(f'Error reading statistics: {e}')
    sys.exit(1)
"
    
    echo ""
    echo "Detailed results available in:"
    echo "  $OUTPUT_DIR/detailed_results.csv"
    echo "  $OUTPUT_DIR/evaluation_statistics.json"
    echo "  $OUTPUT_DIR/dataset_tables.tex"
    
else
    echo ""
    echo "Error: No results were generated. Check the log above for errors."
    exit 1
fi

echo ""
echo "Result aggregation complete!"
