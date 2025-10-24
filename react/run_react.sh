#!/bin/bash

# Run script for ReAct Schema Mapping Method
# Usage: ./run_react.sh [options]

set -e  # Exit on any error

echo "Starting ReAct Schema Mapping Evaluation"
echo "============================================"

# Check if we're in the react directory
if [ ! -f "react_runner.py" ]; then
    echo "Error: react_runner.py not found. Please run this script from the react/ directory."
    echo "Usage: cd react && ./run_react.sh"
    exit 1
fi

# Check if shared dataset loader exists (one level up)
if [ ! -f "../data/dataset_loader.py" ]; then
    echo "Error: shared dataset loader not found at ../data/dataset_loader.py"
    exit 1
fi

# Create data directory if it doesn't exist (one level up)
mkdir -p ../data

# Default parameters - match evaluation script defaults
DATASETS="itops,nl2gql"
DATASETS="itops"
MODELS="granite,llama,mistral-large"
#MODELS="granite"
SCHEMA_DIR="../data/itops_graphql"
QUICK=false
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --datasets)
            DATASETS="$2"
            shift 2
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        --schema-dir)
            SCHEMA_DIR="$2"
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
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --datasets DATASETS    Comma-separated list of datasets (default: itops,nl2gql)"
            echo "  --models MODELS        Comma-separated list of models (default: granite,llama,mistral-large)"
            echo "  --schema-dir DIR       Directory containing GraphQL schemas (default: ../data/graphql)"
            echo "  --quick               Quick evaluation with minimal samples"
            echo "  --verbose             Enable verbose logging"
            echo "  --help                Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --quick                           # Quick evaluation"
            echo "  $0 --datasets itops --models llama   # Specific dataset and model"
            echo "  $0 --verbose                         # Enable debug logging"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Display configuration
echo "Configuration:"
echo "   Datasets: $DATASETS"
echo "   Models: $MODELS"
echo "   Schema directory: $SCHEMA_DIR"
echo "   Quick mode: $QUICK"
echo "   Verbose: $VERBOSE"
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: No virtual environment detected. Consider activating one."
fi

# Check if the ReAct agent files exist
if [ ! -d "app" ]; then
    echo "Error: app directory not found in react directory"
    echo "Make sure the ReAct agent implementation is available"
    exit 1
fi

# Check schema directory
if [ ! -d "$SCHEMA_DIR" ]; then
    echo "Warning: Schema directory $SCHEMA_DIR not found"
    echo "ReAct evaluation will continue but may use mock schemas"
fi

# Build command line arguments
ARGS=""

# Convert comma-separated lists to space-separated for Python argparse
DATASET_ARGS=$(echo $DATASETS | tr ',' ' ')
MODEL_ARGS=$(echo $MODELS | tr ',' ' ')

ARGS="$ARGS --datasets $DATASET_ARGS"
ARGS="$ARGS --models $MODEL_ARGS"
ARGS="$ARGS --schema-dir $SCHEMA_DIR"

if [ "$QUICK" = true ]; then
    ARGS="$ARGS --quick"
fi

if [ "$VERBOSE" = true ]; then
    ARGS="$ARGS --verbose"
fi

echo "Running ReAct evaluation with args: $ARGS"
echo ""

# Run the ReAct evaluation
python3 react_runner.py $ARGS

# Check if results were generated
if [ -d "results" ]; then
    echo ""
    echo "ReAct evaluation completed successfully!"
    echo "Results saved in: ./results/"
    echo ""
    echo "Generated files:"
    ls -la results/ | grep -E "\.(json|jsonl)$"
else
    echo ""
    echo "No results directory found. Check for errors above."
    exit 1
fi

echo ""
echo "ReAct Schema Mapping Evaluation Complete!"
echo "============================================"
