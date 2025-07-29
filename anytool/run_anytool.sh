#!/bin/bash

# Run script for AnyTool Schema Mapping Method
# Usage: ./run_anytool.sh [options]

set -e  # Exit on any error

echo "Starting AnyTool Schema Mapping Evaluation"
echo "=========================================="

# Check if we're in the anytool directory
if [ ! -f "anytool_runner.py" ]; then
    echo "Error: anytool_runner.py not found. Please run this script from the anytool/ directory."
    echo "Usage: cd anytool && ./run_anytool.sh"
    exit 1
fi

# Check if shared dataset loader exists (one level up)
if [ ! -f "../data/dataset_loader.py" ]; then
    echo "Error: shared dataset loader not found at ../data/dataset_loader.py"
    exit 1
fi

# Check if hierarchy file exists
if [ ! -f "../data/hierarchy.json" ]; then
    echo "Error: hierarchy.json not found at ../data/hierarchy.json"
    exit 1
fi

# Create data directory if it doesn't exist (one level up)
mkdir -p ../data

# Default parameters
DATASETS="ITOps"
MODELS="mistral-large"
HIERARCHY_FILE="../data/hierarchy.json"
SCHEMA_DIR="../data/graphql"
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
        --hierarchy-file)
            HIERARCHY_FILE="$2"
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
            echo "  --datasets DATASETS      Comma-separated list of datasets (default: ITOps)"
            echo "  --models MODELS          Comma-separated list of models (default: mistral-large)"
            echo "  --hierarchy-file FILE    Path to hierarchy.json file (default: ../data/hierarchy.json)"
            echo "  --schema-dir DIR         Directory containing GraphQL schemas (default: ../data/graphql)"
            echo "  --quick                  Quick evaluation with minimal samples"
            echo "  --verbose                Enable verbose logging"
            echo "  --help                   Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --quick                                           # Quick evaluation"
            echo "  $0 --datasets ITOps,NL2GQL --models granite          # Multiple datasets and models"
            echo "  $0 --verbose                                         # Enable debug logging"
            echo "  $0 --hierarchy-file /path/to/hierarchy.json          # Custom hierarchy file"
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
echo "   Hierarchy file: $HIERARCHY_FILE"
echo "   Schema directory: $SCHEMA_DIR"
echo "   Quick mode: $QUICK"
echo "   Verbose: $VERBOSE"
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: No virtual environment detected. Consider activating one."
fi

# Check for required Python packages
echo "Checking Python dependencies..."
python3 -c "import langgraph, litellm, langchain_core" 2>/dev/null || {
    echo "Error: Required packages not found. Please install:"
    echo "  pip install langgraph litellm langchain-core"
    exit 1
}

# Check for AnyTool components
if [ ! -f "anytool_agent.py" ]; then
    echo "Warning: anytool_agent.py not found. AnyTool functionality may be limited."
fi

if [ ! -f "$HIERARCHY_FILE" ]; then
    echo "Error: Hierarchy file $HIERARCHY_FILE not found"
    echo "AnyTool requires a valid hierarchy.json file to function"
    exit 1
fi
echo "Hierarchy file found: $HIERARCHY_FILE"

if [ ! -d "$SCHEMA_DIR" ]; then
    echo "Warning: Schema directory $SCHEMA_DIR not found"
    echo "AnyTool evaluation will continue but may use limited schema information"
fi

if [ -f ".env" ]; then
    echo "Found .env file - environment variables will be loaded"
else
    echo "No .env file found - using system environment variables"
fi

# Create necessary directories
mkdir -p results

# Build command line arguments for Python script
ARGS=""

ARGS="$ARGS --datasets $DATASETS"
ARGS="$ARGS --models $MODELS"
ARGS="$ARGS --hierarchy-file $HIERARCHY_FILE"
ARGS="$ARGS --schema-dir $SCHEMA_DIR"

if [ "$QUICK" = true ]; then
    ARGS="$ARGS --quick"
fi

if [ "$VERBOSE" = true ]; then
    ARGS="$ARGS --verbose"
fi

echo "Running AnyTool evaluation with args: $ARGS"
echo ""

# Check if anytool_runner.py has a main function that accepts these arguments
if ! grep -q "def main" anytool_runner.py; then
    echo "Warning: anytool_runner.py may not have a main function with argument parsing."
    echo "You may need to modify anytool_runner.py to accept command line arguments."
fi

# Run the AnyTool evaluation
python3 anytool_runner.py $ARGS

# Check if results were generated
if [ -d "results" ]; then
    echo ""
    echo "AnyTool evaluation completed successfully!"
    echo "Results saved in: ./results/"
    echo ""
    echo "Generated files:"
    ls -la results/ | grep ".json"
    
    # Count results
    result_count=$(ls -1 results/*.json 2>/dev/null | wc -l)
    echo "   Generated $result_count result files"
else
    echo ""
    echo "No results directory found. Check for errors above."
    exit 1
fi

echo ""
echo "AnyTool Schema Mapping Evaluation Complete!"
echo "=========================================="
