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

# Get absolute paths for data directory (one level up)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data"

DATASETS="ITOps"
MODELS="mistral-large"
HIERARCHY_FILE="$DATA_DIR/itops_hierarchy.json"
DATA_DIR_ARG="$DATA_DIR"
SCHEMA_DIR="$DATA_DIR/itops_graphql"
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
        --data-dir)
            DATA_DIR_ARG="$2"
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
            echo "  --hierarchy-file FILE    Path to hierarchy.json file (default: auto-detect)"
            echo "  --data-dir DIR           Directory containing shared data components (default: auto-detect)"
            echo "  --schema-dir DIR         Directory containing GraphQL schemas (default: auto-detect)"
            echo "  --quick                  Quick evaluation with minimal samples"
            echo "  --verbose                Enable verbose logging"
            echo "  --help                   Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --quick                                           # Quick evaluation"
            echo "  $0 --datasets ITOps,NL2GQL --models granite          # Multiple datasets and models"
            echo "  $0 --verbose                                         # Enable debug logging"
            echo "  $0 --hierarchy-file /path/to/hierarchy.json          # Custom hierarchy file"
            echo "  $0 --data-dir /path/to/data                          # Custom data directory"
            echo ""
            echo "Current directory: $SCRIPT_DIR"
            echo "Project root: $PROJECT_ROOT"
            echo "Default data directory: $DATA_DIR"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Auto-detect paths if not explicitly set
if [ -z "$HIERARCHY_FILE" ]; then
    if [ -f "$DATA_DIR/hierarchy.json" ]; then
        HIERARCHY_FILE="$DATA_DIR/hierarchy.json"
    elif [ -f "$SCRIPT_DIR/hierarchy.json" ]; then
        HIERARCHY_FILE="$SCRIPT_DIR/hierarchy.json"
    fi
fi

if [ -z "$DATA_DIR_ARG" ]; then
    if [ -d "$DATA_DIR" ] && [ -f "$DATA_DIR/dataset_loader.py" ]; then
        DATA_DIR_ARG="$DATA_DIR"
    elif [ -d "$SCRIPT_DIR/data" ] && [ -f "$SCRIPT_DIR/data/dataset_loader.py" ]; then
        DATA_DIR_ARG="$SCRIPT_DIR/data"
    fi
fi

if [ -z "$SCHEMA_DIR" ] && [ -n "$DATA_DIR_ARG" ]; then
    if [ -d "$DATA_DIR_ARG/graphql" ]; then
        SCHEMA_DIR="$DATA_DIR_ARG/graphql"
    fi
fi

# Display configuration
echo "Configuration:"
echo "   Datasets: $DATASETS"
echo "   Models: $MODELS"
echo "   Hierarchy file: $HIERARCHY_FILE"
echo "   Data directory: $DATA_DIR_ARG"
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

# Validate files if paths are specified
if [ -n "$HIERARCHY_FILE" ]; then
    if [ ! -f "$HIERARCHY_FILE" ]; then
        echo "Error: Hierarchy file $HIERARCHY_FILE not found"
        echo "AnyTool requires a valid hierarchy.json file to function"
        exit 1
    fi
    echo "Hierarchy file found: $HIERARCHY_FILE"
fi

if [ -n "$DATA_DIR_ARG" ]; then
    if [ ! -d "$DATA_DIR_ARG" ]; then
        echo "Error: Data directory $DATA_DIR_ARG not found"
        exit 1
    fi
    echo "Data directory found: $DATA_DIR_ARG"
    
    if [ ! -f "$DATA_DIR_ARG/dataset_loader.py" ]; then
        echo "Error: dataset_loader.py not found in $DATA_DIR_ARG"
        echo "AnyTool requires the shared dataset loader"
        exit 1
    fi
    echo "Dataset loader found: $DATA_DIR_ARG/dataset_loader.py"
fi

if [ -n "$SCHEMA_DIR" ] && [ ! -d "$SCHEMA_DIR" ]; then
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

# Only add path arguments if they are explicitly set or detected
if [ -n "$HIERARCHY_FILE" ]; then
    ARGS="$ARGS --hierarchy-file $HIERARCHY_FILE"
fi

if [ -n "$DATA_DIR_ARG" ]; then
    ARGS="$ARGS --data-dir $DATA_DIR_ARG"
fi

if [ -n "$SCHEMA_DIR" ]; then
    ARGS="$ARGS --schema-dir $SCHEMA_DIR"
fi

if [ "$QUICK" = true ]; then
    ARGS="$ARGS --quick"
fi

if [ "$VERBOSE" = true ]; then
    ARGS="$ARGS --verbose"
fi

echo "Running AnyTool evaluation with args: $ARGS"
echo ""

# Run the AnyTool evaluation
python3 anytool_runner.py $ARGS

# Check if results were generated
if [ -d "results" ]; then
    echo ""
    echo "AnyTool evaluation completed successfully!"
    echo "Results saved in: ./results/"
    echo ""
    echo "Generated files:"
    ls -la results/ | grep ".json" || echo "   No JSON files found"
    
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
