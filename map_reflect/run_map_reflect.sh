#!/bin/bash

# Run script for MapReflect Schema Mapping Method
# Usage: ./run_map_reflect.sh [options]

set -e  # Exit on any error

echo "Starting MapReflect Schema Mapping Evaluation"
echo "============================================="

# Check if we're in the map_reflect directory
if [ ! -f "map_reflect_runner.py" ]; then
    echo "Error: map_reflect_runner.py not found. Please run this script from the map_reflect/ directory."
    echo "Usage: cd map_reflect && ./run_map_reflect.sh"
    exit 1
fi

# Get absolute paths for data directory (one level up)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data"

DATASETS="ITOps"
MODELS="watsonx/mistralai/mistral-large"
PROPERTY_LOOKUP_FILE="$DATA_DIR/itops_api_lookup.json"
DATA_DIR_ARG="$DATA_DIR"
SCHEMA_DIR="$DATA_DIR/graphql"
REBUILD_INDEX=false
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
        --property-lookup-file)
            PROPERTY_LOOKUP_FILE="$2"
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
        --rebuild-index)
            REBUILD_INDEX=true
            shift
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
            echo "  --datasets DATASETS          Comma-separated list of datasets (default: ITOps)"
            echo "  --models MODELS              Comma-separated list of models (default: watsonx/mistralai/mistral-large)"
            echo "  --property-lookup-file FILE  Path to property lookup JSON file (default: ./api_lookup.json)"
            echo "  --data-dir DIR               Directory containing shared data components (default: ../data)"
            echo "  --schema-dir DIR             Directory containing GraphQL schemas (default: ../data/graphql)"
            echo "  --rebuild-index              Rebuild the ChromaDB index from scratch"
            echo "  --quick                      Quick evaluation with minimal samples"
            echo "  --verbose                    Enable verbose logging"
            echo "  --help                       Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --quick                                           # Quick evaluation"
            echo "  $0 --datasets ITOps,NL2GQL --models granite          # Multiple datasets and models"
            echo "  $0 --verbose --rebuild-index                        # Enable debug logging and rebuild index"
            echo "  $0 --property-lookup-file /path/to/api_lookup.json  # Custom property lookup file"
            echo "  $0 --data-dir /path/to/data                         # Custom data directory"
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

# Convert relative paths to absolute paths if necessary
if [[ "$PROPERTY_LOOKUP_FILE" != /* ]]; then
    PROPERTY_LOOKUP_FILE="$SCRIPT_DIR/$PROPERTY_LOOKUP_FILE"
fi

if [[ "$DATA_DIR_ARG" != /* ]]; then
    DATA_DIR_ARG="$SCRIPT_DIR/$DATA_DIR_ARG"
fi

if [[ "$SCHEMA_DIR" != /* ]]; then
    SCHEMA_DIR="$SCRIPT_DIR/$SCHEMA_DIR"
fi

# Display configuration
echo "Configuration:"
echo "   Datasets: $DATASETS"
echo "   Models: $MODELS"
echo "   Property lookup file: $PROPERTY_LOOKUP_FILE"
echo "   Data directory: $DATA_DIR_ARG"
echo "   Schema directory: $SCHEMA_DIR"
echo "   Rebuild index: $REBUILD_INDEX"
echo "   Quick mode: $QUICK"
echo "   Verbose: $VERBOSE"
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: No virtual environment detected. Consider activating one."
fi

# Check for required Python packages
echo "Checking Python dependencies..."
python3 -c "import langgraph, litellm, langchain_core, chromadb, sentence_transformers, networkx" 2>/dev/null || {
    echo "Error: Required packages not found. Please install:"
    echo "  pip install langgraph litellm langchain-core chromadb sentence-transformers networkx"
    exit 1
}

# Check for MapReflect components
if [ ! -f "map_reflect_agent.py" ]; then
    echo "Error: map_reflect_agent.py not found. MapReflect functionality requires this file."
    exit 1
fi
echo "MapReflect agent found: map_reflect_agent.py"

if [ ! -f "$PROPERTY_LOOKUP_FILE" ]; then
    echo "Error: Property lookup file $PROPERTY_LOOKUP_FILE not found"
    echo "MapReflect requires a valid property lookup JSON file (e.g., api_lookup.json) to function"
    exit 1
fi
echo "Property lookup file found: $PROPERTY_LOOKUP_FILE"

if [ ! -d "$DATA_DIR_ARG" ]; then
    echo "Error: Data directory $DATA_DIR_ARG not found"
    echo "MapReflect requires the shared data directory with dataset_loader.py"
    exit 1
fi
echo "Data directory found: $DATA_DIR_ARG"

if [ ! -f "$DATA_DIR_ARG/dataset_loader.py" ]; then
    echo "Error: dataset_loader.py not found in $DATA_DIR_ARG"
    echo "MapReflect requires the shared dataset loader"
    exit 1
fi
echo "Dataset loader found: $DATA_DIR_ARG/dataset_loader.py"

if [ ! -d "$SCHEMA_DIR" ]; then
    echo "Warning: Schema directory $SCHEMA_DIR not found"
    echo "MapReflect evaluation will continue but may use limited schema information"
fi

if [ -f ".env" ]; then
    echo "Found .env file - environment variables will be loaded"
else
    echo "No .env file found - using system environment variables"
fi

# Check for ChromaDB directory (will be created if doesn't exist)
if [ -d "chroma_dir" ]; then
    if [ "$REBUILD_INDEX" = true ]; then
        echo "ChromaDB index exists and will be rebuilt"
    else
        echo "ChromaDB index exists and will be reused"
    fi
else
    echo "ChromaDB index will be created"
fi

# Create necessary directories
mkdir -p results
mkdir -p chroma_dir

# Build command line arguments for Python script
ARGS=""

ARGS="$ARGS --datasets $DATASETS"
ARGS="$ARGS --models $MODELS"
ARGS="$ARGS --property-lookup-file $PROPERTY_LOOKUP_FILE"
ARGS="$ARGS --data-dir $DATA_DIR_ARG"

if [ -d "$SCHEMA_DIR" ]; then
    ARGS="$ARGS --schema-dir $SCHEMA_DIR"
fi

if [ "$REBUILD_INDEX" = true ]; then
    ARGS="$ARGS --rebuild-index"
fi

if [ "$QUICK" = true ]; then
    ARGS="$ARGS --quick"
fi

if [ "$VERBOSE" = true ]; then
    ARGS="$ARGS --verbose"
fi

echo "Running MapReflect evaluation with args: $ARGS"
echo ""

# Check if map_reflect_runner.py has a main function that accepts these arguments
if ! grep -q "def main" map_reflect_runner.py; then
    echo "Warning: map_reflect_runner.py may not have a main function with argument parsing."
    echo "You may need to modify map_reflect_runner.py to accept command line arguments."
fi

# Run the MapReflect evaluation
python3 map_reflect_runner.py $ARGS

# Check if results were generated
if [ -d "results" ]; then
    echo ""
    echo "MapReflect evaluation completed successfully!"
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
echo "MapReflect Schema Mapping Evaluation Complete!"
echo "=============================================="
