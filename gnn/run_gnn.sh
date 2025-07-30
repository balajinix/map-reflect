#!/bin/bash

# Run script for GNN Schema Mapping Method
# Usage: ./run_gnn.sh [options]

set -e  # Exit on any error

echo "Starting GNN Schema Mapping Evaluation"
echo "=========================================="

# Check if we're in the gnn directory
if [ ! -f "gnn_runner.py" ]; then
    echo "Error: gnn_runner.py not found. Please run this script from the gnn/ directory."
    echo "Usage: cd gnn && ./run_gnn.sh"
    exit 1
fi

# Check if shared dataset loader exists (one level up)
if [ ! -f "../data/dataset_loader.py" ]; then
    echo "Error: shared dataset loader not found at ../data/dataset_loader.py"
    exit 1
fi

# Create data directory if it doesn't exist (one level up)
mkdir -p ../data

# Default parameters
MODELS="llama"
DATASET_NAME="itops"
GNN_MODEL_PATH="../data/gnn_results/${DATASET_NAME}_mapping_gnn_results.pkl"
GRAPH_DATA_PATH="../data/${DATASET_NAME}_graph_data.pkl"
SCHEMA_DIR="../data/${DATASET_NAME}_graphql"
CHROMA_DB_PATH="../data/chroma_db"
COLLECTION_NAME="${DATASET_NAME}_graphql_schemas"
API_LOOKUP_FILE="../data/${DATASET_NAME}_api_lookup.json"
QUICK=false
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET_NAME="$2"
            # Update paths based on dataset name
            GNN_MODEL_PATH="../data/gnn_results/${DATASET_NAME}_mapping_gnn_results.pkl"
            GRAPH_DATA_PATH="../data/${DATASET_NAME}_graph_data.pkl"
            API_LOOKUP_FILE="../data/${DATASET_NAME}_api_lookup.json"
            SCHEMA_DIR="../data/${DATASET_NAME}_graphql"
            shift 2
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        --gnn-model-path)
            GNN_MODEL_PATH="$2"
            shift 2
            ;;
        --graph-data-path)
            GRAPH_DATA_PATH="$2"
            shift 2
            ;;
        --schema-dir)
            SCHEMA_DIR="$2"
            shift 2
            ;;
        --chroma-db-path)
            CHROMA_DB_PATH="$2"
            shift 2
            ;;
        --collection_name)
            COLLECTION_NAME="$2"
            shift 2
            ;;
        --api-lookup-file)
            API_LOOKUP_FILE="$2"
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
            echo "  --dataset NAME           Dataset name (itops, nl2gql, etc.) - updates all paths"
            echo "  --models MODELS          Comma-separated list of models (default: llama)"
            echo "  --gnn-model-path PATH    Path to trained GNN model"
            echo "  --graph-data-path PATH   Path to graph data"
            echo "  --schema-dir DIR         Directory containing GraphQL schemas"
            echo "  --chroma-db-path PATH    Path to ChromaDB database (default: ../data/chroma_db)"
            echo "  --collection_name PATH   collection name"
            echo "  --api-lookup-file PATH   Path to API lookup file"
            echo "  --quick                  Quick evaluation with minimal samples"
            echo "  --verbose                Enable verbose logging"
            echo "  --help                   Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --quick                                      # Quick evaluation with default dataset"
            echo "  $0 --dataset healthcare --verbose              # Use healthcare dataset"
            echo "  $0 --datasets ITOps,NL2GQL --models granite     # Multiple datasets and models"
            echo ""
            echo "Prerequisites:"
            echo "  1. Run GNN training pipeline first: ./setup_gnn.sh"
            echo "  2. Ensure dataset file exists (e.g., ../data/itops_dataset.jsonl)"
            echo "  3. Ensure api_lookup.json exists"
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
echo "   Dataset name: $DATASET_NAME"
echo "   Models: $MODELS"
echo "   GNN model: $GNN_MODEL_PATH"
echo "   Graph data: $GRAPH_DATA_PATH"
echo "   Schema directory: $SCHEMA_DIR"
echo "   ChromaDB path: $CHROMA_DB_PATH"
echo "   Collection Name: $COLLECTION_NAME"
echo "   API lookup file: $API_LOOKUP_FILE"
echo "   Quick mode: $QUICK"
echo "   Verbose: $VERBOSE"
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: No virtual environment detected. Consider activating one."
fi

# Check for GNN components
echo "Checking GNN prerequisites..."

# Check GNN model file
if [ ! -f "$GNN_MODEL_PATH" ]; then
    echo "Error: GNN model not found at $GNN_MODEL_PATH"
    echo "Please run the GNN training pipeline first:"
    echo "  ./setup_gnn.sh --dataset $DATASET_NAME"
    exit 1
else
    echo "GNN model found: $GNN_MODEL_PATH"
fi

# Check graph data file
if [ ! -f "$GRAPH_DATA_PATH" ]; then
    echo "Error: Graph data not found at $GRAPH_DATA_PATH"
    echo "Please run: ./setup_gnn.sh --dataset $DATASET_NAME"
    exit 1
else
    echo "Graph data found: $GRAPH_DATA_PATH"
fi

# Check API lookup file
if [ ! -f "$API_LOOKUP_FILE" ]; then
    echo "Error: API lookup file not found at $API_LOOKUP_FILE"
    echo "Please create the API lookup file with field descriptions"
    exit 1
else
    echo "API lookup file found: $API_LOOKUP_FILE"
fi

# Check for GNN component files
if [ ! -f "gnn_api_retriever.py" ]; then
    echo "Error: gnn_api_retriever.py not found"
    exit 1
else
    echo "GNN API retriever found"
fi

# Check for dataset file
DATASET_FILE="../data/${DATASET_NAME}_dataset.jsonl"
if [ ! -f "$DATASET_FILE" ]; then
    echo "Warning: Dataset file not found at $DATASET_FILE"
else
    echo "Dataset file found: $DATASET_FILE"
fi

# Check dataset loader
if [ ! -f "../data/dataset_loader.py" ]; then
    echo "Error: dataset_loader.py not found at ../data/dataset_loader.py"
    exit 1
else
    echo "Dataset loader found"
fi

# Check schema directory (optional)
if [ ! -d "$SCHEMA_DIR" ]; then
    echo "Warning: Schema directory $SCHEMA_DIR not found"
    echo "GNN evaluation will continue but schema discovery may be limited"
else
    echo "Schema directory found: $SCHEMA_DIR"
fi

# Check ChromaDB directory (optional)
if [ ! -d "$CHROMA_DB_PATH" ]; then
    echo "Warning: ChromaDB directory $CHROMA_DB_PATH not found"
    echo "GNN will create it or use limited functionality"
else
    echo "ChromaDB directory found: $CHROMA_DB_PATH"
fi

if [ ! -d "$COLLECTION_NAME" ]; then
    echo "Warning: Collection name $COLLECTION_NAME not found"
    echo "GNN will create it or use limited functionality"
else
    echo "Collection name found: $COLLECTION_NAME"
fi

# Create necessary directories
mkdir -p results

echo ""
echo "All prerequisites satisfied. Starting GNN evaluation..."

# Build command line arguments
ARGS=""

ARGS="$ARGS --models $MODELS"
ARGS="$ARGS --gnn-model-path $GNN_MODEL_PATH"
ARGS="$ARGS --graph-data-path $GRAPH_DATA_PATH"
ARGS="$ARGS --schema-dir $SCHEMA_DIR"
ARGS="$ARGS --chroma-db-path $CHROMA_DB_PATH"
ARGS="$ARGS --collection_name $COLLECTION_NAME"
ARGS="$ARGS --api-lookup-file $API_LOOKUP_FILE"

if [ "$QUICK" = true ]; then
    ARGS="$ARGS --quick"
fi

if [ "$VERBOSE" = true ]; then
    ARGS="$ARGS --verbose"
fi

echo "Running GNN evaluation with args: $ARGS"
echo ""

# Run the GNN evaluation
python3 gnn_runner.py $ARGS

exit_code=$?

# Check results
if [ $exit_code -eq 0 ]; then
    echo ""
    echo "GNN evaluation completed successfully!"
    
    if [ -d "results" ]; then
        echo "Results saved in: ./results/"
        echo ""
        echo "Generated files:"
        ls -la results/ | grep -E "\.(json|txt|csv)$" | head -10
        
        # Show GNN-specific info
        echo ""
        echo "GNN Configuration Used:"
        echo "   Model: $GNN_MODEL_PATH"
        echo "   Graph data: $GRAPH_DATA_PATH"
        echo "   Dataset: $DATASET_FILE"
    else
        echo "Warning: No results directory found, but script completed successfully."
    fi
else
    echo ""
    echo "GNN evaluation failed with exit code $exit_code"
    echo "Check the error messages above for details."
    exit $exit_code
fi

echo ""
echo "GNN Schema Mapping Evaluation Complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "   Method: Graph Neural Network (GNN)"
echo "   Dataset: $DATASET_NAME"
echo "   Approach: Semantic similarity + field joinability analysis"
echo "   Key advantage: Learns from existing mappings to predict new ones"
echo "   Performance: Check results/*.json for detailed metrics"
