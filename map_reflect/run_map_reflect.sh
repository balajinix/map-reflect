#!/bin/bash

# Run script for MapReflect Schema Mapping Method
# Usage: ./run_map_reflect.sh [options]

set -e  # Exit on any error

echo "Starting MapReflect Schema Mapping Evaluation"
echo "============================================="

# ============================================================================
# CONFIGURATION - Modify these variables to change evaluation settings
# ============================================================================

# Dataset to evaluate
# Options: "itops", "nl2gql", "itops,nl2gql"
DATASETS="nl2gql"

# Available models:
#   - "ibm/granite-4-h-small" (Granite 4 - recommended)
#   - "watsonx/meta-llama/llama-3-3-70b-instruct"
#   - "watsonx/mistralai/mistral-large"
MODELS="ibm/granite-4-h-small"

DATA_DIR="../data"

# Dataset-specific files will be automatically resolved:
#   - Property lookup: ${DATA_DIR}/${dataset}_api_lookup.json
#   - Hierarchy: ${DATA_DIR}/${dataset}_hierarchy.json (optional)
#   - Schema: ${DATA_DIR}/${dataset}_graphql/ (optional)
#   - Evaluation: ${DATA_DIR}/${dataset}_eval.json

# MapReflect options
REBUILD_INDEX=false  # Set to true to rebuild ChromaDB index from scratch
QUICK=false          # Set to true for quick evaluation with minimal samples
VERBOSE=false        # Set to true for detailed logging

# ============================================================================
# END CONFIGURATION
# ============================================================================

# Check if we're in the map_reflect directory
if [ ! -f "map_reflect_runner.py" ]; then
    echo "Error: map_reflect_runner.py not found. Please run this script from the map_reflect/ directory."
    echo "Usage: cd map_reflect && ./run_map_reflect.sh"
    exit 1
fi

# Get absolute paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Parse command line arguments (these override the configuration above)
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
        --data-dir)
            DATA_DIR="$2"
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
            echo "Configuration variables (edit at top of script):"
            echo "  DATASETS = $DATASETS (comma-separated: itops, nl2gql)"
            echo "  MODELS   = $MODELS"
            echo "  DATA_DIR = $DATA_DIR"
            echo ""
            echo "Options:"
            echo "  --datasets DATASETS    Comma-separated list of datasets"
            echo "  --models MODELS        Comma-separated list of models"
            echo "  --data-dir DIR         Directory containing all dataset files"
            echo "  --rebuild-index        Rebuild the ChromaDB index from scratch"
            echo "  --quick                Quick evaluation with minimal samples"
            echo "  --verbose              Enable verbose logging"
            echo "  --help                 Show this help message"
            echo ""
            echo "Dataset Files Required:"
            echo "  For each dataset (e.g., 'itops'), you need:"
            echo "    - \${DATA_DIR}/\${dataset}_api_lookup.json  (required)"
            echo "    - \${DATA_DIR}/\${dataset}_hierarchy.json   (optional)"
            echo "    - \${DATA_DIR}/\${dataset}_graphql/         (optional)"
            echo "    - \${DATA_DIR}/\${dataset}_eval.json        (required by dataset_loader)"
            echo ""
            echo "Examples:"
            echo "  # Run ITOps evaluation"
            echo "  $0 --datasets itops"
            echo ""
            echo "  # Run nl2gql evaluation"
            echo "  $0 --datasets nl2gql"
            echo ""
            echo "  # Run both datasets"
            echo "  $0 --datasets itops,nl2gql"
            echo ""
            echo "  # Multiple models on both datasets"
            echo "  $0 --datasets itops,nl2gql --models 'granite,mistral-large'"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Convert relative paths to absolute paths
if [[ "$DATA_DIR" != /* ]]; then
    DATA_DIR="$SCRIPT_DIR/$DATA_DIR"
fi

# Display configuration
echo ""
echo "Configuration:"
echo "   Datasets: $DATASETS"
echo "   Models: $MODELS"
echo "   Data directory: $DATA_DIR"
echo "   Rebuild index: $REBUILD_INDEX"
echo "   Quick mode: $QUICK"
echo "   Verbose: $VERBOSE"
echo ""

# Validate environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: No virtual environment detected."
fi

# Validate data directory
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    exit 1
fi
echo "Data directory found: $DATA_DIR"

if [ ! -f "$DATA_DIR/dataset_loader.py" ]; then
    echo "Error: dataset_loader.py not found in $DATA_DIR"
    exit 1
fi
echo "Dataset loader found: $DATA_DIR/dataset_loader.py"

# Validate dataset-specific files
IFS=',' read -ra DATASET_ARRAY <<< "$DATASETS"
for dataset in "${DATASET_ARRAY[@]}"; do
    dataset=$(echo "$dataset" | tr '[:upper:]' '[:lower:]' | xargs)  # lowercase and trim
    
    echo ""
    echo "Validating files for dataset: $dataset"
    
    # Check property lookup file
    PROPERTY_FILE="$DATA_DIR/${dataset}_api_lookup.json"
    if [ ! -f "$PROPERTY_FILE" ]; then
        echo "  ERROR: Property lookup file not found: $PROPERTY_FILE"
        echo "  This file is required for MapReflect to work."
        echo ""
        echo "  To generate nl2gql files, run:"
        echo "    python setup_nl2gql_dataset.py --database <cosql.db> --output $DATA_DIR"
        exit 1
    fi
    echo "  Property lookup: $PROPERTY_FILE"
    
    # Check hierarchy file (optional)
    HIERARCHY_FILE="$DATA_DIR/${dataset}_hierarchy.json"
    if [ -f "$HIERARCHY_FILE" ]; then
        echo "  Hierarchy file: $HIERARCHY_FILE"
    else
        echo "  Hierarchy file not found (optional): $HIERARCHY_FILE"
    fi
    
    # Check schema directory (optional)
    SCHEMA_DIR="$DATA_DIR/${dataset}_graphql"
    if [ -d "$SCHEMA_DIR" ]; then
        echo "  Schema directory: $SCHEMA_DIR"
    else
        echo "  Schema directory not found (optional): $SCHEMA_DIR"
    fi
    
    # Check evaluation file
    EVAL_FILE="$DATA_DIR/${dataset}_eval.json"
    if [ -f "$EVAL_FILE" ]; then
        echo "  Evaluation utterances: $EVAL_FILE"
    else
        echo "  Evaluation file not found: $EVAL_FILE"
        echo "  (Will be loaded by dataset_loader.py - may use different naming)"
    fi
done

if [ -f ".env" ]; then
    echo ""
else
    echo ""
    echo "No .env file found - using system environment variables"
fi

# ChromaDB status
if [ -d "$DATA_DIR/chroma_db" ]; then
    if [ "$REBUILD_INDEX" = true ]; then
        echo "ChromaDB index exists and will be rebuilt"
    else
        echo "ChromaDB index exists and will be reused"
        echo "  (Each dataset uses a separate collection: property_index_<dataset>)"
    fi
else
    echo "ChromaDB index will be created"
fi

# Create results directory
mkdir -p results

# Run evaluation for each dataset-model combination
echo ""
echo "Starting evaluations..."
echo "=============================================="

IFS=',' read -ra DATASET_ARRAY <<< "$DATASETS"
IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"

for dataset in "${DATASET_ARRAY[@]}"; do
    dataset=$(echo "$dataset" | tr '[:upper:]' '[:lower:]' | xargs)
    
    for model in "${MODEL_ARRAY[@]}"; do
        model=$(echo "$model" | xargs)
        
        echo ""
        echo "Running: $dataset with $model"
        echo "----------------------------------------------"
        
        # Build dataset-specific arguments
        PROPERTY_FILE="$DATA_DIR/${dataset}_api_lookup.json"
        SCHEMA_DIR="$DATA_DIR/${dataset}_graphql"
        
        ARGS="--datasets $dataset"
        ARGS="$ARGS --models $model"
        ARGS="$ARGS --property-lookup-file $PROPERTY_FILE"
        ARGS="$ARGS --data-dir $DATA_DIR"
        
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
        
        # Run the evaluation
        python3 map_reflect_runner.py $ARGS
        
        if [ $? -ne 0 ]; then
            echo "ERROR: Evaluation failed for $dataset with $model"
            exit 1
        fi
    done
done

if [ -d "results" ]; then
    echo ""
    echo "=============================================="
    echo "MapReflect Evaluation Complete!"
    echo "=============================================="
    echo ""
    echo "Results saved in: ./results/"
    echo ""
    echo "Generated files:"
    ls -lh results/*.json 2>/dev/null || echo "No JSON files found"
    
    result_count=$(ls -1 results/*.json 2>/dev/null | wc -l)
    echo ""
    echo "Generated $result_count result file(s)"
else
    echo ""
    echo "No results directory found. Check for errors above."
    exit 1
fi
