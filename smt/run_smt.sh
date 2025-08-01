#!/bin/bash

# SMT (Schema Mapping Tool) Runner Script
# This script runs SMT evaluation with proper dataset-specific file handling

set -e  # Exit on any error

# Default values
DATASET="ITOps"
MODEL="mistral-large"
SAMPLES=""
TIMEOUT="6000"
VERBOSE=false
QUICK=false
FORCE_REGENERATE=false

# Get script directory and data directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../data"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Function to show usage
usage() {
    echo "SMT (Schema Mapping Tool) Runner"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --dataset DATASET      Dataset to evaluate (default: ITOps)"
    echo "                         Available: ITOps, NL2GQL, GraphQLRestBench"
    echo "  --model MODEL          Model to use (default: mistral-large)"
    echo "                         Available: granite, llama, mistral-large"
    echo "  --samples N            Limit to N samples per dataset"
    echo "  --timeout SECONDS      Timeout per evaluation (default: 6000)"
    echo "  --quick                Quick evaluation mode (limited samples)"
    echo "  --force-regenerate     Force regeneration of all results"
    echo "  --verbose              Enable verbose logging"
    echo "  --help                 Show this help message"
    echo ""
    echo "Data Requirements:"
    echo "  - GraphQL schemas: ../data/{dataset}_graphql/"
    echo "  - Dataset file: ../data/{dataset}_dataset.jsonl"
    echo "  - ChromaDB indices: ../data/chroma_db/"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run with defaults"
    echo "  $0 --dataset ITOps --verbose         # ITOps with verbose output"
    echo "  $0 --quick --samples 10              # Quick test with 10 samples"
    echo "  $0 --dataset NL2GQL --model granite  # NL2GQL with granite model"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --samples)
            SAMPLES="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --quick)
            QUICK=true
            SAMPLES="${SAMPLES:-10}"
            shift
            ;;
        --force-regenerate)
            FORCE_REGENERATE=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "SMT (Schema Mapping Tool) Evaluation"
echo "============================================================"

# Convert dataset to lowercase for file paths
DATASET_LOWER=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')

print_status "Configuration:"
print_status "  Dataset: $DATASET"
print_status "  Model: $MODEL"
if [[ -n "$SAMPLES" ]]; then
    print_status "  Samples: $SAMPLES"
else
    print_status "  Samples: all available"
fi
print_status "  Timeout: ${TIMEOUT}s ($(($TIMEOUT / 60)) minutes)"
print_status "  Quick mode: $QUICK"
print_status "  Force regenerate: $FORCE_REGENERATE"
print_status "  Verbose: $VERBOSE"

# Check if we're in the correct directory
if [[ ! -f "smt_runner.py" ]]; then
    print_error "SMT runner not found. Please run this script from the smt/ directory."
    print_error "Expected file: smt_runner.py"
    exit 1
fi

# Validate dataset
case $DATASET in
    "ITOps"|"NL2GQL"|"GraphQLRestBench")
        print_status "Dataset validated: $DATASET"
        ;;
    *)
        print_error "Invalid dataset: $DATASET"
        print_error "Available datasets: ITOps, NL2GQL, GraphQLRestBench"
        exit 1
        ;;
esac

# Validate model
case $MODEL in
    "granite"|"llama"|"mistral-large")
        print_status "Model validated: $MODEL"
        ;;
    *)
        print_error "Invalid model: $MODEL"
        print_error "Available models: granite, llama, mistral-large"
        exit 1
        ;;
esac

# Check for required dependencies
print_step "1. Checking dependencies"

python -c "import chromadb; print('ChromaDB available')" 2>/dev/null || {
    print_error "ChromaDB not found. Please install with: pip install chromadb"
    exit 1
}

python -c "from sentence_transformers import SentenceTransformer; print('SentenceTransformers available')" 2>/dev/null || {
    print_error "SentenceTransformers not found. Please install with: pip install sentence-transformers"
    exit 1
}

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" && -z "$CONDA_DEFAULT_ENV" ]]; then
    print_warning "No virtual environment detected. Make sure dependencies are installed."
fi

# Check dataset-specific files
print_step "2. Validating dataset files"

# Dataset-specific file paths
DATASET_FILE="$DATA_DIR/${DATASET_LOWER}_dataset.jsonl"
GRAPHQL_DIR="$DATA_DIR/${DATASET_LOWER}_graphql"
CHROMA_DB_PATH="$DATA_DIR/chroma_db"
COLLECTION_NAME="${DATASET_LOWER}_graphql_schemas"

print_status "Dataset files:"
print_status "  Dataset file: $DATASET_FILE"
print_status "  GraphQL schemas: $GRAPHQL_DIR"
print_status "  ChromaDB path: $CHROMA_DB_PATH"
print_status "  Collection name: $COLLECTION_NAME"

# Validate dataset file
if [[ ! -f "$DATASET_FILE" ]]; then
    print_error "Dataset file not found: $DATASET_FILE"
    print_error "Please ensure the dataset file exists with the correct naming convention."
    exit 1
fi

# Validate GraphQL schema directory
if [[ ! -d "$GRAPHQL_DIR" ]]; then
    print_error "GraphQL schema directory not found: $GRAPHQL_DIR"
    print_error "Please ensure the GraphQL schemas are available for the dataset."
    exit 1
fi

# Count GraphQL schema files
SCHEMA_FILES=$(find "$GRAPHQL_DIR" -name "*.graphql" -o -name "*.gql" -o -name "*.schema" | wc -l)
if [[ $SCHEMA_FILES -eq 0 ]]; then
    print_error "No GraphQL schema files found in $GRAPHQL_DIR"
    print_error "Expected files with extensions: .graphql, .gql, .schema"
    exit 1
fi
print_status "Found $SCHEMA_FILES GraphQL schema files"

# Count dataset samples
DATASET_SAMPLES=$(wc -l < "$DATASET_FILE")
print_status "Found $DATASET_SAMPLES samples in dataset"

# Validate ChromaDB
if [[ ! -d "$CHROMA_DB_PATH" ]]; then
    print_error "ChromaDB path not found: $CHROMA_DB_PATH"
    print_error "Please run the setup script first: ./setup_smt.sh"
    exit 1
fi

# Check for setup completion marker
if [[ ! -f "$CHROMA_DB_PATH/.setup_complete" ]]; then
    print_warning "Setup completion marker not found."
    print_warning "Please run ./setup_smt.sh to initialize ChromaDB indices."
fi

# Verify ChromaDB collection exists
print_step "3. Verifying ChromaDB collection"
python -c "
import chromadb
from chromadb.config import Settings
import sys

try:
    client = chromadb.PersistentClient(path='$CHROMA_DB_PATH', settings=Settings(anonymized_telemetry=False))
    collections = [c.name for c in client.list_collections()]
    
    if '$COLLECTION_NAME' in collections:
        collection = client.get_collection('$COLLECTION_NAME')
        count = collection.count()
        print(f'Collection \"$COLLECTION_NAME\" found with {count} documents')
    else:
        print(f'ERROR: Collection \"$COLLECTION_NAME\" not found')
        print(f'Available collections: {collections}')
        sys.exit(1)
        
except Exception as e:
    print(f'ERROR: ChromaDB verification failed: {e}')
    sys.exit(1)
" || {
    print_error "ChromaDB verification failed"
    print_error "Please run ./setup_smt.sh --datasets $DATASET to create the required collections"
    exit 1
}

# Check existing results
print_step "4. Checking existing results"
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

RESULT_PATTERN="${RESULTS_DIR}/${DATASET_LOWER}_${MODEL}_smt_results*.json"
EXISTING_RESULTS=$(ls ${RESULT_PATTERN} 2>/dev/null | wc -l)

if [[ $EXISTING_RESULTS -gt 0 && "$FORCE_REGENERATE" != "true" ]]; then
    print_warning "Found $EXISTING_RESULTS existing result files matching pattern: $RESULT_PATTERN"
    print_warning "Use --force-regenerate to overwrite existing results"
fi

# Build SMT arguments
print_step "5. Building SMT arguments"

SMT_ARGS="--datasets $DATASET --models $MODEL"
SMT_ARGS="$SMT_ARGS --mappings-dir $DATA_DIR/mappings_output"
SMT_ARGS="$SMT_ARGS --schema-dir $GRAPHQL_DIR"
SMT_ARGS="$SMT_ARGS --chroma-db-path $CHROMA_DB_PATH"
SMT_ARGS="$SMT_ARGS --collection-name $COLLECTION_NAME"

if [[ -n "$SAMPLES" ]]; then
    SMT_ARGS="$SMT_ARGS --samples $SAMPLES"
fi

if [[ "$VERBOSE" == "true" ]]; then
    SMT_ARGS="$SMT_ARGS --verbose"
fi

if [[ "$FORCE_REGENERATE" == "true" ]]; then
    SMT_ARGS="$SMT_ARGS --force-regenerate"
fi

print_status "SMT arguments: $SMT_ARGS"

# Run SMT evaluation with timeout
print_step "6. Running SMT evaluation"

START_TIME=$(date +%s)

print_status "Starting SMT evaluation..."
print_status "  Dataset: $DATASET ($DATASET_SAMPLES samples)"
print_status "  GraphQL schemas: $SCHEMA_FILES files"
print_status "  Model: $MODEL"
print_status "  Timeout: ${TIMEOUT}s"

if timeout "$TIMEOUT" python3 smt_runner.py $SMT_ARGS; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    print_status "SMT evaluation completed successfully!"
    print_status "Execution time: ${DURATION}s ($(($DURATION / 60))m $(($DURATION % 60))s)"
    
    # Display results summary
    print_step "7. Results summary"
    
    NEW_RESULTS=$(ls ${RESULT_PATTERN} 2>/dev/null | wc -l)
    print_status "Result files: $NEW_RESULTS"
    
    if [[ $NEW_RESULTS -gt 0 ]]; then
        print_status "Generated files:"
        ls -la ${RESULT_PATTERN} 2>/dev/null || true
        
        # Try to extract basic statistics from the most recent result file
        LATEST_RESULT=$(ls -t ${RESULT_PATTERN} 2>/dev/null | head -1)
        if [[ -n "$LATEST_RESULT" && -f "$LATEST_RESULT" ]]; then
            print_status "Latest result file: $LATEST_RESULT"
            
            # Extract basic statistics
            python -c "
import json
import sys

try:
    with open('$LATEST_RESULT', 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        total = len(data)
        successful = sum(1 for item in data if item.get('success', False))
        print(f'   Total evaluations: {total}')
        print(f'   Successful mappings: {successful}')
        if total > 0:
            print(f'   Success rate: {successful/total:.3f}')
    else:
        print(f'   Result format: {type(data).__name__}')
        
except Exception as e:
    print(f'   Could not parse results: {e}')
"
        fi
    fi
    
else
    EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    if [[ $EXIT_CODE -eq 124 ]]; then
        print_error "SMT evaluation TIMED OUT after ${TIMEOUT}s"
        print_error "Consider increasing timeout with --timeout <seconds>"
    else
        print_error "SMT evaluation FAILED after ${DURATION}s (exit code: $EXIT_CODE)"
    fi
    
    exit $EXIT_CODE
fi

echo ""
echo "============================================================"
print_status "SMT Evaluation Completed Successfully!"
echo "============================================================"

print_status "Next steps:"
echo "  1. Review results in: $RESULTS_DIR/"
echo "  2. Run comparative evaluation: ../run_evaluation.sh --methods SMT --dataset $DATASET"
echo "  3. Add more datasets: $0 --dataset NL2GQL"

echo ""
print_status "Dataset-specific files used:"
echo "  - Dataset: $DATASET_FILE"
echo "  - Schemas: $GRAPHQL_DIR"
echo "  - Collection: $COLLECTION_NAME"

if [[ "$QUICK" == "true" ]]; then
    echo ""
    print_status "Quick mode completed. For full evaluation, run without --quick"
fi
