#!/bin/bash

# SMT (Schema Mapping Tool) Runner Script
# This script runs SMT evaluation with proper dataset-specific file handling

set -e  # Exit on any error

# Default values
DATASET="ITOps"
MODEL="granite"
MODEL="llama"
MODEL="mistral-large"
SAMPLES=""
TIMEOUT="6000"
VERBOSE=false
QUICK=false
FORCE_REGENERATE=false
MAX_WORKERS=""

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
    echo "  --max-workers N        Number of parallel workers (default: auto-detect)"
    echo ""
    echo "  --help                 Show this help message"
    echo ""
    echo "Data Requirements:"
    echo "  - GraphQL schemas: ../data/{dataset}_graphql/"
    echo "  - Dataset file: ../data/{dataset}_dataset.jsonl"
    echo "  - ChromaDB indices: ../data/chroma_db/"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run with defaults (8 threads)"
    echo "  $0 --max-workers 4                   # Use 4 threads"
    echo "  $0 --quick --max-workers 8           # Quick test with 8 threads"
    echo "  $0 --dataset ITOps --verbose         # Debug mode with verbose output"
    echo "  $0 --samples 50 --verbose            # Process only 50 samples with detailed output"
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
        --max-workers)
            MAX_WORKERS="$2"
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

# Auto-detect max workers if not specified
if [[ -z "$MAX_WORKERS" ]]; then
    MAX_WORKERS=$(python -c "import os; print(min(os.cpu_count(), 8))")
fi

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
print_status "  Max workers: $MAX_WORKERS"

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

python -c "import concurrent.futures; print('Concurrent.futures available')" 2>/dev/null || {
    print_error "Concurrent.futures not available"
    exit 1
}

python -c "import threading; print('Threading available')" 2>/dev/null || {
    print_error "Threading not available"
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

# Performance estimation
ESTIMATED_TIME_SEQUENTIAL=$((DATASET_SAMPLES * 5))  # Assume 5 seconds per sample
ESTIMATED_TIME_PARALLEL=$((ESTIMATED_TIME_SEQUENTIAL / MAX_WORKERS))

print_status "Performance estimates:"
print_status "  Sequential time: ~$(($ESTIMATED_TIME_SEQUENTIAL / 60))m"
print_status "  Threaded time ($MAX_WORKERS workers): ~$(($ESTIMATED_TIME_PARALLEL / 60))m"
print_status "  Expected speedup: ${MAX_WORKERS}x"

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

RESULT_PATTERN="${RESULTS_DIR}/${DATASET_LOWER}_${MODEL}_*smt_results*.json*"
EXISTING_RESULTS=$(ls ${RESULT_PATTERN} 2>/dev/null | wc -l)

if [[ $EXISTING_RESULTS -gt 0 && "$FORCE_REGENERATE" != "true" ]]; then
    print_warning "Found $EXISTING_RESULTS existing result files matching pattern"
    print_warning "Use --force-regenerate to overwrite existing results"
    print_status "Existing files:"
    ls -la ${RESULT_PATTERN} 2>/dev/null | head -5
fi

# Build SMT arguments
print_step "5. Building SMT arguments"

SMT_ARGS="--datasets $DATASET --models $MODEL"
SMT_ARGS="$SMT_ARGS --mappings-dir $DATA_DIR/mappings_output"
SMT_ARGS="$SMT_ARGS --schema-dir $GRAPHQL_DIR"
SMT_ARGS="$SMT_ARGS --chroma-db-path $CHROMA_DB_PATH"
SMT_ARGS="$SMT_ARGS --collection-name $COLLECTION_NAME"
SMT_ARGS="$SMT_ARGS --max-workers $MAX_WORKERS"

if [[ -n "$SAMPLES" ]]; then
    SMT_ARGS="$SMT_ARGS --samples $SAMPLES"
fi

if [[ "$VERBOSE" == "true" ]]; then
    SMT_ARGS="$SMT_ARGS --verbose"
fi

if [[ "$FORCE_REGENERATE" == "true" ]]; then
    print_status "Removing existing results..."
    rm -f ${RESULTS_DIR}/${DATASET_LOWER}_${MODEL}_*smt_results*.json*
fi

print_status "SMT arguments: $SMT_ARGS"

# Performance warning for large datasets
if [[ $DATASET_SAMPLES -gt 1000 ]]; then
    print_warning "Large dataset ($DATASET_SAMPLES samples)"
    print_warning "This may take $(($DATASET_SAMPLES * 5 / 60 / MAX_WORKERS)) minutes with $MAX_WORKERS workers"
fi

# Memory warning for many workers
if [[ $MAX_WORKERS -gt 8 ]]; then
    print_warning "Using $MAX_WORKERS workers may consume significant memory"
    print_warning "Consider reducing --max-workers if you encounter memory issues"
fi

# Run SMT evaluation with timeout
print_step "6. Running SMT evaluation (threaded mode)"

START_TIME=$(date +%s)

print_status "Starting SMT evaluation..."
print_status "  Dataset: $DATASET ($DATASET_SAMPLES samples)"
print_status "  GraphQL schemas: $SCHEMA_FILES files"
print_status "  Model: $MODEL"
print_status "  Threading: $MAX_WORKERS workers"
print_status "  Timeout: ${TIMEOUT}s"

# Show progress indicator for long-running tasks
if [[ $DATASET_SAMPLES -gt 100 ]]; then
    print_status "This may take a while. Progress will be shown every 10 completions..."
    if [[ "$VERBOSE" != "true" ]]; then
        print_status "Use --verbose for detailed progress information"
    fi
fi

if timeout "$TIMEOUT" python3 smt_runner.py $SMT_ARGS; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    print_status "SMT evaluation completed successfully!"
    print_status "Execution time: ${DURATION}s ($(($DURATION / 60))m $(($DURATION % 60))s)"
    
    # Calculate actual performance
    if [[ $DURATION -gt 0 ]]; then
        SAMPLES_PER_SECOND=$((DATASET_SAMPLES / DURATION))
        print_status "Performance: ${SAMPLES_PER_SECOND} samples/second"
        
        ESTIMATED_SEQUENTIAL_TIME=$((DATASET_SAMPLES * 5))
        if [[ $ESTIMATED_SEQUENTIAL_TIME -gt 0 ]]; then
            ACTUAL_SPEEDUP=$((ESTIMATED_SEQUENTIAL_TIME / DURATION))
            print_status "Estimated speedup: ${ACTUAL_SPEEDUP}x"
        fi
    fi
    
    # Display results summary
    print_step "7. Results summary"
    
    # Look for both JSONL and JSON files
    NEW_JSONL_RESULTS=$(ls ${RESULTS_DIR}/${DATASET_LOWER}_${MODEL}_*smt_results*.jsonl 2>/dev/null | wc -l)
    NEW_JSON_RESULTS=$(ls ${RESULTS_DIR}/${DATASET_LOWER}_${MODEL}_*smt_results*.json 2>/dev/null | wc -l)
    NEW_METADATA=$(ls ${RESULTS_DIR}/${DATASET_LOWER}_${MODEL}_*metadata*.json 2>/dev/null | wc -l)
    
    print_status "Generated files:"
    print_status "  Results (JSONL): $NEW_JSONL_RESULTS"
    print_status "  Results (JSON): $NEW_JSON_RESULTS"
    print_status "  Metadata: $NEW_METADATA"
    
    if [[ $NEW_JSONL_RESULTS -gt 0 || $NEW_JSON_RESULTS -gt 0 ]]; then
        print_status "File listing:"
        ls -la ${RESULTS_DIR}/${DATASET_LOWER}_${MODEL}_*smt_results*.json* 2>/dev/null | head -5
        ls -la ${RESULTS_DIR}/${DATASET_LOWER}_${MODEL}_*metadata*.json 2>/dev/null | head -3
        
        # Try to extract basic statistics from the most recent result file
        LATEST_RESULT=$(ls -t ${RESULT_PATTERN} 2>/dev/null | head -1)
        if [[ -n "$LATEST_RESULT" && -f "$LATEST_RESULT" ]]; then
            print_status "Latest result file: $LATEST_RESULT"
            
            python -c "
import json
import sys

try:
    with open('$LATEST_METADATA', 'r') as f:
        metadata = json.load(f)
    
    print(f'   Total samples: {metadata.get(\"total_samples\", 0)}')
    print(f'   Completed: {metadata.get(\"completed_samples\", 0)}')
    print(f'   Successful: {metadata.get(\"successful_samples\", 0)}')
    print(f'   Failed: {metadata.get(\"failed_samples\", 0)}')
    print(f'   Overall accuracy: {metadata.get(\"overall_accuracy\", 0.0):.3f}')
    print(f'   Method: {metadata.get(\"method\", \"unknown\")}')
    print(f'   Workers used: {metadata.get(\"max_workers\", \"unknown\")}')
    print(f'   Status: {metadata.get(\"status\", \"unknown\")}')
    
    if 'execution_time' in metadata:
        exec_time = metadata['execution_time']
        print(f'   Execution time: {exec_time:.2f}s')
        
        total_samples = metadata.get('total_samples', 0)
        if exec_time > 0 and total_samples > 0:
            print(f'   Speed: {total_samples/exec_time:.1f} samples/sec')
        
except Exception as e:
    print(f'   Could not parse metadata: {e}')
"
        fi
        
        # Count actual results in JSONL file
        LATEST_JSONL=$(ls -t ${RESULTS_DIR}/${DATASET_LOWER}_${MODEL}_*smt_results*.jsonl 2>/dev/null | head -1)
        if [[ -n "$LATEST_JSONL" && -f "$LATEST_JSONL" ]]; then
            ACTUAL_RESULTS=$(wc -l < "$LATEST_JSONL")
            print_status "Actual results written: $ACTUAL_RESULTS lines in JSONL"
        fi
    fi
    
else
    EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    if [[ $EXIT_CODE -eq 124 ]]; then
        print_error "SMT evaluation TIMED OUT after ${TIMEOUT}s"
        print_error "Consider:"
        print_error "  - Increasing timeout with --timeout <seconds>"
        print_error "  - Reducing workers: --max-workers 4"
        print_error "  - Reducing samples: --samples <number>"
        print_error "  - Running in quick mode: --quick"
    else
        print_error "SMT evaluation FAILED after ${DURATION}s (exit code: $EXIT_CODE)"
        print_error "Consider:"
        print_error "  - Using --verbose for detailed error information"
        print_error "  - Reducing workers: --max-workers 1 (for debugging)"
        print_error "  - Checking that the dataset file format is correct"
        print_error "  - Verifying ChromaDB collection exists"
    fi
    
    # Show any partial results
    PARTIAL_RESULTS=$(ls ${RESULTS_DIR}/${DATASET_LOWER}_${MODEL}_*smt_results*.jsonl 2>/dev/null | wc -l)
    if [[ $PARTIAL_RESULTS -gt 0 ]]; then
        print_warning "Found $PARTIAL_RESULTS partial result files"
        LATEST_PARTIAL=$(ls -t ${RESULTS_DIR}/${DATASET_LOWER}_${MODEL}_*smt_results*.jsonl 2>/dev/null | head -1)
        if [[ -n "$LATEST_PARTIAL" ]]; then
            PARTIAL_COUNT=$(wc -l < "$LATEST_PARTIAL")
            print_warning "Partial results: $PARTIAL_COUNT lines in $LATEST_PARTIAL"
        fi
    fi
    
    exit $EXIT_CODE
fi

echo ""
echo "============================================================"
print_status "SMT Evaluation Completed Successfully!"
echo "============================================================"

print_status "Performance summary:"
print_status "  Workers used: $MAX_WORKERS"
print_status "  Total time: ${DURATION}s"
print_status "  Samples processed: $DATASET_SAMPLES"

print_status "Generated output files:"
echo "  - Incremental results: ${RESULTS_DIR}/${DATASET_LOWER}_${MODEL}_*smt_results*.jsonl"
echo "  - Final results: ${RESULTS_DIR}/${DATASET_LOWER}_${MODEL}_*smt_results*.json"
echo "  - Progress tracking: ${RESULTS_DIR}/${DATASET_LOWER}_${MODEL}_*progress*.json"
echo "  - Metadata: ${RESULTS_DIR}/${DATASET_LOWER}_${MODEL}_*metadata*.json"

print_status "Next steps:"
echo "  1. Review results in: $RESULTS_DIR/"
echo "  2. Run comparative evaluation: ../run_evaluation.sh --methods SMT --dataset $DATASET"
echo "  3. For debugging issues:"
echo "     - $0 --verbose --max-workers 1    # Single-threaded with detailed output"
echo "     - $0 --quick --verbose             # Quick test with detailed output"

echo ""
print_status "Dataset-specific files used:"
echo "  - Dataset: $DATASET_FILE"
echo "  - Schemas: $GRAPHQL_DIR"
echo "  - Collection: $COLLECTION_NAME"

if [[ "$QUICK" == "true" ]]; then
    echo ""
    print_status "Quick mode completed. For full evaluation, run without --quick"
fi

echo ""
print_status "Performance recommendations for your dataset size ($DATASET_SAMPLES samples):"
if [[ $DATASET_SAMPLES -lt 50 ]]; then
    echo "  Recommended: --max-workers 1 (small dataset, debugging easier)"
elif [[ $DATASET_SAMPLES -lt 200 ]]; then
    echo "  Recommended: --max-workers 4 (balanced performance)"
elif [[ $DATASET_SAMPLES -lt 500 ]]; then
    echo "  Recommended: --max-workers 8 (good parallelization)"
else
    echo "  Recommended: --max-workers 8 (maximum reasonable for large dataset)"
    echo "               Consider --quick for testing first"
fi

echo ""
print_status "Troubleshooting tips:"
echo "  - If hanging: try --max-workers 1 --verbose"
echo "  - If memory issues: reduce --max-workers"
echo "  - If timeout: increase --timeout or use --quick"
echo "  - For testing: use --quick --samples 10"
