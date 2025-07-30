#!/bin/bash

# Run script for SMT Schema Mapping Method
# Usage: ./run_smt.sh [options]

set -e  # Exit on any error

echo "🚀 Starting SMT Schema Mapping Evaluation"
echo "=========================================="

# Check if we're in the smt directory
if [ ! -f "smt_runner.py" ]; then
    echo "❌ Error: smt_runner.py not found. Please run this script from the smt/ directory."
    echo "Usage: cd smt && ./run_smt.sh"
    exit 1
fi

# Check if shared dataset loader exists (one level up)
if [ ! -f "../data/dataset_loader.py" ]; then
    echo "❌ Error: shared dataset loader not found at ../data/dataset_loader.py"
    exit 1
fi

# Create data directory if it doesn't exist (one level up)
mkdir -p ../data

# Default parameters
DATASETS="ITOps"
MODELS="llama"
MAPPINGS_DIR="./mappings_output"
SCHEMA_DIR="../data/graphql"
CHROMA_DB_PATH="../data/chroma_db"
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
        --mappings-dir)
            MAPPINGS_DIR="$2"
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
            echo "  --models MODELS          Comma-separated list of models (default: llama)"
            echo "  --mappings-dir DIR       Directory for cached mapping results (default: ./mappings_output)"
            echo "  --schema-dir DIR         Directory containing GraphQL schemas (default: ../../data/graphql)"
            echo "  --chroma-db-path PATH    Path to ChromaDB database (default: ../data/chroma_db)"
            echo "  --quick                  Quick evaluation with minimal samples"
            echo "  --verbose                Enable verbose logging"
            echo "  --help                   Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --quick                                      # Quick evaluation"
            echo "  $0 --datasets ITOps,NL2GQL --models granite     # Multiple datasets and models"
            echo "  $0 --verbose                                    # Enable debug logging"
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Display configuration
echo "📋 Configuration:"
echo "   Datasets: $DATASETS"
echo "   Models: $MODELS"
echo "   Mappings directory: $MAPPINGS_DIR"
echo "   Schema directory: $SCHEMA_DIR"
echo "   ChromaDB path: $CHROMA_DB_PATH"
echo "   Quick mode: $QUICK"
echo "   Verbose: $VERBOSE"
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: No virtual environment detected. Consider activating one."
fi

# Check for SMT components
if [ ! -f "smt_api_retriever.py" ]; then
    echo "Warning: smt_api_retriever.py not found. SMT functionality may be limited."
fi

if [ ! -f "smt_schema_indexer.py" ]; then
    echo "Warning: smt_schema_indexer.py not found. Schema indexing may not work."
fi

# Check schema directory
if [ ! -d "$SCHEMA_DIR" ]; then
    echo "Warning: Schema directory $SCHEMA_DIR not found"
    echo "SMT evaluation will continue but may use mock schemas"
fi

# Check ChromaDB directory
if [ ! -d "$CHROMA_DB_PATH" ]; then
    echo "Warning: ChromaDB directory $CHROMA_DB_PATH not found"
    echo "SMT will create it or use limited functionality"
    # Create the centralized ChromaDB directory
    mkdir -p "$CHROMA_DB_PATH"
    echo "Created centralized ChromaDB directory: $CHROMA_DB_PATH"
fi

# Create necessary directories
mkdir -p "$MAPPINGS_DIR"
mkdir -p results

# Build command line arguments
ARGS=""

ARGS="$ARGS --datasets $DATASETS"
ARGS="$ARGS --models $MODELS"
ARGS="$ARGS --mappings-dir $MAPPINGS_DIR"
ARGS="$ARGS --schema-dir $SCHEMA_DIR"
ARGS="$ARGS --chroma-db-path $CHROMA_DB_PATH"

if [ "$QUICK" = true ]; then
    ARGS="$ARGS --quick"
fi

if [ "$VERBOSE" = true ]; then
    ARGS="$ARGS --verbose"
fi

echo "🔧 Running SMT evaluation with args: $ARGS"
echo ""

# Run the SMT evaluation
python3 smt_runner.py $ARGS

# Check if results were generated
if [ -d "results" ]; then
    echo ""
    echo "✅ SMT evaluation completed successfully!"
    echo "📁 Results saved in: ./results/"
    echo ""
    echo "📊 Generated files:"
    ls -la results/ | grep ".json"
    
    if [ -d "$MAPPINGS_DIR" ]; then
        echo ""
        echo "💾 Cached mappings in: $MAPPINGS_DIR"
        mapping_count=$(ls -1 "$MAPPINGS_DIR"/*.json 2>/dev/null | wc -l)
        echo "   Generated $mapping_count mapping files"
    fi
else
    echo ""
    echo "❌ No results directory found. Check for errors above."
    exit 1
fi

echo ""
echo "🎉 SMT Schema Mapping Evaluation Complete!"
echo "=========================================="
