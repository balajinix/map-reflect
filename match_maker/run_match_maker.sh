#!/bin/bash

# Run script for MatchMaker Schema Matching Method
# Faithful implementation of the paper: "Bootstrapping Self-Improvement of Language Model Programs for Zero-Shot Schema Matching"
# Usage: ./run_match_maker.sh [options]

set -e  # Exit on any error

echo "Starting MatchMaker Schema Matching Evaluation"
echo "=============================================="

# ============================================================================
# CONFIGURATION - Modify these variables to change evaluation settings
# ============================================================================

# Datasets to evaluate
# Options: "itops", "nl2gql", "mimic_omop", or combinations like "itops,nl2gql"
DATASETS="nl2gql"

# Models to use
# Examples:
#   - "watsonx/mistralai/mistral-large"
#   - "watsonx/meta-llama/llama-3-70b-instruct"
#   - "watsonx/ibm/granite-3-3-8b-instruct"
MODELS="watsonx/mistralai/mistral-medium-2505"

# Data directory (contains all dataset files)
DATA_DIR="../data"

# MatchMaker-specific options
REBUILD_INDEX=false     # Set to true to rebuild ColBERT index from scratch
USE_OPTIMIZATION=true   # Set to true to enable self-improvement (paper's key feature)
TOP_K=5                 # Number of candidates to retrieve (paper default: 5)
VERBOSE=false           # Set to true for detailed logging

# ============================================================================
# END CONFIGURATION
# ============================================================================

# Check if we're in the correct directory
if [ ! -f "match_maker_runner.py" ]; then
    echo "Error: match_maker_runner.py not found. Please run this script from the match_maker/ directory."
    echo "Usage: cd match_maker && ./run_match_maker.sh"
    exit 1
fi

# Get absolute paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Parse command line arguments (override configuration)
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
        --no-optimization)
            USE_OPTIMIZATION=false
            shift
            ;;
        --top-k)
            TOP_K="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Configuration variables (edit at top of script):"
            echo "  DATASETS         = $DATASETS (comma-separated: itops, nl2gql, mimic_omop)"
            echo "  MODELS           = $MODELS"
            echo "  DATA_DIR         = $DATA_DIR"
            echo "  REBUILD_INDEX    = $REBUILD_INDEX"
            echo "  USE_OPTIMIZATION = $USE_OPTIMIZATION"
            echo "  TOP_K            = $TOP_K"
            echo ""
            echo "Options:"
            echo "  --datasets DATASETS       Comma-separated list of datasets"
            echo "  --models MODELS           Comma-separated list of models"
            echo "  --data-dir DIR            Directory containing all dataset files"
            echo "  --rebuild-index           Rebuild the ColBERT/ChromaDB index from scratch"
            echo "  --no-optimization         Disable self-improvement optimization"
            echo "  --top-k K                 Number of candidates to retrieve (default: 5)"
            echo "  --verbose                 Enable verbose logging"
            echo "  --help                    Show this help message"
            echo ""
            echo "Dataset Files Required (per dataset):"
            echo "  For each dataset (e.g., 'itops'), you need:"
            echo "    - \${DATA_DIR}/\${dataset}_api_lookup.json  (required)"
            echo "    - \${DATA_DIR}/\${dataset}_hierarchy.json   (optional)"
            echo "    - \${DATA_DIR}/\${dataset}_eval.json        (required)"
            echo ""
            echo "For MIMIC-OMOP dataset:"
            echo "    - \${DATA_DIR}/mimic_omop_lookup.json       (required)"
            echo "    - \${DATA_DIR}/mimic_omop_eval.json         (required)"
            echo ""
            echo "MatchMaker Paper Features:"
            echo "  - Multi-vector embeddings (ColBERT-v2)"
            echo "  - Diverse candidate generation (semantic + reasoning)"
            echo "  - MCQ-based confidence scoring"
            echo "  - Self-improvement via synthetic in-context examples"
            echo ""
            echo "Examples:"
            echo "  # Run on nl2gql dataset"
            echo "  $0 --datasets nl2gql"
            echo ""
            echo "  # Run on multiple datasets"
            echo "  $0 --datasets itops,nl2gql,mimic_omop"
            echo ""
            echo "  # Rebuild index and disable optimization (faster)"
            echo "  $0 --datasets itops --rebuild-index --no-optimization"
            echo ""
            echo "  # Multiple models on MIMIC-OMOP"
            echo "  $0 --datasets mimic_omop --models 'granite,mistral-large'"
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

# Convert relative paths to absolute
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
echo "   Use optimization: $USE_OPTIMIZATION"
echo "   Top-k: $TOP_K"
echo "   Verbose: $VERBOSE"
echo ""

# Validate environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: No virtual environment detected."
    echo "It's recommended to run MatchMaker in a virtual environment."
fi

# Check for required dependencies
echo "Checking dependencies..."
python3 -c "import ragatouille" 2>/dev/null || {
    echo "Warning: RAGatouille (ColBERT-v2) not installed."
    echo "MatchMaker will fall back to sentence-transformers."
    echo "For faithful paper reproduction, install: pip install ragatouille"
}

# Validate data directory
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    exit 1
fi
echo "✓ Data directory found: $DATA_DIR"

# Check for dataset_loader
if [ ! -f "$DATA_DIR/dataset_loader.py" ]; then
    echo "Error: dataset_loader.py not found in $DATA_DIR"
    exit 1
fi
echo "✓ Dataset loader found: $DATA_DIR/dataset_loader.py"

# Validate dataset-specific files
IFS=',' read -ra DATASET_ARRAY <<< "$DATASETS"
for dataset in "${DATASET_ARRAY[@]}"; do
    dataset=$(echo "$dataset" | tr '[:upper:]' '[:lower:]' | xargs)  # lowercase and trim
    
    echo ""
    echo "Validating files for dataset: $dataset"
    
    # Determine property file name based on dataset
    if [ "$dataset" = "mimic_omop" ]; then
        PROPERTY_FILE="$DATA_DIR/mimic_omop_lookup.json"
    else
        PROPERTY_FILE="$DATA_DIR/${dataset}_api_lookup.json"
    fi
    
    # Check property lookup file (required)
    if [ ! -f "$PROPERTY_FILE" ]; then
        echo "  ERROR: Property lookup file not found: $PROPERTY_FILE"
        echo "  This file is required for MatchMaker to work."
        echo ""
        if [ "$dataset" = "mimic_omop" ]; then
            echo "  For MIMIC-OMOP dataset:"
            echo "    Download from: https://github.com/meniData1/MIMIC_2_OMOP"
            echo "    Or prepare source-target property mappings"
        else
            echo "  To generate ${dataset} files, run:"
            echo "    python setup_${dataset}_dataset.py --output $DATA_DIR"
        fi
        exit 1
    fi
    echo "  ✓ Property lookup: $PROPERTY_FILE"
    
    # Check hierarchy file (optional)
    HIERARCHY_FILE="$DATA_DIR/${dataset}_hierarchy.json"
    if [ -f "$HIERARCHY_FILE" ]; then
        echo "  ✓ Hierarchy file: $HIERARCHY_FILE"
    else
        echo "  ℹ Hierarchy file not found (optional): $HIERARCHY_FILE"
    fi
    
    # Check evaluation file (required)
    if [ "$dataset" = "mimic_omop" ]; then
        EVAL_FILE="$DATA_DIR/mimic_omop_eval.json"
    else
        EVAL_FILE="$DATA_DIR/${dataset}_eval.json"
    fi
    
    if [ -f "$EVAL_FILE" ]; then
        echo "  ✓ Evaluation file: $EVAL_FILE"
    else
        echo "  WARNING: Evaluation file not found: $EVAL_FILE"
        echo "  (Will be loaded by dataset_loader - may use different naming)"
    fi
done

# Check environment file
if [ -f ".env" ]; then
    echo ""
    echo "✓ .env file found - environment variables loaded"
else
    echo ""
    echo "ℹ No .env file found - using system environment variables"
fi

# ChromaDB and ColBERT index status
CHROMA_DIR="$DATA_DIR/chroma_db"
if [ -d "$CHROMA_DIR" ]; then
    if [ "$REBUILD_INDEX" = true ]; then
        echo ""
        echo "ChromaDB index exists and will be REBUILT"
        echo "  (This may take several minutes depending on dataset size)"
    else
        echo ""
        echo "✓ ChromaDB index exists and will be reused"
        echo "  (Each dataset uses: matchmaker_<dataset> collection)"
    fi
else
    echo ""
    echo "ChromaDB index will be created"
    mkdir -p "$CHROMA_DIR"
fi

# Self-improvement status
if [ "$USE_OPTIMIZATION" = true ]; then
    echo ""
    echo "Self-improvement optimization: ENABLED"
    echo "  This will:"
    echo "  1. Create evaluation set from source attributes"
    echo "  2. Run MatchMaker and capture execution traces"
    echo "  3. Use LLM Evaluator to score results"
    echo "  4. Bootstrap top-scoring traces as in-context examples"
    echo "  Note: This adds ~2-5 minutes per dataset but improves accuracy"
else
    echo ""
    echo "Self-improvement optimization: DISABLED"
    echo "  (Faster but may have lower accuracy)"
fi

# Create results directory
mkdir -p results

# Run evaluations
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
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Running: $dataset with $model"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # Determine property file
        if [ "$dataset" = "mimic_omop" ]; then
            PROPERTY_FILE="$DATA_DIR/mimic_omop_lookup.json"
        else
            PROPERTY_FILE="$DATA_DIR/${dataset}_api_lookup.json"
        fi
        
        # Build arguments
        ARGS="--datasets $dataset"
        ARGS="$ARGS --models $model"
        ARGS="$ARGS --property-lookup-file $PROPERTY_FILE"
        ARGS="$ARGS --data-dir $DATA_DIR"
        ARGS="$ARGS --top-k $TOP_K"
        
        if [ "$REBUILD_INDEX" = true ]; then
            ARGS="$ARGS --rebuild-index"
        fi
        
        if [ "$USE_OPTIMIZATION" = false ]; then
            ARGS="$ARGS --no-optimization"
        fi
        
        if [ "$VERBOSE" = true ]; then
            ARGS="$ARGS --verbose"
        fi
        
        # Run evaluation
        python3 match_maker_runner.py $ARGS
        
        if [ $? -ne 0 ]; then
            echo "ERROR: Evaluation failed for $dataset with $model"
            exit 1
        fi
        
        echo ""
        echo "✓ Completed: $dataset with $model"
    done
done

# Summary
if [ -d "results" ]; then
    echo ""
    echo "=============================================="
    echo "MatchMaker Evaluation Complete!"
    echo "=============================================="
    echo ""
    echo "Results saved in: ./results/"
    echo ""
    echo "Generated files:"
    ls -lh results/*.json 2>/dev/null || echo "No JSON files found"
    
    result_count=$(ls -1 results/*.json 2>/dev/null | wc -l)
    echo ""
    echo "Generated $result_count result file(s)"
    echo ""
    echo "MatchMaker Features Used:"
    echo "  ✓ Multi-vector embeddings (ColBERT-v2)"
    echo "  ✓ Semantic + Reasoning candidate generation"
    echo "  ✓ MCQ-based confidence scoring"
    if [ "$USE_OPTIMIZATION" = true ]; then
        echo "  ✓ Self-improvement optimization"
    else
        echo "  ✗ Self-improvement (disabled)"
    fi
    echo ""
    echo "Next steps:"
    echo "  - Analyze results: python analyze_results.py"
    echo "  - Compare with baselines"
    echo "  - Generate plots and tables for paper"
else
    echo ""
    echo "ERROR: No results directory found. Check for errors above."
    exit 1
fi
