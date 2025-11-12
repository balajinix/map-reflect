#!/bin/bash

# Run script for ReMatch Schema Matching Method
# Faithful implementation of the paper: "ReMatch: Retrieval Enhanced Schema Matching with LLMs"
# Usage: ./run_rematch.sh [options]

set -e  # Exit on any error

echo "Starting ReMatch Schema Matching Evaluation"
echo "=============================================="

# ============================================================================
# CONFIGURATION - Modify these variables to change evaluation settings
# ============================================================================

# Datasets to evaluate
# Options: "itops", "nl2gql", "mimic_omop", "synthea", or combinations like "itops,nl2gql"
DATASETS="nl2gql"

# Models to use
# Examples:
#   - "gpt-4"
#   - "gpt-4-turbo"
#   - "claude-3-opus"
#   - "watsonx/mistralai/mistral-medium-2505"
MODELS="gpt-4"

# Data directory (contains all dataset files)
DATA_DIR="../data"

# ReMatch-specific options
TOP_J=2                 # Number of candidate target tables to retrieve (paper default: 2)
TOP_K=5                 # Number of top attribute matches to return (paper default: 5)
USE_GUIDANCE=false      # Set to true to enable relevance feedback mechanism
GUIDANCE_FILE=""        # Path to guidance mappings JSON file
VERBOSE=false           # Set to true for detailed logging

# ============================================================================
# END CONFIGURATION
# ============================================================================

# Check if we're in the correct directory
if [ ! -f "rematch_runner.py" ]; then
    echo "Error: rematch_runner.py not found. Please run this script from the rematch/ directory."
    echo "Usage: cd rematch && ./run_rematch.sh"
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
        --top-j)
            TOP_J="$2"
            shift 2
            ;;
        --top-k)
            TOP_K="$2"
            shift 2
            ;;
        --use-guidance)
            USE_GUIDANCE=true
            shift
            ;;
        --guidance-file)
            GUIDANCE_FILE="$2"
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
            echo "  DATASETS       = $DATASETS (comma-separated: itops, nl2gql, mimic_omop, synthea)"
            echo "  MODELS         = $MODELS"
            echo "  DATA_DIR       = $DATA_DIR"
            echo "  TOP_J          = $TOP_J (candidate target tables)"
            echo "  TOP_K          = $TOP_K (top attribute matches)"
            echo "  USE_GUIDANCE   = $USE_GUIDANCE"
            echo "  GUIDANCE_FILE  = $GUIDANCE_FILE"
            echo ""
            echo "Options:"
            echo "  --datasets DATASETS       Comma-separated list of datasets"
            echo "  --models MODELS           Comma-separated list of models"
            echo "  --data-dir DIR            Directory containing all dataset files"
            echo "  --top-j J                 Number of candidate target tables to retrieve (default: 2)"
            echo "  --top-k K                 Number of top attribute matches to return (default: 5)"
            echo "  --use-guidance            Enable relevance feedback mechanism"
            echo "  --guidance-file FILE      Path to guidance mappings JSON file"
            echo "  --verbose                 Enable verbose logging"
            echo "  --help                    Show this help message"
            echo ""
            echo "Dataset Files Required (per dataset):"
            echo "  For each dataset (e.g., 'itops'), you need:"
            echo "    - \${DATA_DIR}/\${dataset}_api_lookup.json  (required)"
            echo "    - \${DATA_DIR}/\${dataset}_eval.json        (required)"
            echo ""
            echo "For MIMIC-OMOP dataset:"
            echo "    - \${DATA_DIR}/mimic_omop_lookup.json       (required)"
            echo "    - \${DATA_DIR}/mimic_omop_eval.json         (required)"
            echo ""
            echo "For Synthea dataset:"
            echo "    - \${DATA_DIR}/synthea_omop_lookup.json     (required)"
            echo "    - \${DATA_DIR}/synthea_eval.json            (required)"
            echo ""
            echo "Guidance File Format (JSON):"
            echo "  {"
            echo "    \"source_table_1\": [\"target_table_1\", \"attribute_pair_1\"],"
            echo "    \"source_table_2\": [\"target_table_2\", \"attribute_pair_2\"]"
            echo "  }"
            echo ""
            echo "ReMatch Paper Features:"
            echo "  - Passage-based document representations"
            echo "  - Two-stage retrieval (tables → attributes)"
            echo "  - LLM-based semantic ranking"
            echo "  - Zero-shot (no training required)"
            echo "  - Optional relevance feedback (guidance)"
            echo ""
            echo "Examples:"
            echo "  # Run on MIMIC-OMOP dataset (paper's main evaluation)"
            echo "  $0 --datasets mimic_omop --models gpt-4"
            echo ""
            echo "  # Run on Synthea dataset (paper's secondary evaluation)"
            echo "  $0 --datasets synthea --models gpt-4"
            echo ""
            echo "  # Run with custom J and K values (paper's grid search)"
            echo "  $0 --datasets mimic_omop --top-j 1 --top-k 3"
            echo ""
            echo "  # Run with guidance (relevance feedback - Section 4.2)"
            echo "  $0 --datasets mimic_omop --use-guidance --guidance-file guidance.json"
            echo ""
            echo "  # Multiple datasets and models"
            echo "  $0 --datasets 'mimic_omop,synthea' --models 'watsonx/mistralai/mistral-medium-2505,watsonx/meta-llama/llama-3-3-70b-instruct'"
            echo ""
            echo "  # Paper reproduction settings:"
            echo "  # - MIMIC (J=1,K=1): $0 --datasets mimic_omop --top-j 1 --top-k 1"
            echo "  # - MIMIC (J=2,K=5): $0 --datasets mimic_omop --top-j 2 --top-k 5"
            echo "  # - Synthea (J=1,K=1): $0 --datasets synthea --top-j 1 --top-k 1"
            echo "  # - With guidance: Add --use-guidance --guidance-file data/guidance.json"
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

if [[ -n "$GUIDANCE_FILE" ]] && [[ "$GUIDANCE_FILE" != /* ]]; then
    GUIDANCE_FILE="$SCRIPT_DIR/$GUIDANCE_FILE"
fi

# Display configuration
echo ""
echo "Configuration:"
echo "   Datasets: $DATASETS"
echo "   Models: $MODELS"
echo "   Data directory: $DATA_DIR"
echo "   Top-J (tables): $TOP_J"
echo "   Top-K (matches): $TOP_K"
echo "   Use guidance: $USE_GUIDANCE"
if [ "$USE_GUIDANCE" = true ] && [ -n "$GUIDANCE_FILE" ]; then
    echo "   Guidance file: $GUIDANCE_FILE"
fi
echo "   Verbose: $VERBOSE"
echo ""

# Validate environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: No virtual environment detected."
    echo "It's recommended to run ReMatch in a virtual environment."
fi

# Check for required dependencies
echo "Checking dependencies..."
python3 -c "import sentence_transformers" 2>/dev/null || {
    echo "Error: sentence-transformers not installed."
    echo "ReMatch requires sentence-transformers for embeddings."
    echo "Install with: pip install sentence-transformers"
    exit 1
}
echo "✓ sentence-transformers installed"

python3 -c "import litellm" 2>/dev/null || {
    echo "Error: litellm not installed."
    echo "ReMatch requires litellm for LLM API calls."
    echo "Install with: pip install litellm"
    exit 1
}
echo "✓ litellm installed"

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

# Validate guidance file if specified
if [ "$USE_GUIDANCE" = true ]; then
    if [ -z "$GUIDANCE_FILE" ]; then
        echo "Warning: --use-guidance specified but no --guidance-file provided"
        echo "Guidance will be enabled but no predefined mappings will be used"
    elif [ ! -f "$GUIDANCE_FILE" ]; then
        echo "Error: Guidance file not found: $GUIDANCE_FILE"
        exit 1
    else
        echo "✓ Guidance file found: $GUIDANCE_FILE"
    fi
fi

# Validate dataset-specific files
IFS=',' read -ra DATASET_ARRAY <<< "$DATASETS"
for dataset in "${DATASET_ARRAY[@]}"; do
    dataset=$(echo "$dataset" | tr '[:upper:]' '[:lower:]' | xargs)  # lowercase and trim
    
    echo ""
    echo "Validating files for dataset: $dataset"
    
    # Determine property file name based on dataset
    if [ "$dataset" = "mimic_omop" ]; then
        PROPERTY_FILE="$DATA_DIR/mimic_omop_lookup.json"
    elif [ "$dataset" = "synthea" ]; then
        PROPERTY_FILE="$DATA_DIR/synthea_omop_lookup.json"
    else
        PROPERTY_FILE="$DATA_DIR/${dataset}_api_lookup.json"
    fi
    
    # Check property lookup file (required)
    if [ ! -f "$PROPERTY_FILE" ]; then
        echo "  ERROR: Property lookup file not found: $PROPERTY_FILE"
        echo "  This file is required for ReMatch to work."
        echo ""
        if [ "$dataset" = "mimic_omop" ]; then
            echo "  For MIMIC-OMOP dataset:"
            echo "    Download from: https://github.com/meniData1/MIMIC_2_OMOP"
            echo "    This is the paper's primary evaluation dataset (Table 1)"
        elif [ "$dataset" = "synthea" ]; then
            echo "  For Synthea dataset:"
            echo "    Use the OMAP benchmark dataset"
            echo "    This is the paper's secondary evaluation dataset (Table 1)"
        else
            echo "  To generate ${dataset} files, run:"
            echo "    python setup_${dataset}_dataset.py --output $DATA_DIR"
        fi
        exit 1
    fi
    echo "  ✓ Property lookup: $PROPERTY_FILE"
    
    # Check evaluation file (required)
    if [ "$dataset" = "mimic_omop" ]; then
        EVAL_FILE="$DATA_DIR/mimic_omop_eval.json"
    elif [ "$dataset" = "synthea" ]; then
        EVAL_FILE="$DATA_DIR/synthea_eval.json"
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
    
    # Check for required API keys based on models
    IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"
    for model in "${MODEL_ARRAY[@]}"; do
        model=$(echo "$model" | xargs)
        
        if [[ "$model" == gpt-* ]] || [[ "$model" == "openai/"* ]]; then
            if [ -z "$OPENAI_API_KEY" ]; then
                echo "  WARNING: OPENAI_API_KEY not set (required for GPT models)"
            fi
        elif [[ "$model" == claude-* ]] || [[ "$model" == "anthropic/"* ]]; then
            if [ -z "$ANTHROPIC_API_KEY" ]; then
                echo "  WARNING: ANTHROPIC_API_KEY not set (required for Claude models)"
            fi
        elif [[ "$model" == watsonx/* ]]; then
            if [ -z "$WATSONX_API_KEY" ]; then
                echo "  WARNING: WATSONX_API_KEY not set (required for WatsonX models)"
            fi
        fi
    done
else
    echo ""
    echo "ℹ No .env file found - using system environment variables"
fi

# ReMatch algorithm status
echo ""
echo "ReMatch Algorithm Configuration:"
echo "  Two-stage retrieval process:"
echo "    Stage 1: Retrieve top-$TOP_J candidate target tables (embedding similarity)"
echo "    Stage 2: LLM ranks attributes within candidates → top-$TOP_K matches"
echo ""
echo "  Paper settings for reproduction:"
echo "    MIMIC-OMOP: J=1,K=1 or J=2,K=5 (Table 2)"
echo "    Synthea:    J=1,K=1 or J=2,K=5 (Table 3)"
echo "    With guidance: +20.38% improvement at J=1,K=1 (Table 5)"
echo ""

if [ "$USE_GUIDANCE" = true ]; then
    echo "  Relevance Feedback (Guidance): ENABLED"
    echo "    Paper Section 4.2: Human matcher provides known mapping"
    echo "    Effect: Includes guided table in candidates + informs LLM"
    echo "    Expected improvement: 7-20% (Table 5)"
else
    echo "  Relevance Feedback (Guidance): DISABLED"
    echo "    Pure zero-shot evaluation (paper's main results)"
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
        echo "┌─────────────────────────────────────────────────────────┐"
        echo "│ Running: $dataset with $model"
        echo "└─────────────────────────────────────────────────────────┘"
        
        # Determine property file
        if [ "$dataset" = "mimic_omop" ]; then
            PROPERTY_FILE="$DATA_DIR/mimic_omop_lookup.json"
        elif [ "$dataset" = "synthea" ]; then
            PROPERTY_FILE="$DATA_DIR/synthea_omop_lookup.json"
        else
            PROPERTY_FILE="$DATA_DIR/${dataset}_api_lookup.json"
        fi
        
        # Build arguments
        ARGS="--datasets $dataset"
        ARGS="$ARGS --models $model"
        ARGS="$ARGS --property-lookup-file $PROPERTY_FILE"
        ARGS="$ARGS --data-dir $DATA_DIR"
        ARGS="$ARGS --top-j $TOP_J"
        ARGS="$ARGS --top-k $TOP_K"
        
        if [ "$USE_GUIDANCE" = true ]; then
            ARGS="$ARGS --use-guidance"
            if [ -n "$GUIDANCE_FILE" ]; then
                ARGS="$ARGS --guidance-file $GUIDANCE_FILE"
            fi
        fi
        
        if [ "$VERBOSE" = true ]; then
            ARGS="$ARGS --verbose"
        fi
        
        # Run evaluation
        python3 rematch_runner.py $ARGS
        
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
    echo "ReMatch Evaluation Complete!"
    echo "=============================================="
    echo ""
    echo "Results saved in: ./results/"
    echo ""
    echo "Generated files:"
    ls -lh results/*rematch*.json 2>/dev/null || echo "No ReMatch result files found"
    
    result_count=$(ls -1 results/*rematch*.json 2>/dev/null | wc -l)
    echo ""
    echo "Generated $result_count ReMatch result file(s)"
    echo ""
    echo "ReMatch Features Used:"
    echo "  ✓ Passage-based document representations"
    echo "  ✓ Two-stage retrieval (J=$TOP_J → K=$TOP_K)"
    echo "  ✓ LLM semantic ranking"
    echo "  ✓ Zero-shot (no training)"
    if [ "$USE_GUIDANCE" = true ]; then
        echo "  ✓ Relevance feedback (guidance)"
    else
        echo "  ✗ Relevance feedback (disabled)"
    fi
    echo ""
    
    # Display paper comparison context
    echo "Paper Results for Comparison (MIMIC-OMOP):"
    echo "  ReMatch (J=1, K=1): Acc@1=0.424"
    echo "  ReMatch (J=2, K=5): Acc@5=0.729"
    echo "  ReMatch + Guidance (J=1): Acc@1=0.539 (+20.38%)"
    echo ""
    echo "Paper Results for Comparison (Synthea):"
    echo "  ReMatch (J=1, K=1): Acc@1=0.562"
    echo "  ReMatch (J=2, K=5): Acc@5=0.743"
    echo ""
    
    echo "Next steps:"
    echo "  - Analyze results: python analyze_results.py"
    echo "  - Compare with SMAT baseline"
    echo "  - Generate plots matching paper figures"
    echo "  - Try different J/K combinations for grid search (Table 2, 3)"
    echo "  - Test guidance mechanism (Table 5)"
else
    echo ""
    echo "ERROR: No results directory found. Check for errors above."
    exit 1
fi