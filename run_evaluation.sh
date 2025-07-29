#!/bin/bash

# Run Schema Mapping Comprehensive Evaluation

set -e  # Exit on any error

echo "Schema Mapping Comprehensive Evaluation"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "eval_schema_mapping.py" ]; then
    echo "Error: eval_schema_mapping.py not found. Please run this script from the root directory."
    exit 1
fi

# Check if shared dataset loader exists
if [ ! -f "data/dataset_loader.py" ]; then
    echo "Error: shared dataset loader not found at data/dataset_loader.py"
    exit 1
fi

# Default parameters
MODE=""
#DATASETS="ITOps,NL2GQL,GraphQLRestBench"
DATASETS="ITOps,NL2GQL"
METHODS="SMT,ReAct,GNN,AnyTool"
#MODELS="granite,llama,mistral-large"
MODELS="mistral-large"
OUTPUT_DIR="./results"
CLEAN=false
VERBOSE=false
FORCE_REGENERATE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            MODE="quick"
            shift
            ;;
        --test)
            MODE="test"
            shift
            ;;
        --force-regenerate)
            FORCE_REGENERATE=true
            shift
            ;;
        --datasets)
            DATASETS="$2"
            shift 2
            ;;
        --methods)
            METHODS="$2"
            shift 2
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [MODE] [OPTIONS]"
            echo ""
            echo "Modes:"
            echo "  --quick              Quick evaluation with existing results"
            echo "  --test               Test functionality with minimal data"
            echo ""
            echo "Options:"
            echo "  --datasets DATASETS  Comma-separated list of datasets (default: ITOps,NL2GQL,GraphQLRestBench)"
            echo "  --methods METHODS    Comma-separated list of methods (default: SMT,ReAct,GNN,AnyTool)"
            echo "  --models MODELS      Comma-separated list of models (default: granite,llama,mistral-large)"
            echo "  --output-dir DIR     Output directory (default: ./results)"
            echo "  --force-regenerate   Force regeneration of all results (ignore existing)"
            echo "  --clean              Clean output directory before starting"
            echo "  --verbose            Enable verbose logging"
            echo "  --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --quick                                # Quick evaluation with existing results"
            echo "  $0 --test                                 # Test with minimal configuration"
            echo "  $0 --methods AnyTool --verbose            # Test AnyTool method with debug info"
            echo "  $0 --datasets ITOps --methods SMT,AnyTool # Compare SMT and AnyTool on ITOps"
            echo "  $0 --methods AnyTool --models mistral-large # Run AnyTool with specific model"
            echo "  $0 --force-regenerate                     # Regenerate all results"
            echo ""
            echo "Available Methods: SMT, ReAct, GNN, AnyTool"
            echo "Available Models: granite, llama, mistral-large"
            echo "Available Datasets: ITOps, NL2GQL, GraphQLRestBench"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set mode-specific defaults
case $MODE in
    "quick")
        echo "Quick Evaluation Mode"
        DATASETS="ITOps"
        MODELS="mistral-large"
        METHODS="AnyTool"
        ;;
    "test")
        echo "Test Mode"
        DATASETS="ITOps"
        MODELS="mistral-large"
        METHODS="AnyTool,GNN"
        VERBOSE=true
        ;;
    "")
        echo "Standard Evaluation Mode"
        echo "   Use --quick or --test for specific modes"
        ;;
esac

# Check for existing result files for information
echo "Checking for existing result files..."

check_method_results() {
    local method=$1
    local method_dir=${method,,}  # Convert to lowercase
    
    if [ -d "${method_dir}/results" ] && [ -n "$(ls -A ${method_dir}/results/*.json 2>/dev/null)" ]; then
        local count=$(ls ${method_dir}/results/*.json 2>/dev/null | wc -l)
        echo "   ${method}: $count result files found"
        
        # Check if method components are available
        case $method in
            "SMT")
                if [ -f "smt/smt_api_retriever.py" ]; then
                    echo "      SMT components available"
                else
                    echo "      SMT components missing"
                fi
                ;;
            "ReAct")
                if [ -f "react/app/agent.py" ]; then
                    echo "      ReAct components available"
                else
                    echo "      ReAct components missing"
                fi
                ;;
            "GNN")
                if [ -f "gnn/gnn_runner.py" ] && [ -f "gnn/gnn_results/mapping_gnn_results.pkl" ]; then
                    echo "      GNN components available"
                else
                    echo "      GNN components missing (run: cd gnn && ./setup_gnn.sh)"
                fi
                ;;
            "AnyTool")
                if [ -f "anytool/anytool_runner.py" ]; then
                    echo "      AnyTool components available"
                else
                    echo "      AnyTool components missing"
                fi
                ;;
        esac
    else
        echo "   ${method}: No result files found"
        
        # Check if method can be run
        case $method in
            "SMT")
                if [ -f "smt/smt_api_retriever.py" ]; then
                    echo "      SMT can be executed"
                else
                    echo "      SMT components missing"
                fi
                ;;
            "ReAct")
                if [ -f "react/app/agent.py" ]; then
                    echo "      ReAct can be executed"
                else
                    echo "      ReAct components missing"
                fi
                ;;
            "GNN")
                if [ -f "gnn/gnn_runner.py" ]; then
                    if [ -f "gnn/gnn_results/mapping_gnn_results.pkl" ]; then
                        echo "      GNN can be executed"
                    else
                        echo "      GNN model missing (run: cd gnn && ./setup_gnn.sh)"
                    fi
                else
                    echo "      GNN components missing"
                fi
                ;;
            "AnyTool")
                if [ -f "anytool/anytool_runner.py" ]; then
                    echo "      AnyTool can be executed"
                else
                    echo "      AnyTool components missing"
                fi
                ;;
        esac
    fi
}

# Check each method that will be evaluated
IFS=',' read -ra METHOD_ARRAY <<< "$METHODS"
for method in "${METHOD_ARRAY[@]}"; do
    check_method_results "$method"
done

# Display configuration
echo ""
echo "Configuration:"
echo "   Mode: ${MODE:-standard}"
echo "   Datasets: $DATASETS"
echo "   Methods: $METHODS"
echo "   Models: $MODELS"
echo "   Output directory: $OUTPUT_DIR"
echo "   Force regenerate: $FORCE_REGENERATE"
echo "   Clean output: $CLEAN"
echo "   Verbose: $VERBOSE"
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: No virtual environment detected. Consider activating one."
fi

# Build command line arguments
ARGS=""

# Convert comma-separated lists to space-separated for Python argparse
DATASET_ARGS=$(echo $DATASETS | tr ',' ' ')
METHOD_ARGS=$(echo $METHODS | tr ',' ' ')
MODEL_ARGS=$(echo $MODELS | tr ',' ' ')

ARGS="$ARGS --datasets $DATASET_ARGS"
ARGS="$ARGS --methods $METHOD_ARGS"
ARGS="$ARGS --models $MODEL_ARGS"
ARGS="$ARGS --output-dir $OUTPUT_DIR"

if [ "$MODE" = "quick" ]; then
    ARGS="$ARGS --quick"
fi

if [ "$FORCE_REGENERATE" = true ]; then
    ARGS="$ARGS --force-regenerate"
fi

if [ "$CLEAN" = true ]; then
    ARGS="$ARGS --clean"
fi

if [ "$VERBOSE" = true ]; then
    ARGS="$ARGS --debug"
fi

echo "Running evaluation with args: $ARGS"
echo ""

# Run the evaluation
python3 eval_schema_mapping.py $ARGS

# Check if results were generated
if [ -d "$OUTPUT_DIR" ]; then
    echo ""
    echo "Evaluation completed successfully!"
    echo "Results saved in: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    ls -la "$OUTPUT_DIR" | grep -E "\.(csv|json)$"
    
    # Show quick statistics if available
    if [ -f "$OUTPUT_DIR/evaluation_statistics.json" ]; then
        echo ""
        echo "Quick Statistics:"
        
        # Extract overall accuracy using Python
        python3 -c "
import json
try:
    with open('$OUTPUT_DIR/evaluation_statistics.json', 'r') as f:
        stats = json.load(f)
    overall = stats.get('overall', {})
    print(f'   Total evaluations: {overall.get(\"total_evaluations\", 0)}')
    print(f'   Successful evaluations: {overall.get(\"successful_evaluations\", 0)}')
    print(f'   Average accuracy: {overall.get(\"avg_accuracy\", 0.0):.3f}')
    
    # Show method performance
    print('   Method Performance:')
    for method, metrics in stats.get('by_method', {}).items():
        acc = metrics.get('avg_accuracy', 0.0)
        sr = metrics.get('success_rate', 0.0)
        print(f'     {method}: Accuracy={acc:.3f}, Success Rate={sr:.3f}')
        
    # Show dataset performance
    print('   Dataset Performance:')
    for dataset, metrics in stats.get('by_dataset', {}).items():
        acc = metrics.get('avg_accuracy', 0.0)
        print(f'     {dataset}: Accuracy={acc:.3f}')
except Exception as e:
    print(f'   Could not parse statistics: {e}')
"
    fi
    
    # Show result files created
    if [ -f "$OUTPUT_DIR/detailed_results.csv" ]; then
        TOTAL_ROWS=$(tail -n +2 "$OUTPUT_DIR/detailed_results.csv" | wc -l)
        echo ""
        echo "Result Details:"
        echo "   Total result rows: $TOTAL_ROWS"
    fi
    
    # Show method-specific result files
    echo ""
    echo "Method-specific result files:"
    for method in "${METHOD_ARRAY[@]}"; do
        method_lower=${method,,}
        if [ -d "${method_lower}/results" ]; then
            result_count=$(ls ${method_lower}/results/*.json 2>/dev/null | wc -l)
            echo "   ${method}: ${result_count} files in ${method_lower}/results/"
        fi
    done
else
    echo ""
    echo "No results directory found. Check for errors above."
    exit 1
fi

echo ""
echo "Schema Mapping Evaluation Complete!"

# Show next steps based on mode
case $MODE in
    "quick")
        echo ""
        echo "Next steps:"
        echo "   Run full evaluation: ./run_evaluation.sh"
        echo "   Test AnyTool specifically: ./run_evaluation.sh --methods AnyTool --verbose"
        echo "   Compare methods: ./run_evaluation.sh --methods SMT,AnyTool --datasets ITOps"
        ;;
    "test")
        echo ""
        echo "Test completed successfully!"
        echo "   Run full evaluation: ./run_evaluation.sh"
        echo "   Compare all methods: ./run_evaluation.sh --methods SMT,ReAct,GNN,AnyTool"
        ;;
    "")
        echo ""
        echo "Next steps:"
        echo "   View detailed results: cat $OUTPUT_DIR/detailed_results.csv"
        echo "   Test individual methods: ./run_evaluation.sh --methods AnyTool --test"
        echo "   Compare specific methods: ./run_evaluation.sh --methods ReAct,AnyTool"
        ;;
esac

# Show usage tips
echo ""
echo "Usage Tips:"
echo "   - Result files are cached automatically - use --force-regenerate to recreate"
echo "   - Individual method results are saved in method/results/ directories"
echo "   - Use --verbose for detailed logging during execution"
echo "   - GNN method requires setup: cd gnn && ./setup_gnn.sh"
echo "   - AnyTool requires API credentials in anytool/.env file"
echo "   - AnyTool supports: granite, llama, qwen, mistral-large models"

echo "=========================================="
