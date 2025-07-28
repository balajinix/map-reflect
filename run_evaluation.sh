#!/bin/bash

# Run Schema Mapping Comprehensive Evaluation
# This script provides easy access to eval_schema_mapping.py with common usage patterns

set -e  # Exit on any error

echo "🚀 Schema Mapping Comprehensive Evaluation"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "eval_schema_mapping.py" ]; then
    echo "❌ Error: eval_schema_mapping.py not found. Please run this script from the root directory."
    exit 1
fi

# Check if shared dataset loader exists
if [ ! -f "data/dataset_loader.py" ]; then
    echo "❌ Error: shared dataset loader not found at data/dataset_loader.py"
    exit 1
fi

# Default parameters
MODE=""
DATASETS="ITOps,NL2GQL,GraphQLRestBench"
METHODS="SMT,ReAct"
#MODELS="granite,llama,qwen"
MODELS="mistral"
OUTPUT_DIR="./results"
GENERATE_LATEX=false
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
        --paper)
            MODE="paper"
            GENERATE_LATEX=true
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
        --generate-latex)
            GENERATE_LATEX=true
            shift
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
            echo "  --paper              Generate full paper analysis with LaTeX tables"
            echo "  --test               Test functionality with minimal data"
            echo ""
            echo "Options:"
            echo "  --datasets DATASETS  Comma-separated list of datasets (default: ITOps,NL2GQL,GraphQLRestBench)"
            echo "  --methods METHODS    Comma-separated list of methods (default: SMT,ReAct)"
            echo "  --models MODELS      Comma-separated list of models (default: granite,llama,qwen)"
            echo "  --output-dir DIR     Output directory (default: ./results)"
            echo "  --generate-latex     Generate LaTeX tables for paper"
            echo "  --force-regenerate   Force regeneration of all results (ignore existing)"
            echo "  --clean              Clean output directory before starting"
            echo "  --verbose            Enable verbose logging"
            echo "  --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --quick                          # Quick evaluation with existing results"
            echo "  $0 --paper                          # Generate paper with LaTeX tables"
            echo "  $0 --test                           # Test with minimal configuration"
            echo "  $0 --methods ReAct,SMT --generate-latex # Analyze ReAct and SMT results"
            echo "  $0 --datasets ITOps --verbose       # Analyze only ITOps dataset with debug info"
            echo "  $0 --force-regenerate --paper       # Regenerate all results and create paper"
            echo ""
            echo "Workflow:"
            echo "  1. The evaluation script will automatically run methods if result files don't exist"
            echo "  2. Use --force-regenerate to ignore existing result files and run methods again"
            echo "  3. Use --paper to generate LaTeX tables for your research paper"
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set mode-specific defaults
case $MODE in
    "quick")
        echo "📋 Quick Evaluation Mode"
        DATASETS="ITOps"
        MODELS="granite"
        METHODS="ReAct"  # Use ReAct for quick mode
        ;;
    "paper")
        echo "📋 Paper Generation Mode"
        GENERATE_LATEX=true
        echo "   Will generate LaTeX tables for AAAI paper"
        ;;
    "test")
        echo "📋 Test Mode"
        DATASETS="ITOps"
        MODELS="granite"
        METHODS="ReAct"  # Use ReAct for test mode
        VERBOSE=true
        ;;
    "")
        echo "📋 Standard Evaluation Mode"
        echo "   Use --quick, --paper, or --test for specific modes"
        ;;
esac

# Check for existing result files for information
echo "🔍 Checking for existing result files..."

REACT_COUNT=0
SMT_COUNT=0

if [ -d "react/results" ] && [ -n "$(ls -A react/results/*.json 2>/dev/null)" ]; then
    REACT_COUNT=$(ls react/results/*.json 2>/dev/null | wc -l)
    echo "   ✅ ReAct: $REACT_COUNT result files found"
else
    echo "   ❌ ReAct: No result files found"
fi

if [ -d "smt/results" ] && [ -n "$(ls -A smt/results/*.json 2>/dev/null)" ]; then
    SMT_COUNT=$(ls smt/results/*.json 2>/dev/null | wc -l)
    echo "   ✅ SMT: $SMT_COUNT result files found"
else
    echo "   ❌ SMT: No result files found"
fi

# Information about method availability
if [[ "$METHODS" == *"ReAct"* ]]; then
    if [ -d "react/app" ] && [ -f "react/app/agent.py" ]; then
        echo "   ✅ ReAct components available for execution"
    else
        echo "   ⚠️  ReAct components not found - will use fallback if no result files exist"
    fi
fi

if [[ "$METHODS" == *"SMT"* ]]; then
    if [ -d "smt" ] && [ -f "smt/smt_api_retriever.py" ]; then
        echo "   ✅ SMT components available for execution"
    else
        echo "   ⚠️  SMT components not found - will use fallback if no result files exist"
    fi
fi

# Display configuration
echo ""
echo "📋 Configuration:"
echo "   Mode: ${MODE:-standard}"
echo "   Datasets: $DATASETS"
echo "   Methods: $METHODS"
echo "   Models: $MODELS"
echo "   Output directory: $OUTPUT_DIR"
echo "   Generate LaTeX: $GENERATE_LATEX"
echo "   Force regenerate: $FORCE_REGENERATE"
echo "   Clean output: $CLEAN"
echo "   Verbose: $VERBOSE"
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: No virtual environment detected. Consider activating one."
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

if [ "$GENERATE_LATEX" = true ]; then
    ARGS="$ARGS --generate-latex"
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

echo "🔧 Running evaluation with args: $ARGS"
echo ""

# Run the evaluation
python3 eval_schema_mapping.py $ARGS

# Check if results were generated
if [ -d "$OUTPUT_DIR" ]; then
    echo ""
    echo "✅ Evaluation completed successfully!"
    echo "📁 Results saved in: $OUTPUT_DIR"
    echo ""
    echo "📊 Generated files:"
    ls -la "$OUTPUT_DIR" | grep -E "\.(csv|json|tex)$"
    
    if [ "$GENERATE_LATEX" = true ]; then
        echo ""
        echo "📝 LaTeX tables for paper:"
        ls -la "$OUTPUT_DIR"/*.tex 2>/dev/null || echo "   No LaTeX files generated"
    fi
    
    # Show quick statistics if available
    if [ -f "$OUTPUT_DIR/evaluation_statistics.json" ]; then
        echo ""
        echo "📈 Quick Statistics:"
        
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
    for method, metrics in stats.get('by_method', {}).items():
        print(f'   {method} accuracy: {metrics.get(\"avg_accuracy\", 0.0):.3f}')
        
    # Show dataset performance
    for dataset, metrics in stats.get('by_dataset', {}).items():
        print(f'   {dataset} accuracy: {metrics.get(\"avg_accuracy\", 0.0):.3f}')
except Exception as e:
    print(f'   Could not parse statistics: {e}')
"
    fi
    
    # Show result files created
    if [ -f "$OUTPUT_DIR/detailed_results.csv" ]; then
        TOTAL_ROWS=$(tail -n +2 "$OUTPUT_DIR/detailed_results.csv" | wc -l)
        echo ""
        echo "📋 Result Details:"
        echo "   Total result rows: $TOTAL_ROWS"
        
        # Show method breakdown
        if command -v cut &> /dev/null && command -v sort &> /dev/null && command -v uniq &> /dev/null; then
            echo "   Methods evaluated:"
            tail -n +2 "$OUTPUT_DIR/detailed_results.csv" | cut -d',' -f2 | sort | uniq -c | sed 's/^/     /'
            echo "   Datasets evaluated:"
            tail -n +2 "$OUTPUT_DIR/detailed_results.csv" | cut -d',' -f1 | sort | uniq -c | sed 's/^/     /'
        fi
    fi
else
    echo ""
    echo "❌ No results directory found. Check for errors above."
    exit 1
fi

echo ""
echo "🎉 Schema Mapping Evaluation Complete!"

# Show next steps based on mode
case $MODE in
    "quick")
        echo ""
        echo "💡 Next steps:"
        echo "   Run full evaluation: ./run_evaluation.sh --paper"
        echo "   Force regenerate results: ./run_evaluation.sh --force-regenerate --paper"
        ;;
    "paper")
        echo ""
        echo "📄 Paper files ready!"
        echo "   Use the generated .tex files in your LaTeX document"
        if [ "$GENERATE_LATEX" = true ]; then
            echo "   LaTeX tables saved to: $OUTPUT_DIR"
            echo "   Include in your paper with: \\input{$OUTPUT_DIR/main_results.tex}"
        fi
        ;;
    "test")
        echo ""
        echo "🧪 Test completed successfully!"
        echo "   Run full evaluation: ./run_evaluation.sh --paper"
        ;;
    "")
        echo ""
        echo "💡 Next steps:"
        echo "   Generate LaTeX tables: ./run_evaluation.sh --generate-latex"
        echo "   Run full paper analysis: ./run_evaluation.sh --paper"
        ;;
esac

# Show usage tips
echo ""
echo "💡 Usage Tips:"
echo "   - Result files are cached automatically - use --force-regenerate to recreate"
echo "   - Individual method results are saved in react/results/ and smt/results/"
echo "   - The evaluation script will run methods if result files don't exist"
echo "   - Use --verbose for detailed logging during execution"

echo "=========================================="
