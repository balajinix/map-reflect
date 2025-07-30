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

# Get absolute paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"

# Check if shared dataset loader exists
if [ ! -f "$DATA_DIR/dataset_loader.py" ]; then
    echo "Error: shared dataset loader not found at $DATA_DIR/dataset_loader.py"
    exit 1
fi

# Default parameters
MODE=""
#DATASETS="ITOps,NL2GQL,GraphQLRestBench"
DATASETS="ITOps"
METHODS="SMT,ReAct,GNN,AnyTool,MapReflect"
#MODELS="granite,llama,mistral-large"
MODELS="mistral-large"
OUTPUT_DIR="./results"
SAMPLES=""  # No limit by default
TIMEOUT="600"
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
        --samples)
            SAMPLES="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
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
            echo "  --datasets DATASETS  Comma-separated list of datasets (default: ITOps)"
            echo "  --methods METHODS    Comma-separated list of methods (default: SMT,ReAct,GNN,AnyTool,MapReflect)"
            echo "  --models MODELS      Comma-separated list of models (default: mistral-large)"
            echo "  --samples N          Limit to N samples per dataset (default: no limit)"
            echo "  --output-dir DIR     Output directory (default: ./results)"
            echo "  --timeout SECONDS    Timeout per method evaluation in seconds (default: 600)"
            echo "  --force-regenerate   Force regeneration of all results (ignore existing)"
            echo "  --clean              Clean output directory before starting"
            echo "  --verbose            Enable verbose logging"
            echo "  --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --quick                                # Quick evaluation with existing results"
            echo "  $0 --test                                 # Test with minimal configuration"
            echo "  $0 --samples 10                           # Limit to 10 samples per dataset"
            echo "  $0 --methods MapReflect --verbose         # Test MapReflect method with debug info"
            echo "  $0 --datasets ITOps --methods SMT,MapReflect # Compare SMT and MapReflect on ITOps"
            echo "  $0 --methods MapReflect --timeout 1200    # Run MapReflect with 20-minute timeout"
            echo "  $0 --force-regenerate                     # Regenerate all results"
            echo ""
            echo "Available Methods: SMT, ReAct, GNN, AnyTool, MapReflect"
            echo "Available Models: granite, llama, mistral-large"
            echo "Available Datasets: ITOps, NL2GQL, GraphQLRestBench"
            echo ""
            echo "Project Root: $SCRIPT_DIR"
            echo "Data Directory: $DATA_DIR"
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
        METHODS="MapReflect"
        SAMPLES="5"  # Use small sample for quick testing
        ;;
    "test")
        echo "Test Mode"
        DATASETS="ITOps"
        MODELS="mistral-large"
        METHODS="MapReflect,AnyTool"
        SAMPLES="3"  # Use very small sample for testing
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
    
    # Handle special case for MapReflect
    if [ "$method" = "MapReflect" ]; then
        method_dir="map_reflect"
    fi
    
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
            "MapReflect")
                if [ -f "map_reflect/map_reflect_runner.py" ] && [ -f "map_reflect/api_lookup.json" ]; then
                    echo "      MapReflect components available"
                else
                    echo "      MapReflect components missing"
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
            "MapReflect")
                if [ -f "map_reflect/map_reflect_runner.py" ]; then
                    if [ -f "map_reflect/api_lookup.json" ]; then
                        echo "      MapReflect can be executed"
                    else
                        echo "      MapReflect property lookup missing (map_reflect/api_lookup.json)"
                    fi
                else
                    echo "      MapReflect components missing"
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

# Check required files
echo ""
echo "Checking required files:"
echo "   Data directory: $DATA_DIR"
if [ -f "$DATA_DIR/hierarchy.json" ]; then
    echo "   Legacy hierarchy file: Found at $DATA_DIR/hierarchy.json"
else
    echo "   Legacy hierarchy file: Missing at $DATA_DIR/hierarchy.json"
fi

# Check dataset-specific hierarchy files if needed
if [[ "$METHODS" == *"AnyTool"* ]] || [[ "$METHODS" == *"MapReflect"* ]]; then
    echo ""
    echo "Checking dataset-specific hierarchy files:"
    IFS=',' read -ra DATASET_ARRAY <<< "$DATASETS"
    for dataset in "${DATASET_ARRAY[@]}"; do
        dataset_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
        hierarchy_file="$DATA_DIR/${dataset_lower}_hierarchy.json"
        
        if [ -f "$hierarchy_file" ]; then
            echo "   ${dataset}: Found hierarchy file at ${hierarchy_file}"
        else
            echo "   ${dataset}: Warning - hierarchy file ${hierarchy_file} not found"
            
            # Check for legacy hierarchy.json as fallback
            if [ -f "$DATA_DIR/hierarchy.json" ]; then
                echo "            Will use legacy hierarchy.json as fallback"
            else
                echo "            No hierarchy file available for this dataset"
            fi
        fi
    done
fi

if [ -f "$DATA_DIR/dataset_loader.py" ]; then
    echo "   Dataset loader: Found at $DATA_DIR/dataset_loader.py"
else
    echo "   Dataset loader: Missing at $DATA_DIR/dataset_loader.py"
    echo "   Error: Cannot proceed without dataset loader"
    exit 1
fi

if [ -d "$DATA_DIR/graphql" ]; then
    echo "   GraphQL schemas: Found at $DATA_DIR/graphql"
else
    echo "   GraphQL schemas: Missing at $DATA_DIR/graphql"
fi

# Check MapReflect specific files if MapReflect is in methods
if [[ "$METHODS" == *"MapReflect"* ]]; then
    if [ -f "map_reflect/api_lookup.json" ]; then
        echo "   MapReflect property lookup: Found at map_reflect/api_lookup.json"
    else
        echo "   MapReflect property lookup: Missing at map_reflect/api_lookup.json"
        echo "   Warning: MapReflect requires api_lookup.json file"
    fi
fi

# Display configuration
echo ""
echo "Configuration:"
echo "   Mode: ${MODE:-standard}"
echo "   Datasets: $DATASETS"
echo "   Methods: $METHODS"
echo "   Models: $MODELS"
if [ -n "$SAMPLES" ]; then
    echo "   Samples per dataset: $SAMPLES"
else
    echo "   Samples per dataset: all available"
fi
echo "   Output directory: $OUTPUT_DIR"
echo "   Timeout: ${TIMEOUT}s ($(($TIMEOUT / 60)) minutes)"
echo "   Force regenerate: $FORCE_REGENERATE"
echo "   Clean output: $CLEAN"
echo "   Verbose: $VERBOSE"
echo "   Project root: $SCRIPT_DIR"
echo "   Data directory: $DATA_DIR"
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: No virtual environment detected. Consider activating one."
fi

# Function to run command with timeout
run_with_timeout() {
    local timeout_duration=$1
    local method_key=$2
    shift 2  # Remove timeout and method_key from arguments
    local cmd="$@"
    
    echo "   Running with ${timeout_duration}s timeout: $cmd"
    
    # Record start time
    local start_time=$(date +%s)
    
    # Run command with timeout
    if timeout "$timeout_duration" bash -c "$cmd"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        method_times["$method_key"]=$duration
        method_status["$method_key"]="SUCCESS"
        echo "   Completed in ${duration}s"
        return 0
    else
        local exit_code=$?
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        if [ $exit_code -eq 124 ]; then
            # Timeout occurred
            method_times["$method_key"]=$timeout_duration
            method_timeouts["$method_key"]=1
            method_status["$method_key"]="TIMEOUT"
            echo "   ⚠️  TIMEOUT after ${timeout_duration}s"
        else
            # Other error
            method_times["$method_key"]=$duration
            method_status["$method_key"]="FAILED"
            echo "   ❌ FAILED after ${duration}s (exit code: $exit_code)"
        fi
        return $exit_code
    fi
}

# Run method-specific evaluations that need to be run from their directories
run_method_evaluation() {
    local method=$1
    local method_lower=${method,,}
    
    # Handle special case for MapReflect
    if [ "$method" = "MapReflect" ]; then
        method_lower="map_reflect"
    fi
    
    echo "Running $method evaluation..."
    
    case $method in
        "AnyTool")
            if [ -f "$SCRIPT_DIR/anytool/anytool_runner.py" ]; then
                echo "   Executing AnyTool from anytool directory"
                
                # Build AnyTool specific arguments with absolute paths
                ANYTOOL_ARGS="--datasets $DATASETS --models $MODELS"
                
                # Add samples parameter if specified
                if [ -n "$SAMPLES" ]; then
                    ANYTOOL_ARGS="$ANYTOOL_ARGS --samples $SAMPLES"
                fi
                
                # Check for dataset-specific hierarchy files
                IFS=',' read -ra DATASET_CHECK <<< "$DATASETS"
                FIRST_DATASET=$(echo "${DATASET_CHECK[0]}" | tr '[:upper:]' '[:lower:]')
                DATASET_HIERARCHY="$DATA_DIR/${FIRST_DATASET}_hierarchy.json"
                
                if [ -f "$DATASET_HIERARCHY" ]; then
                    ANYTOOL_ARGS="$ANYTOOL_ARGS --hierarchy-file $DATASET_HIERARCHY"
                    echo "   Using dataset-specific hierarchy: $DATASET_HIERARCHY"
                elif [ -f "$DATA_DIR/hierarchy.json" ]; then
                    ANYTOOL_ARGS="$ANYTOOL_ARGS --hierarchy-file $DATA_DIR/hierarchy.json"
                    echo "   Using legacy hierarchy: $DATA_DIR/hierarchy.json"
                else
                    echo "   Warning: No hierarchy file found"
                fi
                
                ANYTOOL_ARGS="$ANYTOOL_ARGS --data-dir $DATA_DIR"
                
                if [ -d "$DATA_DIR/graphql" ]; then
                    ANYTOOL_ARGS="$ANYTOOL_ARGS --schema-dir $DATA_DIR/graphql"
                fi
                
                if [ "$VERBOSE" = true ]; then
                    ANYTOOL_ARGS="$ANYTOOL_ARGS --verbose"
                fi
                
                echo "   AnyTool args: $ANYTOOL_ARGS"
                
                # Create method key for timing
                local method_key="AnyTool_${DATASETS}_${MODELS}"
                
                # Run AnyTool evaluation with timeout
                if run_with_timeout "$TIMEOUT" "$method_key" "cd '$SCRIPT_DIR/anytool' && python3 anytool_runner.py $ANYTOOL_ARGS"; then
                    echo "   AnyTool evaluation completed successfully"
                else
                    local exit_code=$?
                    if [ "${method_timeouts[$method_key]}" = "1" ]; then
                        echo "   AnyTool evaluation timed out after ${TIMEOUT}s"
                    else
                        echo "   AnyTool evaluation failed"
                    fi
                    return $exit_code
                fi
                
                # Return to project root
                cd "$SCRIPT_DIR"
            else
                echo "   AnyTool runner not found, skipping"
                return 1
            fi
            ;;
        "MapReflect")
            if [ -f "$SCRIPT_DIR/map_reflect/map_reflect_runner.py" ]; then
                echo "   Executing MapReflect from map_reflect directory"
                
                # Build MapReflect specific arguments with absolute paths
                MAPREFLECT_ARGS="--datasets $DATASETS --models $MODELS"
                
                # Add samples parameter if specified
                if [ -n "$SAMPLES" ]; then
                    MAPREFLECT_ARGS="$MAPREFLECT_ARGS --samples $SAMPLES"
                fi
                
                MAPREFLECT_ARGS="$MAPREFLECT_ARGS --property-lookup-file $SCRIPT_DIR/map_reflect/api_lookup.json"
                MAPREFLECT_ARGS="$MAPREFLECT_ARGS --data-dir $DATA_DIR"
                
                if [ -d "$DATA_DIR/graphql" ]; then
                    MAPREFLECT_ARGS="$MAPREFLECT_ARGS --schema-dir $DATA_DIR/graphql"
                fi
                
                if [ "$VERBOSE" = true ]; then
                    MAPREFLECT_ARGS="$MAPREFLECT_ARGS --verbose"
                fi
                
                echo "   MapReflect args: $MAPREFLECT_ARGS"
                echo "   Note: MapReflect will use dataset-specific hierarchy files (e.g., itops_hierarchy.json)"
                
                # Create method key for timing
                local method_key="MapReflect_${DATASETS}_${MODELS}"
                
                # Run MapReflect evaluation with timeout
                if run_with_timeout "$TIMEOUT" "$method_key" "cd '$SCRIPT_DIR/map_reflect' && python3 map_reflect_runner.py $MAPREFLECT_ARGS"; then
                    echo "   MapReflect evaluation completed successfully"
                else
                    local exit_code=$?
                    if [ "${method_timeouts[$method_key]}" = "1" ]; then
                        echo "   MapReflect evaluation timed out after ${TIMEOUT}s"
                    else
                        echo "   MapReflect evaluation failed"
                    fi
                    return $exit_code
                fi
                
                # Return to project root
                cd "$SCRIPT_DIR"
            else
                echo "   MapReflect runner not found, skipping"
                return 1
            fi
            ;;
        *)
            echo "   Method $method: No specific pre-execution needed"
            ;;
    esac
    
    return 0
}

# Run individual method evaluations if needed
echo "Pre-running method evaluations..."
OVERALL_START_TIME=$(date +%s)

for method in "${METHOD_ARRAY[@]}"; do
    case $method in
        "AnyTool")
            # Always run AnyTool from its directory to ensure proper paths
            if [[ "$METHODS" == *"AnyTool"* ]]; then
                run_method_evaluation "$method"
            fi
            ;;
        "MapReflect")
            # Always run MapReflect from its directory to ensure proper paths
            if [[ "$METHODS" == *"MapReflect"* ]]; then
                run_method_evaluation "$method"
            fi
            ;;
    esac
done

# Build command line arguments for the main evaluation script
ARGS=""

# Convert comma-separated lists to space-separated for Python argparse
DATASET_ARGS=$(echo $DATASETS | tr ',' ' ')
METHOD_ARGS=$(echo $METHODS | tr ',' ' ')
MODEL_ARGS=$(echo $MODELS | tr ',' ' ')

ARGS="$ARGS --datasets $DATASET_ARGS"
ARGS="$ARGS --methods $METHOD_ARGS"
ARGS="$ARGS --models $MODEL_ARGS"
ARGS="$ARGS --output-dir $OUTPUT_DIR"

# Add samples parameter if specified
if [ -n "$SAMPLES" ]; then
    ARGS="$ARGS --max-samples $SAMPLES"
fi

# Add absolute paths for data components
ARGS="$ARGS --data-dir $DATA_DIR"

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

echo "Running main evaluation with args: $ARGS"
echo ""

# Make sure we're in the project root
cd "$SCRIPT_DIR"

# Run the evaluation
MAIN_EVAL_START_TIME=$(date +%s)
echo "Running main evaluation (may use cached results from pre-run methods)..."
python3 eval_schema_mapping.py $ARGS
MAIN_EVAL_END_TIME=$(date +%s)
MAIN_EVAL_DURATION=$((MAIN_EVAL_END_TIME - MAIN_EVAL_START_TIME))

OVERALL_END_TIME=$(date +%s)
OVERALL_DURATION=$((OVERALL_END_TIME - OVERALL_START_TIME))

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
        
        # Handle special case for MapReflect
        if [ "$method" = "MapReflect" ]; then
            method_lower="map_reflect"
        fi
        
        if [ -d "${method_lower}/results" ]; then
            result_count=$(ls ${method_lower}/results/*.json 2>/dev/null | wc -l)
            echo "   ${method}: ${result_count} files in ${method_lower}/results/"
        fi
    done
    
    # Show timing report
    echo ""
    echo "Execution Time Report:"
    echo "======================"
    echo "   Overall execution time: ${OVERALL_DURATION}s ($(($OVERALL_DURATION / 60))m $(($OVERALL_DURATION % 60))s)"
    echo "   Main evaluation time: ${MAIN_EVAL_DURATION}s ($(($MAIN_EVAL_DURATION / 60))m $(($MAIN_EVAL_DURATION % 60))s)"
    echo ""
    echo "   Method-specific times:"
    
    # Sort methods by execution time for better readability
    for method in "${METHOD_ARRAY[@]}"; do
        method_key_pattern="${method}_"
        for key in "${!method_times[@]}"; do
            if [[ "$key" == ${method_key_pattern}* ]]; then
                local duration=${method_times[$key]}
                local status=${method_status[$key]}
                local status_icon=""
                
                case $status in
                    "SUCCESS") status_icon="✅" ;;
                    "TIMEOUT") status_icon="⏰" ;;
                    "FAILED") status_icon="❌" ;;
                    *) status_icon="❓" ;;
                esac
                
                if [ "${method_timeouts[$key]}" = "1" ]; then
                    echo "     ${status_icon} ${key}: ${duration}s (TIMEOUT after ${TIMEOUT}s)"
                else
                    echo "     ${status_icon} ${key}: ${duration}s ($(($duration / 60))m $(($duration % 60))s) - $status"
                fi
            fi
        done
    done
    
    # Summary of timeouts and failures
    local timeout_count=0
    local failed_count=0
    local success_count=0
    
    for key in "${!method_status[@]}"; do
        case ${method_status[$key]} in
            "SUCCESS") ((success_count++)) ;;
            "TIMEOUT") ((timeout_count++)) ;;
            "FAILED") ((failed_count++)) ;;
        esac
    done
    
    if [ $timeout_count -gt 0 ] || [ $failed_count -gt 0 ]; then
        echo ""
        echo "   Summary: $success_count successful, $failed_count failed, $timeout_count timed out"
        
        if [ $timeout_count -gt 0 ]; then
            echo "   ⚠️  Consider increasing timeout with --timeout <seconds> for timed out methods"
        fi
    fi
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
        echo "   Test MapReflect specifically: ./run_evaluation.sh --methods MapReflect --verbose"
        echo "   Compare methods: ./run_evaluation.sh --methods SMT,MapReflect --datasets ITOps"
        ;;
    "test")
        echo ""
        echo "Test completed successfully!"
        echo "   Run full evaluation: ./run_evaluation.sh"
        echo "   Compare all methods: ./run_evaluation.sh --methods SMT,ReAct,GNN,AnyTool,MapReflect"
        ;;
    "")
        echo ""
        echo "Next steps:"
        echo "   View detailed results: cat $OUTPUT_DIR/detailed_results.csv"
        echo "   Test individual methods: ./run_evaluation.sh --methods MapReflect --test"
        echo "   Compare specific methods: ./run_evaluation.sh --methods ReAct,MapReflect"
        ;;
esac

# Show usage tips
echo ""
echo "Usage Tips:"
echo "   - Result files are cached automatically - use --force-regenerate to recreate"
echo "   - Individual method results are saved in method/results/ directories"
echo "   - Use --verbose for detailed logging during execution"
echo "   - Use --timeout <seconds> to adjust timeout per method (default: 600s/10min)"
echo "   - Use --samples N to limit samples per dataset for testing"
echo "   - GNN method requires setup: cd gnn && ./setup_gnn.sh"
echo "   - AnyTool requires API credentials in anytool/.env file"
echo "   - MapReflect requires API credentials and api_lookup.json in map_reflect/ directory"
echo "   - Dataset-specific hierarchy files: use {dataset}_hierarchy.json (e.g., itops_hierarchy.json)"
echo "   - AnyTool supports: granite, llama, qwen, mistral-large models"
echo "   - MapReflect supports: watsonx/mistralai/mistral-large (and other LiteLLM models)"
echo "   - All paths are resolved as absolute paths from project root: $SCRIPT_DIR"
echo ""
echo "Timeout Management:"
echo "   - Each method evaluation has a ${TIMEOUT}s timeout by default"
echo "   - Increase timeout for slower methods: --timeout 1200 (20 minutes)"
echo "   - Check timing report above to adjust timeout appropriately"

echo "=========================================="
