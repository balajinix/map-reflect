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
METHODS="GNN"
#METHODS="SMT,ReAct,GNN,AnyTool,MapReflect"
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
        --dataset)
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
            echo "  --dataset DATASET  Comma-separated list of dataset (default: ITOps)"
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
            echo "  $0 --dataset ITOps --methods SMT,MapReflect # Compare SMT and MapReflect on ITOps"
            echo "  $0 --methods MapReflect --timeout 1200    # Run MapReflect with 20-minute timeout"
            echo "  $0 --force-regenerate                     # Regenerate all results"
            echo ""
            echo "Available Methods: SMT, ReAct, GNN, AnyTool, MapReflect"
            echo "Available Models: granite, llama, mistral-large"
            echo "Available Datasets: ITOps, NL2GQL, GraphQLRestBench"
            echo ""
            echo "Note: Each method will be called individually for each dataset/model combination"
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

# Convert comma-separated lists to arrays
IFS=',' read -ra DATASET_ARRAY <<< "$DATASETS"
IFS=',' read -ra METHOD_ARRAY <<< "$METHODS"
IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"

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
                if [ -f "gnn/gnn_runner.py" ]; then
                    # Check for dataset-specific GNN model files for each dataset
                    for dataset in "${DATASET_ARRAY[@]}"; do
                        local dataset_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
                        local GNN_MODEL_PATH="$DATA_DIR/gnn_results/${dataset_lower}_mapping_gnn_results.pkl"
                        local GNN_GRAPH_PATH="$DATA_DIR/${dataset_lower}_graph_data.pkl"
                        
                        if [ -f "$GNN_MODEL_PATH" ] && [ -f "$GNN_GRAPH_PATH" ]; then
                            echo "      GNN components available for $dataset dataset"
                            echo "         Model: $GNN_MODEL_PATH"
                            echo "         Graph: $GNN_GRAPH_PATH"
                        else
                            echo "      GNN components missing for $dataset dataset"
                            echo "         Expected model: $GNN_MODEL_PATH"
                            echo "         Expected graph: $GNN_GRAPH_PATH"
                            echo "         Run: cd gnn && ./setup_gnn.sh --dataset $dataset_lower"
                        fi
                    done
                else
                    echo "      GNN components missing (gnn_runner.py not found)"
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
                if [ -f "map_reflect/map_reflect_runner.py" ]; then
                    # Check for dataset-specific or legacy api_lookup files
                    for dataset in "${DATASET_ARRAY[@]}"; do
                        local dataset_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
                        local dataset_api_lookup="$DATA_DIR/${dataset_lower}_api_lookup.json"
                        
                        if [ -f "$dataset_api_lookup" ] || [ -f "map_reflect/api_lookup.json" ]; then
                            echo "      MapReflect components available for $dataset"
                        else
                            echo "      MapReflect components missing for $dataset (no api_lookup file found)"
                        fi
                    done
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
                    # Check for dataset-specific GNN model files for each dataset
                    for dataset in "${DATASET_ARRAY[@]}"; do
                        local dataset_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
                        local GNN_MODEL_NEW="$DATA_DIR/gnn_results/${dataset_lower}_mapping_gnn_results.pkl"
                        local GNN_MODEL_OLD="gnn/gnn_results/mapping_gnn_results.pkl"
                        local GNN_GRAPH_NEW="$DATA_DIR/${dataset_lower}_graph_data.pkl"
                        local GNN_GRAPH_OLD="gnn/graph_data.pkl"
                        
                        if [ -f "$GNN_MODEL_NEW" ] && [ -f "$GNN_GRAPH_NEW" ]; then
                            echo "      GNN can be executed for $dataset (new parameterized setup)"
                            echo "         Model: $GNN_MODEL_NEW"
                            echo "         Graph: $GNN_GRAPH_NEW"
                        elif [ -f "$GNN_MODEL_OLD" ] && [ -f "$GNN_GRAPH_OLD" ]; then
                            echo "      GNN can be executed for $dataset (legacy setup)"
                            echo "         Model: $GNN_MODEL_OLD"
                            echo "         Graph: $GNN_GRAPH_OLD"
                            echo "         Consider upgrading: cd gnn && ./setup_gnn.sh --dataset $dataset_lower"
                        else
                            echo "      GNN model missing for $dataset dataset"
                            echo "         Run: cd gnn && ./setup_gnn.sh --dataset $dataset_lower"
                            
                            # Check if setup files exist
                            if [ -f "gnn/setup_gnn.sh" ]; then
                                echo "         Setup script available: gnn/setup_gnn.sh"
                            else
                                echo "         Setup script missing: gnn/setup_gnn.sh"
                            fi
                        fi
                    done
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
                    for dataset in "${DATASET_ARRAY[@]}"; do
                        local dataset_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
                        local dataset_api_lookup="$DATA_DIR/${dataset_lower}_api_lookup.json"
                        
                        if [ -f "$dataset_api_lookup" ] || [ -f "map_reflect/api_lookup.json" ]; then
                            echo "      MapReflect can be executed for $dataset"
                        else
                            echo "      MapReflect property lookup missing for $dataset (${dataset_lower}_api_lookup.json or legacy api_lookup.json)"
                        fi
                    done
                else
                    echo "      MapReflect components missing"
                fi
                ;;
        esac
    fi
}

# Check each method that will be evaluated
for method in "${METHOD_ARRAY[@]}"; do
    check_method_results "$method"
done

# Check required files
echo ""
echo "Checking required files:"
echo "   Data directory: $DATA_DIR"

# Check dataset-specific hierarchy files
if [[ "$METHODS" == *"AnyTool"* ]] || [[ "$METHODS" == *"MapReflect"* ]]; then
    echo ""
    echo "Checking dataset-specific hierarchy files:"
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

# Check dataset-specific api_lookup files for MapReflect
if [[ "$METHODS" == *"MapReflect"* ]]; then
    echo ""
    echo "Checking dataset-specific api_lookup files:"
    for dataset in "${DATASET_ARRAY[@]}"; do
        dataset_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
        api_lookup_file="$DATA_DIR/${dataset_lower}_api_lookup.json"
        
        if [ -f "$api_lookup_file" ]; then
            echo "   ${dataset}: Found api_lookup file at ${api_lookup_file}"
        else
            echo "   ${dataset}: Warning - api_lookup file ${api_lookup_file} not found"
            
            # Check for legacy api_lookup.json as fallback
            if [ -f "map_reflect/api_lookup.json" ]; then
                echo "            Will use legacy map_reflect/api_lookup.json as fallback"
            else
                echo "            No api_lookup file available for this dataset"
            fi
        fi
    done
fi

# Check GNN-specific files for each dataset
if [[ "$METHODS" == *"GNN"* ]]; then
    echo ""
    echo "Checking GNN-specific files:"
    for dataset in "${DATASET_ARRAY[@]}"; do
        dataset_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
        
        # Check for new parameterized GNN files
        gnn_model_file="$DATA_DIR/gnn_results/${dataset_lower}_mapping_gnn_results.pkl"
        gnn_graph_file="$DATA_DIR/${dataset_lower}_graph_data.pkl"
        gnn_api_lookup="$DATA_DIR/${dataset_lower}_api_lookup.json"
        
        echo "   ${dataset} GNN files:"
        
        if [ -f "$gnn_model_file" ]; then
            echo "      ✓ Model: ${gnn_model_file}"
        else
            echo "      ✗ Model missing: ${gnn_model_file}"
        fi
        
        if [ -f "$gnn_graph_file" ]; then
            echo "      ✓ Graph: ${gnn_graph_file}"
        else
            echo "      ✗ Graph missing: ${gnn_graph_file}"
        fi
        
        if [ -f "$gnn_api_lookup" ]; then
            echo "      ✓ API lookup: ${gnn_api_lookup}"
        else
            echo "      ✗ API lookup missing: ${gnn_api_lookup}"
        fi
        
        # Check GNN runner script
        if [ -f "gnn/gnn_runner.py" ]; then
            echo "      ✓ Runner: gnn/gnn_runner.py"
        else
            echo "      ✗ Runner missing: gnn/gnn_runner.py"
        fi
        
        # Provide setup instructions if files are missing
        if [ ! -f "$gnn_model_file" ] || [ ! -f "$gnn_graph_file" ]; then
            echo "      → Run: cd gnn && ./setup_gnn.sh --dataset ${dataset_lower}"
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

# Display configuration
echo ""
echo "Configuration:"
echo "   Mode: ${MODE:-standard}"
echo "   Datasets: $DATASETS (${#DATASET_ARRAY[@]} datasets)"
echo "   Methods: $METHODS (${#METHOD_ARRAY[@]} methods)"
echo "   Models: $MODELS (${#MODEL_ARRAY[@]} models)"
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

# Calculate total combinations
TOTAL_COMBINATIONS=$((${#DATASET_ARRAY[@]} * ${#METHOD_ARRAY[@]} * ${#MODEL_ARRAY[@]}))
echo "Total method evaluations to run: $TOTAL_COMBINATIONS"
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: No virtual environment detected. Consider activating one."
fi

# Initialize associative arrays for tracking timing and status
declare -A method_times
declare -A method_timeouts
declare -A method_status

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
    local dataset=$2
    local model=$3
    local method_lower=${method,,}
    
    # Handle special case for MapReflect
    if [ "$method" = "MapReflect" ]; then
        method_lower="map_reflect"
    fi
    
    echo "Running $method evaluation for dataset: $dataset, model: $model..."
    
    case $method in
        "GNN")
            if [ -f "$SCRIPT_DIR/gnn/gnn_runner.py" ]; then
                echo "   Executing GNN from gnn directory"
                
                local dataset_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
                
                # Build GNN specific arguments with absolute paths
                GNN_ARGS="--dataset $dataset --model $model"
                
                # Use dataset-specific file paths
                GNN_MODEL_PATH="$DATA_DIR/gnn_results/${dataset_lower}_mapping_gnn_results.pkl"
                GNN_GRAPH_PATH="$DATA_DIR/${dataset_lower}_graph_data.pkl"
                GNN_API_LOOKUP="$DATA_DIR/${dataset_lower}_api_lookup.json"
                GNN_SCHEMA_DIR="$DATA_DIR/${dataset_lower}_graphql"
                
                # Add file paths to arguments
                GNN_ARGS="$GNN_ARGS --gnn-model-path $GNN_MODEL_PATH"
                GNN_ARGS="$GNN_ARGS --graph-data-path $GNN_GRAPH_PATH"
                GNN_ARGS="$GNN_ARGS --api-lookup-file $GNN_API_LOOKUP"
                GNN_ARGS="$GNN_ARGS --schema-dir $GNN_SCHEMA_DIR"
                GNN_ARGS="$GNN_ARGS --chroma-db-path $DATA_DIR/chroma_db"
                
                # Add samples parameter if specified
                if [ -n "$SAMPLES" ]; then
                    GNN_ARGS="$GNN_ARGS --quick"  # GNN uses --quick for limited samples
                fi
                
                if [ "$VERBOSE" = true ]; then
                    GNN_ARGS="$GNN_ARGS --verbose"
                fi
                
                echo "   GNN args: $GNN_ARGS"
                echo "   Using dataset-specific files for: $dataset_lower"
                
                # Verify required files exist
                if [ ! -f "$GNN_MODEL_PATH" ]; then
                    echo "   ❌ GNN model missing: $GNN_MODEL_PATH"
                    echo "      Run: cd gnn && ./setup_gnn.sh --dataset $dataset_lower"
                    return 1
                fi
                
                if [ ! -f "$GNN_GRAPH_PATH" ]; then
                    echo "   ❌ GNN graph data missing: $GNN_GRAPH_PATH"
                    echo "      Run: cd gnn && ./setup_gnn.sh --dataset $dataset_lower"
                    return 1
                fi
                
                if [ ! -f "$GNN_API_LOOKUP" ]; then
                    echo "   ❌ GNN API lookup missing: $GNN_API_LOOKUP"
                    echo "      Ensure API lookup file exists for the dataset"
                    return 1
                fi
                
                # Create method key for timing
                local method_key="GNN_${dataset}_${model}"
                
                # Run GNN evaluation with timeout
                if run_with_timeout "$TIMEOUT" "$method_key" "cd '$SCRIPT_DIR/gnn' && python3 gnn_runner.py $GNN_ARGS"; then
                    echo "   GNN evaluation completed successfully"
                else
                    local exit_code=$?
                    if [ "${method_timeouts[$method_key]}" = "1" ]; then
                        echo "   GNN evaluation timed out after ${TIMEOUT}s"
                    else
                        echo "   GNN evaluation failed"
                    fi
                    return $exit_code
                fi
                
                # Return to project root
                cd "$SCRIPT_DIR"
            else
                echo "   GNN runner not found, skipping"
                return 1
            fi
            ;;
        "SMT")
            if [ -f "$SCRIPT_DIR/smt/smt_api_retriever.py" ]; then
                echo "   Executing SMT from smt directory"
                
                # Build SMT specific arguments
                SMT_ARGS="--dataset $dataset --model $model"
                
                # Add samples parameter if specified
                if [ -n "$SAMPLES" ]; then
                    SMT_ARGS="$SMT_ARGS --samples $SAMPLES"
                fi
                
                SMT_ARGS="$SMT_ARGS --data-dir $DATA_DIR"
                
                if [ -d "$DATA_DIR/graphql" ]; then
                    SMT_ARGS="$SMT_ARGS --schema-dir $DATA_DIR/graphql"
                fi
                
                if [ "$VERBOSE" = true ]; then
                    SMT_ARGS="$SMT_ARGS --verbose"
                fi
                
                echo "   SMT args: $SMT_ARGS"
                
                # Create method key for timing
                local method_key="SMT_${dataset}_${model}"
                
                # Run SMT evaluation with timeout
                if run_with_timeout "$TIMEOUT" "$method_key" "cd '$SCRIPT_DIR/smt' && python3 smt_api_retriever.py $SMT_ARGS"; then
                    echo "   SMT evaluation completed successfully"
                else
                    local exit_code=$?
                    if [ "${method_timeouts[$method_key]}" = "1" ]; then
                        echo "   SMT evaluation timed out after ${TIMEOUT}s"
                    else
                        echo "   SMT evaluation failed"
                    fi
                    return $exit_code
                fi
                
                # Return to project root
                cd "$SCRIPT_DIR"
            else
                echo "   SMT runner not found, skipping"
                return 1
            fi
            ;;
        "ReAct")
            if [ -f "$SCRIPT_DIR/react/app/agent.py" ]; then
                echo "   Executing ReAct from react directory"
                
                # Build ReAct specific arguments
                REACT_ARGS="--dataset $dataset --model $model"
                
                # Add samples parameter if specified
                if [ -n "$SAMPLES" ]; then
                    REACT_ARGS="$REACT_ARGS --samples $SAMPLES"
                fi
                
                REACT_ARGS="$REACT_ARGS --data-dir $DATA_DIR"
                
                if [ -d "$DATA_DIR/graphql" ]; then
                    REACT_ARGS="$REACT_ARGS --schema-dir $DATA_DIR/graphql"
                fi
                
                if [ "$VERBOSE" = true ]; then
                    REACT_ARGS="$REACT_ARGS --verbose"
                fi
                
                echo "   ReAct args: $REACT_ARGS"
                
                # Create method key for timing
                local method_key="ReAct_${dataset}_${model}"
                
                # Run ReAct evaluation with timeout
                if run_with_timeout "$TIMEOUT" "$method_key" "cd '$SCRIPT_DIR/react' && python3 app/agent.py $REACT_ARGS"; then
                    echo "   ReAct evaluation completed successfully"
                else
                    local exit_code=$?
                    if [ "${method_timeouts[$method_key]}" = "1" ]; then
                        echo "   ReAct evaluation timed out after ${TIMEOUT}s"
                    else
                        echo "   ReAct evaluation failed"
                    fi
                    return $exit_code
                fi
                
                # Return to project root
                cd "$SCRIPT_DIR"
            else
                echo "   ReAct runner not found, skipping"
                return 1
            fi
            ;;
        "AnyTool")
            if [ -f "$SCRIPT_DIR/anytool/anytool_runner.py" ]; then
                echo "   Executing AnyTool from anytool directory"
                
                # Build AnyTool specific arguments with absolute paths
                ANYTOOL_ARGS="--dataset $dataset --model $model"
                
                # Add samples parameter if specified
                if [ -n "$SAMPLES" ]; then
                    ANYTOOL_ARGS="$ANYTOOL_ARGS --samples $SAMPLES"
                fi
                
                # Check for dataset-specific hierarchy files
                local dataset_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
                DATASET_HIERARCHY="$DATA_DIR/${dataset_lower}_hierarchy.json"
                
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
                local method_key="AnyTool_${dataset}_${model}"
                
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
                MAPREFLECT_ARGS="--dataset $dataset --model $model"
                
                # Add samples parameter if specified
                if [ -n "$SAMPLES" ]; then
                    MAPREFLECT_ARGS="$MAPREFLECT_ARGS --samples $SAMPLES"
                fi
                
                # Check for dataset-specific api_lookup files
                local dataset_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
                DATASET_API_LOOKUP="$DATA_DIR/${dataset_lower}_api_lookup.json"
                
                if [ -f "$DATASET_API_LOOKUP" ]; then
                    MAPREFLECT_ARGS="$MAPREFLECT_ARGS --property-lookup-file $DATASET_API_LOOKUP"
                    echo "   Using dataset-specific api_lookup: $DATASET_API_LOOKUP"
                elif [ -f "map_reflect/api_lookup.json" ]; then
                    MAPREFLECT_ARGS="$MAPREFLECT_ARGS --property-lookup-file $SCRIPT_DIR/map_reflect/api_lookup.json"
                    echo "   Using legacy api_lookup: $SCRIPT_DIR/map_reflect/api_lookup.json"
                else
                    echo "   Warning: No api_lookup file found"
                fi
                
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
                local method_key="MapReflect_${dataset}_${model}"
                
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

# Run individual method evaluations
echo "Running method evaluations..."
OVERALL_START_TIME=$(date +%s)

# Initialize counters for progress tracking
CURRENT_COMBINATION=0
TOTAL_COMBINATIONS=$((${#DATASET_ARRAY[@]} * ${#METHOD_ARRAY[@]} * ${#MODEL_ARRAY[@]}))

# Triple nested loop: for each dataset, method, and model combination
for dataset in "${DATASET_ARRAY[@]}"; do
    for method in "${METHOD_ARRAY[@]}"; do
        for model in "${MODEL_ARRAY[@]}"; do
            CURRENT_COMBINATION=$((CURRENT_COMBINATION + 1))
            echo ""
            echo "=========================================="
            echo "Combination $CURRENT_COMBINATION of $TOTAL_COMBINATIONS"
            echo "Dataset: $dataset | Method: $method | Model: $model"
            echo "=========================================="
            
            # Run the specific method evaluation
            run_method_evaluation "$method" "$dataset" "$model"
            
            # Brief pause between evaluations to avoid overwhelming the system
            sleep 1
        done
    done
done

# Build command line arguments for the main evaluation script (aggregation)
ARGS=""

# Convert arrays back to comma-separated lists for the main evaluation script
DATASET_ARGS=$(IFS=','; echo "${DATASET_ARRAY[*]}")
METHOD_ARGS=$(IFS=','; echo "${METHOD_ARRAY[*]}")
MODEL_ARGS=$(IFS=','; echo "${MODEL_ARRAY[*]}")

ARGS="$ARGS --dataset $DATASET_ARGS"
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

echo ""
echo "=========================================="
echo "Running main evaluation aggregation..."
echo "Args: $ARGS"
echo "=========================================="

# Make sure we're in the project root
cd "$SCRIPT_DIR"

# Run the evaluation
MAIN_EVAL_START_TIME=$(date +%s)
echo "Running main evaluation aggregation (collecting results from individual method runs)..."
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
    echo "   Main evaluation aggregation time: ${MAIN_EVAL_DURATION}s ($(($MAIN_EVAL_DURATION / 60))m $(($MAIN_EVAL_DURATION % 60))s)"
    echo ""
    echo "   Individual method execution times:"
    
    # Group results by method for better readability
    for method in "${METHOD_ARRAY[@]}"; do
        echo "   $method:"
        for dataset in "${DATASET_ARRAY[@]}"; do
            for model in "${MODEL_ARRAY[@]}"; do
                local method_key="${method}_${dataset}_${model}"
                if [[ -n "${method_times[$method_key]}" ]]; then
                    local duration=${method_times[$method_key]}
                    local status=${method_status[$method_key]}
                    local status_icon=""
                    
                    case $status in
                        "SUCCESS") status_icon="✅" ;;
                        "TIMEOUT") status_icon="⏰" ;;
                        "FAILED") status_icon="❌" ;;
                        *) status_icon="❓" ;;
                    esac
                    
                    if [ "${method_timeouts[$method_key]}" = "1" ]; then
                        echo "     ${status_icon} ${dataset}/${model}: ${duration}s (TIMEOUT)"
                    else
                        echo "     ${status_icon} ${dataset}/${model}: ${duration}s ($(($duration / 60))m $(($duration % 60))s) - $status"
                    fi
                fi
            done
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
        echo "   Compare methods: ./run_evaluation.sh --methods SMT,MapReflect --dataset ITOps"
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
echo "   - Each method is now called individually for each dataset/model combination"
echo "   - Total combinations executed: $TOTAL_COMBINATIONS (${#DATASET_ARRAY[@]} datasets × ${#METHOD_ARRAY[@]} methods × ${#MODEL_ARRAY[@]} models)"
echo "   - Result files are cached automatically - use --force-regenerate to recreate"
echo "   - Individual method results are saved in method/results/ directories"
echo "   - Use --verbose for detailed logging during execution"
echo "   - Use --timeout <seconds> to adjust timeout per method (default: 600s/10min)"
echo "   - Use --samples N to limit samples per dataset for testing"
echo "   - GNN method requires setup: cd gnn && ./setup_gnn.sh --dataset <dataset>"
echo "   - AnyTool requires API credentials in anytool/.env file"
echo "   - MapReflect requires API credentials and dataset-specific api_lookup.json in data/ directory"
echo "   - Dataset-specific hierarchy files: use {dataset}_hierarchy.json (e.g., itops_hierarchy.json)"
echo "   - Dataset-specific api_lookup files: use {dataset}_api_lookup.json (e.g., itops_api_lookup.json)"
echo "   - AnyTool supports: granite, llama, qwen, mistral-large models"
echo "   - MapReflect supports: watsonx/mistralai/mistral-large (and other LiteLLM models)"
echo "   - All paths are resolved as absolute paths from project root: $SCRIPT_DIR"
echo ""
echo "File Structure Expected:"
echo "   data/{dataset}_hierarchy.json     - Dataset-specific hierarchy files for AnyTool/MapReflect"
echo "   data/{dataset}_api_lookup.json    - Dataset-specific API lookup files for MapReflect"
echo "   data/{dataset}_dataset.jsonl     - Dataset files (e.g., itops_dataset.jsonl)"
echo "   data/hierarchy.json               - Legacy fallback hierarchy file"
echo "   map_reflect/api_lookup.json       - Legacy fallback API lookup file"
echo ""
echo "Timeout Management:"
echo "   - Each method evaluation has a ${TIMEOUT}s timeout by default"
echo "   - Increase timeout for slower methods: --timeout 1200 (20 minutes)"
echo "   - Check timing report above to adjust timeout appropriately"

echo "=========================================="
