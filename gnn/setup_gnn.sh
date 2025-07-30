#!/bin/bash

# Setup script for GNN Schema Mapping Method
# This script prepares the GNN environment and trains the model

set -e  # Exit on any error

echo "Setting up GNN Schema Mapping Environment"
echo "============================================="

# Check if we're in the gnn directory
if [ ! -f "populate_nx.py" ] || [ ! -f "train_gnn.py" ]; then
    echo "Error: GNN training files not found. Please run this script from the gnn/ directory."
    echo "Expected files: populate_nx.py, train_gnn.py"
    exit 1
fi

# Default parameters
DATASET="itops"
MAPPINGS_DIR="../data/mappings_output"
API_LOOKUP_FILE="../data/itops_api_lookup.json"
OUTPUT_DIR="../data"
FORCE_RETRAIN=false
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --mappings-dir)
            MAPPINGS_DIR="$2"
            shift 2
            ;;
        --api-lookup-file)
            API_LOOKUP_FILE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --force-retrain)
            FORCE_RETRAIN=true
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
            echo "  --dataset NAME           Dataset name for output files (default: itops)"
            echo "  --mappings-dir DIR       Directory containing mapping files (default: ../data/mappings_output)"
            echo "  --api-lookup-file PATH   Path to API lookup file (default: ../data/itops_api_lookup.json)"
            echo "  --output-dir DIR         Output directory for generated files (default: ../data)"
            echo "  --force-retrain          Force retrain GNN model even if it exists"
            echo "  --verbose                Enable verbose logging"
            echo "  --help                   Show this help message"
            echo ""
            echo "This script will:"
            echo "  1. Check for required mapping files"
            echo "  2. Create/verify API lookup file"
            echo "  3. Build the mapping graph (populate_nx.py)"
            echo "  4. Train the GNN model (train_gnn.py)"
            echo "  5. Verify the setup is complete"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create derived filenames based on dataset
GRAPH_DATA_FILE="${OUTPUT_DIR}/${DATASET}_graph_data.pkl"
GNN_RESULTS_DIR="${OUTPUT_DIR}/gnn_results"
GNN_MODEL_FILE="${GNN_RESULTS_DIR}/${DATASET}_mapping_gnn_results.pkl"

# Display configuration
echo "Configuration:"
echo "   Dataset: $DATASET"
echo "   Mappings directory: $MAPPINGS_DIR"
echo "   API lookup file: $API_LOOKUP_FILE"
echo "   Output directory: $OUTPUT_DIR"
echo "   Graph data file: $GRAPH_DATA_FILE"
echo "   GNN model file: $GNN_MODEL_FILE"
echo "   Force retrain: $FORCE_RETRAIN"
echo "   Verbose: $VERBOSE"
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: No virtual environment detected. Consider activating one."
fi

echo ""
echo "Step 0: Checking dependencies..."

# Check Python dependencies
echo "Checking Python packages..."

# Function to check if a Python package is installed
check_python_package() {
    package_name=$1
    if python3 -c "import $package_name" 2>/dev/null; then
        echo "   $package_name"
        return 0
    else
        echo "   $package_name (missing)"
        return 1
    fi
}

# Check core dependencies
missing_packages=""

if ! check_python_package "torch"; then
    missing_packages="$missing_packages torch"
fi

if ! check_python_package "torch_geometric"; then
    missing_packages="$missing_packages torch-geometric"
fi

if ! check_python_package "networkx"; then
    missing_packages="$missing_packages networkx"
fi

if ! check_python_package "sentence_transformers"; then
    missing_packages="$missing_packages sentence-transformers"
fi

if ! check_python_package "sklearn"; then
    missing_packages="$missing_packages scikit-learn"
fi

# Handle missing packages
if [ -n "$missing_packages" ]; then
    echo ""
    echo "Missing required packages:$missing_packages"
    echo ""
    echo "Installing missing packages..."
    
    if [ -f "requirements_gnn.txt" ]; then
        echo "Using requirements_gnn.txt..."
        pip install -r requirements_gnn.txt
    else
        echo "Installing packages individually..."
        for package in $missing_packages; do
            echo "Installing $package..."
            pip install "$package"
        done
    fi
    
    echo ""
    echo "Re-checking dependencies..."
    
    # Re-check core packages
    all_good=true
    for package in torch torch_geometric networkx sentence_transformers sklearn; do
        if ! check_python_package "$package"; then
            all_good=false
        fi
    done
    
    if [ "$all_good" = false ]; then
        echo ""
        echo "Some packages still missing. Please install manually:"
        echo "   pip install torch torch-geometric networkx sentence-transformers scikit-learn"
        echo ""
        echo "Note: GNN will use fallback methods if torch-geometric is not available"
        echo "   You can continue with reduced functionality"
        
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo "All dependencies installed successfully!"
    fi
else
    echo "All required packages are available"
fi

echo ""
echo "Step 1: Checking prerequisites..."

# Check mappings directory
if [ ! -d "$MAPPINGS_DIR" ]; then
    echo "Error: Mappings directory not found: $MAPPINGS_DIR"
    echo "Please ensure you have mapping files from the SMT method or create sample mappings"
    exit 1
fi

# Count mapping files
mapping_count=$(find "$MAPPINGS_DIR" -name "*.json" | wc -l)
if [ "$mapping_count" -eq 0 ]; then
    echo "Error: No JSON mapping files found in $MAPPINGS_DIR"
    echo "Please generate some mapping files first using the SMT method"
    exit 1
fi

echo "Found $mapping_count mapping files in $MAPPINGS_DIR"

# Check/create API lookup file
echo ""
echo "Step 2: Checking API lookup file..."

if [ ! -f "$API_LOOKUP_FILE" ]; then
    echo "API lookup file not found. Creating a sample one..."
else
    echo "API lookup file found: $API_LOOKUP_FILE"
fi

# Validate API lookup file
if ! python3 -c "import json; json.load(open('$API_LOOKUP_FILE'))" 2>/dev/null; then
    echo "Error: API lookup file is not valid JSON"
    exit 1
fi

# Create output directory if it doesn't exist
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# Create GNN results directory if it doesn't exist
if [ ! -d "$GNN_RESULTS_DIR" ]; then
    echo "Creating GNN results directory: $GNN_RESULTS_DIR"
    mkdir -p "$GNN_RESULTS_DIR"
fi

echo ""
echo "Step 3: Building mapping graph..."

# Check if graph data already exists
if [ -f "$GRAPH_DATA_FILE" ] && [ "$FORCE_RETRAIN" = false ]; then
    echo "Graph data already exists ($GRAPH_DATA_FILE)"
    echo "   Use --force-retrain to rebuild"
else
    echo "Running graph population (populate_nx.py)..."
    
    POPULATE_CMD="python3 populate_nx.py --dataset $DATASET --mappings-dir $MAPPINGS_DIR --api-lookup-file $API_LOOKUP_FILE --output-dir $OUTPUT_DIR"
    
    if [ "$VERBOSE" = true ]; then
        $POPULATE_CMD
    else
        $POPULATE_CMD > /dev/null 2>&1
    fi
    
    if [ -f "$GRAPH_DATA_FILE" ]; then
        echo "Graph data created successfully ($GRAPH_DATA_FILE)"
    else
        echo "Error: Failed to create graph data"
        exit 1
    fi
fi

echo ""
echo "Step 4: Training GNN model..."

# Check if GNN model already exists and is loadable
model_needs_training=false

if [ -f "$GNN_MODEL_FILE" ] && [ "$FORCE_RETRAIN" = false ]; then
    echo "Checking existing GNN model..."
    
    # Test if the model can be loaded properly
    if python3 -c "
import pickle
import sys
sys.path.append('.')
try:
    from gnn_model import load_model_safely
    model = load_model_safely('$GNN_MODEL_FILE')
    if model is not None:
        print('Model loads successfully')
        exit(0)
    else:
        print('Model failed to load')
        exit(1)
except Exception as e:
    print(f'Model load error: {e}')
    exit(1)
" 2>/dev/null; then
        echo "GNN model already exists and is loadable"
    else
        echo "Existing GNN model has compatibility issues, will retrain"
        model_needs_training=true
    fi
else
    model_needs_training=true
fi

if [ "$model_needs_training" = true ] || [ "$FORCE_RETRAIN" = true ]; then
    if [ "$FORCE_RETRAIN" = true ]; then
        echo "Force retraining GNN model..."
    else
        echo "Training GNN model (train_gnn.py)..."
    fi
    
    echo "   This may take a few minutes..."
    
    # Remove old model if it exists and has issues
    if [ -f "$GNN_MODEL_FILE" ] && [ "$model_needs_training" = true ]; then
        echo "   Removing incompatible model file..."
        rm -f "$GNN_MODEL_FILE"
    fi
    
    TRAIN_CMD="python3 train_gnn.py --dataset $DATASET --output-dir $OUTPUT_DIR"
    
    if [ "$VERBOSE" = true ]; then
        $TRAIN_CMD
    else
        $TRAIN_CMD > /dev/null 2>&1
    fi
    
    if [ -f "$GNN_MODEL_FILE" ]; then
        echo "GNN model trained successfully"
        
        # Verify the new model can be loaded
        if python3 -c "
import sys
sys.path.append('.')
from gnn_model import load_model_safely
model = load_model_safely('$GNN_MODEL_FILE')
print('New model verified successfully')
" 2>/dev/null; then
            echo "New model verified and loadable"
        else
            echo "New model may have issues but training completed"
        fi
    else
        echo "Error: Failed to train GNN model"
        exit 1
    fi
fi

echo ""
echo "Step 5: Verifying setup..."

# Run verification with parameters
python3 -c "
import os
import json
import pickle
import sys

def check_file(path, description):
    if os.path.exists(path):
        print(f'{description}: {path}')
        return True
    else:
        print(f'{description}: {path} (missing)')
        return False

def check_pickle_file(path, description):
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print(f'{description}: {path} (loadable)')
        return True
    except Exception as e:
        print(f'{description}: {path} (error: {e})')
        return False

def check_api_lookup(path):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        
        num_fields = len(data)
        products = set(meta.get('product_name', 'unknown') for meta in data.values())
        
        print(f'API lookup file: {path}')
        print(f'   Fields: {num_fields}')
        print(f'   Products: {', '.join(sorted(products))}')
        return True
    except Exception as e:
        print(f'API lookup file: {path} (error: {e})')
        return False

print('GNN Setup Verification')
print('=' * 40)

all_good = True

# Check required files
all_good &= check_file('populate_nx.py', 'Graph builder script')
all_good &= check_file('train_gnn.py', 'GNN training script')

# Check generated files
all_good &= check_pickle_file('$GRAPH_DATA_FILE', 'Graph data')
all_good &= check_pickle_file('$GNN_MODEL_FILE', 'GNN model')

# Check API lookup
all_good &= check_api_lookup('$API_LOOKUP_FILE')

# Check mappings directory
mappings_dir = '$MAPPINGS_DIR'
if os.path.exists(mappings_dir):
    mapping_files = [f for f in os.listdir(mappings_dir) if f.endswith('.json')]
    print(f'Mappings directory: {mappings_dir} ({len(mapping_files)} files)')
else:
    print(f'Mappings directory: {mappings_dir} (missing)')
    all_good = False

print('\n' + '=' * 40)
if all_good:
    print('GNN setup verification PASSED')
    print('Ready to run GNN evaluations!')
else:
    print('GNN setup verification FAILED')
    print('Please fix the issues above before proceeding')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "GNN Setup Complete!"
    echo "====================="
    echo ""
    echo "Next Steps:"
    echo "   1. Run GNN evaluation: ./run_gnn.sh --dataset $DATASET"
    echo "   2. Compare with other methods (SMT, React)"
    echo "   3. Check results in results/ directory"
    echo ""
    echo "Usage Examples:"
    echo "   ./run_gnn.sh --dataset $DATASET --quick                    # Quick test"
    echo "   ./run_gnn.sh --dataset $DATASET --verbose                  # Full evaluation with logging"
    echo "   ./run_gnn.sh --help                                        # Show all options"
    echo ""
    echo "GNN Method Summary:"
    echo "   • Uses Graph Neural Networks for field joinability prediction"
    echo "   • Learns from existing mapping patterns"
    echo "   • Combines semantic similarity with learned connectivity patterns"
    echo "   • Provides confidence scores for field mappings"
    echo ""
    echo "Generated Files:"
    echo "   • Graph data: $GRAPH_DATA_FILE"
    echo "   • GNN model: $GNN_MODEL_FILE"
else
    echo ""
    echo "Setup verification failed. Please check the errors above."
    exit 1
fi
