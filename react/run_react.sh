#!/bin/bash

# Run script for ReAct Schema Mapping Method
# Usage: ./run_react.sh [options]

set -e  # Exit on any error

# Default values
DATASETS="ITOps"
MODELS="llama"
VERBOSE=""
QUICK=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Function to show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --datasets DATASETS    Comma-separated list of datasets (default: ITOps)"
    echo "                        Available: ITOps, NL2GQL, GraphQLRestBench"
    echo "  --models MODELS        Comma-separated list of models (default: llama)"
    echo "                        Available: granite, llama, qwen"
    echo "  --verbose             Enable verbose output"
    echo "  --quick               Run quick test with minimal data"
    echo "  --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                          # Run with defaults"
    echo "  $0 --datasets ITOps,NL2GQL --models llama  # Multiple datasets"
    echo "  $0 --verbose --quick                       # Quick verbose run"
}

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
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        --quick)
            QUICK="--quick"
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

# Check if we're in the correct directory
if [[ ! -f "react_runner.py" ]]; then
    print_error "react_runner.py not found. Please run this script from the react/ directory."
    exit 1
fi

# Check if virtual environment is activated (optional check)
if [[ -z "$VIRTUAL_ENV" && -z "$CONDA_DEFAULT_ENV" ]]; then
    print_warning "No virtual environment detected. Make sure dependencies are installed."
fi

# Create results directory if it doesn't exist
mkdir -p results

print_status "Starting ReAct Schema Mapping Evaluation"
print_status "Datasets: $DATASETS"
print_status "Models: $MODELS"
print_status "Verbose: $([ -n "$VERBOSE" ] && echo "enabled" || echo "disabled")"
print_status "Quick mode: $([ -n "$QUICK" ] && echo "enabled" || echo "disabled")"

# Build the command
CMD="python react_runner.py --datasets $DATASETS --models $MODELS"
if [[ -n "$VERBOSE" ]]; then
    CMD="$CMD $VERBOSE"
fi
if [[ -n "$QUICK" ]]; then
    CMD="$CMD $QUICK"
fi

print_status "Executing: $CMD"
echo ""

# Run the command
if eval $CMD; then
    print_status "ReAct evaluation completed successfully!"
    print_status "Results saved in results/ directory"
    
    # Show results summary if available
    if [[ -d "results" && -n "$(ls -A results 2>/dev/null)" ]]; then
        echo ""
        print_status "Generated result files:"
        ls -la results/
    fi
else
    print_error "ReAct evaluation failed with exit code $?"
    exit 1
fi
