#!/bin/bash

# Run script for SMT API Retriever
# Usage: ./run_smt.sh [options]

set -e  # Exit on any error

# Default values
UTTERANCE=""
DATASET=""
MODEL=""
TOP_K="5"
VERBOSE=""
OUTPUT_DIR="mappings_output"

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

print_query() {
    echo -e "${BLUE}[QUERY]${NC} $1"
}

# Function to show usage
usage() {
    echo "Usage: $0 --utterance \"UTTERANCE\" [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  --utterance TEXT       Natural language utterance describing the mapping task"
    echo ""
    echo "Options:"
    echo "  --dataset DATASET      Target dataset (ITOps, NL2GQL, GraphQLRestBench)"
    echo "  --model MODEL          Target model (granite, llama, qwen)"
    echo "  --top-k K              Number of top results to retrieve (default: 5)"
    echo "  --output-dir DIR       Output directory (default: mappings_output)"
    echo "  --verbose              Enable verbose output"
    echo "  --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --utterance \"map concert vulnerabilities to turbo entities\""
    echo "  $0 --utterance \"find user authentication mapping\" --dataset ITOps --verbose"
    echo "  $0 --utterance \"map API endpoints\" --top-k 10 --model llama"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --utterance)
            UTTERANCE="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --top-k)
            TOP_K="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE="--verbose"
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

# Validate required arguments
if [[ -z "$UTTERANCE" ]]; then
    print_error "Missing required argument: --utterance"
    usage
    exit 1
fi

# Check if we're in the correct directory
if [[ ! -f "smt_api_retriever.py" ]]; then
    print_error "smt_api_retriever.py not found. Please run this script from the smt/ directory."
    exit 1
fi

# Check if ChromaDB directory exists
if [[ ! -d "chroma_db" ]]; then
    print_warning "chroma_db directory not found. Make sure the vector database is initialized."
fi

# Check if virtual environment is activated (optional check)
if [[ -z "$VIRTUAL_ENV" && -z "$CONDA_DEFAULT_ENV" ]]; then
    print_warning "No virtual environment detected. Make sure dependencies are installed."
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

print_status "Starting SMT API Retrieval"
print_query "Utterance: \"$UTTERANCE\""
print_status "Top-K results: $TOP_K"
print_status "Output directory: $OUTPUT_DIR"

if [[ -n "$DATASET" ]]; then
    print_status "Target dataset: $DATASET"
fi

if [[ -n "$MODEL" ]]; then
    print_status "Target model: $MODEL"
fi

print_status "Verbose: $([ -n "$VERBOSE" ] && echo "enabled" || echo "disabled")"

# Build the command
CMD="python smt_api_retriever.py --utterance \"$UTTERANCE\""

if [[ -n "$DATASET" ]]; then
    CMD="$CMD --dataset $DATASET"
fi

if [[ -n "$MODEL" ]]; then
    CMD="$CMD --model $MODEL"
fi

if [[ "$TOP_K" != "5" ]]; then
    CMD="$CMD --top-k $TOP_K"
fi

if [[ "$OUTPUT_DIR" != "mappings_output" ]]; then
    CMD="$CMD --output-dir $OUTPUT_DIR"
fi

if [[ -n "$VERBOSE" ]]; then
    CMD="$CMD $VERBOSE"
fi

print_status "Executing: $CMD"
echo ""

# Run the command
if eval $CMD; then
    print_status "SMT API retrieval completed successfully!"
    
    # Show output files if available
    if [[ -d "$OUTPUT_DIR" && -n "$(ls -A $OUTPUT_DIR 2>/dev/null)" ]]; then
        echo ""
        print_status "Generated files in $OUTPUT_DIR/:"
        ls -la "$OUTPUT_DIR"/
        
        # Show source.json and target.json if they exist
        if [[ -f "$OUTPUT_DIR/source.json" ]]; then
            echo ""
            print_status "Source schema preview:"
            head -n 10 "$OUTPUT_DIR/source.json" | sed 's/^/  /'
        fi
        
        if [[ -f "$OUTPUT_DIR/target.json" ]]; then
            echo ""
            print_status "Target schema preview:"
            head -n 10 "$OUTPUT_DIR/target.json" | sed 's/^/  /'
        fi
    fi
else
    print_error "SMT API retrieval failed with exit code $?"
    exit 1
fi

echo ""
print_status "Next steps:"
echo "  1. Review the generated source.json and target.json files"
echo "  2. Run the SMT toolkit with: make run-smt"
echo "  3. Check results in results/ directory"
