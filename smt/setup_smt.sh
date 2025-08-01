#!/bin/bash

# SMT Setup Script - Dataset-specific ChromaDB Index Creation
# This script sets up ChromaDB indices for specific datasets and should be run once during initial setup
# Usage: ./setup_smt.sh [options]

set -e  # Exit on any error

# Default values
DATASETS="ITOps"  # Default to ITOps, but can specify multiple datasets
MAPPINGS_DIR="../data/mappings_output"
CHROMA_DB_PATH="../data/chroma_db"
FORCE_REINDEX=""
VERBOSE=""
ALL_DATASETS=false

# Available datasets
AVAILABLE_DATASETS=("ITOps" "NL2GQL" "GraphQLRestBench")

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

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Function to show usage
usage() {
    echo "SMT Setup Script - Dataset-specific ChromaDB Index Creation"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --datasets DATASETS    Comma-separated list of datasets (default: ITOps)"
    echo "                         Available: ITOps, NL2GQL, GraphQLRestBench"
    echo "  --all-datasets         Setup indices for all available datasets"
    echo "  --mappings-dir DIR     Existing mappings directory (default: ../data/mappings_output)"
    echo "  --chroma-db-path DIR   ChromaDB storage path (default: ../data/chroma_db)"
    echo "  --force-reindex        Force recreation of all indices"
    echo "  --verbose              Enable verbose output"
    echo "  --help                 Show this help message"
    echo ""
    echo "This script creates dataset-specific ChromaDB collections:"
    echo "  1. {dataset}_graphql_schemas    - For schema discovery per dataset"
    echo "  2. smt_mappings                 - For existing mapping reuse (shared)"
    echo ""
    echo "Dataset-specific file structure:"
    echo "  - GraphQL schemas: ../data/{dataset}_graphql/"
    echo "  - Dataset files: ../data/{dataset}_dataset.jsonl"
    echo "  - Collections: {dataset}_graphql_schemas"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Setup ITOps (default)"
    echo "  $0 --datasets ITOps,NL2GQL           # Setup multiple datasets"
    echo "  $0 --all-datasets                    # Setup all available datasets"
    echo "  $0 --force-reindex --verbose         # Force rebuild with verbose output"
    echo "  $0 --datasets GraphQLRestBench       # Setup specific dataset"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --datasets)
            DATASETS="$2"
            shift 2
            ;;
        --all-datasets)
            ALL_DATASETS=true
            shift
            ;;
        --mappings-dir)
            MAPPINGS_DIR="$2"
            shift 2
            ;;
        --chroma-db-path)
            CHROMA_DB_PATH="$2"
            shift 2
            ;;
        --force-reindex)
            FORCE_REINDEX="--force-reindex"
            shift
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

# Set datasets to all available if --all-datasets is specified
if [[ "$ALL_DATASETS" == "true" ]]; then
    DATASETS=$(IFS=','; echo "${AVAILABLE_DATASETS[*]}")
fi

# Convert comma-separated datasets to array
IFS=',' read -ra DATASET_ARRAY <<< "$DATASETS"

echo "============================================================"
echo "SMT ChromaDB Setup - Dataset-specific Index Creation"
echo "============================================================"

print_status "Configuration:"
print_status "  Datasets: $DATASETS (${#DATASET_ARRAY[@]} datasets)"
print_status "  Mappings directory: $MAPPINGS_DIR"
print_status "  ChromaDB path: $CHROMA_DB_PATH"
print_status "  Force reindex: $([ -n "$FORCE_REINDEX" ] && echo "enabled" || echo "disabled")"
print_status "  Verbose: $([ -n "$VERBOSE" ] && echo "enabled" || echo "disabled")"

# Check if we're in the correct directory
if [[ ! -f "smt_schema_indexer.py" ]] || [[ ! -f "smt_mappings_indexer.py" ]]; then
    print_error "Required indexer scripts not found. Please run this script from the smt/ directory."
    exit 1
fi

# Validate datasets
print_step "1. Validating datasets"
for dataset in "${DATASET_ARRAY[@]}"; do
    if [[ ! " ${AVAILABLE_DATASETS[@]} " =~ " ${dataset} " ]]; then
        print_error "Invalid dataset: $dataset"
        print_error "Available datasets: ${AVAILABLE_DATASETS[*]}"
        exit 1
    fi
    print_status "Dataset validated: $dataset"
done

# Check for required dependencies
print_step "2. Checking dependencies"

python -c "import chromadb; print('ChromaDB available')" 2>/dev/null || {
    print_error "ChromaDB not found. Please install with: pip install chromadb"
    exit 1
}

python -c "from sentence_transformers import SentenceTransformer; print('SentenceTransformers available')" 2>/dev/null || {
    print_error "SentenceTransformers not found. Please install with: pip install sentence-transformers"
    exit 1
}

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" && -z "$CONDA_DEFAULT_ENV" ]]; then
    print_warning "No virtual environment detected. Make sure dependencies are installed."
fi

# Validate dataset-specific directories and files
print_step "3. Validating dataset-specific files"

declare -A DATASET_SCHEMA_DIRS
declare -A DATASET_FILES
declare -A SCHEMA_FILE_COUNTS

for dataset in "${DATASET_ARRAY[@]}"; do
    dataset_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
    
    # Dataset-specific paths
    schema_dir="../data/${dataset_lower}_graphql"
    dataset_file="../data/${dataset_lower}_dataset.jsonl"
    
    print_status "Validating $dataset:"
    print_status "  Schema directory: $schema_dir"
    print_status "  Dataset file: $dataset_file"
    
    # Check schema directory
    if [[ ! -d "$schema_dir" ]]; then
        print_error "Schema directory not found for $dataset: $schema_dir"
        print_error "Please ensure GraphQL schemas are available for this dataset."
        exit 1
    fi
    
    # Check dataset file
    if [[ ! -f "$dataset_file" ]]; then
        print_error "Dataset file not found for $dataset: $dataset_file"
        print_error "Please ensure the dataset file exists."
        exit 1
    fi
    
    # Count schema files
    schema_files=$(find "$schema_dir" -name "*.graphql" -o -name "*.gql" -o -name "*.schema" | wc -l)
    if [[ $schema_files -eq 0 ]]; then
        print_error "No GraphQL schema files found for $dataset in $schema_dir"
        print_error "Expected files with extensions: .graphql, .gql, .schema"
        exit 1
    fi
    
    # Count dataset samples
    dataset_samples=$(wc -l < "$dataset_file")
    
    print_status "  Found $schema_files GraphQL schema files"
    print_status "  Found $dataset_samples dataset samples"
    
    # Store for later use
    DATASET_SCHEMA_DIRS["$dataset"]="$schema_dir"
    DATASET_FILES["$dataset"]="$dataset_file"
    SCHEMA_FILE_COUNTS["$dataset"]="$schema_files"
done

# Validate mappings directory
print_status "Validating shared mappings directory:"
if [[ ! -d "$MAPPINGS_DIR" ]]; then
    print_warning "Mappings directory not found: $MAPPINGS_DIR"
    print_warning "Creating directory for future mappings..."
    mkdir -p "$MAPPINGS_DIR"
    MAPPING_FILES=0
else
    MAPPING_FILES=$(find "$MAPPINGS_DIR" -name "*-*.json" | wc -l)
    print_status "Found $MAPPING_FILES existing mapping files"
fi

if [[ $MAPPING_FILES -eq 0 ]]; then
    print_warning "No existing mapping files found in $MAPPINGS_DIR"
    print_warning "Expected files with format: source-target.json"
fi

# Create necessary directories
print_step "4. Setting up directories"
mkdir -p "$CHROMA_DB_PATH"
mkdir -p results
print_status "Created/verified directories"

# Check existing collections for each dataset
print_step "5. Checking existing ChromaDB collections"

declare -A SCHEMA_INDEX_EXISTS
declare -A SCHEMA_COUNTS

if [[ -f "$CHROMA_DB_PATH/chroma.sqlite3" ]] && [[ -z "$FORCE_REINDEX" ]]; then
    print_status "Checking existing ChromaDB indices..."
    
    for dataset in "${DATASET_ARRAY[@]}"; do
        dataset_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
        collection_name="${dataset_lower}_graphql_schemas"
        
        # Check existing collections
        python -c "
import chromadb
from chromadb.config import Settings
try:
    client = chromadb.PersistentClient(path='$CHROMA_DB_PATH', settings=Settings(anonymized_telemetry=False))
    collections = [c.name for c in client.list_collections()]
    
    if '$collection_name' in collections:
        collection = client.get_collection('$collection_name')
        count = collection.count()
        print(f'SCHEMA_EXISTS_{dataset}:{count}')
        
except Exception as e:
    print(f'ERROR_{dataset}:{e}')
" > /tmp/chromadb_check_${dataset}.txt 2>/dev/null

        if grep -q "SCHEMA_EXISTS_${dataset}:" /tmp/chromadb_check_${dataset}.txt; then
            count=$(grep "SCHEMA_EXISTS_${dataset}:" /tmp/chromadb_check_${dataset}.txt | cut -d: -f2)
            print_status "Existing $dataset schema index found: $count documents"
            SCHEMA_INDEX_EXISTS["$dataset"]=true
            SCHEMA_COUNTS["$dataset"]=$count
        else
            SCHEMA_INDEX_EXISTS["$dataset"]=false
        fi
        
        rm -f /tmp/chromadb_check_${dataset}.txt
    done
    
    # Check mappings collection
    MAPPINGS_INDEX_EXISTS=false
    python -c "
import chromadb
from chromadb.config import Settings
try:
    client = chromadb.PersistentClient(path='$CHROMA_DB_PATH', settings=Settings(anonymized_telemetry=False))
    collections = [c.name for c in client.list_collections()]
    
    if 'smt_mappings' in collections:
        mappings_collection = client.get_collection('smt_mappings')
        mappings_count = mappings_collection.count()
        print(f'MAPPINGS_EXISTS:{mappings_count}')
        
except Exception as e:
    print(f'ERROR:{e}')
" > /tmp/chromadb_check_mappings.txt 2>/dev/null

    if grep -q "MAPPINGS_EXISTS:" /tmp/chromadb_check_mappings.txt; then
        MAPPINGS_COUNT=$(grep "MAPPINGS_EXISTS:" /tmp/chromadb_check_mappings.txt | cut -d: -f2)
        print_status "Existing mappings index found: $MAPPINGS_COUNT documents"
        MAPPINGS_INDEX_EXISTS=true
    fi
    
    rm -f /tmp/chromadb_check_mappings.txt
else
    for dataset in "${DATASET_ARRAY[@]}"; do
        SCHEMA_INDEX_EXISTS["$dataset"]=false
    done
    MAPPINGS_INDEX_EXISTS=false
fi

# Index GraphQL schemas for each dataset
print_step "6. Indexing dataset-specific GraphQL schemas"

for dataset in "${DATASET_ARRAY[@]}"; do
    dataset_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
    collection_name="${dataset_lower}_graphql_schemas"
    schema_dir="${DATASET_SCHEMA_DIRS[$dataset]}"
    schema_files="${SCHEMA_FILE_COUNTS[$dataset]}"
    
    print_status "Processing $dataset dataset:"
    
    if [[ "${SCHEMA_INDEX_EXISTS[$dataset]}" == "true" ]] && [[ -z "$FORCE_REINDEX" ]]; then
        print_status "  Schema index already exists (${SCHEMA_COUNTS[$dataset]} documents)"
        print_status "  Use --force-reindex to rebuild"
    else
        print_status "  Creating GraphQL schema index..."
        print_status "    Source: $schema_dir ($schema_files files)"
        print_status "    Target: $CHROMA_DB_PATH/$collection_name collection"
        
        python smt_schema_indexer.py \
            --schema-dir "$schema_dir" \
            --chroma-db-path "$CHROMA_DB_PATH" \
            --collection-name "$collection_name" \
            $FORCE_REINDEX \
            --test-query "turbo entity schema" \
            $VERBOSE
        
        if [[ $? -eq 0 ]]; then
            print_status "  $dataset GraphQL schema indexing completed"
        else
            print_error "  $dataset GraphQL schema indexing failed"
            exit 1
        fi
    fi
done

# Index existing mappings (shared across all datasets)
print_step "7. Indexing existing mappings (shared)"

if [[ "$MAPPINGS_INDEX_EXISTS" == "true" ]] && [[ -z "$FORCE_REINDEX" ]]; then
    print_status "Mappings index already exists (use --force-reindex to rebuild)"
else
    if [[ $MAPPING_FILES -gt 0 ]]; then
        print_status "Creating existing mappings index..."
        print_status "  Source: $MAPPINGS_DIR ($MAPPING_FILES files)"
        print_status "  Target: $CHROMA_DB_PATH/smt_mappings collection"
        
        python smt_mappings_indexer.py \
            --mappings-dir "$MAPPINGS_DIR" \
            --chroma-db-path "$CHROMA_DB_PATH" \
            --collection-name "smt_mappings" \
            $FORCE_REINDEX \
            --test-query "map concert vulnerabilities to turbo entities" \
            $VERBOSE
        
        if [[ $? -eq 0 ]]; then
            print_status "Existing mappings indexing completed"
        else
            print_error "Existing mappings indexing failed"
            exit 1
        fi
    else
        print_warning "No existing mappings to index"
        print_status "Creating empty mappings collection for future use..."
        
        python -c "
import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(path='$CHROMA_DB_PATH', settings=Settings(anonymized_telemetry=False))
try:
    collection = client.create_collection('smt_mappings', metadata={'description': 'SMT mappings for reuse'})
    print('Empty mappings collection created')
except Exception as e:
    if 'already exists' in str(e):
        print('Mappings collection already exists')
    else:
        print(f'Error: {e}')
"
    fi
fi

# Create setup completion marker
echo "$(date): SMT ChromaDB setup completed successfully for datasets: $DATASETS" > "$CHROMA_DB_PATH/.setup_complete"

echo ""
echo "============================================================"
print_status "SMT ChromaDB Setup Completed Successfully!"
echo "============================================================"

print_status "Created collections:"
for dataset in "${DATASET_ARRAY[@]}"; do
    dataset_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
    print_status "  ${dataset_lower}_graphql_schemas  - For $dataset schema discovery"
done
print_status "  smt_mappings                     - For existing mapping reuse (shared)"

echo ""
print_status "Files created:"
print_status "  $CHROMA_DB_PATH/               - ChromaDB database"
print_status "  $CHROMA_DB_PATH/.setup_complete - Setup completion marker"

echo ""
print_status "Dataset-specific collections created:"
for dataset in "${DATASET_ARRAY[@]}"; do
    dataset_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
    schema_files="${SCHEMA_FILE_COUNTS[$dataset]}"
    print_status "  $dataset: ${dataset_lower}_graphql_schemas ($schema_files schema files indexed)"
done

echo ""
print_status "Next steps:"
echo "  1. Run SMT evaluation: ./run_smt.sh --dataset [DATASET] --quick"
echo "  2. Run comparative evaluation: ../run_evaluation.sh --methods SMT --dataset [DATASET]"
echo "  3. The system will now use existing indices for fast retrieval"

echo ""
print_status "Available commands:"
for dataset in "${DATASET_ARRAY[@]}"; do
    echo "  ./run_smt.sh --dataset $dataset --quick --verbose"
done

echo ""
print_status "Maintenance:"
echo "  - To rebuild indices: $0 --datasets $DATASETS --force-reindex"
echo "  - To add new datasets: $0 --datasets [EXISTING],NewDataset"
echo "  - To setup all datasets: $0 --all-datasets"
echo "  - To add new mappings: copy files to $MAPPINGS_DIR and re-run setup"

echo ""
print_status "Test the setup:"
echo "  ./run_smt.sh --dataset ${DATASET_ARRAY[0]} --quick --verbose"
