#!/bin/bash

# SMT Setup Script - One-time ChromaDB Index Creation
# This script sets up both ChromaDB indices and should be run once during initial setup
# Usage: ./setup_smt.sh [options]

set -e  # Exit on any error

# Default values
SCHEMA_DIR="../data/graphql"
MAPPINGS_DIR="../data/mappings_output"
CHROMA_DB_PATH="../data/chroma_db"
FORCE_REINDEX=""
VERBOSE=""

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
    echo "SMT Setup Script - One-time ChromaDB Index Creation"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --schema-dir DIR       GraphQL schema directory (default: ../data/graphql)"
    echo "  --mappings-dir DIR     Existing mappings directory (default: ../data/mappings_output)"
    echo "  --chroma-db-path DIR   ChromaDB storage path (default: ../data/chroma_db)"
    echo "  --force-reindex        Force recreation of all indices"
    echo "  --verbose              Enable verbose output"
    echo "  --help                 Show this help message"
    echo ""
    echo "This script creates two ChromaDB collections:"
    echo "  1. graphql_schemas    - For schema discovery"
    echo "  2. smt_mappings       - For existing mapping reuse"
    echo ""
    echo "Examples:"
    echo "  $0                           # Setup with defaults"
    echo "  $0 --force-reindex --verbose # Force rebuild with verbose output"
    echo "  $0 --schema-dir ./schemas    # Custom schema directory"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --schema-dir)
            SCHEMA_DIR="$2"
            shift 2
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

echo "============================================================"
echo "SMT ChromaDB Setup - One-time Index Creation"
echo "============================================================"

print_status "Configuration:"
print_status "  Schema directory: $SCHEMA_DIR"
print_status "  Mappings directory: $MAPPINGS_DIR"
print_status "  ChromaDB path: $CHROMA_DB_PATH"
print_status "  Force reindex: $([ -n "$FORCE_REINDEX" ] && echo "enabled" || echo "disabled")"
print_status "  Verbose: $([ -n "$VERBOSE" ] && echo "enabled" || echo "disabled")"

# Check if we're in the correct directory
if [[ ! -f "smt_schema_indexer.py" ]] || [[ ! -f "smt_mappings_indexer.py" ]]; then
    print_error "Required indexer scripts not found. Please run this script from the smt/ directory."
    exit 1
fi

# Check for required dependencies
print_step "1. Checking dependencies"

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

# Validate input directories
print_step "2. Validating input directories"

if [[ ! -d "$SCHEMA_DIR" ]]; then
    print_error "Schema directory not found: $SCHEMA_DIR"
    print_error "Please check the path or specify correct directory with --schema-dir"
    exit 1
fi

if [[ ! -d "$MAPPINGS_DIR" ]]; then
    print_error "Mappings directory not found: $MAPPINGS_DIR"
    print_error "Please check the path or specify correct directory with --mappings-dir"
    exit 1
fi

# Count available files
SCHEMA_FILES=$(find "$SCHEMA_DIR" -name "*.graphql" -o -name "*.gql" -o -name "*.schema" | wc -l)
MAPPING_FILES=$(find "$MAPPINGS_DIR" -name "*-*.json" | wc -l)

print_status "Found $SCHEMA_FILES GraphQL schema files"
print_status "Found $MAPPING_FILES existing mapping files"

if [[ $SCHEMA_FILES -eq 0 ]]; then
    print_error "No GraphQL schema files found in $SCHEMA_DIR"
    print_error "Expected files with extensions: .graphql, .gql, .schema"
    exit 1
fi

if [[ $MAPPING_FILES -eq 0 ]]; then
    print_warning "No existing mapping files found in $MAPPINGS_DIR"
    print_warning "Expected files with format: source-target.json"
fi

# Create necessary directories
print_step "3. Setting up directories"
mkdir -p "$CHROMA_DB_PATH"
mkdir -p results
print_status "Created/verified directories"

# Check if indices already exist
SCHEMA_INDEX_EXISTS=false
MAPPINGS_INDEX_EXISTS=false

if [[ -f "$CHROMA_DB_PATH/chroma.sqlite3" ]] && [[ -z "$FORCE_REINDEX" ]]; then
    print_status "Checking existing ChromaDB indices..."
    
    # Check existing collections
    python -c "
import chromadb
from chromadb.config import Settings
try:
    client = chromadb.PersistentClient(path='$CHROMA_DB_PATH', settings=Settings(anonymized_telemetry=False))
    collections = [c.name for c in client.list_collections()]
    
    if 'graphql_schemas' in collections:
        schema_collection = client.get_collection('graphql_schemas')
        schema_count = schema_collection.count()
        print(f'SCHEMA_EXISTS:{schema_count}')
    
    if 'smt_mappings' in collections:
        mappings_collection = client.get_collection('smt_mappings')
        mappings_count = mappings_collection.count()
        print(f'MAPPINGS_EXISTS:{mappings_count}')
        
except Exception as e:
    print(f'ERROR:{e}')
" > /tmp/chromadb_check.txt 2>/dev/null

    if grep -q "SCHEMA_EXISTS:" /tmp/chromadb_check.txt; then
        SCHEMA_COUNT=$(grep "SCHEMA_EXISTS:" /tmp/chromadb_check.txt | cut -d: -f2)
        print_status "Existing schema index found: $SCHEMA_COUNT documents"
        SCHEMA_INDEX_EXISTS=true
    fi
    
    if grep -q "MAPPINGS_EXISTS:" /tmp/chromadb_check.txt; then
        MAPPINGS_COUNT=$(grep "MAPPINGS_EXISTS:" /tmp/chromadb_check.txt | cut -d: -f2)
        print_status "Existing mappings index found: $MAPPINGS_COUNT documents"
        MAPPINGS_INDEX_EXISTS=true
    fi
    
    rm -f /tmp/chromadb_check.txt
fi

# Index GraphQL schemas
print_step "4. Indexing GraphQL schemas"

if [[ "$SCHEMA_INDEX_EXISTS" == "true" ]] && [[ -z "$FORCE_REINDEX" ]]; then
    print_status "Schema index already exists (use --force-reindex to rebuild)"
else
    print_status "Creating GraphQL schema index..."
    print_status "  Source: $SCHEMA_DIR ($SCHEMA_FILES files)"
    print_status "  Target: $CHROMA_DB_PATH/graphql_schemas collection"
    
    python smt_schema_indexer.py \
        --schema-dir "$SCHEMA_DIR" \
        --chroma-db-path "$CHROMA_DB_PATH" \
        --collection-name "graphql_schemas" \
        $FORCE_REINDEX \
        --test-query "turbo entity schema" \
        $VERBOSE
    
    if [[ $? -eq 0 ]]; then
        print_status "GraphQL schema indexing completed"
    else
        print_error "GraphQL schema indexing failed"
        exit 1
    fi
fi

# Index existing mappings
print_step "5. Indexing existing mappings"

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
        print_warning "⚠No existing mappings to index"
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

# Verify setup
print_step "6. Verifying setup"

print_status "Final ChromaDB verification..."
python -c "
import chromadb
from chromadb.config import Settings

try:
    client = chromadb.PersistentClient(path='$CHROMA_DB_PATH', settings=Settings(anonymized_telemetry=False))
    collections = client.list_collections()
    
    print('ChromaDB Collections:')
    total_docs = 0
    
    for collection in collections:
        count = collection.count()
        total_docs += count
        print(f'  📦 {collection.name}: {count} documents')
    
    print(f'📈 Total indexed documents: {total_docs}')
    
    # Test queries
    print('')
    print('Testing search functionality:')
    
    # Test schema search
    try:
        schema_collection = client.get_collection('graphql_schemas')
        schema_results = schema_collection.query(query_texts=['turbo entity'], n_results=2)
        print(f'  Schema search test: Found {len(schema_results[\"ids\"][0])} results')
    except Exception as e:
        print(f'  Schema search test: {e}')
    
    # Test mappings search
    try:
        mappings_collection = client.get_collection('smt_mappings')
        mapping_results = mappings_collection.query(query_texts=['map concert to turbo'], n_results=2)
        print(f'  Mappings search test: Found {len(mapping_results[\"ids\"][0])} results')
    except Exception as e:
        print(f'  Mappings search test: {e}')
        
except Exception as e:
    print(f'Verification failed: {e}')
    exit(1)
" || {
    print_error "Setup verification failed"
    exit 1
}

# Create setup completion marker
echo "$(date): SMT ChromaDB setup completed successfully" > "$CHROMA_DB_PATH/.setup_complete"

echo ""
echo "============================================================"
print_status "SMT ChromaDB Setup Completed Successfully!"
echo "============================================================"

print_status "Two ChromaDB collections created:"
print_status "  graphql_schemas  - For schema discovery"
print_status "  smt_mappings     - For existing mapping reuse"

echo ""
print_status "Files created:"
print_status "  $CHROMA_DB_PATH/               - ChromaDB database"
print_status "  $CHROMA_DB_PATH/.setup_complete - Setup completion marker"

echo ""
print_status "Next steps:"
echo "  1. Run SMT evaluation: ./run_smt.sh --quick"
echo "  2. The system will now use existing indices for fast retrieval"
echo "  3. Re-run this setup only when you add new schemas or mappings"

echo ""
print_status "Maintenance:"
echo "  - To rebuild indices: $0 --force-reindex"
echo "  - To add new mappings: copy files to $MAPPINGS_DIR and re-run setup"
echo "  - To add new schemas: copy files to $SCHEMA_DIR and re-run setup"

echo ""
print_status "Test the setup:"
echo "  ./run_smt.sh --quick --verbose"
