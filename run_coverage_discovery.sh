#!/bin/bash
# Run ITOpsGraph Coverage Discovery Pipeline
# This script generates mappings for uncovered types using UE-Reflect

set -e  # Exit on error

echo "========================================"
echo "ITOpsGraph Coverage Discovery Pipeline"
echo "========================================"
echo ""

# Configuration
TOP_K_CANDIDATES=5
MODEL="watsonx/meta-llama/llama-3-3-70b-instruct"
DATA_DIR="data"
OUTPUT_DIR="map_reflect/coverage_results"

# Step 1: Generate type-level candidates using embedding similarity
echo "[Step 1/4] Generating type-level candidates..."
python scripts/generate_coverage_candidates.py \
    --api-lookup "$DATA_DIR/itopsgraph_api_lookup.json" \
    --output "$DATA_DIR/itopsgraph_coverage_candidates.json" \
    --top-k $TOP_K_CANDIDATES

echo ""
echo "[Step 2/4] Generating field-level validation dataset..."
python scripts/generate_coverage_dataset.py \
    --candidates "$DATA_DIR/itopsgraph_coverage_candidates.json" \
    --api-lookup "$DATA_DIR/itopsgraph_api_lookup.json" \
    --output "$DATA_DIR/itopsgraph_coverage_dataset.jsonl"

echo ""
echo "[Step 3/4] Running UE-Reflect field validation..."
echo "This step may take a long time depending on the number of samples."
echo "Progress is saved via checkpoints - you can resume if interrupted."
echo ""

python -m map_reflect.ue_reflect \
    --datasets ITOpsGraph_Coverage \
    --models "$MODEL" \
    --property-lookup-file "$DATA_DIR/itopsgraph_api_lookup.json" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --top-k 10 \
    --use-sampling

echo ""
echo "[Step 4/4] Extracting validated mappings..."
python scripts/extract_coverage_mappings.py \
    --results-dir "$OUTPUT_DIR" \
    --lookup-file "$DATA_DIR/itops_type_mapping_lookup.json" \
    --output-file "$DATA_DIR/itops_type_mapping_lookup_expanded.json"

echo ""
echo "========================================"
echo "Coverage Discovery Pipeline Complete!"
echo "========================================"
echo ""
echo "Results:"
echo "  - Type candidates: $DATA_DIR/itopsgraph_coverage_candidates.json"
echo "  - Validation dataset: $DATA_DIR/itopsgraph_coverage_dataset.jsonl"
echo "  - UE-Reflect results: $OUTPUT_DIR/"
echo "  - Expanded lookup: $DATA_DIR/itops_type_mapping_lookup_expanded.json"
echo ""
echo "To verify improvement, run:"
echo "  python analyze_type_coverage.py"
