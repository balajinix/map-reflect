#!/bin/bash
# UQ Threshold ablation study for Map-Reflect
# Tests different uncertainty thresholds for triggering reflection

set -e

MODEL="watsonx/meta-llama/llama-3-3-70b-instruct"
DATASET="ITOpsGraph"
LOOKUP="data/itopsgraph_api_lookup.json"
DATA_DIR="data"
OUTPUT_BASE="results/uq_threshold_ablation"

mkdir -p "$OUTPUT_BASE"

echo "=============================================="
echo "UQ Threshold Ablation Study"
echo "=============================================="
echo ""
echo "Testing thresholds: 0.08, 0.10, 0.12, 0.15"
echo ""

# 1. Threshold 0.08 (~50% trigger)
echo -e "\n[1/4] Running: threshold=0.08"
python -m map_reflect.map_reflect_runner \
    --datasets "$DATASET" \
    --models "$MODEL" \
    --property-lookup-file "$LOOKUP" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_BASE/uq_0.08" \
    --neighbor-k 0 \
    --use-uncertainty \
    --use-uq-reflection \
    --uq-threshold 0.08

# 2. Threshold 0.10 (~29% trigger)
echo -e "\n[2/4] Running: threshold=0.10"
python -m map_reflect.map_reflect_runner \
    --datasets "$DATASET" \
    --models "$MODEL" \
    --property-lookup-file "$LOOKUP" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_BASE/uq_0.10" \
    --neighbor-k 0 \
    --use-uncertainty \
    --use-uq-reflection \
    --uq-threshold 0.10

# 3. Threshold 0.12 (~15% trigger)
echo -e "\n[3/4] Running: threshold=0.12"
python -m map_reflect.map_reflect_runner \
    --datasets "$DATASET" \
    --models "$MODEL" \
    --property-lookup-file "$LOOKUP" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_BASE/uq_0.12" \
    --neighbor-k 0 \
    --use-uncertainty \
    --use-uq-reflection \
    --uq-threshold 0.12

# 4. Threshold 0.15 (~4.5% trigger)
echo -e "\n[4/4] Running: threshold=0.15"
python -m map_reflect.map_reflect_runner \
    --datasets "$DATASET" \
    --models "$MODEL" \
    --property-lookup-file "$LOOKUP" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_BASE/uq_0.15" \
    --neighbor-k 0 \
    --use-uncertainty \
    --use-uq-reflection \
    --uq-threshold 0.15

echo -e "\n=============================================="
echo "UQ threshold ablation complete!"
echo "=============================================="

# Print summary
echo -e "\nResults summary:"
for dir in "$OUTPUT_BASE"/*/; do
    name=$(basename "$dir")
    file="$dir/mapreflect_itopsgraph_meta-llama_llama-3-3-70b-instruct.json"
    if [ -f "$file" ]; then
        acc=$(python3 -c "import json; d=json.load(open('$file')); print(f\"{d['metrics']['accuracy']*100:.1f}%\")" 2>/dev/null)
        refl=$(python3 -c "
import json
d=json.load(open('$file'))
count = sum(1 for s in d['results'] for p in s.get('predictions', {}).values() if p.get('reflection'))
print(count)
" 2>/dev/null)
        printf "%-15s %s (reflections: %s)\n" "$name:" "$acc" "$refl"
    fi
done
