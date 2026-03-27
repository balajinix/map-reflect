#!/bin/bash
# Train GNN on ITGraph dataset
# Converts ITGraph mappings to the format expected by populate_nx.py,
# then trains the GNN model.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BASE_DIR"

if [[ -f "${BASE_DIR}/.venv/bin/activate" ]]; then
    source "${BASE_DIR}/.venv/bin/activate"
fi

DATASET="itgraph"
API_LOOKUP="data/itom_hashicorp_api_lookup_v2.json"
MAPPINGS_DIR="data/itgraph_mappings_output"
OUTPUT_DIR="data"

echo "=== Step 1: Convert ITGraph dataset to mapping files ==="
mkdir -p "$MAPPINGS_DIR"

python3 -c "
import json, os, hashlib

with open('data/itgraph_dataset.jsonl') as f:
    samples = [json.loads(l) for l in f if l.strip()]

outdir = '$MAPPINGS_DIR'
count = 0
for s in samples:
    mapping = s.get('mapping', {})
    if not mapping:
        continue

    # Get source and target types from first field
    src_fields = list(mapping.keys())
    tgt_fields = list(mapping.values())
    if not src_fields:
        continue

    src_type = src_fields[0].split('.')[0] if '.' in src_fields[0] else src_fields[0]
    tgt_type = tgt_fields[0].split('.')[0] if '.' in tgt_fields[0] else tgt_fields[0]

    # Create mapping file in expected format
    result = {
        'source_type': src_type,
        'target_type': tgt_type,
        'mapping': mapping,
        'sample_id': s['id'],
    }

    # Generate unique filename
    h = hashlib.md5(json.dumps(mapping, sort_keys=True).encode()).hexdigest()[:8]
    fname = f'{src_type}-{tgt_type}-{h}.json'
    with open(os.path.join(outdir, fname), 'w') as f:
        json.dump(result, f, indent=2)
    count += 1

print(f'Created {count} mapping files in {outdir}')
"

echo ""
echo "=== Step 2: Build mapping graph ==="
cd gnn
python3 populate_nx.py \
    --dataset "$DATASET" \
    --mappings-dir "../$MAPPINGS_DIR" \
    --api-lookup-file "../$API_LOOKUP" \
    --output-dir "../$OUTPUT_DIR"

echo ""
echo "=== Step 3: Train GNN model ==="
python3 train_gnn.py \
    --dataset "$DATASET" \
    --output-dir "../$OUTPUT_DIR"

echo ""
echo "=== Step 4: Verify ==="
ls -la "../$OUTPUT_DIR/${DATASET}_graph_data.pkl" 2>/dev/null
ls -la "../$OUTPUT_DIR/gnn_results/${DATASET}_mapping_gnn_results.pkl" 2>/dev/null

echo ""
echo "=== Done! ==="
echo "GNN model trained on ITGraph dataset."
echo "Run evaluation with:"
echo "  cd gnn && python3 gnn_runner.py --datasets ITGraph_HC --gnn-model-path ../data/gnn_results/${DATASET}_mapping_gnn_results.pkl --graph-data-path ../data/${DATASET}_graph_data.pkl --api-lookup-file ../data/itom_hashicorp_api_lookup_v2.json --data-dir ../data"
