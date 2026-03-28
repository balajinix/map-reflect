#!/bin/bash
# OMOP Signal Ablation (CMS + MIMIC, Llama-3.3-70B)
# Same 9 configs as ITOpsGraph ablation (Table 7)
set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate 2>/dev/null || true
source .env 2>/dev/null || true

MODEL="watsonx/meta-llama/llama-3-3-70b-instruct"
DATA_DIR="data"

for DATASET in OMOP_CMS OMOP_MIMIC; do
    LOOKUP="data/itopsgraph_api_lookup.json"
    OUTPUT_BASE="results/omop_ablation/${DATASET}"
    mkdir -p "$OUTPUT_BASE"

    echo "========================================"
    echo "Ablation: $DATASET"
    echo "========================================"

    run_config() {
        local name="$1"; shift
        local outdir="$OUTPUT_BASE/$name"
        if [ -d "$outdir" ] && ls "$outdir"/*.json 1>/dev/null 2>&1; then
            echo "  [$name] SKIP (exists)"
            return
        fi
        echo "  [$name] Running..."
        mkdir -p "$outdir"
        python -m map_reflect.map_reflect_runner \
            --datasets "$DATASET" --models "$MODEL" \
            --property-lookup-file "$LOOKUP" --data-dir "$DATA_DIR" \
            --output-dir "$outdir" "$@" 2>&1 | tail -3
    }

    run_config "base" --neighbor-k 0
    run_config "self_refl" --neighbor-k 0 --use-self-reflection
    run_config "error_refl" --neighbor-k 0 --use-error-reflection
    run_config "uq_refl" --neighbor-k 0 --use-uncertainty --use-uq-reflection
    run_config "nbr_refl" --use-neighborhood-reflection
    run_config "error_uq" --neighbor-k 0 --use-error-reflection --use-uncertainty --use-uq-reflection
    run_config "error_nbr" --use-error-reflection --use-neighborhood-reflection
    run_config "uq_nbr" --use-uncertainty --use-uq-reflection --use-neighborhood-reflection
    run_config "induced_refl" --use-error-reflection --use-uncertainty --use-uq-reflection --use-neighborhood-reflection
done

echo "OMOP ablation complete"
