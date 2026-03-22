#!/bin/bash
# =============================================================================
# Constrained Assignment Ablation on OMOP + SchemaNet
# =============================================================================
# Expert-curated ground truth datasets. Tests whether constrained
# assignment helps or hurts on properly annotated benchmarks.
#
# OMOP: CMS (38), MIMIC (46), Synthea (12) — healthcare schema mapping
# SchemaNet: bank1_bank2, imdb_sakila — relational DB matching
#
# 4 variants each: base, constrained, uq, constrained+uq
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BASE_DIR"

if [[ -f "${BASE_DIR}/.venv/bin/activate" ]]; then
    source "${BASE_DIR}/.venv/bin/activate"
fi

if [[ -f "${BASE_DIR}/.env" ]]; then
    set -a && source "${BASE_DIR}/.env" && set +a
fi

MODEL="watsonx/meta-llama/llama-3-3-70b-instruct"
OUTPUT_BASE="results/constraint_ablation"
mkdir -p "$OUTPUT_BASE"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $1"; }

run_variant() {
    local dataset="$1"
    local label="$2"
    local lookup="$3"
    local data_dir="$4"
    local extra_args="$5"
    local output_dir="${OUTPUT_BASE}/${label}"
    mkdir -p "$output_dir"

    log "Running: $dataset / $label"
    python -m map_reflect.map_reflect_runner \
        --datasets "$dataset" \
        --models "$MODEL" \
        --property-lookup-file "$lookup" \
        --data-dir "$data_dir" \
        --batch-size 20 \
        --checkpoint-interval 20 \
        --output-dir "$output_dir" \
        $extra_args \
        2>&1 | grep -E "Phase 3.5|Metrics"
    echo
}

log "=========================================="
log "Constrained Assignment Ablation (OMOP + SchemaNet)"
log "=========================================="

# --- OMOP datasets ---
for OMOP_DS in OMOP_CMS OMOP_MIMIC OMOP_Synthea; do
    ds_lower=$(echo "$OMOP_DS" | tr '[:upper:]' '[:lower:]')
    LOOKUP="data/${ds_lower}_api_lookup.json"

    log "=== $OMOP_DS ==="

    run_variant "$OMOP_DS" "${ds_lower}_base" "$LOOKUP" "data" \
        "--no-constrained-assignment --neighbor-k 0"

    run_variant "$OMOP_DS" "${ds_lower}_constrained" "$LOOKUP" "data" \
        "--neighbor-k 0"

    run_variant "$OMOP_DS" "${ds_lower}_uq" "$LOOKUP" "data" \
        "--no-constrained-assignment --use-uq-reflection --neighbor-k 0"

    run_variant "$OMOP_DS" "${ds_lower}_constrained_uq" "$LOOKUP" "data" \
        "--use-uq-reflection --neighbor-k 0"
done

# --- SchemaNet (two cleanest pairs) ---
for SN_DS in schemanet_bank1_bank2 schemanet_imdb_sakila; do
    LOOKUP="data/schemanet/${SN_DS}_api_lookup.json"

    log "=== $SN_DS ==="

    run_variant "$SN_DS" "${SN_DS}_base" "$LOOKUP" "data/schemanet" \
        "--no-constrained-assignment --neighbor-k 0"

    run_variant "$SN_DS" "${SN_DS}_constrained" "$LOOKUP" "data/schemanet" \
        "--neighbor-k 0"
done

log "=========================================="
log "Ablation complete!"
log "=========================================="
