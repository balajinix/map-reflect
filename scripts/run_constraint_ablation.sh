#!/bin/bash
# =============================================================================
# Constrained Assignment Ablation (ITOpsGraph + ITOM_Hashicorp)
# =============================================================================
# Runs 4 variants on ITOpsGraph (521 samples) and 2 on ITOM_Hashicorp (61):
#   1. Base (no reflection, no constraint)
#   2. Constrained only
#   3. UQ only (no constraint)
#   4. Constrained + UQ
#
# Results saved to results/constraint_ablation/ for paper Table comparison.
#
# Usage:
#   ./scripts/run_constraint_ablation.sh
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
log "Constrained Assignment Ablation"
log "Model: $MODEL"
log "=========================================="

# --- ITOpsGraph (521 samples) ---
ITOPS_LOOKUP="data/itopsgraph_api_lookup.json"
ITOPS_DATA="data"

log "=== ITOpsGraph (521 samples) ==="

# 1. Base: no reflection, no constraint
run_variant "ITOpsGraph" "itopsgraph_base" "$ITOPS_LOOKUP" "$ITOPS_DATA" \
    "--no-constrained-assignment --neighbor-k 0"

# 2. Constrained only: no reflection
run_variant "ITOpsGraph" "itopsgraph_constrained" "$ITOPS_LOOKUP" "$ITOPS_DATA" \
    "--neighbor-k 0"

# 3. UQ only: no constraint
run_variant "ITOpsGraph" "itopsgraph_uq" "$ITOPS_LOOKUP" "$ITOPS_DATA" \
    "--no-constrained-assignment --use-uq-reflection --neighbor-k 0"

# 4. Constrained + UQ
run_variant "ITOpsGraph" "itopsgraph_constrained_uq" "$ITOPS_LOOKUP" "$ITOPS_DATA" \
    "--use-uq-reflection --neighbor-k 0"

# --- ITOM_Hashicorp (61 samples) ---
IH_LOOKUP="data/itom_hashicorp_api_lookup.json"
IH_DATA="data"

log "=== ITOM_Hashicorp (61 samples) ==="

# 1. Base
run_variant "ITOM_Hashicorp" "itom_hashi_base" "$IH_LOOKUP" "$IH_DATA" \
    "--no-constrained-assignment --neighbor-k 0"

# 2. Constrained only
run_variant "ITOM_Hashicorp" "itom_hashi_constrained" "$IH_LOOKUP" "$IH_DATA" \
    "--neighbor-k 0"

# 3. UQ only
run_variant "ITOM_Hashicorp" "itom_hashi_uq" "$IH_LOOKUP" "$IH_DATA" \
    "--no-constrained-assignment --use-uq-reflection --neighbor-k 0"

# 4. Constrained + UQ
run_variant "ITOM_Hashicorp" "itom_hashi_constrained_uq" "$IH_LOOKUP" "$IH_DATA" \
    "--use-uq-reflection --neighbor-k 0"

log "=========================================="
log "Ablation complete! Results in $OUTPUT_BASE"
log "=========================================="

# Print summary
log "Summary:"
for dir in "$OUTPUT_BASE"/*/; do
    name=$(basename "$dir")
    for f in "$dir"/*.json; do
        [[ -f "$f" ]] || continue
        [[ "$f" == *checkpoint* ]] && continue
        metrics=$(python3 -c "
import json
d=json.load(open('$f'))
m=d.get('metrics',{})
print(f\"samples={m.get('total_samples',0)} type_acc={m.get('type_accuracy',0):.1%} field_acc={m.get('field_accuracy',0):.1%}\")
" 2>/dev/null)
        printf "  %-35s %s\n" "$name:" "$metrics"
    done
done
