#!/bin/bash
# =============================================================================
# SIMBA Sampling UQ Experiment
# =============================================================================
# Runs Map-Reflect with SIMBA pairwise consistency UQ on 3 datasets:
#   1. ITOpsGraph (521 samples, 1429 fields) — main benchmark
#   2. OMOP-CMS (38 samples, 145 fields) — healthcare vocabulary
#   3. OMOP-MIMIC (46 samples, 119 fields) — healthcare vocabulary
#
# For each dataset, runs 3 configs:
#   - base: no reflection (baseline)
#   - entropy: current heuristic entropy gate
#   - simba: SIMBA pairwise consistency gate (k=3 samples, temp=0.7)
#
# Usage:
#   ./scripts/run_simba_uq.sh
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
OUTPUT_BASE="results/simba_uq"
mkdir -p "$OUTPUT_BASE"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $1"; }

run_config() {
    local dataset="$1" lookup="$2" data_dir="$3" config_name="$4"
    shift 4
    local extra_flags="$@"

    local output_dir="${OUTPUT_BASE}/${dataset,,}_${config_name}"
    mkdir -p "$output_dir"

    log "Running $dataset / $config_name"
    python -m map_reflect.map_reflect_runner \
        --datasets "$dataset" \
        --models "$MODEL" \
        --property-lookup-file "$lookup" \
        --data-dir "$data_dir" \
        --batch-size 20 --checkpoint-interval 20 --resume \
        --output-dir "$output_dir" \
        $extra_flags \
        2>&1 | tee "${output_dir}/run.log" | tail -5
    echo
}

log "=========================================="
log "SIMBA Sampling UQ Experiment"
log "=========================================="

# --- ITOpsGraph ---
ITOPS_LOOKUP="data/itopsgraph_api_lookup.json"
ITOPS_DATA="data"

log "=== ITOpsGraph (521 samples) ==="

# Base: no reflection
run_config "ITOpsGraph" "$ITOPS_LOOKUP" "$ITOPS_DATA" "base"

# Entropy gate (current default)
run_config "ITOpsGraph" "$ITOPS_LOOKUP" "$ITOPS_DATA" "entropy" \
    --use-uncertainty --use-uq-reflection

# SIMBA gate (k=3 samples) — no-cache to force fresh sampling
run_config "ITOpsGraph" "$ITOPS_LOOKUP" "$ITOPS_DATA" "simba" \
    --use-uncertainty --use-uq-reflection --uq-method simba --n-uncertainty-samples 3 --no-cache

# --- OMOP-CMS ---
OMOP_CMS_LOOKUP="data/omop_cms_api_lookup.json"
OMOP_DATA="data"

log "=== OMOP-CMS (38 samples) ==="

run_config "OMOP_CMS" "$OMOP_CMS_LOOKUP" "$OMOP_DATA" "base"

run_config "OMOP_CMS" "$OMOP_CMS_LOOKUP" "$OMOP_DATA" "entropy" \
    --use-uncertainty --use-uq-reflection

run_config "OMOP_CMS" "$OMOP_CMS_LOOKUP" "$OMOP_DATA" "simba" \
    --use-uncertainty --use-uq-reflection --uq-method simba --n-uncertainty-samples 3 --no-cache

# --- OMOP-MIMIC ---
OMOP_MIMIC_LOOKUP="data/omop_mimic_api_lookup.json"

log "=== OMOP-MIMIC (46 samples) ==="

run_config "OMOP_MIMIC" "$OMOP_MIMIC_LOOKUP" "$OMOP_DATA" "base"

run_config "OMOP_MIMIC" "$OMOP_MIMIC_LOOKUP" "$OMOP_DATA" "entropy" \
    --use-uncertainty --use-uq-reflection

run_config "OMOP_MIMIC" "$OMOP_MIMIC_LOOKUP" "$OMOP_DATA" "simba" \
    --use-uncertainty --use-uq-reflection --uq-method simba --n-uncertainty-samples 3 --no-cache

log "=========================================="
log "SIMBA UQ experiment complete"
log "=========================================="

# Print summary
log "Results in $OUTPUT_BASE/"
for ds in itopsgraph omop_cms omop_mimic; do
    echo "=== ${ds} ==="
    for cfg in base entropy simba; do
        result_file=$(ls ${OUTPUT_BASE}/${ds}_${cfg}/mapreflect_*.json 2>/dev/null | head -1)
        if [[ -n "$result_file" ]]; then
            python3 -c "
import json
with open('$result_file') as f:
    d = json.load(f)
m = d.get('metrics', {})
print(f'  {\"$cfg\":8s}  type={m.get(\"type_accuracy\",0)*100:.1f}%  field={m.get(\"field_accuracy\",0)*100:.1f}%  samples={m.get(\"total_samples\",0)}')
"
        fi
    done
done
