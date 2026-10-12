#!/usr/bin/env bash
# Method v5 (docs/naacl/v6/PLAN_PHASE2.md, "Method v5") second-pass runs. One stream per model and step, the ten core
# datasets in order. Method v4.1 flags always (MAPREFLECT_V41=1, MAPREFLECT_CANONICAL_MARGIN=1); the description
# repair MAPREFLECT_FIELD_FIRST=1 on every dataset that has data/<ds>_reflect_lookup_ff1.json (elsewhere the treatment
# changes no prompt). TOKEN_LOG and the raw log (<out>.raw.jsonl) on every call. endpoint_b models run through
# endpoint_b_throttle.py (one shared token bucket). Resumable: the runner skips written keys, a .done marker skips a cell.
#   dev   v5_run.py --actions both --with-check on every development field
#   d3c   the current reflection (v6_d3_run.py arms cur,blind) under the description repair, on the D3 development
#         fields of the datasets the repair changes (the other D3 cells' prompts are unchanged: their D3 rows stand)
#   test  v5_run.py --actions test on every test field; refuses to start without docs/naacl/results/v5_frozen.json
# Resume guard: v5_run.py and v6_d3_run.py write <out>.fingerprint.json (flags, code and input md5s, settings;
# scripts/naacl/run_guard.py) and refuse to resume an output written under another configuration, or one with rows
# but no fingerprint. A refused cell is logged FAIL; write it to a fresh path or set the old file aside.
# Log: results/naacl/v6/v5/run.log
#
# Usage: run_v5.sh <gptoss|qwen35|llama4|granite> <dev|d3c|test> [WORKERS=n] [DATASETS="..."]
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
PY="$(pwd)/.venv/bin/python"
TAG=$1; STEP=$2
D=results/naacl/v6/v5; mkdir -p "$D/tokens" "$D/$STEP"
LOG=$D/run.log
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
WRAP=(); case $TAG in llama4|granite) WRAP=(scripts/naacl/v6_calls/endpoint_b_throttle.py) ;; esac
exec 9>"$D/.stream_${TAG}_${STEP}.lock"; flock -n 9 || { log "stream $TAG $STEP already running; this launch exits"; exit 0; }
WORKERS=${WORKERS:-8}
DATASETS=${DATASETS:-itopsgraph toolbench nl2gql omop_cms omop_mimic omop_synthea valentine_chembl valentine_opendata valentine_tpcdi valentine_wikidata}
export MAPREFLECT_V41=1 MAPREFLECT_CANONICAL_MARGIN=1
if [ "$STEP" = test ] && [ ! -s docs/naacl/results/v5_frozen.json ]; then log "REFUSE test $TAG: no docs/naacl/results/v5_frozen.json"; exit 1; fi
log "=== start v5 $TAG $STEP: $DATASETS (v5_run md5 $(md5sum scripts/naacl/v5_run.py | cut -c1-32), v6_d3_run md5 $(md5sum scripts/naacl/v6_d3_run.py | cut -c1-32), reflect_v2 md5 $(md5sum scripts/naacl/reflect_v2.py | cut -c1-32), field_first md5 $(md5sum scripts/naacl/field_first.py | cut -c1-32)) ==="
for ds in $DATASETS; do
  FFV=0; [ -s "data/${ds}_reflect_lookup_ff1.json" ] && FFV=1
  o=$D/$STEP/${ds}__${TAG}.jsonl
  [ -e "$o.done" ] && { log "skip $STEP $ds $TAG (done)"; continue; }
  case $STEP in
    dev)  args=(scripts/naacl/v5_run.py --dataset "$ds" --model-tag "$TAG" --split dev --actions both --with-check) ;;
    test) args=(scripts/naacl/v5_run.py --dataset "$ds" --model-tag "$TAG" --split test --actions test)
          # the frozen intervention's verifier: the either-order rule needs no current check; a frozen current check does
          [ "$($PY -c "import json; print(json.load(open('docs/naacl/results/v5_frozen.json'))['intervention']['outcome_field'])")" = chk ] \
            && args+=(--with-check) ;;
    d3c)
      [ "$FFV" = 1 ] || { log "skip d3c $ds $TAG (the repair changes no prompt here)"; continue; }
      k=results/naacl/v6/d3/keys/${ds}__${TAG}.txt
      [ -s "$k" ] || { log "skip d3c $ds $TAG (no D3 field)"; continue; }
      args=(scripts/naacl/v6_d3_run.py --dataset "$ds" --model-tag "$TAG" --keys "$k" --arms cur,blind) ;;
    *) echo "unknown step $STEP" >&2; exit 1 ;;
  esac
  log "RUN  $STEP $ds $TAG (ff1 $FFV)"
  if [ "$FFV" = 1 ]; then export MAPREFLECT_FIELD_FIRST=1; else unset MAPREFLECT_FIELD_FIRST; fi
  TOKEN_LOG="$D/tokens/${STEP}__${ds}__${TAG}.jsonl" TOKEN_RUN="v5:$STEP/${ds}__${TAG}" \
  $PY "${WRAP[@]}" "${args[@]}" --workers "$WORKERS" --out "$o" >> "$D/$STEP/${ds}__${TAG}.log" 2>&1 \
    && { log "DONE $STEP $ds $TAG"; date > "$o.done"; } || log "FAIL $STEP $ds $TAG"
done
log "=== end v5 $TAG $STEP ==="
