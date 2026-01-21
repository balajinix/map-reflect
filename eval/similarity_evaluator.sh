#!/bin/bash
# Run similarity-based evaluation on a mapping file
# Usage: ./eval/similarity_evaluator.sh

cd "$(dirname "$0")/.."

python eval/similarity_evaluator.py \
  --mapping_file "data/mappings_output/concert_vulnerability-turbo_entity.json" \
  --api_lookup_file "eval/api_lookup.json" \
  --output "eval/similarity_eval_output.json"
