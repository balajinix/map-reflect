#!/bin/bash

export PYTHONPATH=$(pwd)/../src:$PYTHONPATH

python evaluation-scripts/similarity_evaluator.py \
  --mapping_file "mappings_output/conert_vulnerability-turbo_entity.json" \
  --api_lookup_file "evaluation-scripts/api_lookup.json" \
  --output "evaluation-scripts/similarity_eval_output.json"