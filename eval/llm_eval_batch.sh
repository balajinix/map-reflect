#!/bin/bash
# Run LLM-based batch evaluation on all mapping files
# Usage: ./eval/llm_eval_batch.sh
# Requires: WATSONX_APIKEY and WATSONX_PROJECT_ID in environment

cd "$(dirname "$0")/.."

python eval/llm_eval_batch.py \
  --mapping_dir data/mappings_output \
  --api_lookup_file eval/api_lookup.json \
  --output eval/all_results.json
