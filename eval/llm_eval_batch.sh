#!/bin/bash

export PYTHONPATH=$(pwd)/../src:$PYTHONPATH

python evaluation-scripts/llm_eval_batch.py \
  --mapping_dir mappings_output \
  --api_lookup_file evaluation-scripts/api_lookup.json \
  --output evaluation-scripts/all_results.json

