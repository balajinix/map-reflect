# Results Directory

This directory contains experiment results from various schema mapping methods.

## Syncing Results

Results files are **not committed to git** (to avoid large binary files). When running experiments across multiple machines (local and remote), use `rsync` to sync results:

```bash
# Sync results FROM remote TO local
rsync -avz remote-server:/path/to/graphql-schema-mapping/results/ ./results/

# Sync results FROM local TO remote
rsync -avz ./results/ remote-server:/path/to/graphql-schema-mapping/results/

# Sync all method-specific results directories
rsync -avz remote-server:/path/to/graphql-schema-mapping/map_reflect/results/ ./map_reflect/results/
rsync -avz remote-server:/path/to/graphql-schema-mapping/react/results/ ./react/results/
rsync -avz remote-server:/path/to/graphql-schema-mapping/smt/results/ ./smt/results/
rsync -avz remote-server:/path/to/graphql-schema-mapping/match_maker/results/ ./match_maker/results/
rsync -avz remote-server:/path/to/graphql-schema-mapping/rematch/results/ ./rematch/results/
```

## Result File Naming Convention

Results are named as: `{dataset}_{model}.json` or `{dataset}_{model}.jsonl`

Examples:
- `ITOpsGraph_meta-llama_llama-3-3-70b-instruct.json`
- `NL2GQL_ibm_granite-3-3-8b-instruct.json`
- `OMOP_CMS_watsonx_meta-llama_llama-3-3-70b-instruct.jsonl`

## Checkpoint Files

Checkpoint files (`*_checkpoint.json`) allow resuming interrupted experiments. They are created automatically and can be safely deleted after an experiment completes.

## Method-Specific Results Directories

Each method stores results in its own subdirectory:
- `map_reflect/results/` - UE-Reflect results
- `react/results/` - ReAct agent results
- `smt/results/` - SMT baseline results
- `match_maker/results/` - MatchMaker and DualContext results
- `rematch/results/` - ReMatch results
- `results/` - Consolidated/legacy results
