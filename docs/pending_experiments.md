# Pending Experiments

**Updated:** 2026-04-01

## Currently Running

1. **ITOpsGraph ablation** (6 remaining configs with --no-cache)
   - error_refl done (72.7%), uq_refl running
   - nbr_refl, error_uq, error_nbr, uq_nbr, induced_refl queued
   - ETA: ~3-4 hours

2. **AnyTool ToolBench full** (3190 samples)
   - 882/3190 done
   - ETA: ~40 min

## Queued (after ablation completes)

3. **ITOpsGraph-MB full dataset** (2529 samples, 4621 fields)
   - New dataset: `/home/bganesa1/code/ITOpsGraph/itgraph/data/itgraph_mb_dataset.jsonl`
   - Copy to schema-mapping, then run:
     - Map-Reflect × 3 models (~25 min)
     - MatchMaker × 3 models (~7 hours)
   - Removes "246-sample subset" limitation from paper
   - Uses `itopsgraph_api_lookup.json` (11,110 entries, all 13 products covered)
   - Updates Tables 5-6 in paper

## Completed Today

- MR main table: all 7 datasets × 3 models (results_v2)
- NL2GQL evaluation fix (normalized matching)
- ToolBench MR-Mistral
- GNN strict top-1 evaluation
- ITOpsGraph ablation: base (81.2%), self_refl (79.7%), error_refl (72.7%)
- Full provenance audit (docs/tables_2_3_final_audit.md)
