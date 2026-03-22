# Constrained Assignment in Map-Reflect

## Problem

The MAP phase ranks target candidates for each source field independently.
When a source type has 22 fields but the target type has only 6 fields,
multiple source fields claim the same target field. In the
concert.Application → instana.ApplicationPerspective example:

- `instana_ApplicationPerspective.label` is claimed by 10 source fields
- `instana_ApplicationPerspective.id` is claimed by 5 source fields
- `instana_ApplicationPerspective.boundaryScope` is claimed by 7 source fields
- 3 target fields (`entityType`, `to`, `windowSize`) are never used

This inflates false positives and leaves valid target fields unmapped.

## Relationship to Paper Claims

The paper (Section 3.1) describes Map-Reflect as:

> Phase 1: Map (Greedy One-to-One). For each source type, we compute
> embedding similarity with all types in the target schema and select
> the highest-scoring target type. For each field in the mapped source
> type, we similarly select the best matching field.

> Phase 2: Reflect (Uncertainty-Guided Refinement). The Reflect phase
> selectively refines the greedy mapping using external signals.

The current implementation does Phase 1 correctly at the type level but
not at the field level — each field is ranked independently with no
constraint preventing target reuse. Phase 2 (UQ-triggered reflection)
exists but has no structural signal about contested assignments.

## Solution: Constrained Assignment (Phase 1.5)

Insert a constrained assignment step between MAP ranking and REFLECT.
This enforces one-to-one field mapping and produces structural signals
for reflection.

### Algorithm

```
Input:
  rankings[src_field] = [(target_id, score, reason), ...] for each source field

Output:
  assignments: source_field -> target_field (one-to-one)
  contested: list of (source_field, original_target, evicted_by)
  unmatched: list of source_fields with no good assignment

Algorithm:
  1. Sort source fields by their top-1 confidence score (descending)

  2. For each source field (highest confidence first):
     a. Walk through its ranking list in order:
        - If candidate target is UNCLAIMED:
            Assign source -> target
            Break
        - If candidate target is CLAIMED but current source has
          higher score than the existing claimant:
            Evict the existing claimant (add to eviction queue)
            Assign current source -> target
            Break
        - Otherwise:
            Continue to next candidate
     b. If no candidate could be assigned:
        Mark source as UNMATCHED

  3. Process eviction queue (evicted sources need reassignment):
     Repeat step 2 for evicted sources, but only consider
     targets not yet assigned. Iterate until stable.

  4. Collect signals:
     - contested_fields: sources that were evicted at least once
     - unmatched_fields: sources with no assignment after all rounds
     - target_utilization: fraction of target fields that got assigned
```

### Properties

- **Greedy with preemption**: Higher-confidence sources can take targets
  from lower-confidence ones, ensuring the best matches win.

- **Convergence**: Each eviction reduces the set of available targets for
  the evicted source. The process terminates when no more evictions occur
  (at most N rounds where N = number of source fields).

- **Many-to-one detection**: If a source field exhausts all candidates
  (all taken by higher-confidence sources) and has no viable alternative,
  it becomes UNMATCHED. This signals a genuine many-to-one case where
  multiple source fields legitimately map to the same target concept.

### Integration with Reflection

The constrained assignment produces three concrete signals for REFLECT:

1. **Contested assignments** — source fields that were evicted and
   reassigned to a lower-ranked candidate. These are candidates for
   UQ reflection because their final assignment may not be semantically
   correct (they were pushed to a second-best option).

2. **Unmatched source fields** — fields that could not be assigned to
   any unclaimed target. Reflection should expand the candidate pool
   (neighbor graph traversal) to find targets outside the initial
   retrieval set.

3. **Low target utilization** — if many target fields remain unclaimed,
   the greedy assignment is too conservative. Reflection can try to
   match unmatched sources to unused targets with lower similarity
   thresholds.

These signals replace or complement the entropy-based UQ trigger:
- Entropy measures confidence dispersion across candidates
- Assignment conflict measures structural impossibility (two sources
  wanting the same target)
- Both can fire together: a contested field that also has high entropy
  is the strongest reflection candidate

### Confidence Score Update

After constrained assignment, update confidence scores:

```
For assigned fields (not contested):
  confidence = original top-1 score (unchanged)

For contested fields (reassigned to lower candidate):
  confidence = new_candidate_score * penalty_factor
  where penalty_factor = 0.8 (configurable)
  This reflects reduced confidence due to forced reassignment.

For unmatched fields:
  confidence = 0.0 (triggers reflection)
```

### Where in Code

In `map_reflect_runner.py`, between Phase 3 (processing responses) and
Phase 4 (reflection), approximately line 3200:

```
Phase 1:   Collect ranking prompts (existing)
Phase 2:   Batch inference (existing)
Phase 3:   Parse responses (existing)
Phase 3.5: CONSTRAINED ASSIGNMENT (new)
           - Build assignment matrix from parsed rankings
           - Run greedy assignment with preemption
           - Collect contested/unmatched signals
Phase 4:   REFLECT on contested + unmatched fields (enhanced)
Phase 5:   Assemble results (existing)
```

### Example: concert.Application → instana.ApplicationPerspective

Before (current):
```
concert_Application.id          -> instana_AP.id          (0.9)
concert_Application.name        -> instana_AP.label       (0.9)
concert_Application.latest      -> instana_AP.label       (0.8)  ← COLLISION
concert_Application.status      -> instana_AP.label       (0.8)  ← COLLISION
concert_Application.version     -> instana_AP.label       (0.8)  ← COLLISION
concert_Application.created_on  -> instana_AP.id          (0.8)  ← COLLISION
... (16 more fields, all mapping to id/label/boundaryScope)
```
3 target fields used, 3 unused. Many false positives.

After (with constrained assignment):
```
concert_Application.name        -> instana_AP.label            (0.9) assigned first (highest)
concert_Application.id          -> instana_AP.id               (0.9) assigned
concert_Application.latest      -> instana_AP.windowSize       (0.6) reassigned (label taken)
concert_Application.status      -> instana_AP.entityType       (0.5) reassigned (label taken)
concert_Application.version     -> instana_AP.to               (0.4) reassigned
concert_Application.created_on  -> instana_AP.boundaryScope    (0.5) reassigned (id taken)
... (remaining 16 fields -> UNMATCHED, trigger reflection)
```
6 target fields used. Contested fields flagged for reflection.

### Handling Asymmetric Cardinality

When source has more fields than target (22 vs 6), many source fields
will be UNMATCHED after assignment. This is expected and correct:

- Not every source field has a semantic equivalent in the target
- UNMATCHED fields get confidence 0.0 and are reported as "no mapping"
- This naturally produces the many-to-many coverage the paper describes:
  the system reports which fields CAN be mapped (with confidence) and
  which CANNOT (explicitly unmapped)

When target has more fields than source, some target fields remain
unclaimed. Post-assignment analysis can flag these as potential
expansion opportunities for the source schema.

### Complexity

- Sorting: O(S log S) where S = source fields
- Assignment: O(S × K) where K = candidates per field (typically 5-10)
- Eviction rounds: at most S rounds, each O(S × K)
- Total: O(S² × K) — negligible compared to LLM inference cost

For the concert.Application example: 22 × 10 × 22 = 4,840 operations.

---

## Experimental Results

### Ablation: Constrained vs Base vs UQ (Llama-3.3-70B, Field F1)

| Dataset (samples) | Base | Constrained | UQ | Constr+UQ | GT many-to-one |
|-------------------|------|-------------|-----|-----------|----------------|
| ITOpsGraph (521) | **88.6** | 86.9 | 88.5 | 86.7 | 68 |
| OMOP-CMS (38) | **76.6** | 71.0 | 76.6 | 72.4 | 9 |
| OMOP-MIMIC (46) | **59.7** | 44.5 | **60.5** | 47.1 | 23 |
| OMOP-Synthea (12) | **100.0** | 53.5 | **100.0** | 64.4 | 29 |
| ITOM_Hashicorp (61) | **54.8** | 28.6 | 51.4 | 35.0 | 64 (synthetic) |
| SchemaNet bank (6) | 0.0 | 0.0 | — | — | 4 |
| SchemaNet imdb (5) | 0.0 | 0.0 | — | — | 3 |

API-level accuracy is 100% across all variants on ITOpsGraph, ITOM_Hashicorp,
and SchemaNet.  OMOP API accuracy is 0% in the runner's internal metric due to
a format mismatch (eval/calc_accuracy.py reports the correct numbers).

### Key Findings

1. **Constrained assignment reduces field F1 on every dataset.**
   The drop is proportional to the number of legitimate many-to-one mappings
   in the ground truth.  OMOP-Synthea has 29 ground truth collisions (multiple
   source fields correctly mapping to the same OMOP target column); enforcing
   one-to-one breaks these correct mappings, dropping F1 from 100% to 53.5%.

2. **UQ reflection is neutral or slightly positive** on all datasets.
   It fires selectively and does not degrade performance — consistent with
   the paper's ablation findings.

3. **Constrained + UQ does not recover the loss.** UQ reflection cannot
   undo the structural damage of forcing one-to-one assignment when the
   ground truth is many-to-one.

4. **Collision count drops dramatically** (ITOpsGraph: 68→7, OMOP-Synthea:
   29→5), confirming the mechanism works as designed.  The problem is that
   many of those "collisions" are correct many-to-one mappings.

### Conclusion

**Constrained assignment is NOT the right default for field-level mapping.**

Schema mapping inherently involves many-to-one relationships:
- Multiple source columns join to the same target FK
- Several source attributes represent facets of one target concept
- OMOP's CDM maps multiple proprietary fields to standardized columns

Enforcing one-to-one at the field level breaks these legitimate patterns.

**When to use constrained assignment:**
- Schema federation/query rewriting where duplicate mappings break joins
- Generating deduplicated field correspondence tables
- As an analysis tool to detect over-reliance on a single target field

**Recommended default:** Base + UQ (the paper's current recommendation).
The base MAP phase allows many-to-one naturally; UQ catches genuinely
uncertain predictions without forcing structural constraints.

### Implementation Note

The `--no-constrained-assignment` flag (default: constraint ON) should be
changed to `--constrained-assignment` (default: OFF) to reflect that the
base configuration without constraint is the recommended default.  The
flag was added as `use_constrained_assignment: bool = True` in
`MapReflectConfig`; this should be flipped to `False`.
