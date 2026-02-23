# Map-Reflect Example: coverage_03641

## Overview
This example demonstrates all three reflection signals working together to correct an initial mapping error.

**Task**: Map fields from `cloudability.Aggregate` to equivalent fields in the target product `turbonomic`

**Query**: "find types that can be mapped to cloudability Aggregate"

**Ground Truth Mapping**:
| Source Field | Target Field |
|--------------|--------------|
| cloudability_Aggregate.name | turbonomic_StatFilterEntry.displayName |
| cloudability_Aggregate.type | turbonomic_StatFilterEntry.type |
| cloudability_Aggregate.value | turbonomic_StatFilterEntry.value |

---

## Stage 1: Initial Mapping (WRONG)

For the field `cloudability_Aggregate.name`, the model produces this **incorrect** initial ranking:

| Rank | Candidate | Score | Reason |
|------|-----------|-------|--------|
| 1 | cloudability_Aggregate.type | 0.8 | Same prefix, similar semantic meaning |
| 2 | cloudability_Aggregate.value | 0.7 | Same prefix |
| 3 | cloudability_Measure_MeasureInfo.type | 0.6 | Similar semantic meaning to 'type' |
| 4 | cloudability_Dimension_MeasureInfo.type | 0.6 | Similar semantic meaning |
| 5 | cloudability_Filter_MetaInfo.value | 0.6 | Similar semantic meaning |
| 6 | **turbonomic_StatFilterEntry.displayName** | 0.5 | Similar to 'name', different prefix |
| 7 | turbonomic_StatFilterEntry.type | 0.5 | Different prefix |
| 8 | turbonomic_StatFilterEntry.value | 0.5 | Different prefix |

**Problem**: The correct answer (`turbonomic_StatFilterEntry.displayName`) is ranked #6 with score 0.5, while the top candidate is from the WRONG product (source instead of target).

---

## Stage 2: Three Signals Trigger Reflection

### Signal 1: ERROR (Source/Target Confusion)
```
ERROR: Top prediction 'cloudability_Aggregate.type' is from SOURCE product
'CLOUDABILITY', but should be from TARGET product 'TURBONOMIC'.
The model matched by name similarity instead of finding the semantic
equivalent in the target schema.
```

### Signal 2: HIGH_UQ (Uncertainty = 0.285)
- Uncertainty threshold: 0.15
- Measured uncertainty: **0.285** (HIGH)
- Method: Heuristics-based (score gaps, semantic similarity)
- Triggers reflection because uncertainty > threshold

### Signal 3: NEIGHBORHOOD (Score Gap Analysis)
Score gaps between top candidates:
- Rank 1→2: 0.8 - 0.7 = **0.1** (< 0.2 threshold → close competition)
- Rank 5→6: 0.6 - 0.5 = **0.1** (< 0.2 threshold → correct answer is competitive)

The small score gaps indicate ambiguity that neighborhood context from the property graph can help resolve.

---

## Stage 3: Reflection Analysis

The model receives the detected errors and performs self-reflection:

> **Analysis**: "The initial ranking shows HIGH UNCERTAINTY, indicating potential issues with the top-ranked candidates. The top prediction, 'cloudability_Aggregate.type', is from the SOURCE product 'CLOUDABILITY', but it should be from the TARGET product 'TURBONOMIC'. This suggests that the model matched by name similarity instead of finding the semantic equivalent in the target schema. Furthermore, the warning about name similarity instead of semantic meaning and the high uncertainty score suggest that the ranking may not accurately reflect the best matches. A closer examination of the candidates and their semantic meanings is necessary to provide a more accurate ranking."

---

## Stage 4: Revised Ranking (CORRECT)

After reflection, the model produces the **corrected** ranking:

| Rank | Candidate | Score | Reason |
|------|-----------|-------|--------|
| **1** | **turbonomic_StatFilterEntry.displayName** | **0.8** | Similar semantic meaning to 'name', from TARGET product |
| 2 | cloudability_Aggregate.type | 0.7 | From SOURCE product, deprioritized |
| 3 | cloudability_Aggregate.value | 0.6 | From SOURCE product |
| 4 | cloudability_Measure_MeasureInfo.type | 0.5 | Not from TARGET product |
| 5 | cloudability_Dimension_MeasureInfo.type | 0.5 | Not from TARGET product |

**Result**: Correct answer now ranked #1 with score 0.8

---

## Property Graph Context (Neighborhood Expansion)

The property graph connects fields across products based on:
1. **Same field name** (e.g., "type", "value", "name")
2. **Same product** (sibling fields)
3. **Semantic similarity** (e.g., "name" ↔ "displayName")

### Neighbors of `cloudability_Aggregate.name`:

**Same-type neighbors** (cloudability_Aggregate.*):
- cloudability_Aggregate.type
- cloudability_Aggregate.value
- cloudability_Aggregate.label
- cloudability_Aggregate.description
- cloudability_Aggregate.data_type

**Cross-product neighbors** (fields with similar names):
- turbonomic_StatsEntry.name
- turbonomic_StatInput.name
- concert_Access_points.name
- watsonx-orchestrate_SkillsetsEntry.name

**Semantic neighbors** (displayName ~ name):
- turbonomic_StatFilterEntry.displayName ← **correct target**
- turbonomic_ActionSchedule.displayName
- turbonomic_BusinessAccount.displayName
- turbonomic_ContainerClusterEntity.displayName

---

## Schema Information

### Source: cloudability_Aggregate
| Field | Type | Description |
|-------|------|-------------|
| name | String | Name of the aggregate |
| type | String | Type of aggregate |
| value | Int | Aggregate value |
| label | String | Display label |
| description | String | Description text |
| data_type | String | Data type |

### Target: turbonomic_StatFilterEntry
| Field | Type | Description |
|-------|------|-------------|
| displayName | String | Human-readable name |
| type | String | Filter type |
| value | String | Filter value |

---

## Final Results

| Metric | Before Reflection | After Reflection |
|--------|-------------------|------------------|
| Accuracy | 0/3 (0%) | 3/3 (100%) |
| Mean Uncertainty | 0.285 | 0.135 |
| Correct Rank of Target | #6 | #1 |

All three fields were correctly mapped after reflection:
- ✅ cloudability_Aggregate.name → turbonomic_StatFilterEntry.displayName
- ✅ cloudability_Aggregate.type → turbonomic_StatFilterEntry.type
- ✅ cloudability_Aggregate.value → turbonomic_StatFilterEntry.value

---

## Figure Suggestion

Create a flow diagram showing:

```
┌─────────────────────────────────────────────────────────────────┐
│                    INITIAL MAPPING STAGE                        │
├─────────────────────────────────────────────────────────────────┤
│  Query: "map cloudability_Aggregate.name to target"             │
│                                                                 │
│  Initial Ranking:                                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ #1  cloudability_Aggregate.type        [0.8] ← WRONG    │   │
│  │ #2  cloudability_Aggregate.value       [0.7]            │   │
│  │ ...                                                      │   │
│  │ #6  turbonomic_StatFilterEntry.displayName [0.5]        │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SIGNAL DETECTION                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │    ERROR     │  │   HIGH_UQ    │  │ NEIGHBORHOOD │          │
│  │    🔴        │  │     🟡       │  │     🔵       │          │
│  │              │  │              │  │              │          │
│  │ Source/Target│  │ Uncertainty  │  │  Score Gap   │          │
│  │  Confusion   │  │   = 0.285    │  │   < 0.2      │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    REFLECTION STAGE                             │
├─────────────────────────────────────────────────────────────────┤
│  LLM Self-Reflection:                                           │
│  "Top prediction is from SOURCE product but should be from      │
│   TARGET product TURBONOMIC. Model matched by name similarity   │
│   instead of semantic meaning..."                               │
│                                                                 │
│  Property Graph Context:                                        │
│  • "name" semantically similar to "displayName"                 │
│  • turbonomic_StatFilterEntry has displayName, type, value     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CORRECTED OUTPUT                             │
├─────────────────────────────────────────────────────────────────┤
│  Revised Ranking:                                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ #1  turbonomic_StatFilterEntry.displayName [0.8] ← ✓    │   │
│  │ #2  cloudability_Aggregate.type            [0.7]        │   │
│  │ #3  cloudability_Aggregate.value           [0.6]        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Accuracy: 0% → 100%                                            │
└─────────────────────────────────────────────────────────────────┘
```
