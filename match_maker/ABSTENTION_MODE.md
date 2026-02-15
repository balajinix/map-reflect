# MatchMaker Abstention Mode Configuration

## Overview

The MatchMaker schema matching algorithm uses a confidence scoring mechanism to decide whether to make a match or abstain. The original paper implementation uses a "None of the above" option that competes with actual candidates, and if "None" scores highest, the model abstains from making a match.

However, in practice, this can lead to excessive abstentions when:
1. The LLM is conservative about assigning high confidence scores
2. The candidates are reasonable but not perfect matches
3. The prompt encourages abstention for imperfect matches

This document describes the configurable `abstention_mode` parameter that controls this behavior.

## Abstention Modes

### `strict` (Original Paper Behavior)
- **Description**: Abstain if the "None of the above" confidence score is higher than the best candidate's score.
- **Use case**: When you want high precision and are willing to accept lower recall (fewer predictions but more confident ones).
- **Behavior**: `abstain = (none_confidence >= best_candidate_confidence)`

### `lenient` (Default, Recommended)
- **Description**: Only abstain if the best candidate's confidence is below a threshold AND "None" scores higher.
- **Use case**: When you want a balance between making predictions and avoiding clearly wrong matches.
- **Behavior**: `abstain = (best_confidence < threshold) AND (none_confidence > best_confidence)`
- **Default threshold**: 20.0 (out of 100)

### `disabled`
- **Description**: Never abstain - always select the best candidate regardless of confidence.
- **Use case**: When you want maximum recall and will post-process results, or for analysis purposes.
- **Behavior**: `abstain = False` (always select best candidate)

### `threshold`
- **Description**: Abstain if the best candidate's confidence is below a fixed threshold.
- **Use case**: When you want a simple confidence-based cutoff without the "None" option comparison.
- **Behavior**: `abstain = (best_confidence < threshold)`

## Configuration

### Command Line
```bash
python match_maker_runner.py \
    --property-lookup-file ../data/itops_api_lookup.json \
    --data-dir ../data \
    --datasets ITOps \
    --models watsonx/meta-llama/llama-3-3-70b-instruct \
    --abstention-mode lenient \
    --min-confidence 20
```

### Programmatic
```python
from match_maker_agent import MatchMakerAgent

agent = MatchMakerAgent(
    property_lookup_file="path/to/lookup.json",
    model="watsonx/meta-llama/llama-3-3-70b-instruct",
    dataset="ITOps",
    abstention_mode="lenient",      # 'strict', 'lenient', 'disabled', 'threshold'
    min_confidence_threshold=20.0    # Used by 'lenient' and 'threshold' modes
)
```

## Prompt Changes

The confidence scoring prompt was also updated to be less aggressive about recommending "None of the above":

**Before:**
```
If no option is a good match, score "None of the above" highest.
```

**After:**
```
For "None of the above": Only score it high (>50) if ALL other options are clearly wrong or incompatible.
Prefer making a match when there's a reasonable candidate over abstaining.
```

This change, combined with the `lenient` abstention mode, significantly reduces excessive abstentions while still allowing the model to abstain when truly no good match exists.

## Impact on Results

- **Strict mode**: May result in many abstentions (0% mapping rate in some cases)
- **Lenient mode**: Balances making predictions with avoiding low-confidence matches
- **Disabled mode**: Always makes a prediction (100% mapping rate)
- **Threshold mode**: Similar to lenient but ignores "None" score entirely

## Recommendations

1. **Start with `lenient` mode** (the default) for most evaluation tasks
2. Use `disabled` mode for analysis and debugging
3. Use `strict` mode only if you have high-quality training data and need maximum precision
4. Adjust `min_confidence_threshold` based on your quality requirements (lower = more matches, higher = more abstentions)
