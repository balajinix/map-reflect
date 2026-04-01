# Tag Abstraction Layer Design

## Overview

The goal is to create a higher-level abstraction layer that hides the complexity of the tagkey/tag relationship and provides a more intuitive API for users. Instead of requiring users to understand and manage the tagkey/tag pattern, we provide semantic field names that automatically handle the underlying tagkey filtering.

## Problem Statement

### Current State

Users must understand and manually manage the tagkey/tag relationship:

```graphql
# ❌ Complex and error-prone
query CostByTeam {
  llmFriendlyCostReport(
    dimensions: [tag21]
    match: { tagkey21: { eq: "team" } } # Must remember to filter by tagkey
  )
}
```

### Desired State

Users work with semantic field names:

```graphql
# ✅ Clean and intuitive
query CostByTeam {
  llmFriendlyCostReport(
    dimensions: [team]
    match: { team: { ne: null } } # Simple filter on semantic field
  )
}
```

## Implementation Challenges

### 1. Runtime Transformation Required

- **Constraint**: Mapping must happen at runtime in the StepZen layer
- **Challenge**: Cannot use Python preprocessing like our current camelCase → snake_case transformation
- **Solution**: Must implement in StepZen helper functions

### 2. Multi-Argument Helper Functions

- **Current State**: Helper functions operate on one argument at a time (dimension, metrics, match, etc.)
- **Required Change**: Need helper functions that can operate on multiple arguments simultaneously
- **Challenge**: Need to understand StepZen's "sequence" directive for coordinating multiple transformations

### 3. Response Mapping

- **Status**: Relatively straightforward
- **Approach**: Map tag21 values back to "team" field in response
- **Priority**: Lower concern compared to input transformation

## Proposed Architecture

### 1. Semantic Field Mappings

```yaml
# Configuration-driven mappings
semantic_fields:
  team:
    dimension: tag21
    filter_key: tagkey21
    filter_value: team
    description: "Team ownership"

  application:
    dimension: tag24
    filter_key: tagkey24
    filter_values: [app, application]
    description: "Application name"

  environment:
    dimension: tag20
    filter_key: tagkey20
    filter_values:
      [
        environment,
        cldy:aws:accountleveltag:environment,
        cldy:azure:resourcetag:environment,
      ]
    description: "Environment (prod, staging, dev)"

  customer_project:
    dimension: category16
    filter_key: category16
    filter_value: not applicable
    filter_operator: ne
    description: "Customer project name"
```

### 2. Input Transformation Process

#### Step 1: Dimension Transformation

```graphql
# Transform semantic dimensions to actual tag dimensions
dimensions: [team, application, vendor]
→ dimensions: [tag21, tag24, vendor]
```

#### Step 2: Filter Transformation

```graphql
# Transform semantic filters to tagkey filters
match: { team: { ne: null } }
→ match: { tagkey21: { eq: "team" } }
```

#### Step 3: Combined Filter Logic

```graphql
# Handle complex cases like "team = 'flagship'"
match: { team: { eq: "flagship" } }
→ match: {
    AND: [
      { tagkey21: { eq: "team" } },
      { tag21: { eq: "flagship" } }
    ]
  }
```

### 3. Response Transformation

```graphql
# Map tag fields back to semantic names
{
  tag21: "flagship",
  tag24: "biit",
  vendor: "Amazon"
}
→ {
  team: "flagship",
  application: "biit",
  vendor: "Amazon"
}
```

## StepZen Implementation Strategy

### 1. Multi-Argument Helper Functions

Current approach:

```graphql
# Single argument transformation
dimensions: [transformDimensions($dimensions)]
```

Required approach:

```graphql
# Multi-argument coordination
@sequence(
  transformDimensionsAndFilters($dimensions, $match)
)
```

### 2. Helper Function Design

```graphql
# Pseudo-code for multi-argument helper
function transformDimensionsAndFilters(dimensions, match) {
  // Transform dimensions
  const transformedDimensions = dimensions.map(dim => {
    if (dim in SEMANTIC_MAPPINGS) {
      return SEMANTIC_MAPPINGS[dim].dimension
    }
    return dim
  })

  // Transform filters
  const additionalFilters = []
  dimensions.forEach(dim => {
    if (dim in SEMANTIC_MAPPINGS) {
      const mapping = SEMANTIC_MAPPINGS[dim]
      additionalFilters.push({
        field: mapping.filter_key,
        operator: mapping.filter_operator || 'eq',
        value: mapping.filter_value
      })
    }
  })

  // Merge with existing filters
  const transformedMatch = mergeFilters(match, additionalFilters)

  return {
    dimensions: transformedDimensions,
    match: transformedMatch
  }
}
```

### 3. Schema Changes

#### Input Schema

```graphql
input LlmFriendlyCostReportInput {
  dimensions: [Dimension!]!
  metrics: [Metric!]!
  match: MatchInput
  # ... other fields
}

enum Dimension {
  # Standard dimensions
  vendor
  service_name
  region

  # Semantic tag dimensions
  team
  application
  environment
  customer_project

  # Raw tag dimensions (for advanced users)
  tag1
  tag2
  # ... etc
}

input MatchInput {
  # Standard filters
  vendor: StringFilterInput
  service_name: StringFilterInput

  # Semantic tag filters
  team: StringFilterInput
  application: StringFilterInput
  environment: StringFilterInput
  customer_project: StringFilterInput

  # Raw tag filters (for advanced users)
  tagkey1: StringFilterInput
  tagkey2: StringFilterInput
  # ... etc
}
```

#### Output Schema

```graphql
type LlmFriendlyCostReportResult {
  # Standard fields
  vendor: String
  service_name: String
  region: String

  # Semantic tag fields
  team: String
  application: String
  environment: String
  customer_project: String

  # Raw tag fields (for advanced users)
  tag1: String
  tag2: String
  # ... etc
}
```

## Implementation Phases

### Phase 1: Foundation

1. **Research StepZen sequence directive** for multi-argument coordination
2. **Create semantic field mappings** configuration
3. **Implement basic dimension transformation** helper
4. **Test with simple cases** (team dimension only)

### Phase 2: Filter Transformation

1. **Implement filter transformation** helper
2. **Handle complex filter logic** (AND conditions)
3. **Support multiple filter values** (app/application)
4. **Test with various filter scenarios**

### Phase 3: Response Mapping

1. **Implement response transformation** helper
2. **Map tag fields to semantic names**
3. **Maintain backward compatibility**
4. **Test end-to-end scenarios**

### Phase 4: Production

1. **Add comprehensive semantic fields** based on analysis
2. **Update documentation** to prefer semantic fields
3. **Monitor performance** and optimize
4. **Consider deprecating raw tag fields** (optional)

## Benefits

### 1. User Experience

- **Intuitive API**: No need to understand tagkey/tag relationships
- **Semantic naming**: `team` instead of `tag21`
- **Reduced errors**: Automatic tagkey filtering
- **Cleaner queries**: Simpler, more readable GraphQL

### 2. Maintainability

- **Centralized configuration**: Easy to add new semantic fields
- **Backward compatibility**: Raw tag fields still available
- **Configuration-driven**: No code changes for new mappings
- **Documentation**: Clear semantic meaning for each field

### 3. Flexibility

- **Advanced users**: Can still use raw tag fields
- **Future-proof**: Easy to extend with new semantic fields
- **Multi-cloud support**: Handle different tagkey patterns
- **Complex scenarios**: Support AND/OR filter logic

## Risks and Mitigation

### 1. Performance Impact

- **Risk**: Additional transformation overhead
- **Mitigation**: Optimize helper functions, cache transformations

### 2. Complexity

- **Risk**: More complex helper logic
- **Mitigation**: Thorough testing, clear documentation, gradual rollout

### 3. Backward Compatibility

- **Risk**: Breaking existing queries
- **Mitigation**: Maintain raw tag fields, provide migration guide

### 4. StepZen Limitations

- **Risk**: Sequence directive complexity
- **Mitigation**: Research thoroughly, test with StepZen team if needed

## Next Steps

1. **Research StepZen sequence directive** for multi-argument coordination
2. **Create proof of concept** with team field transformation
3. **Test with real data** to validate approach
4. **Design helper function architecture** based on StepZen capabilities
5. **Implement phase 1** with basic dimension transformation

## Questions for Investigation

1. **StepZen sequence directive**: How does it work for coordinating multiple argument transformations?
2. **Performance**: What's the overhead of multi-argument helper functions?
3. **Complexity**: How complex can helper functions be before performance degrades?
4. **Testing**: How can we test these transformations effectively?
5. **Rollout**: What's the best strategy for rolling out semantic fields?

## Conclusion

The tag abstraction layer concept provides a significant improvement in API usability while maintaining the flexibility of the underlying tag system. The main challenge is implementing the multi-argument transformation logic in StepZen, but the benefits justify the effort. The configuration-driven approach makes it maintainable and extensible for future needs.
