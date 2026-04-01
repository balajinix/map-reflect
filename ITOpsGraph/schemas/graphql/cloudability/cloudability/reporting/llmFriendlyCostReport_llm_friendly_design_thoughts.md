# Designing an LLM-Friendly GraphQL Cost Report Query

## 1. Current Interface: Challenges for LLMs

- **Mix of Enums and Strings:**
  - `dimensions` and `metrics` use enums, but `sort.field` is a string, and filter fields are camelCase strings.
  - LLMs may confuse which fields are valid where, or use the wrong format.
- **Field Name Ambiguity:**
  - Field names in enums, filters, and result types are similar but not always identical (e.g., camelCase vs. UPPER_SNAKE_CASE).
  - LLMs may generate invalid field names or mix up formats.
- **Argument Placement:**
  - Some arguments (like `dateRange`) are top-level, others are nested in filter objects.
  - This inconsistency can confuse LLMs about where to place certain fields.
- **Documentation Overload:**
  - Heavy reliance on docstrings to explain mapping and requirements, which LLMs may not always parse or follow correctly.
- **Loose Typing in Some Inputs:**
  - `sort.field` is a string, not an enum, so LLMs may invent or misspell field names.

## 2. Principles for LLM-Optimized GraphQL Design

- **Consistency:** Use the same format (preferably enums) for all field selectors (dimensions, metrics, sort, filter keys).
- **Explicitness:** Minimize optional/implicit behavior; make all required fields explicit and clearly typed.
- **Flat Structure:** Prefer fewer levels of nesting unless conceptually necessary.
- **Self-Describing Types:** Use enums and input types that are discoverable and introspectable.
- **Single Source of Truth:** Field names should be defined in one place and reused everywhere.
- **Clear Error Feedback:** Schema should provide clear errors for invalid field names or combinations.

## 3. Proposed LLM-Friendly Query Interface

```graphql
# Enums for all selectable fields
enum CostReportField {
  VENDOR
  ACCOUNT_ID
  # ... all valid fields, one enum for all selectors
}

enum CostReportMetric {
  COST
  USAGE_HOURS
  # ...
}

enum SortDirection {
  ASC
  DESC
}

input CostReportSort {
  field: CostReportField!
  direction: SortDirection = ASC
}

input CostReportFilter {
  field: CostReportField!
  operator: FilterOperator! # e.g., EQ, GT, IN, etc.
  value: String # or a union/scalar for type safety
}

input CostReportRequest {
  dateRange: DateRangeInput!
  select: [CostReportField!]! # dimensions
  metrics: [CostReportMetric!]!
  filter: [CostReportFilter!]
  sort: [CostReportSort!]
  limit: Int
}

type Query {
  llmFriendlyCostReport(request: CostReportRequest!): CostReportResult
}
```

### Key Features

- **All selectors use enums:** No ambiguity about valid field names.
- **Single input object:** All arguments are grouped, reducing confusion about argument placement.
- **Explicit filter structure:** Each filter specifies the field, operator, and value.
- **Consistent sort input:** Sort fields use the same enum as selectors.
- **Introspectable and discoverable:** LLMs can query the schema for valid fields and operators.

## 4. Mapping to Backend API

- **Field Mapping:**
  - Enum values map directly to backend field names via a lookup table.
  - Backend adapters can translate enums to the required API field names.
- **Filter/Sort Translation:**
  - Each filter and sort input is converted to the backend's expected format (e.g., CSV, JSON).
- **Validation:**
  - Backend or middleware can validate that only selected fields are used in sort/filter.
- **Extensibility:**
  - Adding new fields or metrics is a matter of updating the enums and mapping logic.

## 5. Expert Commentary

### Generative AI & Tool-Calling Perspective

- **Enum-Driven Design:** LLMs are much less likely to hallucinate or invent field names when enums are used everywhere.
- **Single Input Object:** Reduces the chance of argument misplacement and makes prompt generation more reliable.
- **Explicitness:** LLMs perform better when all requirements are explicit and discoverable via introspection.
- **Error Reduction:** Strong typing and enums reduce the risk of invalid queries.

### GraphQL Schema Design Perspective

- **Discoverability:** Enums and input types make the schema self-documenting and easy to explore.
- **Consistency:** Using the same enum for selectors, filters, and sorts ensures a single source of truth.
- **Maintainability:** Schema changes (new fields, deprecations) are easier to manage and communicate.
- **Validation:** GraphQL can provide immediate feedback on invalid field/operator combinations.

## 6. Conclusion

A truly LLM-friendly GraphQL interface for cost reporting should:

- Use enums for all field selectors.
- Group all arguments in a single input object.
- Be explicit and consistent in structure and naming.
- Provide clear, introspectable types for all inputs.

This approach will maximize LLM accuracy, reduce hallucination, and make the API easier to use for both humans and generative models.
