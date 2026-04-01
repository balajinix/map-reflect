# Migration Plan: Move `dateRange` into Filter Input for `llmFriendlyCostReport`

This document outlines the steps required to move the `dateRange` argument from a top-level argument to a field within the `match` (filter) input object for the `llmFriendlyCostReport` query.

---

## 1. Update Main Query Definition (`llm_report_query.graphql`)

- Remove the top-level `dateRange` argument from the `llmFriendlyCostReport` query.
- Ensure the `match` argument (of type `ReportFilter`) is required if date range is required.
- Update documentation/comments to reflect the new usage (date range is now part of the filter).
- Example change:

  ```graphql
  # Before:
  llmFriendlyCostReport(
    dateRange: DateRangeInput!
    ...
    match: ReportFilter
    ...
  )

  # After:
  llmFriendlyCostReport(
    ...
    match: ReportFilter!
    ...
  )
  ```

---

## 2. Update Input Types (`llm_inputs.graphql` and Generator)

- Add a `dateRange: DateRangeInput` field to the `ReportFilter` input type.
- Ensure `DateRangeInput` is still defined and referenced correctly.
- Update the schema generator (`generate_schema.py`) to include this field in the generated `ReportFilter`.
- Example addition:
  ```graphql
  input ReportFilter {
    ...
    dateRange: DateRangeInput
    ...
  }
  ```

---

## 3. Update Helpers and Backend Mapping (`llm_helpers.graphql` and Generator)

- Update the `_convertFilter` helper to extract and process `dateRange` from the filter object.
- Update the backend argument preparation logic to use `filter.dateRange.start` and `filter.dateRange.end` instead of top-level arguments.
- Update the schema generator to reflect these changes in the generated helpers.

---

## 4. Update Documentation and Examples

- Update any documentation, usage examples, and inline comments to show how to specify `dateRange` inside the filter object.
- Example:
  ```graphql
  match: {
    dateRange: { start: "2024-01-01", end: "2024-01-31" },
    ...
  }
  ```

---

## 5. Update Test/Reference Files

- Update any test queries, sample operations, or reference files to use the new filter structure.

---

## 6. Validation

- Run `stepzen build` and test key queries to ensure the new structure works as expected.
- Verify that the backend receives the correct date range values from within the filter object.

---

## Notes

- Ensure all places that reference `dateRange` as a top-level argument are updated.
- Consider making `match` required if `dateRange` is required for all queries.
- Communicate this change to any consumers of the API, as it is a breaking change.
