#!/usr/bin/env python3
"""
Schema Generation Script for Cloudability GraphQL API

This script fetches live measures data from the Cloudability API via StepZen
and generates up-to-date GraphQL schema files.
"""

import json
import subprocess
import sys
import re
from typing import Dict, List, Set, Tuple, Any
from pathlib import Path


def run_stepzen_query() -> Dict[str, Any]:
    """Fetch measures data using StepZen CLI."""
    print("Fetching measures data from Cloudability API...")

    try:
        result = subprocess.run(
            ["stepzen", "request", "-f", "../operations/measures.graphql"],
            capture_output=True,
            text=True,
            check=True,
        )

        # Parse JSON response
        data = json.loads(result.stdout)
        return data

    except subprocess.CalledProcessError as e:
        print(f"Error running StepZen command: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw output: {result.stdout}")
        sys.exit(1)


def should_include_field(measure: Dict[str, Any]) -> bool:
    """Include all fields - let users and LLMs decide what's useful based on the field names and descriptions."""
    return True


def camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case."""
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase."""
    components = name.split("_")
    return components[0] + "".join(word.capitalize() for word in components[1:])


def generate_enum_name(name: str) -> str:
    """Generate GraphQL enum value from backend field name."""
    return name  # Use snake_case instead of UPPER_SNAKE_CASE for consistency


def escape_description(desc: str) -> str:
    """Escape description for GraphQL comments."""
    if not desc:
        return ""
    # Replace newlines and clean up whitespace
    cleaned = re.sub(r"\s+", " ", desc.strip())
    # Escape quotes
    cleaned = cleaned.replace('"', '\\"')
    return cleaned


def generate_enums(dimensions: List[Dict], metrics: List[Dict]) -> str:
    """Generate the ReportDimension and ReportMetric enums."""

    content = [
        "# generated_llm_enums.graphql",
        "",
        "# Auto-generated from live Cloudability API measures data",
        "# DO NOT EDIT MANUALLY - Use generate_schema.py to regenerate",
        "",
    ]

    # Generate ReportDimension enum
    content.extend(
        [
            '"""',
            "Identifies dimensions available for reporting and filtering.",
            '"""',
            "enum ReportDimension {",
        ]
    )

    for dim in sorted(dimensions, key=lambda x: x["name"]):
        if should_include_field(dim):
            enum_name = generate_enum_name(dim["name"])
            label = dim.get("label", dim["name"])
            desc = escape_description(dim.get("description", ""))

            if desc:
                content.append(f'  """')
                content.append(f"  {label} - {desc}")
                content.append(f'  """')
            else:
                content.append(f'  """')
                content.append(f"  {label}")
                content.append(f'  """')
            content.append(f"  {enum_name}")

    content.extend(["}", ""])

    # Generate ReportMetric enum
    content.extend(
        [
            '"""',
            "Identifies metrics available for reporting.",
            '"""',
            "enum ReportMetric {",
        ]
    )

    for metric in sorted(metrics, key=lambda x: x["name"]):
        if should_include_field(metric):
            enum_name = generate_enum_name(metric["name"])
            label = metric.get("label", metric["name"])
            desc = escape_description(metric.get("description", ""))

            if desc:
                content.append(f'  """')
                content.append(f"  {label} - {desc}")
                content.append(f'  """')
            else:
                content.append(f'  """')
                content.append(f"  {label}")
                content.append(f'  """')
            content.append(f"  {enum_name}")

    content.extend(["}", ""])

    return "\n".join(content)


def get_filter_type(data_type: str) -> str:
    """Determine the appropriate filter input type based on data type."""
    if data_type in ["integer", "int"]:
        return "IntFilterInput"
    elif data_type in ["float", "double", "currency"]:
        return "FloatFilterInput"
    else:
        return "StringFilterInput"


def generate_inputs(dimensions: List[Dict], metrics: List[Dict]) -> str:
    """Generate the input types for filtering."""

    content = [
        "# generated_llm_inputs.graphql",
        "",
        "# Auto-generated from live Cloudability API measures data",
        "# DO NOT EDIT MANUALLY - Use generate_schema.py to regenerate",
        "",
        '"""',
        "Input for specifying a date range.",
        '"""',
        "input DateRangeInput {",
        '  """',
        "  Start date in YYYY-MM-DD format.",
        '  """',
        "  start: String!",
        '  """',
        "  End date in YYYY-MM-DD format.",
        '  """',
        "  end: String!",
        "}",
        "",
        "input StringFilterInput {",
        "  eq: String",
        "  ne: String",
        "  strictEq: String",
        "  strictNe: String",
        "  in: [String!]",
        "  nin: [String!]",
        "  like: String",
        "  notLike: String",
        "}",
        "",
        "input FloatFilterInput {",
        "  eq: Float",
        "  ne: Float",
        "  in: [Float!]",
        "  nin: [Float!]",
        "  gt: Float",
        "  gte: Float",
        "  lt: Float",
        "  le: Float",
        "}",
        "",
        "input IntFilterInput {",
        "  eq: Int",
        "  ne: Int",
        "  in: [Int!]",
        "  nin: [Int!]",
        "  gt: Int",
        "  gte: Int",
        "  lt: Int",
        "  le: Int",
        "}",
        "",
        '"""',
        "Sort directions for report sorting.",
        '"""',
        "enum SortDirection {",
        '  """',
        "  Sort in ascending order. Default.",
        '  """',
        "  ASC",
        '  """',
        "  Sort in descending order.",
        '  """',
        "  DESC",
        "}",
        "",
        '"""',
        "Input for specifying sorting criteria for the cost report.",
        '"""',
        "input ReportSortInput {",
        '  """',
        "  The field to sort by. Must be included in the 'dimensions'/'metrics' arguments for the report.",
        '  """',
        "  field: String!",
        '  """',
        "  The sort direction. Default is ASC.",
        '  """',
        "  direction: SortDirection",
        "}",
        "",
        '"""',
        "Main input for filtering the cost report.",
        '"""',
        "input ReportFilter {",
    ]

    # Add dimension filters
    all_fields = dimensions + metrics
    for field in sorted(all_fields, key=lambda x: x["name"]):
        if should_include_field(field):
            field_name = field[
                "name"
            ]  # Use snake_case directly instead of converting to camelCase
            filter_type = get_filter_type(field.get("data_type", "string"))
            label = field.get("label", field["name"])
            backend_name = field["name"]

            content.append(f'  """')
            content.append(f"  Filter by: {label}")
            content.append(f'  """')
            content.append(f"  {field_name}: {filter_type}")

    content.extend(["}", ""])

    return "\n".join(content)


def get_graphql_type(data_type: str) -> str:
    """Map backend data type to GraphQL type."""
    type_mapping = {
        "string": "String",
        "integer": "Int",
        "int": "Int",
        "float": "Float",
        "double": "Float",
        "currency": "Float",
        "boolean": "Boolean",
        "date": "String",
        "datetime": "String",
    }
    return type_mapping.get(data_type, "String")


def generate_outputs(dimensions: List[Dict], metrics: List[Dict]) -> str:
    """Generate the output types for report results."""

    content = [
        "# generated_llm_outputs.graphql",
        "",
        "# Auto-generated from live Cloudability API measures data",
        "# DO NOT EDIT MANUALLY - Use generate_schema.py to regenerate",
        "",
        '"""',
        "Represents a single row in the cost report.",
        '"""',
        "type ReportRow {",
    ]

    # Add all fields (dimensions and metrics)
    all_fields = dimensions + metrics
    for field in sorted(all_fields, key=lambda x: x["name"]):
        if should_include_field(field):
            field_name = field["name"]
            graphql_type = get_graphql_type(field.get("data_type", "string"))
            label = field.get("label", field["name"])
            field_type = field.get("type", "unknown")

            content.append(f'  """({field_type.title()}) {label}"""')
            content.append(f"  {field_name}: {graphql_type}")

    content.extend(["}", ""])

    # Add the main report response type
    content.extend(
        [
            '"""',
            "Response structure for cost reports.",
            '"""',
            "type LLMCostReport {",
            '  """',
            "  Total number of results available (before limit/offset).",
            '  """',
            "  total_results: Int",
            '  """',
            "  Number of results returned in this response.",
            '  """',
            "  limit: Int",
            '  """',
            "  Starting position of results (0-indexed).",
            '  """',
            "  offset: Int",
            '  """',
            "  Array of report data rows.",
            '  """',
            "  results: [ReportRow]",
            '  """',
            "  Metadata about the report request and available fields.",
            '  """',
            "  meta: Meta",
            "}",
            "",
        ]
    )

    return "\n".join(content)


def generate_helpers(dimensions: List[Dict], metrics: List[Dict]) -> str:
    """Generate the helper functions with field mappings."""

    content = [
        "# generated_llm_helpers.graphql",
        "",
        "# Auto-generated from live Cloudability API measures data",
        "# DO NOT EDIT MANUALLY - Use generate_schema.py to regenerate",
        "",
        "type DimensionsCsvOutput {",
        "  dimensions: String",
        "}",
        "",
        "type MetricsCsvOutput {",
        "  metrics: String",
        "}",
        "",
        "type FiltersArrayOutput {",
        "  filters: [String]",
        "}",
        "",
        "type SortCsvOutput {",
        "  sort: String",
        "}",
        "",
        "type _Backend_CostReportArguments {",
        "  dimensions: String",
        "  metrics: String",
        "  filters: [String]",
        "  sort: String",
        "}",
        "",
        "extend type Query {",
        "  _convertDimensionListToCsv(list: [ReportDimension!]!): DimensionsCsvOutput",
        "    @value(",
        "      script: {",
        "        language: ECMASCRIPT,",
        '        src: """',
        "          var dimensionEnumList = list;",
        "          var apiDimensionNames = [];",
        "          if (dimensionEnumList && typeof dimensionEnumList.length === 'number') {",
        "            for (var i = 0; i < dimensionEnumList.length; i++) {",
        "              // Enum values are already snake_case, use directly",
        "              apiDimensionNames.push(dimensionEnumList[i]);",
        "            }",
        "          }",
        "          ({ dimensions: apiDimensionNames.join(',') });",
        '        """',
        "      }",
        "    )",
        "",
        "  _convertMetricListToCsv(list: [ReportMetric!]!): MetricsCsvOutput",
        "    @value(",
        "      script: {",
        "        language: ECMASCRIPT,",
        '        src: """',
        "          var metricEnumList = list;",
        "          var apiMetricNames = [];",
        "          if (metricEnumList && typeof metricEnumList.length === 'number') {",
        "            for (var i = 0; i < metricEnumList.length; i++) {",
        "              // Enum values are already snake_case, use directly",
        "              apiMetricNames.push(metricEnumList[i]);",
        "            }",
        "          }",
        "          ({ metrics: apiMetricNames.join(',') });",
        '        """',
        "      }",
        "    )",
        "",
        "  _convertFilter(filter: ReportFilter!={}): FiltersArrayOutput",
        "    @value(",
        "      script: {",
        "        language: ECMASCRIPT,",
        '        src: """',
        "          var inputFilterObject = filter;",
        "          var operatorMap = {",
        '            "eq": "==", "ne": "!=", "strictEq": "===", "strictNe": "!==",',
        '            "in": "[]=", "nin": "[]!=", "like": "=@", "notLike": "!=@",',
        '            "gt": ">", "gte": ">=", "lt": "<", "le": "<="',
        "          };",
        "          var filters = [];",
        "          ",
        "          // Process each field in the filter object",
        "          for (var fieldName in inputFilterObject) {",
        "            if (inputFilterObject.hasOwnProperty(fieldName)) {",
        "              var filterValue = inputFilterObject[fieldName];",
        "              // Field names are already snake_case, use directly",
        "              var apiFieldName = fieldName;",
        "              ",
        "              if (apiFieldName && filterValue) {",
        "                // Process each operator in the filter value",
        "                for (var operator in filterValue) {",
        "                  if (filterValue.hasOwnProperty(operator)) {",
        "                    var value = filterValue[operator];",
        "                    var apiOperator = operatorMap[operator];",
        "                    ",
        "                    if (apiOperator && value !== null && value !== undefined) {",
        "                      if (Array.isArray(value)) {",
        "                        // Handle array values (for 'in' and 'nin' operators)",
        "                        var valueStr = value.join(',');",
        "                        filters.push(apiFieldName + apiOperator + valueStr);",
        "                      } else {",
        "                        // Handle single values",
        "                        filters.push(apiFieldName + apiOperator + String(value));",
        "                      }",
        "                    }",
        "                  }",
        "                }",
        "              }",
        "            }",
        "          }",
        "          ",
        "          ({ filters: filters });",
        '        """',
        "      }",
        "    )",
        "",
        "  _convertSortListToCsv(list: [ReportSortInput!]): SortCsvOutput",
        "    @value(",
        "      script: {",
        "        language: ECMASCRIPT,",
        '        src: """',
        "          var sortList = list;",
        "          var sortParts = [];",
        "          if (sortList && typeof sortList.length === 'number') {",
        "            for (var i = 0; i < sortList.length; i++) {",
        "              var item = sortList[i];",
        "              if (item && item.field) {",
        "                // Field names are already snake_case, use directly",
        "                var backendFieldName = item.field;",
        "                // Backend API requires explicit ASC or DESC - default to ASC if not specified",
        "                var direction = (item.direction && String(item.direction).toUpperCase() === 'DESC') ? 'DESC' : 'ASC';",
        "                sortParts.push(backendFieldName + direction);",
        "              }",
        "            }",
        "          }",
        "          ({ sort: sortParts.join(',') });",
        '        """',
        "      }",
        "    )",
        "}",
        "",
    ]

    return "\n".join(content)


def generate_analysis(dimensions: List[Dict], metrics: List[Dict]) -> str:
    """Generate human-readable analysis of the fields."""

    lines = [
        "Field Analysis Report",
        "=" * 50,
        "",
        f"Generated on: {__import__('datetime').datetime.now().isoformat()}",
        "",
        f"Total Dimensions: {len([d for d in dimensions if should_include_field(d)])}",
        f"Total Metrics: {len([m for m in metrics if should_include_field(m)])}",
        "",
        "DIMENSIONS",
        "-" * 20,
        "",
    ]

    for dim in sorted(dimensions, key=lambda x: x["name"]):
        if should_include_field(dim):
            lines.append(f"• {dim['name']} ({dim.get('data_type', 'string')})")
            lines.append(f"  Label: {dim.get('label', 'N/A')}")
            if dim.get("description"):
                lines.append(f"  Description: {dim['description'][:100]}...")
            lines.append("")

    lines.extend(["", "METRICS", "-" * 20, ""])

    for metric in sorted(metrics, key=lambda x: x["name"]):
        if should_include_field(metric):
            lines.append(f"• {metric['name']} ({metric.get('data_type', 'string')})")
            lines.append(f"  Label: {metric.get('label', 'N/A')}")
            if metric.get("description"):
                lines.append(f"  Description: {metric['description'][:100]}...")
            lines.append("")

    return "\n".join(lines)


def main():
    """Main execution function."""
    print("🚀 Starting Cloudability GraphQL Schema Generation")
    print("=" * 60)

    # Fetch data from API
    data = run_stepzen_query()

    # Save raw data for reference
    with open("measures_data.json", "w") as f:
        json.dump(data, f, indent=2)
    print("✅ Saved raw measures data to measures_data.json")

    # Extract measures
    measures = data.get("data", {}).get("measures", [])
    if not measures:
        print("❌ No measures data found in API response")
        sys.exit(1)

    # Separate dimensions and metrics
    dimensions = [m for m in measures if m.get("type") == "dimension"]
    metrics = [m for m in measures if m.get("type") == "metric"]

    print(f"📊 Found {len(dimensions)} dimensions and {len(metrics)} metrics")

    # Filter out test/internal fields
    filtered_dimensions = [d for d in dimensions if should_include_field(d)]
    filtered_metrics = [m for m in metrics if should_include_field(m)]

    print(
        f"🔍 After filtering: {len(filtered_dimensions)} dimensions and {len(filtered_metrics)} metrics"
    )

    # Generate schema files
    print("\n📝 Generating schema files...")

    # Generate enums
    enums_content = generate_enums(filtered_dimensions, filtered_metrics)
    with open("generated_llm_enums.graphql", "w") as f:
        f.write(enums_content)
    print("✅ Generated generated_llm_enums.graphql")

    # Generate inputs
    inputs_content = generate_inputs(filtered_dimensions, filtered_metrics)
    with open("generated_llm_inputs.graphql", "w") as f:
        f.write(inputs_content)
    print("✅ Generated generated_llm_inputs.graphql")

    # Generate outputs
    outputs_content = generate_outputs(filtered_dimensions, filtered_metrics)
    with open("generated_llm_outputs.graphql", "w") as f:
        f.write(outputs_content)
    print("✅ Generated generated_llm_outputs.graphql")

    # Generate helpers
    helpers_content = generate_helpers(filtered_dimensions, filtered_metrics)
    with open("generated_llm_helpers.graphql", "w") as f:
        f.write(helpers_content)
    print("✅ Generated generated_llm_helpers.graphql")

    # Generate analysis
    analysis_content = generate_analysis(filtered_dimensions, filtered_metrics)
    with open("field_analysis.txt", "w") as f:
        f.write(analysis_content)
    print("✅ Generated field_analysis.txt")

    print("\n🎉 Schema generation completed successfully!")
    print("\n📋 Next steps:")
    print("1. Review the generated files in this directory")
    print("2. Compare with existing schema files")
    print("3. Test in development environment")
    print("4. Manually copy files to replace existing schema")
    print("\nSee README.md for detailed integration instructions.")


if __name__ == "__main__":
    main()
