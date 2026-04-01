# Schema Generation Tools

This directory contains tools to automatically generate GraphQL schema files from the live Cloudability API measures data, ensuring the schema stays in sync with backend capabilities.

## Problem Statement

The current schema has hardcoded enums and types that can drift out of sync with the actual Cloudability API. New dimensions, metrics, and fields are added to the backend but not reflected in our GraphQL schema, leading to:

- Missing functionality for new fields
- Maintenance overhead keeping enums updated
- Potential runtime errors from outdated mappings

## Solution

The `generate_schema.py` script fetches live measures data from the Cloudability API and generates up-to-date schema files automatically.

## Generated Files

The script generates the following files in this directory:

### Core Schema Files

- `generated_llm_enums.graphql` - ReportDimension and ReportMetric enums with current backend fields
- `generated_llm_inputs.graphql` - ReportFilter input types with all available filter fields
- `generated_llm_outputs.graphql` - ReportRow output type with all available result fields
- `generated_llm_helpers.graphql` - Helper functions with updated field mappings

### Reference Files

- `measures_data.json` - Raw measures data from the API (for debugging/reference)
- `field_analysis.txt` - Human-readable analysis of dimensions vs metrics

## Usage

### Prerequisites

- Python 3.7+ with `requests` library
- StepZen CLI configured and authenticated
- Access to the Cloudability API through your StepZen configuration

### Running the Generator

**Option 1: Full Update (Recommended)**

```bash
# From the reporting/build directory
./update_schema.sh
```

This convenience script will:

1. Generate new schema files
2. Show comparison with existing files
3. Provide integration instructions

**Option 2: Manual Steps**

```bash
# Generate schema files
python generate_schema.py

# Compare with existing (optional)
python compare_schemas.py
```

The generation process will:

1. Fetch current measures data using StepZen
2. Analyze dimensions vs metrics
3. Generate all schema files
4. Create reference documentation

### Integrating Generated Files

**⚠️ Important: Always review generated files before replacing existing schema!**

1. **Review the generated files** in this directory
2. **Compare with existing files** to understand what changed
3. **Test the changes** in a development environment first
4. **Manually copy/replace** the files in the main schema:

```bash
# Example integration (review first!)
cp generated_llm_enums.graphql ../llm_enums.graphql
cp generated_llm_inputs.graphql ../llm_inputs.graphql
cp generated_llm_outputs.graphql ../llm_outputs.graphql
cp generated_llm_helpers.graphql ../llm_helpers.graphql
```

### Validation

After integrating:

1. Run `stepzen build` to check for GraphQL syntax errors
2. Test key queries to ensure functionality works
3. Verify new fields are accessible in your reports

## Customization

The generator includes several customization options:

- **Complete field inclusion**: All API fields are included - users and LLMs can decide what's useful based on field names and descriptions
- **Naming conventions**: Adjust enum naming in `generate_enum_name()`
- **Documentation**: Customize field descriptions in `generate_field_docs()`

## Maintenance

Run this generator periodically (monthly/quarterly) or when:

- New Cloudability features are released
- You notice missing fields in reports
- Backend API capabilities change

## Troubleshooting

**"No measures data found"**: Check StepZen authentication and API connectivity
**"Generated files are empty"**: Verify the measures query returns valid data
**"GraphQL syntax errors"**: Review field names for invalid characters

### Comparing Changes

Before integrating, use the comparison tool to see what changed:

```bash
python compare_schemas.py
```

This will show you:

- New fields/enums that were added
- Old fields/enums that were removed
- Summary of changes for each file

## Files in this Directory

### Scripts

- `update_schema.sh` - Convenience script that runs generation + comparison
- `generate_schema.py` - Main generation script
- `compare_schemas.py` - Comparison tool to see what changed
- `README.md` - This documentation

### Generated Files (created by scripts)

- `generated_*.graphql` - Generated schema files
- `measures_data.json` - Raw API data (for debugging/reference)
- `field_analysis.txt` - Field analysis report
