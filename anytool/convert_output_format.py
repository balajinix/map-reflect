#!/usr/bin/env python3
"""
Convert AnyTool output format to be compatible with evaluation scripts.

AnyTool uses human-readable field names in predicted_mapping:
  {"CMS Beneficiary Summary": {"target_property": "beneficiary_summary", ...}}

Evaluation expects schema field names:
  {"omop_PERSON.person_id": "cms_BENEFICIARYSUMMARY.desynpuf_id"}

This script converts by matching human-readable names to actual schema field names
from expected_source and expected_target properties.
"""

import json
import re
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse


def normalize_name(name: str) -> str:
    """Normalize a name for fuzzy matching."""
    # Convert to lowercase, remove special chars, normalize spaces
    name = str(name).lower()
    name = re.sub(r'[^a-z0-9\s]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def extract_field_name(prop_name: str) -> str:
    """Extract just the field name from schema.field format."""
    if '.' in prop_name:
        return prop_name.split('.')[-1].lower()
    return prop_name.lower()


def build_field_lookup(schema_info: Dict) -> Dict[str, str]:
    """Build a lookup from normalized names to actual field names."""
    lookup = {}

    if not isinstance(schema_info, dict):
        return lookup

    schema_id = schema_info.get('id', '')
    schema_name = schema_info.get('name', '')

    # Add schema-level entries
    if schema_id:
        lookup[normalize_name(schema_id)] = schema_id
        lookup[normalize_name(schema_name)] = schema_id

    # Add field-level entries
    properties = schema_info.get('properties', [])
    for prop in properties:
        if isinstance(prop, dict):
            prop_name = prop.get('name', '')
            prop_desc = prop.get('description', '')

            if prop_name:
                # Full name
                lookup[normalize_name(prop_name)] = prop_name

                # Field name only
                field_only = extract_field_name(prop_name)
                lookup[field_only] = prop_name

                # Description-based (first part before semicolon)
                if prop_desc and ';' in prop_desc:
                    desc_field = prop_desc.split(';')[-1].strip()
                    if desc_field:
                        lookup[normalize_name(desc_field)] = prop_name

    return lookup


# Minimum token length to use in substring matching.  Short tokens like "id",
# "at", "on" would match almost everything and produce spurious mappings.
_MIN_FUZZY_TOKEN_LEN = 6


def find_best_match(human_name: str, lookup: Dict[str, str], schema_id: str = '') -> Optional[str]:
    """Find the best matching schema field name for a human-readable name.

    Matching is attempted in order from most to least precise:
      1. Direct normalised-name match (exact, safe)
      2. Individual word match with a minimum length guard (avoids short tokens
         like "id" matching every field that contains the letters "id")
      3. Substring match with the same length guard on both the key and the
         query token (prevents "id" in "person_id" matching "administration")
      4. No synthetic placeholder is generated - returning None is safer than
         inventing a field name that will silently inflate false-positive counts.
    """
    normalized = normalize_name(human_name)

    # 1. Direct match
    if normalized in lookup:
        return lookup[normalized]

    # 2. Individual word match - only for tokens long enough to be distinctive
    words = normalized.split()
    for word in words:
        if len(word) >= _MIN_FUZZY_TOKEN_LEN and word in lookup:
            return lookup[word]

    # 3. Substring match - both the lookup key and the query token must be long
    #    enough that a containment relationship is meaningful
    for key, value in lookup.items():
        if len(key) < _MIN_FUZZY_TOKEN_LEN:
            continue
        if key in normalized and len(key) >= _MIN_FUZZY_TOKEN_LEN:
            return value
        if normalized in key and len(normalized) >= _MIN_FUZZY_TOKEN_LEN:
            return value

    # 4. No match found - do not fabricate a field name
    return None


def convert_predicted_mapping(
    pred_mapping: Dict,
    source_schema: Dict,
    target_schema: Dict
) -> Dict[str, str]:
    """Convert AnyTool predicted_mapping to evaluation-compatible format."""

    source_lookup = build_field_lookup(source_schema)
    target_lookup = build_field_lookup(target_schema)

    source_id = source_schema.get('id', '') if isinstance(source_schema, dict) else ''
    target_id = target_schema.get('id', '') if isinstance(target_schema, dict) else ''

    converted = {}

    for src_human, tgt_info in pred_mapping.items():
        # Get target field
        if isinstance(tgt_info, dict):
            tgt_human = tgt_info.get('target_property', str(tgt_info))
        else:
            tgt_human = str(tgt_info)

        # Skip unknown/failed mappings
        if 'unknown' in str(src_human).lower() and 'unknown' in str(tgt_human).lower():
            continue

        # Find matching schema field names
        src_field = find_best_match(src_human, source_lookup, source_id)
        tgt_field = find_best_match(tgt_human, target_lookup, target_id)

        if src_field and tgt_field:
            converted[src_field] = tgt_field

    return converted


def convert_result(result: Dict) -> Dict:
    """Convert a single result entry."""
    converted = result.copy()

    pred_mapping = result.get('predicted_mapping', {})
    source_schema = result.get('expected_source', {})
    target_schema = result.get('expected_target', {})

    if pred_mapping and source_schema and target_schema:
        converted['predicted_mapping'] = convert_predicted_mapping(
            pred_mapping, source_schema, target_schema
        )

        # Also add source_schema and target_schema fields for API matching
        if isinstance(source_schema, dict):
            converted['source_schema'] = source_schema.get('id', source_schema.get('name', ''))
        if isinstance(target_schema, dict):
            converted['target_schema'] = target_schema.get('id', target_schema.get('name', ''))

    return converted


def convert_file(input_path: Path, output_path: Path = None) -> Dict:
    """Convert an AnyTool result file."""
    with open(input_path) as f:
        data = json.load(f)

    if 'results' in data:
        converted_results = [convert_result(r) for r in data['results']]
        data['results'] = converted_results
    elif isinstance(data, list):
        data = [convert_result(r) for r in data]

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Converted: {input_path} -> {output_path}")

    return data


def main():
    parser = argparse.ArgumentParser(description='Convert AnyTool output format')
    parser.add_argument('--input', '-i', type=str, help='Input file or directory')
    parser.add_argument('--output', '-o', type=str, help='Output file or directory')
    parser.add_argument('--inplace', action='store_true', help='Modify files in place')
    parser.add_argument('--backup', action='store_true', help='Create backup before modifying')
    args = parser.parse_args()

    if not args.input:
        # Default: convert all files in results directory
        results_dir = Path(__file__).parent / 'results'
        files = list(results_dir.glob('*.json'))
    elif Path(args.input).is_dir():
        files = list(Path(args.input).glob('*.json'))
    else:
        files = [Path(args.input)]

    for input_file in files:
        if 'checkpoint' in input_file.name:
            continue

        if args.inplace:
            if args.backup:
                backup_path = input_file.with_suffix('.json.bak')
                import shutil
                shutil.copy(input_file, backup_path)
            output_file = input_file
        elif args.output:
            if Path(args.output).is_dir():
                output_file = Path(args.output) / input_file.name
            else:
                output_file = Path(args.output)
        else:
            output_file = input_file.with_name(input_file.stem + '_converted.json')

        try:
            convert_file(input_file, output_file)
        except Exception as e:
            print(f"Error converting {input_file}: {e}")


if __name__ == '__main__':
    main()
