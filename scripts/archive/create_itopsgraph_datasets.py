#!/usr/bin/env python3
"""
Create ITOpsGraph evaluation datasets from itopsgraph_mappings.json.

This script creates:
1. itopsgraph_utterance_dataset.jsonl - Natural language utterance-based evaluation
2. itopsgraph_attribute_dataset.jsonl - Attribute-based evaluation (like OMOP)

Usage:
    python scripts/create_itopsgraph_datasets.py --nl2cypher-dir /path/to/nl2cypher
"""

import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict


# Utterance templates for generating natural language descriptions
UTTERANCE_TEMPLATES = [
    "map {source_product} {source_type}.{source_field} to {target_product} {target_type}.{target_field}",
    "map {source_product} {source_field} with {target_product} {target_field}",
    "find {target_product} fields equivalent to {source_product} {source_type} {source_field}",
    "match {source_product} {source_type}.{source_field} to {target_product} schema",
    "link {source_product} {source_field} to {target_product} {target_field}",
    "connect {source_product} {source_type} {source_field} with {target_product} {target_type} {target_field}",
]


def load_mappings(filepath: Path) -> List[Dict[str, Any]]:
    """Load itopsgraph_mappings.json."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_api_lookup(filepath: Path) -> Dict[str, Dict[str, Any]]:
    """Load API lookup for enriching dataset."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_property_name(property_name: str, schema_name: str) -> Tuple[str, str, str]:
    """
    Parse property name into (product, type, field).

    Input formats:
        - "Type.field" (needs schema context)
        - "product_Type.field"
    """
    # Remove _schema suffix from schema name
    product = schema_name.replace('_schema', '')

    if '.' in property_name:
        parts = property_name.split('.', 1)
        type_name = parts[0]
        field_name = parts[1] if len(parts) > 1 else ''
    else:
        type_name = ''
        field_name = property_name

    return product, type_name, field_name


def extract_cross_product_mappings(
    mappings_data: List[Dict[str, Any]],
    min_confidence: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Extract cross-product mappings from the mappings data.

    Returns list of mapping records with source/target info.
    """
    cross_mappings = []

    for schema_entry in mappings_data:
        schema_name = schema_entry.get('name', '')
        properties = schema_entry.get('properties', {})

        source_product = schema_name.replace('_schema', '')

        for source_property, target_mappings in properties.items():
            source_product_name, source_type, source_field = parse_property_name(
                source_property, schema_name
            )

            for mapping in target_mappings:
                mapping_type = mapping.get('mapping_type', '')
                confidence = mapping.get('confidence', 0.0)

                # Skip identity mappings and low-confidence mappings
                if mapping_type == 'identity':
                    continue
                if confidence < min_confidence:
                    continue

                target_table = mapping.get('mapTable', '')
                target_property = mapping.get('mapProperty', '')
                target_product = target_table.replace('_schema', '')

                # Parse target property
                target_product_name, target_type, target_field = parse_property_name(
                    target_property, target_table
                )

                cross_mappings.append({
                    'source_product': source_product,
                    'source_type': source_type,
                    'source_field': source_field,
                    'source_property': source_property,
                    'target_product': target_product,
                    'target_type': target_type,
                    'target_field': target_field,
                    'target_property': target_property,
                    'confidence': confidence,
                    'justification': mapping.get('justification', ''),
                    'method': mapping.get('method', ''),
                    'equivalence_score': mapping.get('equivalence_score', 0.0),
                    'joinability_score': mapping.get('joinability_score', 0.0),
                })

    return cross_mappings


def generate_utterance(mapping: Dict[str, Any], template_idx: int = 0) -> str:
    """Generate a natural language utterance for a mapping."""
    template = UTTERANCE_TEMPLATES[template_idx % len(UTTERANCE_TEMPLATES)]

    return template.format(
        source_product=mapping['source_product'],
        source_type=mapping['source_type'],
        source_field=mapping['source_field'],
        target_product=mapping['target_product'],
        target_type=mapping['target_type'],
        target_field=mapping['target_field'],
    )


def create_utterance_dataset(
    cross_mappings: List[Dict[str, Any]],
    output_file: Path,
    api_lookup: Dict[str, Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Create utterance-based dataset.

    Each record has a natural language utterance describing the mapping task.
    """
    dataset = []

    for i, mapping in enumerate(cross_mappings):
        # Generate utterance using rotating templates
        utterance = generate_utterance(mapping, i)

        # Create mapping dict (source_field -> target_field)
        source_key = f"{mapping['source_product']}_{mapping['source_type']}.{mapping['source_field']}"
        target_key = f"{mapping['target_product']}_{mapping['target_type']}.{mapping['target_field']}"

        record = {
            'id': f"itopsgraph_utt_{i:04d}",
            'utterance': utterance,
            'source_schema': mapping['source_product'],
            'target_schema': mapping['target_product'],
            'mapping': {
                source_key: target_key
            },
            'metadata': {
                'confidence': mapping['confidence'],
                'justification': mapping['justification'],
                'method': mapping.get('method', ''),
                'equivalence_score': mapping.get('equivalence_score', 0.0),
                'joinability_score': mapping.get('joinability_score', 0.0),
                'source_type': mapping['source_type'],
                'target_type': mapping['target_type'],
            }
        }

        dataset.append(record)

    # Write to JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in dataset:
            f.write(json.dumps(record) + '\n')

    print(f"Created utterance dataset with {len(dataset)} samples: {output_file}")
    return dataset


def create_attribute_dataset(
    cross_mappings: List[Dict[str, Any]],
    output_file: Path,
    api_lookup: Dict[str, Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Create attribute-based dataset (similar to OMOP format).

    Each record uses the source attribute as the "utterance" field.
    """
    dataset = []

    # Group mappings by source-target product pair
    product_pairs = defaultdict(list)
    for mapping in cross_mappings:
        pair_key = (mapping['source_product'], mapping['target_product'])
        product_pairs[pair_key].append(mapping)

    record_idx = 0
    for (source_product, target_product), mappings in product_pairs.items():
        # Collect all source and target properties for this pair
        source_properties = []
        target_properties = []

        for mapping in mappings:
            source_key = f"{mapping['source_product']}_{mapping['source_type']}.{mapping['source_field']}"
            target_key = f"{mapping['target_product']}_{mapping['target_type']}.{mapping['target_field']}"

            # Get property info from API lookup if available
            source_info = api_lookup.get(source_key, {}) if api_lookup else {}
            target_info = api_lookup.get(target_key, {}) if api_lookup else {}

            source_properties.append({
                'name': source_key,
                'description': source_info.get('description', f"Field {source_key}")
            })

            target_properties.append({
                'name': target_key,
                'description': target_info.get('description', f"Field {target_key}")
            })

        # Create one record per mapping in this pair
        for mapping in mappings:
            source_key = f"{mapping['source_product']}_{mapping['source_type']}.{mapping['source_field']}"
            target_key = f"{mapping['target_product']}_{mapping['target_type']}.{mapping['target_field']}"

            record = {
                'id': f"itopsgraph_attr_{record_idx:04d}",
                'utterance': source_key,  # Attribute name as utterance
                'source_schema': {
                    'id': source_product,
                    'name': source_product,
                    'properties': source_properties
                },
                'target_schema': {
                    'id': target_product,
                    'name': target_product,
                    'properties': target_properties
                },
                'mapping': {
                    source_key: target_key
                },
                'metadata': {
                    'confidence': mapping['confidence'],
                    'justification': mapping['justification'],
                    'source_type': mapping['source_type'],
                    'target_type': mapping['target_type'],
                }
            }

            dataset.append(record)
            record_idx += 1

    # Write to JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in dataset:
            f.write(json.dumps(record) + '\n')

    print(f"Created attribute dataset with {len(dataset)} samples: {output_file}")
    return dataset


def analyze_mappings(cross_mappings: List[Dict[str, Any]]):
    """Print analysis of the mapping data."""
    print("\n" + "=" * 60)
    print("Mapping Analysis")
    print("=" * 60)

    # Count by product pairs
    product_pairs = defaultdict(int)
    for mapping in cross_mappings:
        pair = (mapping['source_product'], mapping['target_product'])
        product_pairs[pair] += 1

    print(f"\nTotal cross-product mappings: {len(cross_mappings)}")
    print(f"\nProduct pair distribution:")
    for (src, tgt), count in sorted(product_pairs.items(), key=lambda x: -x[1]):
        print(f"  {src} -> {tgt}: {count}")

    # Confidence distribution
    confidences = [m['confidence'] for m in cross_mappings]
    if confidences:
        print(f"\nConfidence distribution:")
        print(f"  Min: {min(confidences):.2f}")
        print(f"  Max: {max(confidences):.2f}")
        print(f"  Mean: {sum(confidences)/len(confidences):.2f}")

    # Unique products
    source_products = set(m['source_product'] for m in cross_mappings)
    target_products = set(m['target_product'] for m in cross_mappings)
    print(f"\nUnique source products: {len(source_products)}")
    print(f"Unique target products: {len(target_products)}")
    print(f"All products: {sorted(source_products | target_products)}")


def main():
    parser = argparse.ArgumentParser(
        description='Create ITOpsGraph evaluation datasets from mappings'
    )
    parser.add_argument(
        '--nl2cypher-dir',
        type=str,
        default='/Users/balaji/code/solis-dal-dataset/nl2cypher',
        help='Path to nl2cypher repository'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Output directory for generated files'
    )
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.0,
        help='Minimum confidence threshold for including mappings'
    )

    args = parser.parse_args()

    nl2cypher_dir = Path(args.nl2cypher_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load mappings
    mappings_file = nl2cypher_dir / 'data' / 'itopsgraph_mappings.json'
    print(f"Loading mappings from: {mappings_file}")
    mappings_data = load_mappings(mappings_file)

    # Load API lookup if available
    api_lookup_file = output_dir / 'itopsgraph_api_lookup.json'
    api_lookup = None
    if api_lookup_file.exists():
        print(f"Loading API lookup from: {api_lookup_file}")
        api_lookup = load_api_lookup(api_lookup_file)

    # Extract cross-product mappings
    print(f"\nExtracting cross-product mappings (min_confidence={args.min_confidence})...")
    cross_mappings = extract_cross_product_mappings(
        mappings_data,
        min_confidence=args.min_confidence
    )

    # Analyze mappings
    analyze_mappings(cross_mappings)

    # Create utterance dataset
    print("\n" + "=" * 60)
    print("Phase 2.1: Creating utterance-based dataset")
    print("=" * 60)
    utterance_file = output_dir / 'itopsgraph_utterance_dataset.jsonl'
    create_utterance_dataset(cross_mappings, utterance_file, api_lookup)

    # Create attribute dataset
    print("\n" + "=" * 60)
    print("Phase 2.2: Creating attribute-based dataset")
    print("=" * 60)
    attribute_file = output_dir / 'itopsgraph_attribute_dataset.jsonl'
    create_attribute_dataset(cross_mappings, attribute_file, api_lookup)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Files created:")
    print(f"  - {utterance_file}")
    print(f"  - {attribute_file}")


if __name__ == '__main__':
    main()
