#!/usr/bin/env python3
"""
Generate field-level validation dataset for ITOpsGraph coverage expansion.

This script creates samples for UE-Reflect to validate type mappings
by checking if fields can be mapped between source and target types.

Input: data/itopsgraph_coverage_candidates.json
Output: data/itopsgraph_coverage_dataset.jsonl
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'


def load_candidates(candidates_file: Path) -> Dict:
    """Load the candidate type pairs."""
    with open(candidates_file, 'r') as f:
        return json.load(f)


def load_api_lookup(lookup_file: Path) -> Dict:
    """Load the API lookup file."""
    with open(lookup_file, 'r') as f:
        return json.load(f)


def get_type_fields(api_lookup: Dict, product: str, type_name: str) -> List[Dict]:
    """
    Get all fields for a given type from the API lookup.
    Returns list of field dicts with name, description, type info.
    """
    fields = []
    # type_name is already in format "product_TypeName"
    type_prefix = f"{type_name}."

    for key, value in api_lookup.items():
        if not isinstance(value, dict):
            continue

        full_name = value.get('full_name', '')
        if full_name.startswith(type_prefix) or key.startswith(type_prefix):
            fields.append({
                'name': value.get('field_name', ''),
                'full_name': full_name or key,
                'description': value.get('description', ''),
                'field_type': value.get('field_type', ''),
                'is_list': value.get('is_list', False),
                'is_required': value.get('is_required', False)
            })

    return fields


def create_sample(
    sample_id: str,
    source_type: str,
    source_product: str,
    source_fields: List[Dict],
    target_type: str,
    target_product: str,
    target_fields: List[Dict],
    similarity: float
) -> Dict:
    """
    Create a dataset sample for field-level validation.

    The sample format is compatible with the existing UE-Reflect pipeline,
    but focuses on validating whether ANY field mapping is possible.
    """
    # Create utterance describing the mapping task
    source_type_short = source_type.split('_', 1)[-1] if '_' in source_type else source_type
    target_type_short = target_type.split('_', 1)[-1] if '_' in target_type else target_type

    utterance = f"map {source_product} {source_type_short} fields to {target_product} {target_type_short}"

    # Create ground truth mapping as empty - this is a discovery task
    # The mapping will be filled in by UE-Reflect
    mapping = {}

    # For evaluation, we expect the model to find at least one valid field mapping
    # if the types are truly compatible

    return {
        'id': sample_id,
        'utterance': utterance,
        'source_schema': source_product,
        'target_schema': target_product,
        'source_type': source_type,
        'target_type': target_type,
        'source_fields': [f['full_name'] for f in source_fields],
        'target_fields': [f['full_name'] for f in target_fields],
        'mapping': mapping,
        'metadata': {
            'type': 'coverage_validation',
            'similarity': similarity,
            'source_type_short': source_type_short,
            'target_type_short': target_type_short,
            'source_field_count': len(source_fields),
            'target_field_count': len(target_fields)
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Generate coverage validation dataset')
    parser.add_argument('--candidates', type=str,
                        default=str(DATA_DIR / 'itopsgraph_coverage_candidates.json'),
                        help='Path to candidates file')
    parser.add_argument('--api-lookup', type=str,
                        default=str(DATA_DIR / 'itopsgraph_api_lookup.json'),
                        help='Path to API lookup file')
    parser.add_argument('--output', type=str,
                        default=str(DATA_DIR / 'itopsgraph_coverage_dataset.jsonl'),
                        help='Output dataset file path')
    parser.add_argument('--min-source-fields', type=int, default=2,
                        help='Minimum number of source fields required')
    parser.add_argument('--min-target-fields', type=int, default=2,
                        help='Minimum number of target fields required')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to generate')
    args = parser.parse_args()

    print("=" * 70)
    print("Generating Coverage Validation Dataset")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    candidates_data = load_candidates(Path(args.candidates))
    api_lookup = load_api_lookup(Path(args.api_lookup))

    candidates = candidates_data.get('candidates', {})
    print(f"  Source types with candidates: {len(candidates)}")
    print(f"  API lookup entries: {len(api_lookup)}")

    # Generate samples
    print("\nGenerating samples...")
    samples = []
    skipped_no_fields = 0
    sample_idx = 0

    for source_type, source_info in candidates.items():
        source_product = source_info['source_product']

        # Get source fields from API lookup
        source_fields = get_type_fields(api_lookup, source_product, source_type)

        if len(source_fields) < args.min_source_fields:
            skipped_no_fields += 1
            continue

        for candidate in source_info['candidates']:
            target_type = candidate['target_type']
            target_product = candidate['target_product']
            similarity = candidate['similarity']

            # Get target fields from API lookup
            target_fields = get_type_fields(api_lookup, target_product, target_type)

            if len(target_fields) < args.min_target_fields:
                continue

            sample = create_sample(
                sample_id=f"coverage_{sample_idx:05d}",
                source_type=source_type,
                source_product=source_product,
                source_fields=source_fields,
                target_type=target_type,
                target_product=target_product,
                target_fields=target_fields,
                similarity=similarity
            )

            samples.append(sample)
            sample_idx += 1

            if args.max_samples and sample_idx >= args.max_samples:
                break

        if args.max_samples and sample_idx >= args.max_samples:
            break

    print(f"  Generated {len(samples)} samples")
    print(f"  Skipped {skipped_no_fields} source types (insufficient fields)")

    # Calculate statistics
    if samples:
        avg_source_fields = sum(s['metadata']['source_field_count'] for s in samples) / len(samples)
        avg_target_fields = sum(s['metadata']['target_field_count'] for s in samples) / len(samples)
        avg_similarity = sum(s['metadata']['similarity'] for s in samples) / len(samples)

        print(f"\n  Average source fields per sample: {avg_source_fields:.1f}")
        print(f"  Average target fields per sample: {avg_target_fields:.1f}")
        print(f"  Average type similarity: {avg_similarity:.3f}")

        # Distribution by source product
        from collections import Counter
        source_products = Counter(s['source_schema'] for s in samples)
        target_products = Counter(s['target_schema'] for s in samples)

        print("\n  Samples by source product:")
        for product, count in source_products.most_common(10):
            print(f"    {product}: {count}")

        print("\n  Samples by target product:")
        for product, count in target_products.most_common(10):
            print(f"    {product}: {count}")

    # Save dataset
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    print(f"\nDataset saved to: {output_path}")
    print(f"Total samples: {len(samples)}")


if __name__ == '__main__':
    main()
