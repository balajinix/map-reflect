#!/usr/bin/env python3
"""
Create evaluation dataset from batch coverage validation results.

Converts the 532 validated mappings into a proper evaluation dataset
with ground truth field-level mappings.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'


def load_batch_results(results_file: Path) -> List[Dict]:
    """Load batch validation results."""
    with open(results_file, 'r') as f:
        data = json.load(f)
    return data.get('results', [])


def load_api_lookup(lookup_file: Path) -> Dict:
    """Load API lookup for field information."""
    with open(lookup_file, 'r') as f:
        return json.load(f)


def create_eval_sample(result: Dict, api_lookup: Dict) -> Dict:
    """Convert a batch result into an evaluation sample."""
    source_type = result.get('source_type', '')
    target_type = result.get('target_type', '')
    source_product = result.get('source_product', '')
    target_product = result.get('target_product', '')

    # Get field mappings from batch result
    field_mappings = result.get('mappings', {})

    # Convert to full field names (product_Type.field format)
    mapping = {}
    source_type_short = source_type.split('_', 1)[-1] if '_' in source_type else source_type
    target_type_short = target_type.split('_', 1)[-1] if '_' in target_type else target_type

    for source_field, target_field in field_mappings.items():
        # Build full field names
        source_full = f"{source_type_short}.{source_field}"
        target_full = f"{target_type_short}.{target_field}"

        # Add to mapping with product prefixes
        mapping[f"{source_product}_{source_full}"] = f"{target_product}_{target_full}"

    # Create retrieval-style utterance (doesn't give away the target)
    # The model must retrieve the target type from the corpus
    utterance = f"find types that can be mapped to {source_product} {source_type_short}"

    return {
        'id': result.get('sample_id', ''),
        'utterance': utterance,
        'source_schema': source_product,
        'target_schema': target_product,
        'expected_target_type': f"{target_product}_{target_type_short}",
        'mapping': mapping,
        'metadata': {
            'source_type': source_type_short,
            'target_type': target_type_short,
            'confidence': result.get('confidence', 0),
            'similarity': result.get('similarity', 0),
            'validation_reason': result.get('reason', ''),
            'type': 'coverage_eval'
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Create coverage evaluation dataset')
    parser.add_argument('--batch-results', type=str,
                        default=str(DATA_DIR / 'batch_coverage_results_0.5.json'),
                        help='Batch validation results file')
    parser.add_argument('--api-lookup', type=str,
                        default=str(DATA_DIR / 'itopsgraph_api_lookup.json'),
                        help='API lookup file')
    parser.add_argument('--output', type=str,
                        default=str(DATA_DIR / 'itopsgraph_coverage_eval_dataset.jsonl'),
                        help='Output evaluation dataset')
    parser.add_argument('--min-confidence', type=float, default=0.6,
                        help='Minimum confidence threshold')
    parser.add_argument('--min-mappings', type=int, default=1,
                        help='Minimum number of field mappings required')
    args = parser.parse_args()

    print("=" * 70)
    print("Creating Coverage Evaluation Dataset")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    results = load_batch_results(Path(args.batch_results))
    api_lookup = load_api_lookup(Path(args.api_lookup))
    print(f"  Total batch results: {len(results)}")

    # Filter to valid results
    valid_results = [
        r for r in results
        if r.get('valid', False)
        and r.get('confidence', 0) >= args.min_confidence
        and len(r.get('mappings', {})) >= args.min_mappings
    ]
    print(f"  Valid results (confidence >= {args.min_confidence}): {len(valid_results)}")

    # Create evaluation samples
    print("\nCreating evaluation samples...")
    eval_samples = []
    for result in valid_results:
        sample = create_eval_sample(result, api_lookup)
        if sample['mapping']:  # Only include if has mappings
            eval_samples.append(sample)

    print(f"  Evaluation samples created: {len(eval_samples)}")

    # Statistics
    total_mappings = sum(len(s['mapping']) for s in eval_samples)
    avg_mappings = total_mappings / len(eval_samples) if eval_samples else 0
    print(f"  Total field mappings: {total_mappings}")
    print(f"  Avg mappings per sample: {avg_mappings:.1f}")

    # Product distribution
    print("\n  By source product:")
    from collections import Counter
    source_counts = Counter(s['source_schema'] for s in eval_samples)
    for product, count in source_counts.most_common():
        print(f"    {product}: {count}")

    # Save dataset
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        for sample in eval_samples:
            f.write(json.dumps(sample) + '\n')

    print(f"\nDataset saved to: {output_path}")

    # Show sample
    if eval_samples:
        print("\nSample entry:")
        sample = eval_samples[0]
        print(f"  ID: {sample['id']}")
        print(f"  Utterance: {sample['utterance']}")
        print(f"  Mapping: {list(sample['mapping'].items())[:2]}")


if __name__ == '__main__':
    main()
