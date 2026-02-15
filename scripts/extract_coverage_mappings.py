#!/usr/bin/env python3
"""
Extract validated mappings from UE-Reflect coverage results.

This script parses UE-Reflect results and filters to find type mappings
where at least one field mapping has low uncertainty (high confidence).

Input: map_reflect/coverage_results/*.json
Output: Updates data/itops_type_mapping_lookup.json with validated mappings
"""

import json
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple
import glob

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'

# Configuration
UNCERTAINTY_THRESHOLD = 0.5  # Max uncertainty to consider field mapping valid
MIN_VALID_FIELDS = 1  # Minimum valid field mappings to confirm type mapping
MIN_SCORE_THRESHOLD = 0.6  # Minimum score for a field mapping to be considered


def load_results(results_dir: Path) -> List[Dict]:
    """Load all UE-Reflect result files from the directory."""
    results = []
    pattern = str(results_dir / "ue_reflect_itopsgraph_coverage*.json")

    for filepath in glob.glob(pattern):
        print(f"  Loading: {filepath}")
        with open(filepath, 'r') as f:
            data = json.load(f)
            results.append(data)

    return results


def load_existing_lookup(lookup_file: Path) -> Dict:
    """Load existing type mapping lookup."""
    if not lookup_file.exists():
        return {}
    with open(lookup_file, 'r') as f:
        return json.load(f)


def extract_valid_field_mappings(
    sample_result: Dict,
    uncertainty_threshold: float = UNCERTAINTY_THRESHOLD,
    min_score: float = MIN_SCORE_THRESHOLD
) -> List[Dict]:
    """
    Extract valid field mappings from a sample result.

    A field mapping is valid if:
    1. Uncertainty is below threshold
    2. Top score is above minimum threshold

    Returns list of {source_field, target_field, score, uncertainty}
    """
    valid_mappings = []

    predictions = sample_result.get('predictions', {})

    for source_field, pred_info in predictions.items():
        uncertainty = pred_info.get('uncertainty', 1.0)

        # Skip high uncertainty predictions
        if uncertainty > uncertainty_threshold:
            continue

        # Get the top prediction from revised ranking
        reflection = pred_info.get('reflection', {})
        revised_ranking = reflection.get('revised_ranking', [])

        if not revised_ranking:
            continue

        # Check if top prediction meets score threshold
        top_pred = revised_ranking[0]
        if isinstance(top_pred, dict):
            target_field = top_pred.get('id', '')
            score = top_pred.get('score', 0)

            try:
                score = float(score) if score is not None else 0
            except (ValueError, TypeError):
                score = 0

            if score >= min_score and target_field:
                valid_mappings.append({
                    'source_field': source_field,
                    'target_field': target_field,
                    'score': score,
                    'uncertainty': uncertainty,
                    'reason': top_pred.get('reason', '')
                })

    return valid_mappings


def validate_type_mapping(
    sample_result: Dict,
    uncertainty_threshold: float = UNCERTAINTY_THRESHOLD,
    min_score: float = MIN_SCORE_THRESHOLD,
    min_valid_fields: int = MIN_VALID_FIELDS
) -> Tuple[bool, List[Dict]]:
    """
    Validate if a type mapping should be accepted.

    Returns (is_valid, valid_field_mappings)
    """
    valid_mappings = extract_valid_field_mappings(
        sample_result,
        uncertainty_threshold=uncertainty_threshold,
        min_score=min_score
    )

    is_valid = len(valid_mappings) >= min_valid_fields

    return is_valid, valid_mappings


def extract_type_from_field(field_name: Any) -> str:
    """Extract type name from a full field name like 'product_Type.field'."""
    field_name = str(field_name) if field_name is not None else ''
    if '.' in field_name:
        return field_name.split('.')[0]
    return field_name


def main():
    parser = argparse.ArgumentParser(description='Extract validated mappings from UE-Reflect results')
    parser.add_argument('--results-dir', type=str,
                        default=str(BASE_DIR / 'map_reflect' / 'coverage_results'),
                        help='Directory containing UE-Reflect results')
    parser.add_argument('--lookup-file', type=str,
                        default=str(DATA_DIR / 'itops_type_mapping_lookup.json'),
                        help='Type mapping lookup file to update')
    parser.add_argument('--output-file', type=str,
                        default=str(DATA_DIR / 'itops_type_mapping_lookup_expanded.json'),
                        help='Output file for expanded lookup')
    parser.add_argument('--uncertainty-threshold', type=float, default=UNCERTAINTY_THRESHOLD,
                        help='Maximum uncertainty for valid field mapping')
    parser.add_argument('--min-score', type=float, default=MIN_SCORE_THRESHOLD,
                        help='Minimum score for valid field mapping')
    parser.add_argument('--min-valid-fields', type=int, default=MIN_VALID_FIELDS,
                        help='Minimum valid field mappings to confirm type mapping')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be added without modifying files')
    args = parser.parse_args()

    print("=" * 70)
    print("Extracting Validated Mappings from UE-Reflect Results")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Uncertainty threshold: {args.uncertainty_threshold}")
    print(f"  Min score threshold: {args.min_score}")
    print(f"  Min valid fields: {args.min_valid_fields}")

    # Load results
    print("\nLoading UE-Reflect results...")
    results_dir = Path(args.results_dir)
    all_results = load_results(results_dir)

    if not all_results:
        print("  No results found!")
        return

    total_samples = sum(len(r.get('results', [])) for r in all_results)
    print(f"  Loaded {len(all_results)} result files with {total_samples} total samples")

    # Load existing lookup
    print("\nLoading existing type mapping lookup...")
    lookup_file = Path(args.lookup_file)
    existing_lookup = load_existing_lookup(lookup_file)
    print(f"  Existing entries: {len(existing_lookup)}")

    # Process results and extract valid type mappings
    print("\nProcessing samples...")
    valid_type_mappings = []
    invalid_type_mappings = []
    type_pair_field_mappings = defaultdict(list)  # (source_type, target_type) -> field_mappings

    for result_data in all_results:
        for sample_result in result_data.get('results', []):
            sample_id = sample_result.get('sample_id', 'unknown')
            source_schema = sample_result.get('expected_source', '')
            target_schema = sample_result.get('expected_target', '')

            # Handle dict format for expected_source/target
            if isinstance(source_schema, dict):
                source_schema = source_schema.get('name', source_schema.get('id', ''))
            if isinstance(target_schema, dict):
                target_schema = target_schema.get('name', target_schema.get('id', ''))

            # Reconstruct full type names from field names
            # Field names are like "TypeName.field", need to prepend product name
            ground_truth = sample_result.get('ground_truth_mapping', {})
            source_type = ''
            if ground_truth:
                first_field = list(ground_truth.keys())[0]
                if '.' in first_field:
                    type_part = first_field.split('.')[0]
                    source_type = f"{source_schema}_{type_part}"

            # Get target type from predictions (first valid mapping)
            predictions = sample_result.get('predictions', {})
            target_type = ''
            for pred_info in predictions.values():
                refl = pred_info.get('reflection', {})
                ranking = refl.get('revised_ranking', [])
                if ranking and isinstance(ranking[0], dict):
                    top_target = ranking[0].get('id', '')
                    if '.' in top_target:
                        target_type_part = top_target.split('.')[0]
                        target_type = target_type_part  # Already has product prefix
                        break

            is_valid, field_mappings = validate_type_mapping(
                sample_result,
                uncertainty_threshold=args.uncertainty_threshold,
                min_score=args.min_score,
                min_valid_fields=args.min_valid_fields
            )

            if is_valid:
                valid_type_mappings.append({
                    'sample_id': sample_id,
                    'source_type': source_type,
                    'target_type': target_type,
                    'field_mappings': field_mappings
                })
                # Aggregate field mappings by type pair
                type_pair_key = (source_type, target_type)
                type_pair_field_mappings[type_pair_key].extend(field_mappings)
            else:
                invalid_type_mappings.append({
                    'sample_id': sample_id,
                    'source_type': source_type,
                    'target_type': target_type,
                    'field_mappings': field_mappings  # May be empty or below threshold
                })

    print(f"  Valid type mappings: {len(valid_type_mappings)}")
    print(f"  Invalid type mappings: {len(invalid_type_mappings)}")

    # Build new lookup entries
    print("\nBuilding new lookup entries...")
    new_entries = {}

    for (source_type, target_type), field_mappings in type_pair_field_mappings.items():
        # source_type is now the full type name like "instana_TypeName"
        # Skip if empty
        if not source_type:
            continue

        # Skip if source type already in lookup
        if source_type in existing_lookup:
            continue

        # Get the best target for this source type
        # (highest average score across field mappings)
        targets_scores = defaultdict(list)
        for fm in field_mappings:
            target_type_from_field = extract_type_from_field(fm['target_field'])
            targets_scores[target_type_from_field].append(fm['score'])

        if not targets_scores:
            continue

        # Find best target
        best_target = max(targets_scores.keys(), key=lambda t: sum(targets_scores[t]) / len(targets_scores[t]))

        # Collect methods (we'll label as UE-Reflect discovery)
        methods = ['UE-Reflect (Coverage)']

        if source_type not in new_entries:
            new_entries[source_type] = {}

        if best_target not in new_entries[source_type]:
            new_entries[source_type][best_target] = methods
        else:
            # Merge methods
            for m in methods:
                if m not in new_entries[source_type][best_target]:
                    new_entries[source_type][best_target].append(m)

    print(f"  New type entries to add: {len(new_entries)}")

    # Statistics by product
    by_product = defaultdict(int)
    for source_type in new_entries:
        if '_' in source_type:
            product = source_type.split('_')[0]
            by_product[product] += 1

    if by_product:
        print("\n  New entries by source product:")
        for product, count in sorted(by_product.items(), key=lambda x: -x[1]):
            print(f"    {product}: {count}")

    # Show some examples
    if new_entries:
        print("\n  Sample new entries:")
        for i, (source_type, targets) in enumerate(list(new_entries.items())[:5]):
            print(f"    {source_type} -> {list(targets.keys())}")

    if args.dry_run:
        print("\n[DRY RUN] No files modified.")
        return

    # Merge with existing lookup
    merged_lookup = existing_lookup.copy()
    for source_type, targets in new_entries.items():
        if source_type not in merged_lookup:
            merged_lookup[source_type] = targets
        else:
            # Merge targets
            for target, methods in targets.items():
                if target not in merged_lookup[source_type]:
                    merged_lookup[source_type][target] = methods
                else:
                    # Merge methods
                    for m in methods:
                        if m not in merged_lookup[source_type][target]:
                            merged_lookup[source_type][target].append(m)

    # Save updated lookup
    output_path = Path(args.output_file)
    with open(output_path, 'w') as f:
        json.dump(merged_lookup, f, indent=2)

    print(f"\nExpanded lookup saved to: {output_path}")
    print(f"  Original entries: {len(existing_lookup)}")
    print(f"  New entries added: {len(new_entries)}")
    print(f"  Total entries: {len(merged_lookup)}")

    # Also save detailed field mappings for analysis
    field_mappings_file = output_path.parent / 'coverage_field_mappings.json'
    field_mappings_output = {
        'metadata': {
            'uncertainty_threshold': args.uncertainty_threshold,
            'min_score': args.min_score,
            'min_valid_fields': args.min_valid_fields,
            'total_valid_type_mappings': len(valid_type_mappings),
            'total_invalid_type_mappings': len(invalid_type_mappings)
        },
        'valid_mappings': valid_type_mappings,
        'new_lookup_entries': new_entries
    }

    with open(field_mappings_file, 'w') as f:
        json.dump(field_mappings_output, f, indent=2)

    print(f"  Field mappings saved to: {field_mappings_file}")


if __name__ == '__main__':
    main()
