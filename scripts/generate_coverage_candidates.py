#!/usr/bin/env python3
"""
Generate candidate type pairs for ITOpsGraph coverage expansion.

This script finds candidate target types for each uncovered source type
using semantic embedding similarity of type names and descriptions.

Output: data/itopsgraph_coverage_candidates.json
"""

import json
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'

# Products with underscores that need special prefix matching
PRODUCTS_WITH_UNDERSCORES = {'hcp_terraform', 'watsonx-orchestrate'}

# Configuration
TOP_K_TYPE_CANDIDATES = 5  # Max target types to consider per source type


def extract_product_from_type(type_name: str) -> str:
    """Extract product name from type like 'concert_Application' -> 'concert'."""
    for known_prod in sorted(PRODUCTS_WITH_UNDERSCORES, key=len, reverse=True):
        if type_name.startswith(known_prod + '_'):
            return known_prod

    if '_' in type_name:
        return type_name.split('_', 1)[0]
    return 'unknown'


def load_api_lookup(lookup_file: Path) -> Dict:
    """Load the API lookup file."""
    with open(lookup_file, 'r') as f:
        return json.load(f)


def load_type_coverage() -> Dict:
    """Load the type coverage analysis file."""
    coverage_file = DATA_DIR / 'type_coverage_analysis.json'
    with open(coverage_file, 'r') as f:
        return json.load(f)


def get_types_by_product(api_lookup: Dict) -> Dict[str, Dict[str, Dict]]:
    """
    Organize types by product from the API lookup.
    Returns: {product: {type_name: {fields: [...], description: str}}}
    """
    types_by_product = defaultdict(lambda: defaultdict(lambda: {'fields': [], 'description': ''}))

    for key, value in api_lookup.items():
        if not isinstance(value, dict):
            continue

        product = value.get('product_name', '')
        type_name = value.get('type_name', '')
        field_name = value.get('field_name', '')
        type_desc = value.get('type_description', '')

        if product and type_name:
            full_type = f"{product}_{type_name}"
            types_by_product[product][full_type]['fields'].append(field_name)
            if type_desc and not types_by_product[product][full_type]['description']:
                types_by_product[product][full_type]['description'] = type_desc

    return dict(types_by_product)


def get_uncovered_types(coverage_data: Dict) -> Set[str]:
    """Get set of uncovered types from coverage analysis."""
    uncovered = set()
    for product, types in coverage_data.get('uncovered_types_by_product', {}).items():
        uncovered.update(types)
    return uncovered


def create_type_embeddings(
    types_info: Dict[str, Dict],
    model: SentenceTransformer
) -> Tuple[List[str], np.ndarray]:
    """
    Create embeddings for all types based on name and description.
    Returns: (type_names, embeddings_matrix)
    """
    type_names = []
    texts = []

    for type_name, info in types_info.items():
        # Create text representation: type name + description + field names
        type_base = type_name.split('_', 1)[-1] if '_' in type_name else type_name
        desc = info.get('description', '') or ''
        fields = info.get('fields', [])

        # Combine type name, description, and field names for embedding
        text_parts = [type_base]
        if desc:
            text_parts.append(desc)
        if fields:
            text_parts.append(' '.join(fields[:10]))  # Limit fields to avoid too long text

        text = ' -- '.join(text_parts)
        type_names.append(type_name)
        texts.append(text)

    # Generate embeddings
    print(f"Generating embeddings for {len(texts)} types...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    return type_names, embeddings


def find_candidate_targets(
    source_type: str,
    source_product: str,
    source_embedding: np.ndarray,
    all_type_names: List[str],
    all_embeddings: np.ndarray,
    types_by_product: Dict[str, Dict],
    top_k: int = TOP_K_TYPE_CANDIDATES
) -> List[Dict]:
    """
    Find top-K candidate target types for a source type.
    Excludes types from the same product.
    """
    # Calculate cosine similarities
    # Normalize embeddings
    source_norm = source_embedding / (np.linalg.norm(source_embedding) + 1e-8)
    all_norms = all_embeddings / (np.linalg.norm(all_embeddings, axis=1, keepdims=True) + 1e-8)

    similarities = np.dot(all_norms, source_norm)

    # Get sorted indices
    sorted_indices = np.argsort(-similarities)

    candidates = []
    for idx in sorted_indices:
        target_type = all_type_names[idx]
        target_product = extract_product_from_type(target_type)

        # Skip same product
        if target_product == source_product:
            continue

        # Skip if same type (shouldn't happen, but safety check)
        if target_type == source_type:
            continue

        # Get type info
        target_info = types_by_product.get(target_product, {}).get(target_type, {})

        candidates.append({
            'target_type': target_type,
            'target_product': target_product,
            'similarity': float(similarities[idx]),
            'target_fields': target_info.get('fields', []),
            'target_description': target_info.get('description', '')
        })

        if len(candidates) >= top_k:
            break

    return candidates


def main():
    parser = argparse.ArgumentParser(description='Generate coverage candidate type pairs')
    parser.add_argument('--api-lookup', type=str,
                        default=str(DATA_DIR / 'itopsgraph_api_lookup.json'),
                        help='Path to API lookup file')
    parser.add_argument('--output', type=str,
                        default=str(DATA_DIR / 'itopsgraph_coverage_candidates.json'),
                        help='Output file path')
    parser.add_argument('--top-k', type=int, default=TOP_K_TYPE_CANDIDATES,
                        help='Number of candidate targets per source type')
    parser.add_argument('--max-types', type=int, default=None,
                        help='Maximum number of source types to process (for testing)')
    args = parser.parse_args()

    print("=" * 70)
    print("Generating Coverage Candidate Type Pairs")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    api_lookup = load_api_lookup(Path(args.api_lookup))
    coverage_data = load_type_coverage()

    print(f"  API lookup entries: {len(api_lookup)}")

    # Get types organized by product
    types_by_product = get_types_by_product(api_lookup)
    all_types_info = {}
    for product, types in types_by_product.items():
        all_types_info.update(types)

    print(f"  Total types: {len(all_types_info)}")
    print(f"  Products: {len(types_by_product)}")

    # Get uncovered types
    uncovered_types = get_uncovered_types(coverage_data)
    print(f"  Uncovered types: {len(uncovered_types)}")

    # Filter to only process uncovered types that exist in our lookup
    uncovered_with_info = [t for t in uncovered_types if t in all_types_info]
    print(f"  Uncovered types with field info: {len(uncovered_with_info)}")

    if args.max_types:
        uncovered_with_info = uncovered_with_info[:args.max_types]
        print(f"  Processing first {args.max_types} types (testing mode)")

    # Load embedding model
    print("\nLoading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create embeddings for all types
    all_type_names, all_embeddings = create_type_embeddings(all_types_info, model)

    # Create index mapping
    type_to_idx = {name: idx for idx, name in enumerate(all_type_names)}

    # Find candidates for each uncovered type
    print(f"\nFinding candidates for {len(uncovered_with_info)} uncovered types...")
    candidates_by_source = {}

    for i, source_type in enumerate(uncovered_with_info):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(uncovered_with_info)} types...")

        source_product = extract_product_from_type(source_type)
        source_idx = type_to_idx.get(source_type)

        if source_idx is None:
            continue

        source_embedding = all_embeddings[source_idx]
        source_info = all_types_info.get(source_type, {})

        candidates = find_candidate_targets(
            source_type=source_type,
            source_product=source_product,
            source_embedding=source_embedding,
            all_type_names=all_type_names,
            all_embeddings=all_embeddings,
            types_by_product=types_by_product,
            top_k=args.top_k
        )

        if candidates:
            candidates_by_source[source_type] = {
                'source_product': source_product,
                'source_fields': source_info.get('fields', []),
                'source_description': source_info.get('description', ''),
                'candidates': candidates
            }

    # Calculate statistics
    total_candidates = sum(len(v['candidates']) for v in candidates_by_source.values())

    print(f"\n{'=' * 70}")
    print("Results Summary")
    print(f"{'=' * 70}")
    print(f"  Source types processed: {len(candidates_by_source)}")
    print(f"  Total candidate pairs: {total_candidates}")
    print(f"  Average candidates per source: {total_candidates / len(candidates_by_source):.2f}" if candidates_by_source else "N/A")

    # Show distribution by source product
    by_source_product = defaultdict(int)
    for source_type in candidates_by_source:
        product = extract_product_from_type(source_type)
        by_source_product[product] += 1

    print("\n  Source types by product:")
    for product, count in sorted(by_source_product.items(), key=lambda x: -x[1]):
        print(f"    {product}: {count}")

    # Save results
    output_data = {
        'metadata': {
            'total_source_types': len(candidates_by_source),
            'total_candidate_pairs': total_candidates,
            'top_k': args.top_k,
            'embedding_model': 'all-MiniLM-L6-v2'
        },
        'candidates': candidates_by_source
    }

    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
