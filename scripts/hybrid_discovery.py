#!/usr/bin/env python3
"""
Hybrid Discovery: Combine exact name matching with semantic similarity.

Strategy:
1. First: Try exact field name match in target product
2. If multiple matches: Use type-level context to disambiguate
3. Fallback: Semantic similarity on field name + description

This should handle the 76% exact name matches well.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_property_lookup(filepath: str) -> Dict:
    """Load property lookup with field metadata."""
    with open(filepath) as f:
        return json.load(f)


def find_matching_fields_by_name(
    source_field: str,
    target_product: str,
    property_data: Dict
) -> List[Tuple[str, str, float]]:
    """Find fields in target product with same field name.

    Returns list of (field_id, type_name, score)
    """
    # Extract just the field name (last part after last dot)
    source_name = source_field.split('.')[-1].lower()

    matches = []
    for field_id, data in property_data.items():
        if data.get('product_name', '').lower() != target_product.lower():
            continue

        field_name = data.get('field_name', field_id.split('.')[-1]).lower()

        # Exact match
        if field_name == source_name:
            matches.append((field_id, data.get('type_name', ''), 1.0))
        # Partial match (one contains the other)
        elif source_name in field_name or field_name in source_name:
            matches.append((field_id, data.get('type_name', ''), 0.8))

    return matches


def find_matching_fields_semantic(
    source_field: str,
    source_description: str,
    target_product: str,
    property_data: Dict,
    model: SentenceTransformer,
    embeddings_cache: Dict,
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """Find semantically similar fields using embeddings."""

    source_name = source_field.split('.')[-1]
    source_text = f"{source_name} {source_description}".strip()
    source_emb = model.encode([source_text], convert_to_numpy=True)[0]

    candidates = []
    for field_id, data in property_data.items():
        if data.get('product_name', '').lower() != target_product.lower():
            continue

        if field_id in embeddings_cache:
            emb = embeddings_cache[field_id]
        else:
            field_name = data.get('field_name', field_id.split('.')[-1])
            desc = data.get('description', '')
            text = f"{field_name} {desc}".strip()
            emb = model.encode([text], convert_to_numpy=True)[0]
            embeddings_cache[field_id] = emb

        sim = np.dot(source_emb, emb) / (np.linalg.norm(source_emb) * np.linalg.norm(emb))
        candidates.append((field_id, float(sim)))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_k]


def rank_by_type_similarity(
    source_type: str,
    matches: List[Tuple[str, str, float]],
    model: SentenceTransformer
) -> List[Tuple[str, float]]:
    """Rank matches by how similar their type is to source type."""

    if not matches:
        return []

    source_type_name = source_type.split('_')[-1] if '_' in source_type else source_type
    source_emb = model.encode([source_type_name], convert_to_numpy=True)[0]

    scored = []
    for field_id, type_name, name_score in matches:
        type_name_clean = type_name.split('_')[-1] if '_' in type_name else type_name
        type_emb = model.encode([type_name_clean], convert_to_numpy=True)[0]
        type_sim = np.dot(source_emb, type_emb) / (np.linalg.norm(source_emb) * np.linalg.norm(type_emb))

        # Combine name match score with type similarity
        combined_score = 0.7 * name_score + 0.3 * type_sim
        scored.append((field_id, float(combined_score)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def hybrid_discovery(
    source_field: str,
    source_type: str,
    source_description: str,
    target_product: str,
    property_data: Dict,
    model: SentenceTransformer,
    embeddings_cache: Dict,
    top_k: int = 5
) -> List[Tuple[str, float, str]]:
    """Hybrid approach: exact name match first, then semantic.

    Returns list of (field_id, score, method)
    """

    # Step 1: Try exact/partial name match
    name_matches = find_matching_fields_by_name(source_field, target_product, property_data)

    if name_matches:
        # If single match, return it
        if len(name_matches) == 1:
            return [(name_matches[0][0], name_matches[0][2], 'exact_name')]

        # Multiple matches: rank by type similarity
        ranked = rank_by_type_similarity(source_type, name_matches, model)
        return [(fid, score, 'name+type') for fid, score in ranked[:top_k]]

    # Step 2: Fallback to semantic similarity
    semantic_matches = find_matching_fields_semantic(
        source_field, source_description, target_product,
        property_data, model, embeddings_cache, top_k
    )
    return [(fid, score, 'semantic') for fid, score in semantic_matches]


def evaluate_hybrid(
    dataset_path: str,
    property_lookup_path: str,
    output_path: str,
    top_k: int = 5,
    max_samples: int = None
):
    """Evaluate hybrid discovery on dataset."""

    print("Loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Loading property lookup...")
    property_data = load_property_lookup(property_lookup_path)

    print("Loading dataset...")
    samples = []
    with open(dataset_path) as f:
        for line in f:
            samples.append(json.loads(line))

    if max_samples:
        samples = samples[:max_samples]

    print(f"Evaluating {len(samples)} samples...")

    embeddings_cache = {}
    results = []
    correct_top1 = 0
    correct_top5 = 0
    method_counts = {'exact_name': 0, 'name+type': 0, 'semantic': 0}
    method_correct = {'exact_name': 0, 'name+type': 0, 'semantic': 0}

    for sample in tqdm(samples):
        mapping = sample['mapping']
        source_field = list(mapping.keys())[0]
        expected_target = list(mapping.values())[0]
        target_product = sample['target_schema']

        # Get source type and description
        source_type = source_field.split('.')[0]  # e.g., "turbonomic_Action"
        source_data = property_data.get(source_field, {})
        source_desc = source_data.get('description', '')

        # Run hybrid discovery
        predictions = hybrid_discovery(
            source_field, source_type, source_desc,
            target_product, property_data, model,
            embeddings_cache, top_k
        )

        pred_ids = [p[0] for p in predictions]
        method_used = predictions[0][2] if predictions else 'none'

        top1_correct = predictions[0][0] == expected_target if predictions else False
        top5_correct = expected_target in pred_ids

        if top1_correct:
            correct_top1 += 1
        if top5_correct:
            correct_top5 += 1

        if method_used in method_counts:
            method_counts[method_used] += 1
            if top1_correct:
                method_correct[method_used] += 1

        result = {
            'id': sample['id'],
            'utterance': sample['utterance'],
            'source_field': source_field,
            'expected_target': expected_target,
            'predictions': [(p[0], p[1], p[2]) for p in predictions],
            'method_used': method_used,
            'top1_correct': top1_correct,
            'top5_correct': top5_correct
        }
        results.append(result)

    # Summary
    n = len(samples)
    summary = {
        'total_samples': n,
        'top1_accuracy': correct_top1 / n,
        'top5_accuracy': correct_top5 / n,
        'method_counts': method_counts,
        'method_accuracy': {
            k: method_correct[k] / method_counts[k] if method_counts[k] > 0 else 0
            for k in method_counts
        }
    }

    print("\n" + "="*60)
    print("HYBRID DISCOVERY RESULTS")
    print("="*60)
    print(f"Total samples: {n}")
    print(f"Top-1 Accuracy: {100*summary['top1_accuracy']:.1f}%")
    print(f"Top-5 Accuracy: {100*summary['top5_accuracy']:.1f}%")

    print("\nAccuracy by Method:")
    for method, count in method_counts.items():
        if count > 0:
            acc = method_correct[method] / count
            print(f"  {method}: {100*acc:.1f}% ({method_correct[method]}/{count} samples)")

    # Show examples
    print("\n" + "-"*60)
    print("EXAMPLE PREDICTIONS")
    print("-"*60)

    for r in results[:10]:
        print(f"\nSource: {r['source_field']}")
        print(f"Expected: {r['expected_target']}")
        print(f"Method: {r['method_used']}")
        print(f"Top predictions:")
        for pred, score, method in r['predictions'][:3]:
            marker = "✓" if pred == r['expected_target'] else " "
            print(f"  {marker} {pred} ({score:.3f})")
        print(f"Correct: {'✓' if r['top1_correct'] else '✗'}")

    # Save results
    output = {'summary': summary, 'results': results}
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description='Hybrid Discovery Evaluation')
    parser.add_argument('--dataset', default='data/itopsgraph_discovery_dataset.jsonl')
    parser.add_argument('--property-lookup', default='data/itopsgraph_api_lookup.json')
    parser.add_argument('--output', default='results/hybrid_discovery_results.json')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--max-samples', type=int, default=None)

    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    evaluate_hybrid(
        args.dataset, args.property_lookup, args.output,
        args.top_k, args.max_samples
    )


if __name__ == '__main__':
    main()
