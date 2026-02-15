#!/usr/bin/env python3
"""
Semantic Discovery: Find target field mappings using embedding similarity.

Approach:
1. Embed source field (name + description)
2. Find most similar target fields in target product
3. No LLM calls - pure embedding-based matching

This tests whether semantic similarity alone can discover mappings.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_property_lookup(filepath: str) -> Dict:
    """Load property lookup with field metadata."""
    with open(filepath) as f:
        return json.load(f)


def build_field_embeddings(property_data: Dict, model: SentenceTransformer) -> Tuple[Dict, np.ndarray, List[str]]:
    """Build embeddings for all fields.

    Returns:
        - field_info: Dict mapping field_id to metadata
        - embeddings: numpy array of embeddings
        - field_ids: list of field IDs in same order as embeddings
    """
    field_info = {}
    texts = []
    field_ids = []

    for field_id, data in property_data.items():
        # Create text representation: name + description
        name = data.get('field_name', field_id.split('.')[-1])
        desc = data.get('description', '')
        type_name = data.get('type_name', '')
        product = data.get('product_name', '')

        # Combine for embedding
        text = f"{name} {desc}".strip()
        if not text:
            text = field_id

        field_info[field_id] = {
            'name': name,
            'description': desc,
            'type_name': type_name,
            'product': product,
            'text': text
        }
        texts.append(text)
        field_ids.append(field_id)

    print(f"Encoding {len(texts)} fields...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    return field_info, embeddings, field_ids


def find_similar_fields(
    source_field: str,
    target_product: str,
    field_info: Dict,
    embeddings: np.ndarray,
    field_ids: List[str],
    model: SentenceTransformer,
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """Find most similar fields in target product."""

    # Get source field text
    if source_field in field_info:
        source_text = field_info[source_field]['text']
    else:
        # Fallback: use field name
        source_text = source_field.split('.')[-1]

    # Embed source
    source_emb = model.encode([source_text], convert_to_numpy=True)[0]

    # Find target fields in target product
    target_indices = []
    for i, fid in enumerate(field_ids):
        if field_info[fid]['product'].lower() == target_product.lower():
            target_indices.append(i)

    if not target_indices:
        return []

    # Compute similarities
    target_embeddings = embeddings[target_indices]
    similarities = np.dot(target_embeddings, source_emb) / (
        np.linalg.norm(target_embeddings, axis=1) * np.linalg.norm(source_emb)
    )

    # Get top-k
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        orig_idx = target_indices[idx]
        field_id = field_ids[orig_idx]
        sim = similarities[idx]
        results.append((field_id, float(sim)))

    return results


def evaluate_discovery(
    dataset_path: str,
    property_lookup_path: str,
    output_path: str,
    top_k: int = 5,
    max_samples: int = None
):
    """Evaluate semantic discovery on dataset."""

    print("Loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Loading property lookup...")
    property_data = load_property_lookup(property_lookup_path)

    print("Building field embeddings...")
    field_info, embeddings, field_ids = build_field_embeddings(property_data, model)

    print("Loading dataset...")
    samples = []
    with open(dataset_path) as f:
        for line in f:
            samples.append(json.loads(line))

    if max_samples:
        samples = samples[:max_samples]

    print(f"Evaluating {len(samples)} samples...")

    results = []
    correct_top1 = 0
    correct_top5 = 0
    correct_type = 0

    for sample in tqdm(samples):
        # Get source and target info
        mapping = sample['mapping']
        source_field = list(mapping.keys())[0]
        expected_target = list(mapping.values())[0]
        target_product = sample['target_schema']

        # Find similar fields
        predictions = find_similar_fields(
            source_field, target_product,
            field_info, embeddings, field_ids,
            model, top_k
        )

        # Check accuracy
        pred_ids = [p[0] for p in predictions]

        top1_correct = predictions[0][0] == expected_target if predictions else False
        top5_correct = expected_target in pred_ids

        # Check if correct type is in predictions
        expected_type = expected_target.split('.')[0]  # e.g., "concert_PackageRecommendation"
        pred_types = set(p.split('.')[0] for p in pred_ids)
        type_correct = expected_type in pred_types

        if top1_correct:
            correct_top1 += 1
        if top5_correct:
            correct_top5 += 1
        if type_correct:
            correct_type += 1

        result = {
            'id': sample['id'],
            'utterance': sample['utterance'],
            'source_field': source_field,
            'expected_target': expected_target,
            'predictions': predictions,
            'top1_correct': top1_correct,
            'top5_correct': top5_correct,
            'type_correct': type_correct
        }
        results.append(result)

    # Summary
    n = len(samples)
    summary = {
        'total_samples': n,
        'top1_accuracy': correct_top1 / n,
        'top5_accuracy': correct_top5 / n,
        'type_accuracy': correct_type / n,
        'top_k': top_k
    }

    print("\n" + "="*60)
    print("SEMANTIC DISCOVERY RESULTS")
    print("="*60)
    print(f"Total samples: {n}")
    print(f"Top-1 Accuracy: {100*summary['top1_accuracy']:.1f}%")
    print(f"Top-5 Accuracy: {100*summary['top5_accuracy']:.1f}%")
    print(f"Type Accuracy (correct type in top-5): {100*summary['type_accuracy']:.1f}%")

    # Show some examples
    print("\n" + "-"*60)
    print("EXAMPLE PREDICTIONS")
    print("-"*60)

    for r in results[:10]:
        print(f"\nUtterance: {r['utterance'][:70]}...")
        print(f"Source: {r['source_field']}")
        print(f"Expected: {r['expected_target']}")
        print(f"Top predictions:")
        for pred, sim in r['predictions'][:3]:
            marker = "✓" if pred == r['expected_target'] else " "
            print(f"  {marker} {pred} ({sim:.3f})")
        print(f"Correct: {'✓' if r['top1_correct'] else '✗'}")

    # Save results
    output = {
        'summary': summary,
        'results': results
    }
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Semantic Discovery Evaluation')
    parser.add_argument('--dataset', default='data/itopsgraph_discovery_dataset.jsonl',
                        help='Path to discovery dataset')
    parser.add_argument('--property-lookup', default='data/itopsgraph_api_lookup.json',
                        help='Path to property lookup')
    parser.add_argument('--output', default='results/semantic_discovery_results.json',
                        help='Output path for results')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of top predictions to consider')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples to evaluate')

    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    evaluate_discovery(
        args.dataset,
        args.property_lookup,
        args.output,
        args.top_k,
        args.max_samples
    )


if __name__ == '__main__':
    main()
