#!/usr/bin/env python3
"""
Evaluate GNN on SchemaNet datasets with Field-Level Accuracy.
Produces metrics comparable to Map-Reflect baselines.

For each source field, predicts the target field it should map to.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np

try:
    from gnn_model import MappingKGGNN, load_model_safely
except ImportError:
    print("Error: gnn_model.py not found")
    sys.exit(1)


def load_dataset(dataset_file):
    """Load SchemaNet dataset."""
    samples = []
    with open(dataset_file) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def load_gnn_model_and_graph(model_file, graph_data_file):
    """Load trained GNN model and graph data."""
    model = load_model_safely(model_file)
    with open(graph_data_file, 'rb') as f:
        graph_data = pickle.load(f)
    return model, graph_data


def get_target_field_indices(graph, node_list, target_schema):
    """Get all target field indices with their node names."""
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    target_fields = []

    for node in node_list:
        node_data = graph.nodes[node]
        if node_data.get('node_type') != 'field':
            continue
        if node_data.get('product') != target_schema:
            continue

        if node in node_to_idx:
            # Build the full field name as in ground truth: "bank2_branch.branch_id"
            type_name = node_data.get('type', '')
            field_name = node_data.get('name', '')
            full_type = f"{target_schema}_{type_name}"
            full_field = f"{full_type}.{field_name}"
            target_fields.append({
                'node': node,
                'idx': node_to_idx[node],
                'full_field': full_field,
                'type': full_type
            })

    return target_fields


def get_source_field_index(graph, node_list, source_field, source_schema):
    """Get node index for a specific source field.

    source_field format: "bank1_branch.branch_id" or "mimic_iii_admissions.row_id"
    source_schema: The schema name (e.g., "bank1" or "mimic_iii")
    """
    node_to_idx = {node: i for i, node in enumerate(node_list)}

    parts = source_field.split('.')
    type_full = parts[0]  # "bank1_branch" or "mimic_iii_admissions"
    field_name = parts[1] if len(parts) > 1 else parts[0]

    # Use provided schema to extract type name correctly
    type_name = type_full.replace(f"{source_schema}_", "")

    for node in node_list:
        node_data = graph.nodes[node]
        if node_data.get('node_type') != 'field':
            continue
        if node_data.get('product') != source_schema:
            continue
        if node_data.get('type') != type_name:
            continue
        if node_data.get('name') != field_name:
            continue

        if node in node_to_idx:
            return node_to_idx[node]

    return None


def predict_target_field(model, data, src_idx, target_fields):
    """Predict the best target field for a source field."""
    if src_idx is None or not target_fields:
        return None, None, 0.0

    model.eval()

    # Create edges from source to all target fields
    edges = [[src_idx, tf['idx']] for tf in target_fields]
    query_edges = torch.tensor(edges).t()

    with torch.no_grad():
        try:
            probs = model.predict_links(data.x, data.train_pos_edge_index, query_edges)
            probs = probs.squeeze()

            # Get best prediction
            best_idx = probs.argmax().item()
            best_score = probs[best_idx].item()
            best_field = target_fields[best_idx]

            return best_field['full_field'], best_field['type'], best_score
        except Exception as e:
            return None, None, 0.0


def evaluate_dataset(dataset_name, schemanet_dir, gnn_results_dir, data_dir):
    """Evaluate GNN on a single SchemaNet dataset (field-level)."""
    dataset_file = Path(schemanet_dir) / f"{dataset_name}_dataset.jsonl"
    model_file = Path(gnn_results_dir) / f"{dataset_name}_mapping_gnn_results.pkl"
    graph_data_file = Path(data_dir) / f"{dataset_name}_graph_data.pkl"

    for f in [dataset_file, model_file, graph_data_file]:
        if not f.exists():
            print(f"  File not found: {f}")
            return None

    samples = load_dataset(dataset_file)
    model, graph_data = load_gnn_model_and_graph(model_file, graph_data_file)

    data = graph_data['data']
    node_list = graph_data['node_list']
    graph = graph_data['graph']

    target_schema = samples[0].get('target_schema')
    target_fields = get_target_field_indices(graph, node_list, target_schema)

    # Evaluate each field mapping
    field_correct = 0
    type_correct = 0
    total = 0
    results = []

    for sample in samples:
        source_schema = sample.get('source_schema')
        for source_field, expected_target in sample.get('mapping', {}).items():
            src_idx = get_source_field_index(graph, node_list, source_field, source_schema)

            predicted_field, predicted_type, score = predict_target_field(
                model, data, src_idx, target_fields
            )

            # Extract expected type from expected_target (e.g., "bank2_branch.branch_id" -> "bank2_branch")
            expected_type = expected_target.split('.')[0]

            is_field_correct = (predicted_field == expected_target)
            is_type_correct = (predicted_type == expected_type)

            if is_field_correct:
                field_correct += 1
            if is_type_correct:
                type_correct += 1
            total += 1

            results.append({
                'source_field': source_field,
                'expected_field': expected_target,
                'expected_type': expected_type,
                'predicted_field': predicted_field,
                'predicted_type': predicted_type,
                'score': score,
                'field_correct': is_field_correct,
                'type_correct': is_type_correct
            })

    field_accuracy = field_correct / total if total > 0 else 0.0
    type_accuracy = type_correct / total if total > 0 else 0.0

    return {
        'dataset': dataset_name,
        'field_correct': field_correct,
        'type_correct': type_correct,
        'total': total,
        'field_accuracy': field_accuracy,
        'type_accuracy': type_accuracy,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate GNN on SchemaNet (field-level)')
    parser.add_argument('--schemanet-dir', type=str, default='../data/schemanet')
    parser.add_argument('--gnn-results-dir', type=str, default='../data/gnn_results')
    parser.add_argument('--data-dir', type=str, default='../data')
    parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()

    DATASETS = [
        "schemanet_bank1_bank2",
        "schemanet_imdb_sakila",
        "schemanet_cms_omop",
        "schemanet_mimic_iii_omop",
        "schemanet_synthea_omop",
        "schemanet_cprd_gold_omop",
        "schemanet_cprd_aurum_omop",
    ]

    print("=" * 70)
    print("GNN Field-Level Accuracy on SchemaNet")
    print("=" * 70)
    print()

    all_results = []
    total_field_correct = 0
    total_type_correct = 0
    total_samples = 0

    for dataset in DATASETS:
        print(f"Evaluating {dataset}...")
        result = evaluate_dataset(
            dataset, args.schemanet_dir, args.gnn_results_dir, args.data_dir
        )

        if result:
            all_results.append(result)
            total_field_correct += result['field_correct']
            total_type_correct += result['type_correct']
            total_samples += result['total']
            print(f"  Field Acc: {result['field_accuracy']:.1%} ({result['field_correct']}/{result['total']})")
            print(f"  Type Acc:  {result['type_accuracy']:.1%} ({result['type_correct']}/{result['total']})")
        else:
            print(f"  Skipped")
        print()

    # Summary table
    print("=" * 70)
    print("Summary: Field-Level and Type-Level Accuracy")
    print("=" * 70)
    print(f"{'Dataset':<25} {'Field Acc':>12} {'Type Acc':>12} {'Samples':>10}")
    print("-" * 70)

    for result in all_results:
        name = result['dataset'].replace('schemanet_', '')
        print(f"{name:<25} {result['field_accuracy']:>11.1%} {result['type_accuracy']:>11.1%} {result['total']:>10}")

    print("-" * 70)
    overall_field = total_field_correct / total_samples if total_samples > 0 else 0.0
    overall_type = total_type_correct / total_samples if total_samples > 0 else 0.0
    print(f"{'Overall':<25} {overall_field:>11.1%} {overall_type:>11.1%} {total_samples:>10}")
    print()

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                'summary': {
                    'total_field_correct': total_field_correct,
                    'total_type_correct': total_type_correct,
                    'total_samples': total_samples,
                    'overall_field_accuracy': overall_field,
                    'overall_type_accuracy': overall_type
                },
                'datasets': all_results
            }, f, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
