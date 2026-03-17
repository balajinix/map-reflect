#!/usr/bin/env python3
"""
Evaluate GNN on SchemaNet datasets with Type-Level Accuracy.
Produces metrics comparable to Map-Reflect and MatchMaker baselines.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np

# Import GNN model
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
    # Load model
    model = load_model_safely(model_file)

    # Load graph data
    with open(graph_data_file, 'rb') as f:
        graph_data = pickle.load(f)

    return model, graph_data


def build_type_to_fields(graph, node_list, target_schema):
    """Build mapping from target types to their field node indices."""
    type_to_fields = defaultdict(list)
    node_to_idx = {node: i for i, node in enumerate(node_list)}

    for node in node_list:
        node_data = graph.nodes[node]
        if node_data.get('node_type') != 'field':
            continue
        if node_data.get('product') != target_schema:
            continue

        type_name = node_data.get('type', '')
        full_type = f"{target_schema}_{type_name}"

        if node in node_to_idx:
            type_to_fields[full_type].append(node_to_idx[node])

    return type_to_fields


def get_source_field_indices(graph, node_list, source_type, source_schema):
    """Get node indices for all fields in a source type."""
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    type_name = source_type.replace(f"{source_schema}_", "")

    field_indices = []
    for node in node_list:
        node_data = graph.nodes[node]
        if node_data.get('node_type') != 'field':
            continue
        if node_data.get('product') != source_schema:
            continue
        if node_data.get('type') != type_name:
            continue

        if node in node_to_idx:
            field_indices.append(node_to_idx[node])

    return field_indices


def predict_target_type(model, data, graph, node_list, source_type, source_schema, target_schema, type_to_fields):
    """Use GNN to predict the best target type for a source type."""
    model.eval()

    source_field_indices = get_source_field_indices(graph, node_list, source_type, source_schema)

    if not source_field_indices:
        return None, 0.0

    type_scores = {}

    with torch.no_grad():
        for target_type, target_field_indices in type_to_fields.items():
            if not target_field_indices:
                continue

            # Create all possible source-target field pairs
            edges = []
            for src_idx in source_field_indices:
                for tgt_idx in target_field_indices:
                    edges.append([src_idx, tgt_idx])

            if not edges:
                continue

            query_edges = torch.tensor(edges).t()

            try:
                probs = model.predict_links(data.x, data.train_pos_edge_index, query_edges)
                # Aggregate: mean of top-k scores per source field
                if len(probs) > 0:
                    # Average the max score for each source field
                    probs_matrix = probs.view(len(source_field_indices), len(target_field_indices))
                    max_per_source = probs_matrix.max(dim=1)[0]
                    score = max_per_source.mean().item()
                    type_scores[target_type] = score
            except Exception as e:
                continue

    if not type_scores:
        return None, 0.0

    best_type = max(type_scores, key=type_scores.get)
    best_score = type_scores[best_type]

    return best_type, best_score


def evaluate_dataset(dataset_name, schemanet_dir, gnn_results_dir, data_dir):
    """Evaluate GNN on a single SchemaNet dataset."""
    # File paths
    dataset_file = Path(schemanet_dir) / f"{dataset_name}_dataset.jsonl"
    model_file = Path(gnn_results_dir) / f"{dataset_name}_mapping_gnn_results.pkl"
    graph_data_file = Path(data_dir) / f"{dataset_name}_graph_data.pkl"

    # Check files exist
    if not dataset_file.exists():
        print(f"  Dataset not found: {dataset_file}")
        return None
    if not model_file.exists():
        print(f"  Model not found: {model_file}")
        return None
    if not graph_data_file.exists():
        print(f"  Graph data not found: {graph_data_file}")
        return None

    # Load data
    samples = load_dataset(dataset_file)
    model, graph_data = load_gnn_model_and_graph(model_file, graph_data_file)

    data = graph_data['data']
    node_list = graph_data['node_list']
    graph = graph_data['graph']

    # Determine target schema from first sample
    target_schema = samples[0].get('target_schema')

    # Build type-to-fields mapping for target schema
    type_to_fields = build_type_to_fields(graph, node_list, target_schema)

    # Evaluate each sample
    correct = 0
    total = 0
    results = []

    for sample in samples:
        source_type = sample.get('source_type')
        source_schema = sample.get('source_schema')
        expected_target = sample.get('expected_target_type')

        predicted_type, score = predict_target_type(
            model, data, graph, node_list,
            source_type, source_schema, target_schema, type_to_fields
        )

        is_correct = (predicted_type == expected_target)
        if is_correct:
            correct += 1
        total += 1

        results.append({
            'id': sample.get('id'),
            'source_type': source_type,
            'expected': expected_target,
            'predicted': predicted_type,
            'score': score,
            'correct': is_correct
        })

    accuracy = correct / total if total > 0 else 0.0

    return {
        'dataset': dataset_name,
        'correct': correct,
        'total': total,
        'accuracy': accuracy,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate GNN on SchemaNet datasets')
    parser.add_argument('--schemanet-dir', type=str, default='../data/schemanet',
                        help='Directory containing SchemaNet data')
    parser.add_argument('--gnn-results-dir', type=str, default='../data/gnn_results',
                        help='Directory containing trained GNN models')
    parser.add_argument('--data-dir', type=str, default='../data',
                        help='Directory containing graph data files')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for detailed results (JSON)')

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

    print("=" * 60)
    print("GNN Type-Level Accuracy on SchemaNet")
    print("=" * 60)
    print()

    all_results = []
    total_correct = 0
    total_samples = 0

    for dataset in DATASETS:
        print(f"Evaluating {dataset}...")
        result = evaluate_dataset(
            dataset, args.schemanet_dir, args.gnn_results_dir, args.data_dir
        )

        if result:
            all_results.append(result)
            total_correct += result['correct']
            total_samples += result['total']
            print(f"  Accuracy: {result['accuracy']:.1%} ({result['correct']}/{result['total']})")
        else:
            print(f"  Skipped (missing files)")
        print()

    # Summary table
    print("=" * 60)
    print("Summary: Type-Level Accuracy")
    print("=" * 60)
    print(f"{'Dataset':<30} {'Accuracy':>10} {'Correct':>10}")
    print("-" * 60)

    for result in all_results:
        name = result['dataset'].replace('schemanet_', '')
        print(f"{name:<30} {result['accuracy']:>9.1%} {result['correct']:>5}/{result['total']:<4}")

    print("-" * 60)
    overall_acc = total_correct / total_samples if total_samples > 0 else 0.0
    print(f"{'Overall':<30} {overall_acc:>9.1%} {total_correct:>5}/{total_samples:<4}")
    print()

    # Save detailed results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                'summary': {
                    'total_correct': total_correct,
                    'total_samples': total_samples,
                    'overall_accuracy': overall_acc
                },
                'datasets': all_results
            }, f, indent=2)
        print(f"Detailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
