#!/usr/bin/env python3
"""
Merge ground truth from dataset files into result files.
This allows proper evaluation of existing result files.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List

def load_ground_truth(dataset_file: Path) -> Dict[str, Dict[str, Any]]:
    """Load ground truth mappings from dataset JSONL file."""
    ground_truth = {}

    with open(dataset_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            sample_id = data.get("id", "")
            mapping = data.get("mapping", {})

            if sample_id and mapping:
                ground_truth[sample_id] = mapping

    return ground_truth

def merge_ground_truth_into_results(result_file: Path, ground_truth: Dict[str, Dict[str, Any]]) -> int:
    """Merge ground truth into result file and return count of updated samples."""
    with open(result_file, 'r') as f:
        data = json.load(f)

    results = data.get("results", [])
    updated_count = 0

    for result in results:
        sample_id = result.get("sample_id", "")

        if sample_id in ground_truth:
            result["ground_truth_mapping"] = ground_truth[sample_id]
            updated_count += 1
        elif not result.get("ground_truth_mapping"):
            result["ground_truth_mapping"] = {}

    # Save updated file
    with open(result_file, 'w') as f:
        json.dump(data, f, indent=2)

    return updated_count

def main():
    if len(sys.argv) < 2:
        print("Usage: python merge_ground_truth.py <dataset_name>")
        print("Example: python merge_ground_truth.py OMOP_MIMIC")
        print("\nAvailable datasets: OMOP_MIMIC, OMOP_CMS, OMOP_Synthea")
        sys.exit(1)

    dataset_name = sys.argv[1]

    # Define paths
    data_dir = Path(__file__).parent
    anytool_results_dir = data_dir.parent / "anytool" / "results"

    # Dataset file mapping
    dataset_files = {
        "OMOP_MIMIC": "omop_mimic_dataset.jsonl",
        "OMOP_CMS": "omop_cms_dataset.jsonl",
        "OMOP_Synthea": "omop_synthea_dataset.jsonl"
    }

    if dataset_name not in dataset_files:
        print(f"Unknown dataset: {dataset_name}")
        print(f"Available: {list(dataset_files.keys())}")
        sys.exit(1)

    # Load ground truth
    dataset_file = data_dir / dataset_files[dataset_name]
    if not dataset_file.exists():
        print(f"Dataset file not found: {dataset_file}")
        sys.exit(1)

    print(f"Loading ground truth from: {dataset_file}")
    ground_truth = load_ground_truth(dataset_file)
    print(f"Loaded {len(ground_truth)} ground truth mappings")

    # Find and update result files in multiple directories
    result_dirs = [
        data_dir.parent / "anytool" / "results",
        data_dir.parent / "gnn" / "results",
        data_dir.parent / "map_reflect" / "results",
        data_dir.parent / "smt" / "results",
        data_dir.parent / "react" / "results"
    ]

    total_files = 0
    for results_dir in result_dirs:
        if not results_dir.exists():
            continue

        result_files = list(results_dir.glob(f"{dataset_name}*.json"))
        result_files = [f for f in result_files if "checkpoint" not in f.name]

        for result_file in result_files:
            print(f"\nProcessing: {results_dir.name}/{result_file.name}")
            updated_count = merge_ground_truth_into_results(result_file, ground_truth)
            print(f"  Updated {updated_count} samples with ground truth")
            total_files += 1

    if total_files == 0:
        print(f"No result files found for {dataset_name}")
        sys.exit(1)

if __name__ == "__main__":
    main()
