#!/usr/bin/env python3
"""
Run missing samples and merge with existing results.

Usage:
    python scripts/run_and_merge_missing.py \
        --existing-results results/ablation/uq/mapreflect_itopsgraph_meta-llama_llama-3-3-70b-instruct.json \
        --missing-samples data/missing_samples.jsonl \
        --output-dir results/uq_method_comparison/sampling \
        --uq-method sampling
"""

import json
import subprocess
import argparse
import sys
from pathlib import Path
from datetime import datetime


def run_missing_samples(missing_file: Path, output_dir: Path, uq_method: str, n_samples: int = 3):
    """Run the runner on missing samples."""
    cmd = [
        "python", "-m", "map_reflect.map_reflect_runner",
        "--datasets", "ITOpsGraph",
        "--models", "watsonx/meta-llama/llama-3-3-70b-instruct",
        "--property-lookup-file", "data/itopsgraph_api_lookup.json",
        "--data-dir", "data",
        "--neighbor-k", "0",
        "--use-uncertainty",
        "--uq-method", uq_method,
        "--n-uncertainty-samples", str(n_samples),
        "--eval-file", str(missing_file),
        "--output-dir", str(output_dir)
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def merge_results(existing_file: Path, new_file: Path, output_file: Path):
    """Merge existing results with new results."""
    with open(existing_file) as f:
        existing = json.load(f)

    with open(new_file) as f:
        new = json.load(f)

    # Get existing sample IDs
    existing_ids = set(r['sample_id'] for r in existing['results'])

    # Add new results
    added = 0
    for result in new['results']:
        if result['sample_id'] not in existing_ids:
            existing['results'].append(result)
            added += 1

    # Sort by sample_id
    existing['results'].sort(key=lambda x: x['sample_id'])

    # Update metrics
    total_samples = len(existing['results'])
    total_fields = sum(len(r.get('predictions', {})) for r in existing['results'])
    correct_predictions = sum(r.get('correct_count', 0) for r in existing['results'])

    existing['metrics']['total_samples'] = total_samples
    existing['metrics']['total_fields'] = total_fields
    existing['metrics']['correct_predictions'] = correct_predictions
    existing['metrics']['accuracy'] = correct_predictions / total_fields if total_fields > 0 else 0

    # Recalculate mean uncertainty
    all_uncertainties = []
    for result in existing['results']:
        for pred in result.get('predictions', {}).values():
            if 'uncertainty' in pred:
                all_uncertainties.append(pred['uncertainty'])

    if all_uncertainties:
        import numpy as np
        existing['metrics']['mean_uncertainty'] = float(np.mean(all_uncertainties))

    # Update timestamp
    existing['timestamp'] = datetime.now().isoformat()

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(existing, f, indent=2)

    print(f"Merged {added} new results")
    print(f"Total samples: {total_samples}")
    print(f"Total fields: {total_fields}")
    print(f"Accuracy: {existing['metrics']['accuracy']*100:.1f}%")

    return output_file


def main():
    parser = argparse.ArgumentParser(description="Run missing samples and merge results")
    parser.add_argument("--existing-results", type=Path, required=True, help="Existing results JSON file")
    parser.add_argument("--missing-samples", type=Path, required=True, help="Missing samples JSONL file")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--uq-method", type=str, default="sampling", help="UQ method")
    parser.add_argument("--n-uncertainty-samples", type=int, default=3, help="Number of UQ samples")
    parser.add_argument("--skip-run", action="store_true", help="Skip running, just merge")
    args = parser.parse_args()

    temp_output_dir = args.output_dir / "temp_missing"
    temp_output_dir.mkdir(parents=True, exist_ok=True)

    # Run missing samples
    if not args.skip_run:
        print(f"Running missing samples from {args.missing_samples}...")
        success = run_missing_samples(
            args.missing_samples,
            temp_output_dir,
            args.uq_method,
            args.n_uncertainty_samples
        )
        if not success:
            print("Error running missing samples")
            sys.exit(1)

    # Find the new results file
    new_results = list(temp_output_dir.glob("*.json"))
    if not new_results:
        print("No new results found")
        sys.exit(1)

    new_results_file = new_results[0]
    print(f"New results file: {new_results_file}")

    # Merge results
    output_file = args.output_dir / args.existing_results.name
    merge_results(args.existing_results, new_results_file, output_file)

    print(f"\nMerged results saved to: {output_file}")


if __name__ == "__main__":
    main()
