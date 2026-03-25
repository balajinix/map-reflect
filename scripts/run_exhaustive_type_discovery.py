#!/usr/bin/env python3
"""
Exhaustive type-level discovery for ITOM-Hashicorp mappings.

For each source API type, runs Map-Reflect and MatchMaker to discover
which target API types are predicted as matches. Records all predictions
to build a comprehensive type mapping lookup with method provenance.

This replaces the sample-by-sample approach: instead of evaluating
pre-defined pairs, we let each method freely discover matches.

Usage:
    python scripts/run_exhaustive_type_discovery.py                    # run all
    python scripts/run_exhaustive_type_discovery.py --method mr        # map-reflect only
    python scripts/run_exhaustive_type_discovery.py --method mm        # matchmaker only
    python scripts/run_exhaustive_type_discovery.py --direction itom2hashi  # one direction
    python scripts/run_exhaustive_type_discovery.py --resume           # resume from checkpoint
"""

import json
import os
import sys
import argparse
import time
from pathlib import Path
from collections import defaultdict
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results" / "type_discovery"

sys.path.insert(0, str(BASE_DIR))

# Load env
env_file = BASE_DIR / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

ITOM_PRODUCTS = {"instana", "concert", "turbonomic"}
HASHI_PRODUCTS = {"infragraph", "packer", "aws", "azure", "gcp", "hcp", "hcp_terraform"}


def load_api_lookup():
    path = DATA_DIR / "itom_hashicorp_api_lookup_v2.json"
    if not path.exists():
        path = DATA_DIR / "itom_hashicorp_api_lookup.json"
    with open(path) as f:
        return json.load(f)


def get_types_by_product(api_lookup):
    """Group types by product with field info."""
    types = defaultdict(lambda: defaultdict(list))
    for key, meta in api_lookup.items():
        p = meta.get("product_name", "")
        t = meta.get("type_name", "")
        if p and t:
            types[p][t].append(key)
    return types


def build_discovery_samples(api_lookup, direction="itom2hashi"):
    """Build one sample per source type for type discovery.

    Each sample has the source type's fields as source_schema fields,
    and ALL target product fields available for matching.
    """
    types = get_types_by_product(api_lookup)

    if direction == "itom2hashi":
        src_products = ITOM_PRODUCTS
        tgt_products = HASHI_PRODUCTS
    elif direction == "hashi2itom":
        src_products = HASHI_PRODUCTS
        tgt_products = ITOM_PRODUCTS
    else:
        src_products = ITOM_PRODUCTS | HASHI_PRODUCTS
        tgt_products = ITOM_PRODUCTS | HASHI_PRODUCTS

    # Filter to types with >= 3 fields (skip tiny enum/wrapper types)
    MIN_FIELDS = 3
    samples = []
    sample_id = 0

    for src_prod in sorted(src_products):
        for src_type, src_fields in sorted(types.get(src_prod, {}).items()):
            if len(src_fields) < MIN_FIELDS:
                continue

            # Build mapping: all source fields -> placeholder target
            # (methods will predict actual targets)
            mapping = {}
            for sf in src_fields:
                mapping[sf] = sf  # placeholder, not used for discovery

            for tgt_prod in sorted(tgt_products):
                if tgt_prod == src_prod:
                    continue  # skip same-product

                utterance = f"Map {src_prod} {src_type} to {tgt_prod} schema"

                sample = {
                    "id": f"disc_{sample_id:05d}",
                    "utterance": utterance,
                    "source_schema": src_prod,
                    "target_schema": tgt_prod,
                    "expected_target_type": "",  # unknown - that's what we're discovering
                    "mapping": mapping,
                    "metadata": {
                        "source_type": src_type,
                        "target_type": "",
                        "relationship": "discovery",
                        "description": f"Type discovery: {src_prod}.{src_type} -> {tgt_prod}",
                        "source_entity": src_type,
                        "target_entity": "",
                    },
                }
                samples.append(sample)
                sample_id += 1

    return samples


def run_map_reflect_discovery(samples, api_lookup, model_id, checkpoint_path=None):
    """Run Map-Reflect on discovery samples to find type matches.

    Returns dict of {(src_product, src_type): [(tgt_product, tgt_type, score, method_model)]}
    """
    try:
        from map_reflect.map_reflect_runner import MapReflectRunner
    except ImportError:
        print("  Cannot import MapReflectRunner, trying direct LLM approach...")
        return run_map_reflect_direct(samples, api_lookup, model_id, checkpoint_path)

    discoveries = defaultdict(list)
    # ... would need to adapt MapReflectRunner for discovery mode
    return discoveries


def run_map_reflect_direct(samples, api_lookup, model_id, checkpoint_path=None):
    """Run Map-Reflect using the existing runner on the dataset."""
    import subprocess
    import tempfile

    # Write samples as a temporary dataset
    dataset_path = DATA_DIR / "itom_hashicorp_discovery_dataset.jsonl"
    with open(dataset_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    lookup_path = DATA_DIR / "itom_hashicorp_api_lookup_v2.json"
    if not lookup_path.exists():
        lookup_path = DATA_DIR / "itom_hashicorp_api_lookup.json"

    print(f"  Running Map-Reflect on {len(samples)} samples with {model_id}...")
    print(f"  Dataset: {dataset_path}")

    cmd = [
        sys.executable, "-m", "map_reflect.map_reflect_runner",
        "--datasets", "ITOM_Hashicorp_Discovery",
        "--models", model_id,
        "--property-lookup-file", str(lookup_path),
        "--data-dir", str(DATA_DIR),
        "--batch-size", "20",
        "--checkpoint-interval", "10",
        "--resume",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BASE_DIR))
    if result.returncode != 0:
        print(f"  Map-Reflect failed: {result.stderr[-500:]}")
    else:
        print(f"  Map-Reflect completed")

    return {}


def extract_discoveries_from_results(result_dir, api_lookup):
    """Extract type-level discoveries from method result files.

    Scans all result files and extracts which (source_type, target_type)
    pairs each method predicted.
    """
    types_by_product = get_types_by_product(api_lookup)

    # Build field -> (product, type) lookup
    field_to_type = {}
    for key, meta in api_lookup.items():
        p = meta.get("product_name", "")
        t = meta.get("type_name", "")
        if p and t:
            field_to_type[key] = (p, t)

    discoveries = defaultdict(lambda: defaultdict(set))  # (src_p, src_t) -> (tgt_p, tgt_t) -> {methods}

    for result_file in sorted(result_dir.glob("*.json")):
        if "checkpoint" in result_file.name:
            continue

        try:
            with open(result_file) as f:
                data = json.load(f)
        except Exception:
            continue

        results = data.get("results", []) if isinstance(data, dict) else data
        if not isinstance(results, list):
            continue

        # Determine method name from file
        method = "Unknown"
        fname = result_file.name.lower()
        if "mapreflect" in fname or "map_reflect" in fname:
            method = "Map-Reflect"
        elif "matchmaker" in fname or "match_maker" in fname:
            method = "MatchMaker"
        elif "react" in fname:
            method = "ReAct"
        elif "rematch" in fname:
            method = "ReMatch"
        elif "smt" in fname:
            method = "SMT"
        elif "anytool" in fname:
            method = "AnyTool"
        elif "gnn" in fname:
            method = "GNN"

        # Determine model from file
        model = "Unknown"
        if "llama" in fname:
            model = "Llama"
        elif "granite" in fname:
            model = "Granite"
        elif "mistral" in fname:
            model = "Mistral"
        elif method == "GNN":
            model = "(no LLM)"

        method_model = f"{method} ({model})"

        for result in results:
            # Handle predicted_mapping format
            predicted = result.get("predicted_mapping", {})
            if predicted:
                for src_field, tgt_info in predicted.items():
                    if isinstance(tgt_info, dict):
                        tgt_field = tgt_info.get("target_property", "")
                    else:
                        tgt_field = str(tgt_info)
                    if not tgt_field:
                        continue

                    src_pt = field_to_type.get(src_field)
                    tgt_pt = field_to_type.get(tgt_field)
                    if src_pt and tgt_pt and src_pt != tgt_pt:
                        discoveries[src_pt][tgt_pt].add(method_model)

            # Handle predictions format (Map-Reflect)
            predictions = result.get("predictions", {})
            if predictions and not predicted and isinstance(predictions, dict):
                for src_field, pred_data in predictions.items():
                    if not isinstance(pred_data, dict):
                        continue
                    ranking = pred_data.get("initial_ranking", [])
                    if ranking and isinstance(ranking[0], dict):
                        tgt_field = ranking[0].get("id", "")
                        if tgt_field and tgt_field != "NO_MATCH":
                            src_pt = field_to_type.get(src_field)
                            tgt_pt = field_to_type.get(tgt_field)
                            if src_pt and tgt_pt and src_pt != tgt_pt:
                                discoveries[src_pt][tgt_pt].add(method_model)

    return discoveries


def build_lookup_from_discoveries(discoveries):
    """Build the type mapping lookup from discoveries."""
    lookup = defaultdict(lambda: defaultdict(list))

    for (src_p, src_t), targets in discoveries.items():
        src_key = f"{src_p}_{src_t}"
        for (tgt_p, tgt_t), methods in targets.items():
            tgt_key = f"{tgt_p}_{tgt_t}"
            for m in sorted(methods):
                if m not in lookup[src_key][tgt_key]:
                    lookup[src_key][tgt_key].append(m)

    # Also add bare type names for backwards compatibility
    for (src_p, src_t), targets in discoveries.items():
        for (tgt_p, tgt_t), methods in targets.items():
            for m in sorted(methods):
                if m not in lookup[src_t][tgt_t]:
                    lookup[src_t][tgt_t].append(m)

    return {k: dict(v) for k, v in sorted(lookup.items())}


def main():
    parser = argparse.ArgumentParser(description="Exhaustive type discovery for ITOM-Hashicorp")
    parser.add_argument("--method", choices=["mr", "mm", "all"], default="all")
    parser.add_argument("--direction", choices=["itom2hashi", "hashi2itom", "both"], default="itom2hashi")
    parser.add_argument("--model", default="watsonx/meta-llama/llama-3-3-70b-instruct")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--extract-only", action="store_true",
                        help="Only extract discoveries from existing result files")
    args = parser.parse_args()

    print("=" * 60)
    print("ITOM-Hashicorp Exhaustive Type Discovery")
    print("=" * 60)

    api_lookup = load_api_lookup()
    print(f"API lookup: {len(api_lookup)} fields")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.extract_only:
        print("\nExtracting discoveries from existing result files...")

        # Scan all result directories
        all_discoveries = defaultdict(lambda: defaultdict(set))

        for result_dir in [
            BASE_DIR / "map_reflect" / "results",
            BASE_DIR / "match_maker" / "results",
            BASE_DIR / "react" / "results",
            BASE_DIR / "rematch" / "results",
            BASE_DIR / "smt" / "results",
            BASE_DIR / "anytool" / "results",
            BASE_DIR / "gnn" / "results",
            BASE_DIR / "results",
            BASE_DIR / "remote_results" / "map_reflect",
            BASE_DIR / "remote_results" / "match_maker",
            BASE_DIR / "remote_results" / "react",
            BASE_DIR / "remote_results" / "rematch",
            BASE_DIR / "remote_results" / "smt",
            BASE_DIR / "remote_results" / "anytool",
            BASE_DIR / "remote_results" / "gnn",
        ]:
            if result_dir.exists():
                print(f"  Scanning {result_dir}...")
                disc = extract_discoveries_from_results(result_dir, api_lookup)
                for src, targets in disc.items():
                    for tgt, methods in targets.items():
                        all_discoveries[src][tgt].update(methods)

        # Build and save lookup
        lookup = build_lookup_from_discoveries(all_discoveries)

        # Merge with existing lookup
        existing_path = DATA_DIR / "itopsgraph_type_mapping_lookup.json"
        if existing_path.exists():
            with open(existing_path) as f:
                existing = json.load(f)
            for src, targets in existing.items():
                for tgt, methods in targets.items():
                    if src not in lookup:
                        lookup[src] = {}
                    if tgt not in lookup[src]:
                        lookup[src][tgt] = []
                    for m in methods:
                        if m not in lookup[src][tgt]:
                            lookup[src][tgt].append(m)

        out_path = DATA_DIR / "itopsgraph_type_mapping_lookup.json"
        with open(out_path, "w") as f:
            json.dump(lookup, f, indent=2)

        # Stats
        total_pairs = sum(len(v) for v in lookup.values())
        all_methods = set()
        for targets in lookup.values():
            for methods in targets.values():
                all_methods.update(methods)

        # Count cross-product discoveries
        cross = 0
        for src, targets in all_discoveries.items():
            src_p = src[0]
            for tgt, methods in targets.items():
                tgt_p = tgt[0]
                is_cross = (
                    (src_p in ITOM_PRODUCTS and tgt_p in HASHI_PRODUCTS) or
                    (src_p in HASHI_PRODUCTS and tgt_p in ITOM_PRODUCTS)
                )
                if is_cross:
                    cross += 1

        print(f"\n{'='*60}")
        print(f"Saved: {out_path}")
        print(f"Total source types: {len(lookup)}")
        print(f"Total type mappings: {total_pairs}")
        print(f"Cross-product discoveries: {cross}")
        print(f"Methods found: {sorted(all_methods)}")
        return

    # Generate discovery samples
    print(f"\nGenerating discovery samples (direction: {args.direction})...")
    samples = build_discovery_samples(api_lookup, args.direction)
    print(f"  {len(samples)} discovery samples")

    if len(samples) > 5000:
        print(f"  WARNING: {len(samples)} samples is very large.")
        print(f"  Consider filtering to specific products or types.")
        print(f"  Use --direction itom2hashi to limit scope.")

    # Save the discovery dataset
    dataset_path = DATA_DIR / "itom_hashicorp_discovery_dataset.jsonl"
    with open(dataset_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"  Saved: {dataset_path}")

    # Run methods
    if args.method in ("mr", "all"):
        run_map_reflect_direct(samples, api_lookup, args.model)

    if args.method in ("mm", "all"):
        print(f"\n  Running MatchMaker on {len(samples)} samples...")
        import subprocess
        cmd = [
            sys.executable, "-m", "match_maker.match_maker_runner",
            "--datasets", "ITOM_Hashicorp_Discovery",
            "--models", args.model,
            "--property-lookup-file", str(DATA_DIR / "itom_hashicorp_api_lookup_v2.json"),
            "--data-dir", str(DATA_DIR),
            "--top-k", "5",
            "--abstention-mode", "lenient",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BASE_DIR))
        if result.returncode != 0:
            print(f"  MatchMaker failed: {result.stderr[-500:]}")
        else:
            print(f"  MatchMaker completed")

    print("\nDone. Run with --extract-only to build the lookup from results.")


if __name__ == "__main__":
    main()
